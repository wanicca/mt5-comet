#%% load libraries

from datasets import load_dataset,load_metric
from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast, AdamW, AddedToken,
    get_linear_schedule_with_warmup
)
from transformers_ptuning import PTuningWrapper
import torch
import csv 
import json
from torch.utils.tensorboard import SummaryWriter
import os,time
import argparse
from tqdm.auto import tqdm
from transformers_ptuning.ptuning_wrapper import LSTMEmbeddingPromptEncoder
#%% argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=None, type=int,
                    help='node rank for distributed training')
args = parser.parse_args([])
local_rank = args.local_rank
#%%
cfg = {}
old_vars = set()
old_vars.update(k for k in globals() if not k.startswith('_'))
#%% some hyper-parameters
underlying_model_name = "t5-base"
learning_rate = 6.25e-05
iterations = 10000
cycle = 1000
warm_up_steps = 0.002*iterations
weight_decay = 0.01
batch_size = 64
#Cut down memory usage by accumulating tiny steps for multiple backwards;
#Should be divided exactly by batch_size
accumulation_tiny_steps = 1 
shuffle = True
shuffle_evaluation=False
validation_size = 10000
validation_num_generation = 10
generation_params = {
    "max_length":50,
    "early_stopping":True,
    "num_beams":5,
    "num_return_sequences":5,
}
ddp = args.local_rank is not None
device = 'cuda'
log_dir = 'logs/'
dataset_path = "data/atomic/v4_atomic_all_agg.csv"
prompt_length = 3
atomic_relation_prompt_lengths = {
    "oEffect":prompt_length,
    "oReact":prompt_length,
    "oWant":prompt_length,
    "xAttr":prompt_length,
    "xEffect":prompt_length,
    "xIntent":prompt_length,
    "xNeed":prompt_length,
    "xReact":prompt_length,
    "xWant":prompt_length
}
#%%
new_vars = set(k for k in globals() if not k.startswith('_'))
cfg_vars = new_vars-old_vars
cfg = {k:v for k,v in globals().items() if k in cfg_vars }
#%% load atomic data
def load_atomic_dataset(path):

    data={}
    with open(path) as source_file:
        source_reader = csv.reader(source_file)
        # Read first line to get column name
        source_line = next(source_reader)
        event_colname = source_line[0]
        categories_colname = source_line[1:10]
        prefix_colname = source_line[10]
        split_colname = source_line[11]
        for source_line in source_reader:
            # get every column
            event = source_line[0]
            annotationss = [
                json.loads(raw_anns) for raw_anns in source_line[1:10]]
            event_prefix = source_line[10]
            event_split = source_line[11]
            if event_split not in data:
                data[event_split] = []
            d = {"event":event}
            d.update({category:annotations for 
                category,annotations in zip(categories_colname,annotationss)})
            data[event_split].append(d)
    return data

# atomic_dataset = load_atomic_dataset(dataset_path)
atomic_dataset = load_dataset("atomic")
placeholder_token = "<extra_id_0>"
#%% dpp initialize
is_main_process = (not ddp or local_rank == 0) 
if ddp:
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    print("launch process",local_rank)
    world_size = torch.distributed.get_world_size()
#%% tokenizer & model
tokenizer = T5TokenizerFast.from_pretrained(underlying_model_name)
model = T5ForConditionalGeneration.from_pretrained(underlying_model_name)
for param in model.parameters():
    param.requires_grad = False
allowed_out_token_length = len(tokenizer)
def clip_logits(logits):
    return logits[:,:,:allowed_out_token_length]
#可以在这里就加钩子，因为模型不会替换模块，只会更新权重
clip_logits_hook = model.get_output_embeddings().register_forward_hook(
    lambda m,i,o:clip_logits(o)
)



# add new tokens
# added_tokens = [ 
#     AddedToken(token,lstrip=True,
#         rstrip=False)
#     for token in 
#         list(atomic_relation_mappings.values())
# ]
# tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
# model.resize_token_embeddings(len(tokenizer))
embedding_dim = model.config.hidden_size
wrapped_models = {}
atomic_relation_mappings = {}
def get_prompt_token_fn(id_offset,length):
    return lambda x: (x>=id_offset)&(x<id_offset+length)
for rel in atomic_relation_prompt_lengths:
    id_offset = len(tokenizer)
    length = atomic_relation_prompt_lengths[rel]
    atomic_relation_mappings[rel] = " ".join(f"<{rel}_{i}>" for i in range(length))
    prompt_encoder = LSTMEmbeddingPromptEncoder(length,embedding_dim,id_offset)
    added_tokens = [ 
        AddedToken(f"<{rel}_{i}>",lstrip=True,
            rstrip=False)
        for i in 
            range(length)
    ]
    tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
    wrapped_models[rel] = PTuningWrapper(model,prompt_encoder,prompt_token_fn=get_prompt_token_fn(id_offset,length))
    
#%% Aggregate instances of queries and corresponding responses
# (str)split_name -> (dict) query -> (list) response 
print("building query responses")
atomic_query_responses = {}
for split_name,split_data in atomic_dataset.items():
    atomic_query_responses[split_name] = {}
    for d in split_data:
        for rel in atomic_relation_mappings:
            if rel not in atomic_query_responses[split_name]:
                atomic_query_responses[split_name][rel] = {}
            if len(d[rel])>0: 
                rel_tokens = atomic_relation_mappings[rel]
                query = f"{d['event']} {rel_tokens} {placeholder_token}"
                if query not in atomic_query_responses[split_name][rel]:
                    atomic_query_responses[split_name][rel][query] = []
                for response in d[rel]:
                    atomic_query_responses[split_name][rel][query].append((f"<extra_id_0> {response}",response))
                #didn't convert ___ to <blank>
                #didn't normalize to lowercase

#flatten
print("building flattened pairs")
atomic_flattened = {}
for split_name,rel_queries_responses in atomic_query_responses.items():
    atomic_flattened[split_name] = {}
    for rel,queries_responses in rel_queries_responses.items():
        atomic_flattened[split_name][rel] = []
        for query,responses in queries_responses.items():
            for response,_ in responses:
                atomic_flattened[split_name][rel].append((query,response))

#%% Prepare training data

def collate_fn_for_flattened(batch):
    queries,responses = zip(*batch)
    new_batch = tokenizer(list(queries),return_tensors='pt',padding='longest')
    with tokenizer.as_target_tokenizer():
        outputs = tokenizer(list(responses),return_tensors='pt',padding='longest')
        labels = outputs['input_ids']
        labels[labels==tokenizer.pad_token_id] = -100
        new_batch['labels']=labels
    return new_batch

# def collate_fn_for_generation(batch):
#     queries,references = zip(*batch)
#     new_batch = tokenizer(queries,return_tensors='pt',padding='longest')
#     return new_batch,references
#%% build dataloader
#%% dataloader and  parallel
node_batch_size = batch_size//accumulation_tiny_steps
train_sampler = {}
train_dataloader = {}
dev_dataloader = {}
for rel in atomic_relation_mappings:
    train_sampler[rel] = None
    if shuffle:
        train_sampler[rel] = torch.utils.data.RandomSampler(atomic_flattened['train'][rel])
    if ddp:
        assert node_batch_size%world_size == 0
        node_batch_size = node_batch_size//world_size
        train_sampler[rel] = torch.utils.data.DistributedSampler(atomic_flattened['train'][rel],shuffle=shuffle)
    train_dataloader[rel] = torch.utils.data.DataLoader(atomic_flattened['train'][rel],
        batch_size=node_batch_size,sampler=train_sampler[rel],
        collate_fn=collate_fn_for_flattened)
    if is_main_process:
        dev_dataloader[rel] = torch.utils.data.DataLoader(atomic_flattened['validation'][rel],
            batch_size=node_batch_size,shuffle=shuffle_evaluation,
            collate_fn=collate_fn_for_flattened)

# %% prepare for training
model_name = os.path.join(f"t5-base",f"prompt_length_{prompt_length}",f"{learning_rate}_{cycle}_{iterations}_"
    f"{time.strftime('%Y%m%d %a %H:%M:%S')}")
if is_main_process:
    sw = SummaryWriter(os.path.join(log_dir,model_name))
    serialization_dir = os.path.join(log_dir,model_name)
    tokenizer.save_pretrained(serialization_dir)
    with open(os.path.join(serialization_dir,'exp_config.json'),'w') as f:
        import json
        json.dump(cfg,f,ensure_ascii=False,indent=4)
for wrapped_model in wrapped_models.values():
    wrapped_model.to(device=device)

if ddp:
    for rel in wrapped_models:
        wrapped_models[rel] = torch.nn.parallel.DistributedDataParallel(wrapped_models[rel],device_ids=[local_rank])
# no_decay = ['bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
#     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]


#%%
for rel,wrapped_model in wrapped_models.items():
    optimizer_grouped_parameters = [
        {"params":[p for p in wrapped_model.parameters() if p.requires_grad]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate,eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,warm_up_steps,iterations)
    step = 0
    best_dev_loss = 1e10
    train_iter = iter(train_dataloader[rel])
    if is_main_process:
        pbar = tqdm(total=iterations,dynamic_ncols=True,desc=rel)
    while step <= iterations:
        if is_main_process and (step % cycle == 0 and step > 0): #validation
            with torch.no_grad():
                if ddp:
                    wrapped_model.module.update_model_weight()
                else:
                    wrapped_model.update_model_weight()
                for p in model.parameters():
                    p.requires_grad = False
                model.eval()
                pbar.set_description(f'validating...{rel}')
                dev_allset_micro_loss = 0.
                dev_token_loss = 0.
                dev_token_count = 0
                dev_sample_loss = 0. #avg on sample
                dev_sample_count = 0
                for batch in tqdm(dev_dataloader[rel],desc=f'validating ...',leave=False):
                    if dev_sample_count>=validation_size:
                        break
                    batch = {k:v.to(device=device) for k,v in batch.items()}
                    result = model(**batch)
                    # logits = clip_logits(result['logits'])
                    logits = result['logits']
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1,logits.size(2)),
                        batch['labels'].reshape(-1,),
                        reduction='none'
                    ).reshape(logits.size(0),-1)
                    labels_mask = (batch['labels'] != -100) 
                    dev_token_loss += loss.sum().item()
                    dev_token_count += labels_mask.sum().item()
                    dev_sample_loss += (loss.sum(dim=-1)/labels_mask.sum(dim=-1)).sum().item()
                    dev_sample_count += logits.size(0)
                    del result
                    del loss
                    del labels_mask
                dev_micro_avg_loss = dev_token_loss/dev_token_count
                dev_macro_avg_loss = dev_sample_loss/dev_sample_count
                sw.add_scalar(f'dev/{rel}/micro_avg_loss',dev_micro_avg_loss,step)
                sw.add_scalar(f'dev/{rel}/macro_avg_loss',dev_macro_avg_loss,step)
                if dev_micro_avg_loss < best_dev_loss:
                    best_dev_loss = dev_micro_avg_loss
                    model.save_pretrained(serialization_dir)
                generation_results = \
                "|Queries|Generation Results|\n"\
                "|-|-|\n"
                for i,key in enumerate(tqdm(atomic_query_responses['validation'][rel])):
                    if i==validation_num_generation:
                        break
                    results = tokenizer.batch_decode(
                        model.generate(**tokenizer(key,return_tensors='pt').to(device=device),**generation_params),
                        skip_special_tokens=True
                    )
                    generation_results+=f"|`{key}`|`{str(results)}`|\n"
                sw.add_text(f'dev/{rel}/generation_samples',generation_results,step)
        model.train()
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.)
        for tiny_step in range(accumulation_tiny_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader[rel])
                batch = next(train_iter)
            batch = {k:v.to(device=device) for k,v in batch.items()}
            result = wrapped_model(**batch)
            loss = result['loss']/accumulation_tiny_steps
            # logits = clip_logits(result['logits'])
            # loss = torch.nn.functional.cross_entropy(
            #     logits.reshape(-1,logits.size(2)),
            #     batch['labels'].reshape(-1,)
            # )/accumulation_tiny_steps
            loss.backward()
            batch_loss += loss.item()
        optimizer.step()
        scheduler.step()
        step+=1
        if ddp:
            # loss = loss.detach()
            losses = [torch.zeros_like(batch_loss) for i in range(world_size)]
            torch.distributed.all_gather(tensor_list=losses,tensor=batch_loss)
            batch_loss = torch.stack(losses).mean()
        if is_main_process:
            pbar.set_description(f'training...{rel}')
            pbar.update()
            sw.add_scalar(f'train/{rel}/loss',batch_loss.item(),global_step=step)
        del result
        del loss
    if is_main_process:
        pbar.close()
# %%
