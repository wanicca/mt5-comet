#%% load libraries

from datasets import load_dataset,load_metric
from transformers import (
    MT5ForConditionalGeneration, MT5TokenizerFast, AdamW, AddedToken,
    get_linear_schedule_with_warmup
)
import torch
from torch.utils.tensorboard import SummaryWriter
import os,time
import argparse
from tqdm.auto import tqdm
#%% argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=None, type=int,
                    help='node rank for distributed training')
args = parser.parse_args()
local_rank = args.local_rank
#%% some hyper-parameters
underlying_model_name = "google/mt5-small"
learning_rate = 6.25e-05
iterations = 50000
cycle = 500
warm_up_steps = 0.002*iterations
weight_decay = 0.01
batch_size = 64
shuffle = True
shuffle_evaluation=False
validation_size = 10000
validation_num_generation = 10
generation_params = {
    "max_length":50,
    "early_stopping":True
}
ddp = args.local_rank is not None
device = 'cuda'
log_dir = 'logs/'
#%% load atomic data
atomic_dataset = load_dataset('atomic')
atomic_relation_mappings = {
    "oEffect":"<oEffect>",
    "oReact":"<oReact>",
    "oWant":"<oWant>",
    "xAttr":"<xAttr>",
    "xEffect":"<xEffect>",
    "xIntent":"<xIntent>",
    "xNeed":"<xNeed>",
    "xReact":"<xReact>",
    "xWant":"<xWant>"
}
gen_token = "<gen>"
#%% dpp initialize
is_main_process = (not ddp or local_rank == 0) 
if ddp:
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    print("launch process",local_rank)
    world_size = torch.distributed.get_world_size()
#%% Aggregate instances of queries and corresponding responses
# (str)split_name -> (dict) query -> (list) response 
print("building query responses")
atomic_query_responses = {}
for split_name,split_data in atomic_dataset.items():
    atomic_query_responses[split_name] = {}
    for d in split_data:
        for rel in atomic_relation_mappings:
            if len(d[rel])>0: 
                rel_token = atomic_relation_mappings[rel]
                query = f"{d['event']} {rel_token} {gen_token}"
                if query not in atomic_query_responses[split_name]:
                    atomic_query_responses[split_name][query] = []
                atomic_query_responses[split_name][query].extend(d[rel])
                #didn't convert ___ to <blank>
                #didn't normalize to lowercase

#flatten
print("building flattened pairs")
atomic_flattened = {}
for split_name,queries_responses in atomic_query_responses.items():
    atomic_flattened[split_name] = []
    for query,responses in queries_responses.items():
        for response in responses:
            atomic_flattened[split_name].append((query,response))
#%% tokenizer & model
tokenizer = MT5TokenizerFast.from_pretrained(underlying_model_name)
model = MT5ForConditionalGeneration.from_pretrained(underlying_model_name)
# add new tokens
# added_tokens = list(atomic_relation_mappings.values()) + [gen_token]
added_tokens = [ 
    AddedToken(token,lstrip=True,
        rstrip=False)
    for token in 
        list(atomic_relation_mappings.values())+
        [gen_token]
]
tokenizer.add_special_tokens({"additional_special_tokens":added_tokens})
model.resize_token_embeddings(len(tokenizer))
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
node_batch_size = batch_size
train_sampler = None
if shuffle:
    train_sampler = torch.utils.data.RandomSampler(atomic_flattened['train'])
if ddp:
    assert batch_size%world_size == 0
    node_batch_size = batch_size//world_size
    train_sampler = torch.utils.data.DistributedSampler(atomic_flattened['train'],shuffle=shuffle)
train_dataloader = torch.utils.data.DataLoader(atomic_flattened['train'],
    batch_size=node_batch_size,sampler=train_sampler,
    collate_fn=collate_fn_for_flattened)
if is_main_process:
    dev_dataloader = torch.utils.data.DataLoader(atomic_flattened['validation'],
        batch_size=node_batch_size,shuffle=shuffle_evaluation,
        collate_fn=collate_fn_for_flattened)

# %% prepare for training
model_name = os.path.join("atomic-mt5",f"{learning_rate}_{cycle}_{iterations}_"
    f"{time.strftime('%Y%m%d %a %H:%M:%S')}")
if is_main_process:
    sw = SummaryWriter(os.path.join(log_dir,model_name))
    serialization_dir = os.path.join(log_dir,model_name)
    tokenizer.save_pretrained(serialization_dir)
model = model.to(device=device)
model_ = model #for generation
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,lr=learning_rate,eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer,warm_up_steps,iterations)
step = 0
best_dev_loss = 1e10

#%%
train_iter = iter(train_dataloader)
if is_main_process:
    pbar = tqdm(total=iterations,dynamic_ncols=True)
while step <= iterations:
    if is_main_process and (step % cycle == 0 and step > 0): #validation
        with torch.no_grad():
            model.eval()
            pbar.set_description('validating...')
            dev_allset_micro_loss = 0.
            dev_token_loss = 0.
            dev_token_count = 0
            dev_sample_loss = 0. #avg on sample
            dev_sample_count = 0
            for batch in tqdm(dev_dataloader,desc=f'validating ...',leave=False):
                if dev_sample_count>=validation_size:
                    break
                batch = {k:v.to(device=device) for k,v in batch.items()}
                result = model(**batch)
                loss = torch.nn.functional.cross_entropy(
                    result['logits'].reshape(-1,result['logits'].size(2)),
                    batch['labels'].reshape(-1,),
                    reduction='none'
                ).reshape(result['logits'].size(0),-1)
                labels_mask = (batch['labels'] != -100) 
                dev_token_loss += loss.sum().item()
                dev_token_count += labels_mask.sum().item()
                dev_sample_loss += (loss.sum(dim=-1)/labels_mask.sum(dim=-1)).sum().item()
                dev_sample_count += result['logits'].size(0)
                del result
                del loss
                del labels_mask
            dev_micro_avg_loss = dev_token_loss/dev_token_count
            dev_macro_avg_loss = dev_sample_loss/dev_sample_count
            sw.add_scalar('dev/micro_avg_loss',dev_micro_avg_loss,step)
            sw.add_scalar('dev/macro_avg_loss',dev_macro_avg_loss,step)
            if dev_micro_avg_loss < best_dev_loss:
                best_dev_loss = dev_micro_avg_loss
                model_.save_pretrained(serialization_dir)
            generation_results = \
            "|Queries|Generation Results|\n"\
            "|-|-|\n"
            for i,key in enumerate(tqdm(atomic_query_responses['validation'])):
                if i==validation_num_generation:
                    break
                results = tokenizer.batch_decode(
                    model_.generate(**tokenizer(key,return_tensors='pt').to(device=device),**generation_params),
                    skip_special_tokens=True
                )
                generation_results+=f"|`{key}`|`{str(results)}`|\n"
            sw.add_text('dev/generation_samples',generation_results,step)
    model.train()
    optimizer.zero_grad()
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dataloader)
        batch = next(train_iter)
    batch = {k:v.to(device=device) for k,v in batch.items()}
    result = model(**batch)
    loss = result['loss']
    loss.backward()
    optimizer.step()
    scheduler.step()
    step+=1
    if ddp:
        loss = loss.detach()
        losses = [torch.zeros_like(loss) for i in range(world_size)]
        torch.distributed.all_gather(tensor_list=losses,tensor=loss)
        loss = torch.stack(losses).mean()
    if is_main_process:
        pbar.set_description('training...')
        pbar.update()
        sw.add_scalar('train/loss',loss.item(),global_step=step)
    del result
    del loss
pbar.close()
# %%
