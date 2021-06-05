#%% load libraries

import json
from datasets import load_dataset,load_metric
from transformers import (
    MT5ForConditionalGeneration, MT5TokenizerFast
)
import os,time
import argparse
from tqdm.auto import tqdm

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--load_path', default=None, type=str,
                    help='path to load saved model')
args = parser.parse_args()
load_path = args.load_path
#%% some hyper-parameters
num_generations = 0 #0 means no limit
generation_params = {
    "max_length":50,
    "early_stopping":True
}
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
# print("building flattened pairs")
# atomic_flattened = {}
# for split_name,queries_responses in atomic_query_responses.items():
#     atomic_flattened[split_name] = []
#     for query,responses in queries_responses.items():
#         for response in responses:
#             atomic_flattened[split_name].append((query,response))
#%% tokenizer & model
tokenizer = MT5TokenizerFast.from_pretrained(load_path)
model = MT5ForConditionalGeneration.from_pretrained(load_path)



#%%
results = []
device = 'cuda:0'
model = model.to(device)
for i,(query,responses) in enumerate(tqdm(atomic_query_responses['test'].items())):
    if num_generations>0 and i>= num_generations:
        break
    inputs = tokenizer(query,return_tensors='pt').to(device=device)
    hyps = model.generate(**inputs,**generation_params)
    hyps = tokenizer.batch_decode(hyps,skip_special_tokens=True)
    results.append({
        "query":query,
        "hyps":hyps,
        "refs":responses
    })
#%%
with open(os.path.join(load_path,"generations.json"),'w') as f:
    json.dump(results,f,ensure_ascii=False,indent=2)
# %%
refs = {r['query']:r['refs'] for r in results}
hyps = {r['query']:r['hyps'] for r in results}
from evaluation.eval import QGEvalCap
QGEval = QGEvalCap("",refs,hyps,os.path.join(load_path,"eval_result.jsonl"))
score_dict, scores_dict = QGEval.evaluate()
# %%
