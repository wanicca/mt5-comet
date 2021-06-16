# check head-tail pairs that have more than one relation
#%%
from datasets import load_dataset
#%%
atomic_dataset = load_dataset("atomic")
# %%
atomic_relations = [
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xWant"
]
overlapped = {} # (head,tail)->relations
for split_name,split_data in atomic_dataset.items():
    for d in split_data:
        h = d['event']
        for rel in atomic_relations:
            for t in d[rel]:
                if t.lower()!='none':
                    if not (h,t) in overlapped:
                        overlapped[(h,t)] = set()
                    overlapped[(h,t)].add(rel)
        # tail_sets = {rel: set(t for t in d[rel] if t.lower()!='none') for rel in atomic_relation_mappings}
        # for i in range(len(atomic_relations)):
        #     for j in range(i+1,len(atomic_relations)):
        #         if ((tail_sets[atomic_relations[i]] &  tail_sets[atomic_relations[j]])>0):
        #             for tail in tail_sets
# %%
overlapped = {k:v for k,v in overlapped.items() if len(v)>1}
# %%
count_overlapped = {tuple(sorted(v)):0 for k,v in overlapped.items()}
for k,v in overlapped.items():
    key = tuple(sorted(v))
    count_overlapped[key]+=1

count_triples = 0
for split_name,split_data in atomic_dataset.items():
    for d in split_data:
        h = d['event']
        for rel in atomic_relations:
            for t in d[rel]:
                if (h,t) in overlapped:
                    count_triples+=1
"""
count_overlapped
{('xAttr', 'xReact'): 5160,
 ('oWant', 'xWant'): 3115,
 ('oReact', 'xReact'): 4383,
 ('xAttr', 'xIntent'): 123,
 ('xNeed', 'xWant'): 190,
 ('oReact', 'xAttr', 'xReact'): 409,
 ('oEffect', 'xEffect'): 1351,
 ('xIntent', 'xReact'): 132,
 ('xEffect', 'xWant'): 36,
 ('xIntent', 'xWant'): 758,
 ('oReact', 'xIntent', 'xReact'): 30,
 ('oWant', 'xNeed'): 80,
 ('xIntent', 'xNeed'): 53,
 ('oReact', 'xAttr'): 444,
 ('xEffect', 'xReact'): 52,
 ('oEffect', 'oReact'): 19,
 ('xAttr', 'xEffect'): 21,
 ('oWant', 'xAttr'): 2,
 ('oEffect', 'xAttr'): 3,
 ('oReact', 'xIntent'): 25,
 ('xAttr', 'xNeed'): 4,
 ('xAttr', 'xIntent', 'xReact'): 40,
 ('xAttr', 'xNeed', 'xWant'): 1,
 ('xAttr', 'xEffect', 'xIntent', 'xReact'): 1,
 ('xAttr', 'xEffect', 'xReact'): 13,
 ('oWant', 'xIntent'): 46,
 ('oReact', 'oWant', 'xReact'): 6,
 ('oEffect', 'oWant'): 9,
 ('xIntent', 'xNeed', 'xWant'): 4,
 ('oWant', 'xIntent', 'xWant'): 73,
 ('oEffect', 'xNeed'): 4,
 ('oReact', 'xAttr', 'xIntent'): 7,
 ('oWant', 'xReact'): 11,
 ('oReact', 'oWant'): 6,
 ('oEffect', 'xReact'): 7,
 ('oReact', 'xEffect', 'xReact'): 6,
 ('xReact', 'xWant'): 9,
 ('xEffect', 'xIntent'): 7,
 ('oReact', 'xReact', 'xWant'): 5,
 ('oWant', 'xNeed', 'xWant'): 5,
 ('oEffect', 'oReact', 'xEffect', 'xReact'): 1,
 ('oEffect', 'oReact', 'xReact'): 1,
 ('oEffect', 'oWant', 'xEffect'): 1,
 ('oReact', 'xWant'): 2,
 ('oEffect', 'xWant'): 5,
 ('xNeed', 'xReact'): 3,
 ('xEffect', 'xNeed'): 5,
 ('oEffect', 'xIntent'): 4,
 ('oWant', 'xEffect'): 3,
 ('oReact', 'xEffect'): 3,
 ('oEffect', 'oWant', 'xWant'): 1,
 ('oWant', 'xEffect', 'xWant'): 1,
 ('oEffect', 'xEffect', 'xWant'): 1,
 ('oReact', 'xAttr', 'xEffect', 'xReact'): 1,
 ('xAttr', 'xWant'): 4,
 ('oWant', 'xIntent', 'xNeed'): 1,
 ('oReact', 'xAttr', 'xIntent', 'xReact'): 2,
 ('oEffect', 'oReact', 'xAttr', 'xReact'): 1}

len(overlapped)
16690

count_triples
37781
"""
# %%
