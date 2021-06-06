# +
import sys
import argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from comet.utils.rwfiles import read_jsonl, remove_prefix
from comet.evaluation.eval import QGEvalCap
from tabulate import tabulate


# -

def get_refs_preds(l, type=1):
    if type==1:
        tails = l["fact"]["tails"]
        head = l["fact"]["head"]
        prompt = l["prompt"]
        generations = l["generations"]
        gens = [remove_prefix(g, prompt).strip() for g in generations]
    if type==2:
        tails = l["tails"]
        head = l["head"]
        gens = l["generations"]
    if type==3:
        tails = l["fact"]["tails"]
        head = l["fact"]["head"]
        gens = l["generations"]
    if type==4:
        tails = [l["target"]]
        head = l["source"]
        gens = l["generations"]

    return gens, tails, head

def get2(l):
    return list(zip(*l))[1]


# +
import sacrebleu

def topk_eval(output_name, data, data_type, k, cbs=False):

    topk_gts = {}
    topk_res = {}
    topk_exact_match = []
    topk_exact_match_not_none = []
    topk_bleu_score = []

    topk_is_head = []

    for i, l in enumerate(data):
        (gens, tails, head) = get_refs_preds(l, type=data_type)
        #print("Gens:", gens)
        #print("Tails:", tails)
        #print("Head:", head)
        new_tails = []
        for t in tails:
            end_index = t.index("[EOS]") if "[EOS]" in t else len(t)
            t = t[:end_index]
            new_tails.append(t)
        tails = new_tails

        sentence_tails = [t.lower().strip() for t in tails]
        #print("sentence_tails:", sentence_tails)
        split_tails = [t.lower().split() for t in tails]

        for (j, g) in enumerate(gens[:k]):
            #print("g:",g)
            #if g == "none" or g == "هیچ یک" or g == "<object>none</object>":
            #    continue
            start_index = g.index("[GEN]") if "[GEN]" in g else 0
            end_index = g.index("[EOS]") if "[EOS]" in g else 0
            if start_index > 0 and end_index > 0:
                g = g[start_index+5:end_index].strip()
            elif end_index > 0:
                g = g[:end_index].strip()
            #print("g2:",g)
            key = str(i) + "_" + str(j)
            topk_gts[key] = sentence_tails
            topk_res[key] = [g.lower()]
            
            #print("g.lower()", g.lower())
            #print("split_tails", split_tails)
            #print("g.lower().split", g.lower().split())

            #b = sacrebleu.sentence_bleu(sentence_tails, 
            #                  [g.lower()])
            #print("b1:",b.score)

            b = sentence_bleu(split_tails, 
                              g.lower().split(), 
                              weights=(0.5, 0.5))
            #print("b2:",b)
            
            topk_bleu_score.append((l, b))
            if g in sentence_tails:
                topk_exact_match.append((l, 1))
                if g != "none" and g != "هیچ یک" and g != "<object>none</object>":
                    #print("Exact match between", g, " and ", sentence_tails)
                    topk_exact_match_not_none.append((l, 1))
            else:
                topk_exact_match.append((l, 0))
                if g != "none" and g != "هیچ یک" and g != "<object>none</object>":
                    topk_exact_match_not_none.append((l, 0))
            if g == head:
                topk_is_head.append((l, 1))
            else:
                topk_is_head.append((l, 0))

    print("---------------TOP K={}---------------".format(k))
    print("Exact Match:", np.mean(get2(topk_exact_match)))
    print("Exact Match Not None", np.mean(get2(topk_exact_match_not_none)))
    print("Mean sent BLEU score", np.mean(get2(topk_bleu_score)))
    QGEval = QGEvalCap(output_name, topk_gts, topk_res, calc_bert_score=cbs)
    scores,_ = QGEval.evaluate()
    scores["Exact_match"] = np.mean(get2(topk_exact_match))
    scores["Exact_match_not_none"] = np.mean(get2(topk_exact_match_not_none))
    scores["Mean sent BLEU score"] = np.mean(get2(topk_bleu_score))
    scores["Data rows"] = len(data)
    scores["Records"] = len(topk_gts)
    scores["TopK"] = k
    #scores["TailIsHead"] = np.mean(get2(topk_is_head))
    print(scores)
    return scores


import click
@click.command()
@click.argument("mixture", type=str)
@click.argument("pred_file", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
)
@click.option(
    "--output_name",
    default="eval_results",
    type=str,
    help="The name of results"
)
@click.option(
    "--data_type",
    default=1,
    type=int,
    help="The format of predictions file (refer to get_refs_preds function)"
)
@click.option(
    "--cbs",
    default=False,
    type=bool,
    help="Calculate Bert Score"
)
def eval(pred_file, output_name, data_type=1, topk=1, cbs=False):
    print("EVAL =================,", pred_file)
    if type(pred_file) is str:
        data = read_jsonl(pred_file)
        print("Len data:", len(data))
        return topk_eval(output_name, data, data_type, k=topk, cbs=cbs)
    else:
        src = pred_file["source"]
        target = pred_file["target"]
        gens = pred_file["gens"]
        with open(src, "r") as f:
            src_lines = f.readlines()
        with open(target, "r") as f:
            target_lines = f.readlines()
        with open(gens, "r") as f:
            gens_lines = f.readlines()
        print("src", len(src_lines), " t:", len(target_lines), "gen:", len(target_lines))
        
        data = []
        old_s, old_t, old_g = "","",""
        dups = 0
        new_gens = 0
        for s,t,g in zip(src_lines, target_lines, gens_lines):
            if s != old_s:
                d = {}
                d["source"] = s
                d["target"] = t
                d["generations"] = [g]
                data.append(d)
            elif g != old_g:
                d["generations"].append(g)
                #print("new gen for ", d)
                new_gens += 1
            else:
                #print("duplicate ", s) 
                dups += 1
            old_s, old_t, old_g = s, t, g
        print("New gens:", new_gens, " duplicates: ", dups)

        print("len of data:", len(data))
        #return ""
        return topk_eval(output_name, data, data_type, k=topk, cbs=cbs)

            
            

def toRow(name, results, columns):
    return [name] + [format(float(results[c]), '#.3f') for c in columns]


if __name__ == "__main__":
    eval()
 
