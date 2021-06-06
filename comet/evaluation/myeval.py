# +
import sys
import csv
import glob
import pandas as pd
import pickle
import re
import argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from comet.utils.rwfiles import *
from comet.evaluation.eval import QGEvalCap
from tabulate import tabulate
from pathlib import Path
import datetime
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
    if type==5:
        tails = l["refs"]
        head = l["query"]
        gens = l["hyps"]

    return gens, tails, head

def get2(l):
    return list(zip(*l))[1]


# +
import sacrebleu

def topk_eval(out, data, data_type, k, cbs=False):

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
    #QGEval = QGEvalCap(out, topk_gts, topk_res, calc_bert_score=cbs)
    #scores,_ = QGEval.evaluate()
    scores = {}
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
@click.argument("pred_file", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
)
@click.option(
    "--out",
    default="",
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
@click.option(
    "--ignore_inputs",
    default=False,
    type=bool,
    help="If there is no input file, set it to True"
)
@click.option(
    "--task",
    default="",
    type=str,
    help=""
)
@click.option(
    "--recalc",
    default=True,
    type=bool,
    help="If set to false it uses and shows stored results"
)
def eval(path, pred_file, out, data_type=1, topk=1, cbs=False, 
        ignore_inputs=False,
        task="",
        recalc=True):
    if type(pred_file) is str:
        fname = path + "/" + pred_file
        if Path(fname).is_file():
            if ".json" in fname:
                ext = Path(fname).suffix
                ckp = re.findall(r"\d+", fname)[-1] #checkpoint_step
                pre = pred_file.split(ckp)[0]
                if task == "": task = pre
                print("Loading ", fname)
                if ext == ".jsonl":
                    data = read_jsonl(path + "/" + pred_file)
                elif ext == ".json":
                    data = read_json(path + "/" + pred_file)
                print("Len data:", len(data))
                s = topk_eval(out, data, data_type, k=topk, cbs=cbs)
            else:
                ckp = re.findall(r"\d+", fname)[-1] #checkpoint_step
                pre = pred_file.split(ckp)[0]
                if task == "": task = pre
                inps = glob.glob(f"{path}/{pre}inputs")
                inp_file = "None"
                inp_dict={}
                if inps and not ignore_inputs:
                    inp_file = inps[0]  # f"{path}/{exp}_inputs"
                    with open(inp_file) as f:
                        input_lines = f.readlines()
                    for inp in input_lines:
                        inp_dict[inp] = {"targets": [], "gens": []}

                inps = glob.glob(f"{path}/{pre}targets")
                target_file = inps[0]  # f"{path}/{exp}_inputs"
                with open(target_file) as f:
                    target_lines = f.readlines()
                preds_file = f"{path}/{pred_file}"
                with open(preds_file) as f:
                    pred_lines = f.readlines()

                if not inp_dict:
                    input_lines = [str(k) for k in range(len(target_lines))]
                    inp_dict = data = {
                        str(k): {"targets": [], "gens": []} for k in range(len(target_lines))
                    }

                print("===================================")
                print(Path(inp_file).stem)
                print(Path(target_file).stem)
                print(Path(preds_file).stem)
                print("=========== Started ========================")
                _match = 0
                _match_none = 0
                for inp, target, pred in zip(input_lines, target_lines, pred_lines):
                    inp_dict[inp]["targets"].append(target.strip())
                    inp_dict[inp]["gens"].append(pred.strip())
                    if pred.strip() == "none" or pred.strip() == "هیچ یک":
                        _match_none += 1
                    elif target.strip() == pred.strip():
                        _match += 1

                print("len unique inputs:", len(inp_dict))
                print("Nones:{} {:.2f}".format(_match_none, _match_none/len(input_lines)))
                print("Matches:{} {:.2f}".format(_match, _match/len(input_lines)))
                cc = 1
                extra = ""
                res_name = "res_" + pred_file + extra
                res_fname = f"{path}/{res_name}.jsonl"
                em = 0
                show_top = False
                with open(res_fname, "w") as f:
                    for head, val in inp_dict.items():
                        cc += 1
                        if show_top and cc < 10 and gens[0] != "none":
                            print(head, ":", tails, "--", gens[0])
                        d = {}
                        d["head"] = head.strip()
                        d["tails"] = val["targets"]
                        d["generations"] = val["gens"]
                        json.dump(d, f)
                        f.write("\n")

                data = read_jsonl(res_fname)
                s = topk_eval(out, data, data_type = 2, k=1, cbs=cbs)
                # -
            # Saving Results
            mydate = datetime.datetime.today()
            today = mydate.strftime("%Y-%m-%d")

            ig = "_ignore_inputs" if ignore_inputs else ""
            pickle_fname = f"{path}/{pred_file}" + ig + "_res.pickle"

            P = list(reversed(fname.split("/")))
            p1 = P[1]
            p2 = P[2]
            p3 = P[3]

            res_exist = False


            if Path(pickle_fname).is_file() and not recalc:
                print("============= Results already exists!")
                with open(pickle_fname, "rb") as handle:
                    M = pickle.load(handle)
                res_exist = True
            else:
                M = {}
                M["Task"] = task
                M["P0"] = P[0] 
                for key, val in s.items():
                    try:
                        val = float(val)
                        val = round(val, 3)
                    except ValueError:
                        pass
                    M[key] = val

                for i, st in enumerate(task.split("_")):
                    if st:
                        M["T" + str(i)] = st
                M["Date"] = today
                M["Checkpoint"] = ckp
                for i, p in enumerate(P[:-3]):
                    M["P" + str(i)] = p
                with open(pickle_fname, "wb") as handle:
                    pickle.dump(M, handle, protocol=pickle.HIGHEST_PROTOCOL)
            csv_name = out if out else p2 + "_" + p1
            all_fname = f"{path}/{csv_name}.csv"
            print("==============  CSV output =================+++++++++++++++")
            print(all_fname)
            print("===============================+++++++++++++++")
            Path(all_fname).parent.mkdir(parents=True, exist_ok=True)
            if not res_exist or recalc or not Path(all_fname).is_file():
                res_fname = (
                    path + "/"
                    + p1
                    + "_"
                    + p2
                    + "_"
                    + pred_file
                    + " ("
                    + today
                    + ")"
                )
                Path(res_fname).parent.mkdir(parents=True, exist_ok=True)
                if not Path(all_fname).is_file():
                    with open(all_fname, "w") as f:  # You will need 'wb' mode in Python 2.x
                        w = csv.DictWriter(f, M.keys())
                        w.writeheader()
                        w.writerow(M)
                else:
                    with open(all_fname, "a") as f:  # You will need 'wb' mode in Python 2.x
                        w = csv.DictWriter(f, fieldnames=M.keys())
                        w.writerow(M)
                    csv_df = pd.read_csv(all_fname,  error_bad_lines=False)
                    final_df = csv_df.sort_values(by=['Exact_match_not_none'], ascending=False)
                    #group = final_df.groupby("P0", as_index = False)
                    #final_df = group.first() #.reset_index()
                    final_df.to_csv(all_fname, index=False)

                with open(res_fname, "w") as out:
                    print(res_fname)
                    print(res_fname, file=out)
                    for x in M:
                        print(x, ":", M[x])
                        print(x, ":", M[x], file=out)
                print("results were written in", res_fname)
        else:
            raise FileNotFoundError
            return
    else:
        src = path + "/" + pred_file["source"]
        target = path + "/" + pred_file["target"]
        gens = path + "/" + pred_file["gens"]
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
        return topk_eval(out, data, data_type, k=topk, cbs=cbs)

def toRow(name, results, columns):
    return [name] + [format(float(results[c]), '#.3f') for c in columns]


if __name__ == "__main__":
    eval()
 
