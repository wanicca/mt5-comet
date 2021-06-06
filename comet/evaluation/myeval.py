# +
import sys
import csv
import glob
import click
import pandas as pd
import re
from os.path import expanduser
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from comet.utils.myfiles import *
from comet.evaluation.eval import QGEvalCap
from tabulate import tabulate
from pathlib import Path
import datetime
from comet.utils.myutils import *
import sacrebleu
# -
def get_refs_preds(l, type, keys):
    keys = arg2dict(keys)
    head_key,tails_key,gens_key= keys["head"],keys["tails"],keys["gens"]
    if type==1:
        tails = l["fact"][tails_key]
        head = l["fact"][head_key]
        gens = l[gens_key]
        if "prompt" in l:
            prompt = l["prompt"]
            gens = [remove_prefix(g, prompt).strip() for g in gens]
    if type==2:
        tails = l[tails_key]
        head = l[head_key]
        gens = l[gens_key]
    if type==3:
        tails = [l[tails_key]]
        head = l[head_key]
        gens = l[gens_key]

    return gens, tails, head

def get2(l):
    if l:
        return list(zip(*l))[1]
    else:
        return 0


def topk_eval(out, data, data_type, k, keys, cbs=False):

    topk_gts = {}
    topk_res = {}
    topk_exact_match = []
    topk_exact_match_not_none = []
    topk_bleu_score = []

    topk_is_head = []

    for i, l in enumerate(data):
        (gens, tails, head) = get_refs_preds(l, type=data_type, keys=keys)
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
    QGEval = QGEvalCap(out, topk_gts, topk_res, calc_bert_score=cbs)
    scores,_ = QGEval.evaluate()
    scores["Exact_match"] = np.mean(get2(topk_exact_match))
    scores["Exact_match_not_none"] = np.mean(get2(topk_exact_match_not_none))
    scores["Mean sent BLEU score"] = np.mean(get2(topk_bleu_score))
    scores["Data rows"] = len(data)
    scores["Records"] = len(topk_gts)
    scores["TopK"] = k
    return scores


@click.command()
@click.argument("input_files_pattern", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--out",
    default="",
    type=str,
    help="The name for result csv file"
)
@click.option(
    "--data_type",
    default=2,
    type=int,
    help="The format of predictions file (refer to get_refs_preds function)"
)
@click.option(
    "--cbs",
    "-b",
    is_flag=True,
    help="Calculate Bert Score"
)
@click.option(
    "--ignore_inputs",
    "-i",
    is_flag=True,
    help="If there is no input file"
)
@click.option(
    "--task",
    default="",
    type=str,
    help=""
)
@click.option(
    "--clear",
    "-c",
    is_flag=True,
    help="It clears the last results and recalculate them"
)
@click.option(
    "--append",
    "-a",
    is_flag=True,
    help="It appends the new result to the end of output file"
)
@click.option(
    "--keys",
    default="head:head,tails:tails,gens:generations",
    type=str,
    help="A map for head, tails and generations keys, the default is: head:head,tails:tails,gens:generations"
)
def eval(path, input_files_pattern, out, data_type=2, topk=1, cbs=False, 
        ignore_inputs=False,
        task="",
        clear=False,
        append=False,
        keys=""):

    pred_inps = glob.glob(f"{path}/*{input_files_pattern}*")
    if not pred_inps:
        print(f"No file was found using *{input_files_pattern}* pattern")
        return
    for pred_inp in pred_inps:
        pred_file = Path(pred_inp).name
        print(pred_file)
        fname = path + "/" + pred_file
        ext = Path(fname).suffix
        # split file path into folders
        P = list(reversed(fname.split("/")))
        p1 = P[1]
        p2 = P[2] if len(P) > 1 else ""
        p3 = P[3] if len(P) > 2 else ""
        res_exist = False
        out = out if out else p2 + "_" + p1
        if not out.endswith(".csv"):
           out += ".csv"
        if not "/" in out:
            out_fname = f"{path}/{out}"
        else:
            home = expanduser("~")
            out = out.replace("~/", home + "/")
            out_fname = out 

        if Path(out_fname).is_file() and not (clear or append):
            print(f"{out} already exists! use '--clear' or '--append' to reevaluate")
            return
        if ".json" in ext:
            ckp = re.findall(r"\d+", fname)[-1] #checkpoint_step
            pre = pred_file.split(ckp)[0]
            if task == "": task = pre
            print("Loading ", fname)
            if ext == ".jsonl":
                data = read_jsonl(path + "/" + pred_file)
            elif ext == ".json":
                data = read_json(path + "/" + pred_file)
            print("Len data:", len(data))
            scores = topk_eval(out, data, data_type, k=topk, keys=keys, cbs=cbs)
        else: 
            #when there are seperated input, predictions and targets files
            ckp = re.findall(r"\d+", fname)[-1] #checkpoint_step 
            pre = pred_file.split(ckp)[0] # task name before step
            if task == "": task = pre
            inps = glob.glob(f"{path}/{pre}inputs")
            inp_file = "None"
            inp_dict={}
            if inps and not ignore_inputs:
                inp_file = inps[0]  
                with open(inp_file) as f:
                    input_lines = f.readlines()
                for inp in input_lines:
                    inp_dict[inp] = {"targets": [], "gens": []}

            inps = glob.glob(f"{path}/{pre}targets")
            if not inps:
                print(f"!! target file {pre}targets for {pred_file} is missing!!!")
                continue
            target_file = inps[0]  
            with open(target_file) as f:
                target_lines = f.readlines()
            preds_file = f"{path}/{pred_file}"
            with open(preds_file, encoding="utf-8") as f:
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
            res_name = pred_file + extra
            res_fname = f"{path}/{res_name}.jsonl"
            em = 0
            show_top = False
            kd = arg2dict(keys)
            head_key,tails_key,gens_key= kd["head"],kd["tails"],kd["gens"]
            with open(res_fname, "w", encoding="utf-8") as f:
                for head, val in inp_dict.items():
                    cc += 1
                    if show_top and cc < 10 and gens[0] != "none":
                        print(head, ":", tails, "--", gens[0])
                    d = {}
                    d[head_key] = head.strip()
                    d[tails_key] = val["targets"]
                    d[gens_key] = val["gens"]
                    json.dump(d, f)
                    f.write("\n")

            data = read_jsonl(res_fname)
            scores = topk_eval(out, data, data_type = 2, k=1, keys=keys, cbs=cbs)
            # -
        # Saving Results
        mydate = datetime.datetime.today()
        today = mydate.strftime("%Y-%m-%d")

        M = {}
        M["Task"] = task
        M["P0"] = P[0] 
        if not scores:
            print("No score")
        for key, val in scores.items():
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
        print("==============  CSV output =================")
        print(out_fname)
        print("============================================")
        Path(out_fname).parent.mkdir(parents=True, exist_ok=True)
        if clear:
            with open(out_fname, "w") as f:  # You will need 'wb' mode in Python 2.x
                w = csv.DictWriter(f, M.keys())
                w.writeheader()
                w.writerow(M)
        elif append:
            with open(out_fname, "a") as f:  # You will need 'wb' mode in Python 2.x
                w = csv.DictWriter(f, fieldnames=M.keys())
                w.writerow(M)
            csv_df = pd.read_csv(out_fname,  error_bad_lines=False)
            final_df = csv_df.sort_values(by=['Exact_match_not_none'], ascending=False)
            #group = final_df.groupby("P0", as_index = False)
            #final_df = group.first() #.reset_index()
            final_df.to_csv(out_fname, index=False)

        #eval_file(path, pred_file, out, data_type, topk, cbs)




if __name__ == "__main__":
    eval()
 
