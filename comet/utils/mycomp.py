import pandas as pd
import click
import glob
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np


@click.command()
@click.argument("predfile", type=str)
# @click.argument("results_dir", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
)
@click.option("--unicode", "-m", is_flag=True, help="")
@click.option(
    "--report",
    "-r",
    is_flag=True,
    help="If set it reports matches in each group",
)
@click.option(
    "--confusion",
    default="",
    type=str,
    help="outputs labels that were confused example label1-label2",
)
@click.option(
    "--calc_nones",
    "-c",
    is_flag=True,
    help="If set it reportes matches with/without none targets",
)
def comp(predfile, path, unicode, report, confusion, calc_nones):
    inps = glob.glob(f"{path}/*{predfile}*targets")
    target_file = inps[0]
    print("input file:", target_file)
    inps = glob.glob(f"{path}/*{predfile}*predictions")
    pred_file = inps[0]
    name = Path(pred_file).name
    fid = name.split("predictions")[0]
    print("input file:", pred_file)
    inps = glob.glob(f"{path}/*{predfile}*inputs")
    input_file = inps[0]
    print("input file:", input_file)
    pred_file2 = ""
    if Path(f"{path}/comp").exists():
        inps = glob.glob(f"{path}/comp/*{predfile}*predictions")
        pred_file2 = inps[0]
        print("pred file 2:", pred_file2)
        inps = glob.glob(f"{path}/comp/*{predfile}*inputs")
        input_file2 = inps[0]
        print("input file:", input_file2)
    inpcol = []
    inpcol2 = []
    predcol = []
    predcol2 = []
    targcol = []
    f = open(input_file, "r")
    for x in f:
        text = x
        if unicode:
            text = text[2:-1]
            text = text.encode("raw_unicode_escape")
            text = text.decode("unicode_escape")
            text = text.encode("raw_unicode_escape")
            text = text.decode()
        inpcol.append(text.strip())

    f = open(pred_file, "r")
    for x in f:
        predcol.append(x.strip())

    if pred_file2:
        f = open(input_file2, "r")
        for x in f:
            text = x
            if unicode:
                text = text[2:-1]
                text = text.encode("raw_unicode_escape")
                text = text.decode("unicode_escape")
                text = text.encode("raw_unicode_escape")
                text = text.decode()
            inpcol2.append(text.strip())
        f = open(pred_file2, "r")
        for x in f:
            predcol2.append(x.strip())

    f = open(target_file, "r")
    for x in f:
        targcol.append(x.strip())
    data = []
    m = 0
    n = 0
    p = 0
    q = 0
    N = len(targcol)
    c = 0
    groups = {}
    g_total = {}
    confdata = []
    if confusion:
        label1, label2 = confusion.split("-")
    for i in tqdm(range(N)):
        is_none = False
        g_col = targcol[i].strip()
        match = False
        if "--none" in inpcol[i] or (
            predcol[i].strip() == "none" and predcol[i].strip() == "هیچ یک"
        ):
            n += 1
            is_none = True
            # g_col = predcol[i].strip() + "_none"
        if predcol[i].strip() == targcol[i].strip():
            match = True
            if is_none:
                p += 1
            else:
                q += 1
        if calc_nones:
            if g_col in groups and match:
                groups[g_col] += 1
            elif match:
                groups[g_col] = 1

            if g_col in g_total:
                g_total[g_col] += 1
            else:
                g_total[g_col] = 1

        if not pred_file2:
            data.append([targcol[i], predcol[i], inpcol[i], "", ""])
        elif predcol2[i] == predcol[i]:
            data.append(
                [targcol[i], predcol[i], predcol2[i], inpcol1[i], inpcol2[i]]
            )
            c += 1
        if confusion and targcol[i] == label1 and predcol[i] == label2:
            confdata.append([targcol[i], predcol[i], inpcol[i]])

    m = p + q
    confdata.insert(0, ["target", "prediction", "input"])
    data.insert(
        0,
        [
            N,
            "{} {:.2f}".format(m, m / N),
            "{} {:.2f}".format(n, n / N),
            "{} {:.2f}".format(c, c / N),
            "-",
        ],
    )
    pd.DataFrame(
        data,
        columns=[
            "target_text",
            "pred_text",
            "pred_text2",
            "input_text1",
            "input_text2",
        ],
    ).to_csv(f"{path}/{fid}merge.csv", encoding="utf-8")
    if confusion:
        pd.DataFrame(confdata).to_csv(f"{path}/{fid}{label1}_{label2}.csv")
    if calc_nones:
        print(f"Ratio of nones: ({n} | {n/N:.2f})")
        print(f"Ratio of matches: ({m} | {m/N:.2f})")
        print(f"Ratio of not none matches: ({q} | {q/N:.2f})")
        print(f"Ratio of none matches: ({p} | {p/N:.2f})")
    if report:
        print("Preparing reports ....")
        labels = list(set(targcol))
        cm = confusion_matrix(targcol, predcol, labels=labels)
        df = pd.DataFrame(cm, columns=labels)
        df.insert(loc=0, column="rels", value=labels)
        df.to_csv(path + "/" + fid + "cm.csv", index=False)
        rep = classification_report(targcol, predcol)
        print(rep)
        report_dict = classification_report(targcol, predcol, output_dict=True)
        repdf = pd.DataFrame(report_dict)
        repdf = repdf.round(2).transpose()
        repdf.insert(
            loc=0,
            column="class",
            value=labels + ["accuracy", "macro avg", "weighted avg"],
        )
        repdf.to_csv(path + "/" + fid + "results.csv", index=False)
    if groups:
        sg = 0
        sn = 0
        for g, v in groups.items():
            N = g_total[g]
            sg += v
            sn += N
            print(f"{g} : ( {v}/{N} = {v/N:.2f} )")

        print(f"total v: {sg}, total n: {sn}")


if __name__ == "__main__":
    comp()
