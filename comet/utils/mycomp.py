import pandas as pd
import shutil
import click
import glob
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import comet.utils.confmat as confmat


@click.command()
@click.argument("predfile", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
)
@click.option(
    "--results_dir",
    default="/home/pouramini/images",
    type=str,
    help=""
)
@click.option(
    "--unicode",
    default="file2",
    type=str,
    help="specify the file which needs unicode decoding (file1 or file2 or both)"
)
@click.option(
    "--report",
    default="",
    type=str,
    help="If set it reports matches in each group, set a title for reports with this option",
)
@click.option(
    "--confusion",
    default="",
    type=str,
    help="outputs labels that were confused example label1-label2",
)
@click.option(
    "--target_dir",
    default="",
    type=str,
    help=""
)
@click.option(
    "--comp_dir",
    default="",
    type=str,
    help=""
)
@click.option(
    "--src_df",
    default="self",
    type=str,
    help="Source dataset or dataframe "
)
@click.option(
    "--target_col",
    default="target_text",
    type=str,
    help=""
)
def comp(predfile, path, results_dir, unicode, report, confusion, target_dir, comp_dir, src_df, target_col):
    if not target_dir:
        target_dir = path
    inps = glob.glob(f"{target_dir}/*targets")
    target_file = inps[0]
    print("input file:", target_file)
    inps = glob.glob(f"{path}/*{predfile}*predictions")
    pred_file = inps[0]
    name = Path(pred_file).name
    if not report:
        fid = name.split("predictions")[0] 
    elif report=="self":
        fid = (Path(path).parent.parent.stem + "_" 
               + Path(path).parent.stem + "_" + Path(path).stem)
    else:
        fid = report.lower().replace(" ","_").replace(",","")
    print("input file:", pred_file)
    inps = glob.glob(f"{target_dir}/*inputs")
    input_file = inps[0]
    print("input file:", input_file)
    pred_file2 = ""
    if comp_dir:
        p = Path(f"{comp_dir}").resolve()
        comp_dir = str(p)
        print("Folder for comparison:", comp_dir)
        inps = glob.glob(f"{comp_dir}/*{predfile}*predictions")
        pred_file2 = inps[0]
        print("pred file 2:", pred_file2)
        inps = glob.glob(f"{comp_dir}/*inputs")
        input_file2 = inps[0]
        print("input file:", input_file2)
    inpcol = []
    inpcol2 = []
    predcol = []
    predcol2 = []
    targcol = []
    f = open(input_file, "r")
    ncc = 0
    def persianReplace(text):
        text = text.strip().replace("PersonX's","ش").replace("PersonY","رضا")
        text = text.strip().replace("PersonX","علی").replace("PersonZ","حمید")
        return text

    for x in f:
        text = x
        if unicode == "both" or unicode == "file1":
            text = text[2:-1]
            text = text.encode("raw_unicode_escape")
            text = text.decode("unicode_escape")
            text = text.encode("raw_unicode_escape")
            text = text.decode()
            text = persianReplace(text)
        text = text.strip()
        inpcol.append(text)

    f = open(pred_file, "r")
    for x in f:
        predcol.append(x.strip())

    if pred_file2:
        f = open(input_file2, "r")
        for x in f:
            text = x
            if unicode == "both" or unicode == "file2":
                text = text[2:-1]
                text = text.encode("raw_unicode_escape")
                text = text.decode("unicode_escape")
                text = text.encode("raw_unicode_escape")
                text = text.decode()
                text = persianReplace(text)
            text = text.strip()
            inpcol2.append(text.strip())
        f = open(pred_file2, "r")
        for x in f:
            predcol2.append(x.strip())

    f = open(target_file, "r")
    for x in f:
        targcol.append(x.strip())
    data = []
    confdata = []
    N = len(targcol)
    def hasNone(p):
        return "--none" in p or (
            p.strip() == "none" or p.strip() == "هیچ یک"
        )
    if confusion:
        label1, label2 = confusion.split("-")
    for i in tqdm(range(N)):
        if (targcol[i] == "target_text" or "input_text" in inpcol[i]) and i < 2:
            continue
        if not pred_file2:
            data.append([targcol[i], predcol[i], inpcol[i], "", ""])
        else: #if predcol2[i] == predcol[i]:
            data.append(
                [targcol[i], predcol[i], inpcol[i], predcol2[i], inpcol2[i]]
            )
        if confusion and targcol[i] == label1 and predcol[i] == label2:
            confdata.append([targcol[i], predcol[i], inpcol[i]])
            if is_none:
                ncc += 1

    confdata.insert(0, ["target", "prediction", "input"])
    df = pd.DataFrame(
        data,
        columns=[
            target_col,
            "pred_text1",
            "input_text",
            "pred_text2",
            "input_text_fa",
        ],
    )
    if src_df and not report:
        if src_df=="parent":
            parent_dir = str(Path(path).parent)
            inps = glob.glob(f"{parent_dir}/*tsv")
            src_df = inps[0]
        elif src_df == "self":
            src_df = path + "/src_df.tsv"
        print("source df:", src_df)
        srcdf = pd.read_table(src_df)
        print(f"comparing lengths {len(df)} ?? {len(srcdf)}")
        assert len(df) == len(srcdf), "the number of source dataset with output perdictions must be the same"
        #qdf2 = srcdf.merge(qdf["pred_text1"], left_index=True, right_index = True)
        print("Results were merged with source dataset")
        df = df[[target_col, "pred_text1", "pred_text2"]]
        df = pd.concat([df.reset_index(drop=True),srcdf.reset_index(drop=True)], axis=1)
        #df = df.rename(columns={"input_text":"input_text", "input_text_fa":"input_text_fa"})
        #df = df.merge(srcdf, on=target_col, how = 'left')
        df = df[["prefix","input_text", target_col, "input_text_fa", "target_text_fa", "pred_text1", "pred_text2"]]
        print("Size of dataframe after merge with source:", len(df))
    dup_df = df.drop_duplicates(subset=['input_text',target_col])
    dups = len(df) - len(dup_df)
    dup_df.to_csv(f"{path}/dups1_{fid}.csv", encoding="utf-8")
    dup_df = df.drop_duplicates(subset=['input_text_fa',target_col])
    dup_df.to_csv(f"{path}/dups2_{fid}.csv", encoding="utf-8")
    dups2 = len(df) - len(dup_df)
    if comp_dir:
        comp_col = np.where(df['pred_text1'] == df['pred_text2'], "=", "//")
        df.insert(loc=0, column='compare', value=comp_col)
    df.to_csv(f"{path}/merge_{fid}.csv", encoding="utf-8")
    df.columns = df.columns.astype(str)
    if confusion:
        pd.DataFrame(confdata).to_csv(f"{path}/{label1}_{label2}_confusion.csv", index=False)
    match = []
    nones = ["none", "هیچ یک"]
    N = len(df)
    N2 = df['input_text'].nunique()
    mdf = (df[df['pred_text1'] == df[target_col]])
    mdf.to_csv(f"{path}/match_{fid}.tsv", sep="\t", encoding="utf-8")
    m = len(mdf)
    m2 = len(df[df['pred_text2'] == df[target_col]])
    b2df = (df[(df['pred_text2'] == df[target_col]) 
             & (df['pred_text1'] != df[target_col])
             & ~df[target_col].isin(nones)])
    b2df = b2df[["input_text", "input_text_fa", "pred_text1", "pred_text2", target_col]]
    b2df.to_csv(f"{path}/b2_{fid}.tsv", sep="\t", encoding="utf-8")
    b2 = len(b2df)

    b1df = (df[(df['pred_text1'] == df[target_col]) 
             & (df['pred_text2'] != df[target_col])
             & ~df[target_col].isin(nones)])
    b1df = b1df[["input_text", "input_text_fa", "pred_text1", "pred_text2", target_col]]
    b1df.to_csv(f"{path}/b1_{fid}.tsv", sep="\t", encoding="utf-8")
    b1 = len(b1df)

    p = len(df[(df['pred_text1'] == df[target_col]) & df[target_col].isin(nones)])
    p2 = len(df[(df['pred_text2'] == df[target_col]) & df[target_col].isin(nones)])
    qdf = (df[(df['pred_text1'] == df[target_col]) & ~df[target_col].isin(nones)])
    qdf.to_csv("nn_match_" + Path(path).parent.parent.stem + ".tsv", sep= "\t")
    q = len(qdf)
    q2 = len(df[(df['pred_text2'] == df[target_col]) & ~df[target_col].isin(nones)])
    n = len(df[df['pred_text1'] == 'none'])
    n2 = len(df[df['pred_text2'] == 'none'])
    tn = len(df[df[target_col] == 'none'])
    c = len(df[(df['pred_text1'] == df['pred_text2']) & df['pred_text1'].isin(nones)])
    cnndf = df[(df['pred_text1'] == df['pred_text2']) & ~df['pred_text1'].isin(nones)]
    #cnndf = cnndf[['input_text', 'input_text_fa','pred_text1', target_col]]
    cnndf = cnndf.drop_duplicates(subset=['input_text', 'pred_text1'])
    cnndf.to_csv(f"{path}/common_{fid}.tsv", sep="\t", encoding="utf-8")
    cnn = len(cnndf)

    ct = c + cnn
    acdf = df[(df['pred_text1'] == df['pred_text2']) & 
            (df['pred_text1'] == df[target_col]) & 
            ~df['pred_text1'].isin(nones)]
    acdf = acdf[['input_text', 'input_text_fa', target_col]]
    acdf.to_csv(f"{path}/all_common_{fid}.csv", encoding="utf-8")
    ac = len(acdf)
    match.append(
            [
                "file1",
                N,
                N2,
                dups,
                "{} {:.2f} {:.2f}".format(m, m / N, m /N2),
                "{} {:.2f} {:.2f} ".format(p, p / N, p/N2),
                "{} {:.2f} {:.2f}".format(q, q / N, q/N2),
                "{} {:.2f} {:.2f}".format(n, n / N, n/N2),
                "{} {:.2f} {:.2f}".format(tn, tn / N, tn/N2),
                "{} {:.2f} {:.2f}".format(c, c / N, c/N2),
                "{} {:.2f} {:.2f}".format(cnn, cnn / N, cnn/N2),
                "{} {:.2f} {:.2f}".format(ct, ct / N, ct/N2),
                "{} {:.2f} {:.2f}".format(ac, ac / N, ac/N2),
                "{} {:.2f} {:.2f}".format(b2, b2 / N, b2/N2),
                "{} {:.2f} {:.2f}".format(b1, b1 / N, b1/N2),
            ],
        )
    match.append(
            [
                "file2",
                N,
                N2,
                dups2,
                "{} {:.2f} {:.2f}".format(m2, m2 / N, m2 /N2),
                "{} {:.2f} {:.2f} ".format(p2, p2 / N, p2/N2),
                "{} {:.2f} {:.2f}".format(q2, q2 / N, q2/N2),
                "{} {:.2f} {:.2f}".format(n2, n2 / N, n2/N2),
                "{} {:.2f} {:.2f}".format(tn, tn / N, tn/N2),
                "{} {:.2f} {:.2f}".format(c, c / N, c/N2),
                "{} {:.2f} {:.2f}".format(cnn, cnn / N, cnn/N2),
                "{} {:.2f} {:.2f}".format(ct, ct / N, ct/N2),
                "{} {:.2f} {:.2f}".format(ac, ac / N, ac/N2),
                "{} {:.2f} {:.2f}".format(b2, b2 / N, b2/N2),
                "{} {:.2f} {:.2f}".format(b1, b1 / N, b1/N2),
            ],
        )
    mdf = pd.DataFrame(
        match,
        columns=[
            "file:",
            "input rows:",
            "input unique:",
            "duplicates:",
            "match_target:",
            "none_match:",
            "not_none_match:",
            "pred nones:",
            "target nones:",
            "common nones:",
            "common not nones:",
            "common: ",
            "all common: ",
            "2 is better: ",
            "1 is better: ",
        ],
    )
    if not report:
        print("===================== Matches Report ========================")
        print(mdf.T)
        mdf.to_csv(f"{path}/report_{fid}.csv", encoding="utf-8")
        print("====== ", str(Path(pred_file2).parent.stem) + "/" + str(Path(pred_file2).stem))
        print("")
    if confusion:
        print(f"Ratio of none confusions between {label1} and {label2}: ({ncc} | {ncc/len(confdata):.2f})")
    print("============================================================")
    if report and not confusion:
        print("Preparing reports ....")
        repfile=open(path + "/000_" + report, "w")
        repfile.close()
        labels = list(set(targcol))
        if len(labels) == 9:
            labels = ["xAttr", "xIntent", "xNeed", "xReact", "xEffect", "xWant", "oReact", "oEffect", "oWant"]
        cm = confusion_matrix(targcol, predcol, labels=labels)
        df = pd.DataFrame(cm, columns=labels)
        df.insert(loc=0, column="rels", value=labels)
        df.to_csv(path + "/cm_" + fid + ".csv", index=False)
        rep = classification_report(targcol, predcol, labels=labels, zero_division=1)
        print(rep)
        report_dict = classification_report(targcol, predcol, output_dict=True, labels=labels, zero_division=1)
        repdf = pd.DataFrame(report_dict)
        repdf = repdf.round(2).transpose()
        repdf.insert(
            loc=0,
            column="class",
            value=labels + ["accuracy", "macro avg", "weighted avg"],
        )
        repdf.to_csv(path + "/report_" + fid + ".csv", index=False)
        title = fid if report == "self" else report
        acc_, cr = confmat.report(y_test=targcol, y_pred=predcol, labels=labels, title=title, image=path + "/image_" + fid + ".png")
        if results_dir:
            shutil.copy(path + "/image_" + fid + ".png", results_dir)


if __name__ == "__main__":
    comp()
