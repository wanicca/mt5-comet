import csv
from pathlib import Path
import click
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

@click.command()
@click.argument("fname", type=str)
@click.option("--out", default="", type=str, help="")
@click.option(
    "--path", envvar="PWD", type=click.Path(),
)
@click.option(
    "--split",
    default=0,
    type=float,
    help=""
)
@click.option(
    "--truncate",
    default=-1,
    type=int,
    help=""
)
@click.option(
    "--newcol",
    default="",
    type=str,
    help=""
)
@click.option(
    "--rename",
    default="",
    type=str,
    help="Rename columns example 'oldcol:newcolname, oldcol2:newcol2'"
)

def convert(fname, out, path, split, truncate, newcol, rename):
    fname = path + "/" + fname
    if not out:
        ext = ".tsv" if fname.endswith("csv") else ".csv"
    else:
        ext = "." + out

    out = path + "/" + Path(fname).stem + ext
    print("fname:", fname)
    if fname.endswith("csv"):
        df = pd.read_csv(fname, keep_default_na=False, encoding='utf8', quoting=csv.QUOTE_ALL, error_bad_lines=False)
    else:
        df = pd.read_csv(fname, sep="\t", keep_default_na=False, encoding='utf8', quoting=csv.QUOTE_ALL, error_bad_lines=False)
    if split > 0:
        train_fname = path + "/" + Path(fname).stem + "_train.csv" 
        test_fname = path + "/" + Path(fname).stem + "_test.csv" 
        train, test = train_test_split(df, test_size=split)
        train.to_csv(train_fname, index=False, encoding='utf-8') 
        test.to_csv(test_fname, index=False, encoding='utf-8') 
        print(train_fname)
        print(test_fname)
        return
    if rename:
        res = dict(map(str.strip, sub.split(':', 1)) for sub in rename.split(',') if ':' in sub)
        df = df.rename(columns=res)
        if not newcol:
           df.to_csv(fname, index=False, encoding='utf-8') 
           print(res, " were renamed")
           return

    if newcol:
       name_value = newcol.split("-")
       df[name_value[0]]=name_value[1]
       print(out)
       df.to_csv(fname, index=False, encoding='utf-8') 
       print(name_value, " was added")
       return

    if truncate > 0:
        df = df.truncate(after = truncate)
        out = path + "/" + Path(fname).stem + "_" + str(truncate) + ext

    print("out:", out)
    if ext == ".tsv":
        with open(out, "w") as tsvout:
            tsvout = csv.writer(tsvout, delimiter="\t")
            for index, row in tqdm(df.iterrows(), total=len(df)):
                tsvout.writerow(row)
    elif ext == ".csv":
        with open(out, "w") as fou:
            cw = csv.writer(fou, escapechar="\\")
            for index, row in tqdm(df.iterrows(), total=len(df)):
                cw.writerow(filecontents)
    elif ext == ".json":
        dlist = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            d = {}
            item = {}
            en = str(row["en"]).replace('"','')
            fa = str(row["fa"]).replace('"','')
            if len(en.strip()) < 15 or len(fa.strip()) < 15:
                continue
            item["en"] = en
            item["fa"] = fa
            d["translation"] = item
            dlist.append(d)
        dd = {"data":dlist}
        with open(out, 'w', encoding='utf8') as jsonfile:
            json.dump(dd, jsonfile, ensure_ascii=False)
    else:
        print("Unsuported output format")

if __name__ == "__main__":
    convert()
