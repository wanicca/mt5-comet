import csv
from pathlib import Path
import click
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import comet.utils.myutils as mu
import comet.utils.myfiles as mf


@click.command()
@click.argument("fname", type=str)
@click.option("--out", default="", type=str, help="")
@click.option(
    "--path", envvar="PWD", type=click.Path(),
)
@click.option("--split", default=0, type=float, help="")
@click.option("--truncate", default=-1, type=int, help="")
@click.option(
    "--newcol",
    default="",
    type=str,
    help="A fixed value column like columnname-fixedvalue or a column from another file like anotherfile@columnname or combination of other columns like columnname-col1+col2",
)
@click.option(
    "--rename",
    default="",
    type=str,
    help="Rename columns example 'oldcol:newcolname, oldcol2:newcol2'",
)
@click.option(
    "--col2file",
    "-f",
    is_flag=True,
    help="If set, it writes --newcol to a file with the name of newcolname",
)
def convert(fname, out, path, split, truncate, newcol, rename, col2file):
    fname = path + "/" + fname
    print("newcol:", newcol)
    if col2file and not newcol:
        print("Please enter your desired column by --newcol option")
        return

    suffix = Path(fname).suffix
    if not out:
        ext = ".tsv" if fname.endswith("csv") else ".csv"
    else:
        ext = "." + out

    sep = "," if suffix == ".csv" else "\t"
    out = path + "/" + Path(fname).stem + ext
    print("fname:", fname)
    if fname.endswith("csv"):
        df = pd.read_csv(
            fname,
            keep_default_na=False,
            encoding="utf8",
            quoting=csv.QUOTE_ALL,
            error_bad_lines=False,
        )
    else:
        df = pd.read_csv(
            fname,
            sep="\t",
            keep_default_na=False,
            encoding="utf8",
            quoting=csv.QUOTE_ALL,
            error_bad_lines=False,
        )
    if split > 0:
        train_fname = path + "/" + Path(fname).stem + "_train" + suffix
        test_fname = path + "/" + Path(fname).stem + "_test" + suffix
        train, test = train_test_split(df, test_size=split)
        train.columns = df.columns
        test.columns = df.columns
        train.to_csv(train_fname, sep=sep, index=False, encoding="utf-8")
        test.to_csv(test_fname, sep=sep, index=False, encoding="utf-8")
        print(train_fname)
        print(test_fname)
        return

    if rename:
        res = mu.arg2dict(rename)
        df = df.rename(columns=res)
        if not newcol:
            df.to_csv(fname, sep=sep, index=False, encoding="utf-8")
            print(res, " were renamed")
            return

    if newcol:
        if not "@" in newcol:
            name_value = newcol.split("-")
            newcol_name = name_value[0]
            newval = name_value[1]
            if newval in df.columns:
                newval = df[newval]
            elif "+" in newval:
                cols = newval.split("+")
                newval = df[cols].agg("--".join, axis=1)
        else:
            name_value = newcol.split("@")
            of_name = name_value[0]
            newcol_name = name_value[1]
            suffix = Path(path + "/" + of_name).suffix
            if suffix == ".jsonl":
                data = mf.read_jsonl(path + "/" + of_name)
                newval = []
                for row in data:
                    newval.append(row[newcol_name])
        if not col2file:
            df[newcol_name] = newval
            df.to_csv(fname, sep=sep, index=False, encoding="utf-8")
            print(name_value, " was added")
        else:
            with open(path + "/" + newcol_name, "w") as f:
                for item in newval:
                    f.write("%s\n" % item)
            print(name_value, " was written")
        return

    if truncate > 0:
        df = df.truncate(after=truncate)
        tn = "k".join(str(truncate).rsplit("000", 1))
        tn = tn.replace("kk", "m") + suffix
        out = path + "/" + Path(fname).stem + "_" + tn
        print("out:", out)
        df.to_csv(out, sep=sep, index=False, encoding="utf-8")
        return

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
            en = str(row["en"]).replace('"', "")
            fa = str(row["fa"]).replace('"', "")
            if len(en.strip()) < 15 or len(fa.strip()) < 15:
                continue
            item["en"] = en
            item["fa"] = fa
            d["translation"] = item
            dlist.append(d)
        dd = {"data": dlist}
        with open(out, "w", encoding="utf8") as jsonfile:
            json.dump(dd, jsonfile, ensure_ascii=False)
    else:
        print("Unsuported output format")


if __name__ == "__main__":
    convert()
