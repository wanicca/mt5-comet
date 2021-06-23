import csv
from pathlib import Path
import click
import json
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import comet.utils.myutils as mu
import comet.utils.myfiles as mf


@click.command()
@click.argument("fname", type=str)
@click.option("--out", default="", type=str, help="")
@click.option(
    "--path", envvar="PWD", type=click.Path(),
)
@click.option(
    "--filter_none",
    default="",
    type=str,
    help="A column that you want to filter none values"
)
@click.option(
    "--replace",
    default="",
    type=str,
    help="--replace=col-eq | neq-oldvalue-newvalue"
)
@click.option("--split", default=0, type=float, help="")
@click.option(
    "--evenly_per",
    default="",
    type=str,
    help="Sample with equal number for each specified group, ex: --evenly_per=age"
)
@click.option(
    "--prop_to",
    default="",
    type=str,
    help="Sample proportioned to the specified column, ex: --prop_to=age"
)
@click.option("--sample", default=-1, type=int, help="")
@click.option(
    "--selcol",
    default="",
    type=str,
    help="A fixed value column like columnname-fixedvalue or a column from another file like anotherfile@columnname or combination of other columns like columnname-col1+col2",
)
@click.option(
    "--rename",
    default="",
    type=str,
    help="Rename columns example 'oldcol:selcolname, oldcol2:selcol2'",
)
@click.option(
    "--col2file",
    "-f",
    is_flag=True,
    help="If set, it writes --selcol to a file with the name of selcolname",
)
@click.option(
    "--new",
    "-n",
    is_flag=True,
    help="If set, a new file is created"
)
@click.option(
    "--no_header",
    "-nh",
    is_flag=True,
    help="If set no header is added to the output"
)
@click.option(
    "--groups",
    default="",
    type=str,
    help="Reports the number in each category grouped by --groups"
)
def convert(fname, out, path, filter_none, replace, split, evenly_per, prop_to, sample, selcol, rename, col2file, new, no_header, groups):
    fname = path + "/" + fname
    print("selcol:", selcol)
    if col2file and not selcol:
        print("Please enter your desired column by --selcol option")
        return

    suffix = Path(fname).suffix
    if not out:
        ext = ".tsv" if fname.endswith("csv") else ".csv"
    else:
        ext = "." + out

    sep = "," if suffix == ".csv" else "\t"
    out = path + "/" + Path(fname).stem + ext
    print("fname:", fname)
    if new:
        df = pd.DataFrame()
        out = fname
    elif fname.endswith("csv"):
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
    if filter_none:
        l1 = len(df)
        print("Before filter:", l1)
        df=df.loc[df[filter_none].str.strip() != "none"]
        l2 = len(df)
        print("After filter:", l2)
        print(l1-l2, " items were filtered")
        out = path + "/" + Path(fname).stem + "_nn" + suffix 
        df.to_csv(out, sep=sep, index=False, encoding="utf-8", header=not no_header)
        return
    if replace:
        col, opr, oldval, newval = replace.split("-")
        if opr == "eq":
            df.loc[df[col] == oldval, col] = newval
            vlen = len(df[df[col] == newval])
        else:
            df.loc[df[col] != oldval, col] = newval
            vlen = len(df[df[col] == oldval])
        sample_df = df.groupby(col).sample(n=vlen, random_state=1)
        out = path + "/" + Path(fname).stem + "_" + oldval + "_" + newval + suffix
        print(out)
        sample_df.to_csv(out, sep=sep, index=False, encoding="utf-8", header=not no_header)
        return
    if evenly_per:
        sample_df = df.groupby(evenly_per).sample(n=sample, random_state=1)
        tn = "k".join(str(sample).rsplit("000", 1))
        tn = tn.replace("kk", "m") + "_per_" + evenly_per + suffix
        out = path + "/" + Path(fname).stem + "_" + tn
        sample_df = shuffle(sample_df)
        print("out:", out)
        sample_df.to_csv(out, sep=sep, index=False, encoding="utf-8", header=not no_header)
        return
    if prop_to:
        probs = df[prop_to].map(df[prop_to].value_counts())
        sample_df = df.sample(n=sample, weights=probs)
        tn = "k".join(str(sample).rsplit("000", 1))
        tn = tn.replace("kk", "m") + "_propto_" + prop_to + ("_nn" if filter_none else "") + suffix
        out = path + "/" + Path(fname).stem + "_" + tn
        print("out:", out)
        sample_df.to_csv(out, sep=sep, index=False, encoding="utf-8", header=not no_header)
        return
    if groups:
        counts = df.value_counts(groups.split(","))
        percents = counts/len(df)
        gs = counts.keys()
        for g, c, p in zip(gs, counts, percents):
            print(g, ":", c, f"{p:.2f}%")

        return
    if split > 0:
        train_fname = path + "/" + Path(fname).stem + "_train" + suffix
        test_fname = path + "/" + Path(fname).stem + "_test" + suffix
        train, test = train_test_split(df, test_size=split)

        train.columns = df.columns
        test.columns = df.columns
        train.to_csv(train_fname, sep=sep, index=False, encoding="utf-8", header=not no_header)
        test.to_csv(test_fname, sep=sep, index=False, encoding="utf-8", header=not no_header)
        print(train_fname)
        print(test_fname)
        return

    if rename:
        res = mu.arg2dict(rename)
        df = df.rename(columns=res)
        if not selcol:
            df.to_csv(fname, sep=sep, index=False, encoding="utf-8", header=not no_header)
            print(res, " were renamed")
            return

    if selcol:
        if not "@" in selcol:
            name_value = selcol.split("-")
            selcol_name = name_value[0]
            newval = name_value[1]
            if newval in df.columns:
                newval = df[newval]
            elif "+" in newval:
                cols = newval.split("+")
                newval = df[cols].agg("--".join, axis=1)
        else:
            name_value = selcol.split("@")
            of_name = name_value[0]
            selcol_name = name_value[1]
            suffix = Path(path + "/" + of_name).suffix
            if suffix == ".jsonl":
                data = mf.read_jsonl(path + "/" + of_name)
                newval = []
                for row in data:
                    newval.append(row[selcol_name])
        if not col2file:
            df[selcol_name] = newval
            df.to_csv(fname, sep=sep, index=False, encoding="utf-8", header=not no_header)
            print(name_value, " was added")
        else:
            with open(path + "/" + selcol_name, "w") as f:
                for item in newval:
                    f.write("%s\n" % item)
            print(name_value, " was written")
        return

    if sample > 0:
        df = df.sample(n=sample)
        tn = "k".join(str(sample).rsplit("000", 1))
        tn = tn.replace("kk", "m") + suffix
        out = path + "/" + Path(fname).stem + "_" + tn
        print("out:", out)
        df.to_csv(out, sep=sep, index=False, encoding="utf-8", header=not no_header)
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
