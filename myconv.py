import csv
from pathlib import Path
import click
import json
import pandas as pd
from tqdm import tqdm

@click.command()
@click.argument("fname", type=str)
@click.option("--out", default="", type=str, help="")
@click.option(
    "--path", envvar="PWD", type=click.Path(),
)
def convert(fname, out, path):
    fname = path + "/" + fname
    if not out:
        ext = ".tsv" if fname.endswith("csv") else ".csv"
    else:
        ext = "." + out

    out = path + "/" + Path(fname).stem + ext
    print("fname:", fname)
    print("out:", out)
    if ext == ".tsv":
        with open(fname, "r") as csvin:
            with open(out, "w") as tsvout:
                csvin = csv.reader(csvin)
                tsvout = csv.writer(tsvout, delimiter="\t")
                for row in csvin:
                    tsvout.writerow(row)
    elif ext == ".csv":
        # read tab-delimited file
        with open(fname, "r") as fin:
            cr = csv.reader(fin, delimiter="\t")
            filecontents = [line for line in cr]

        # write comma-delimited file (comma is the default delimiter)
        with open(out, "w") as fou:
            cw = csv.writer(fou, escapechar="\\")
            cw.writerows(filecontents)
    elif ext == ".json":
        df = pd.read_csv(fname)
        dlist = []
        for index, row in tqdm(df.iterrows(), total=len(df)):
            d = {}
            item = {}
            item["input"] = str(row["input_text"])
            item["target"] = str(row["target_text"])
            d["prefix"] = row["prefix"]
            d["relation"] = item
            dlist.append(d)
        dd = {"data":dlist}
        with open(out, 'w') as jsonfile:
            json.dump(dd, jsonfile)
    else:
        print("Unsuported output format")

if __name__ == "__main__":
    convert()
