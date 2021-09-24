from pathlib import Path
import click
import pandas as pd
from tqdm import tqdm

ulist = [
    "other",
    "xAttr",
    "xIntent",
    "xNeed",
    "xWant",
    "xReact",
    "xEffect",
    "oReact",
    "oEffect",
    "oWant",
]


@click.command()
@click.argument("fname", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)",
)
@click.option("--input_col", default="input_text", type=str, help="")
@click.option("--target_col", default="target_text", type=str, help="")
@click.option("--prefix", default="", type=str, help="")
@click.option("--size", default=5000, type=int, help="")
def main(fname, path, input_col, target_col, prefix, size):
    fname = path + "/" + fname
    df = pd.read_table(fname)
    df = df.groupby("prefix").sample(n=size, random_state=1)
    df = df.sample(frac=1, random_state=1)
    tn = "k".join(str(size).rsplit("000", 1))
    tn = tn.replace("kk", "m")
    if not prefix:
        out = "target_score_" + input_col + "_" + target_col + "_" + tn
    else:
        out = (
            "target_score_"
            + prefix
            + "_"
            + input_col
            + "_"
            + target_col
            + "_"
            + tn
        )
    out = path + "/" + out + ".tsv"
    df.to_csv(out, sep="\t")
    with open(f"{path}/labels.txt", "w") as f:
        for label in ulist:
            print(label.strip(), " ", ulist.index(label), file=f)
    print(out, ":", len(df))
    with open(f"{path}/{input_col}.src", "w") as src, open(
        f"{path}/{input_col}.trg", "w"
    ) as trg:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            inp = row[input_col]
            target = row[target_col]
            s = f"{inp} <mask> {target}"
            print(s, file=src)
            l = ulist.index(row["prefix"])
            assert l in range(10), "Uknown lable"
            print(l, file=trg)


if __name__ == "__main__":
    main()
