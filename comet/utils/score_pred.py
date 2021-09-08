from pathlib import Path
import click
import pandas as pd
from tqdm import tqdm
ulist = ["other", "xAttr", "xIntent", "xNeed", "xWant", "xReact", "xEffect", "oReact", "oEffect", "oWant"]
@click.command()
@click.argument("fname", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
@click.option(
    "--input_col",
    default="input_text",
    type=str,
    help=""
)
@click.option(
    "--pred_col",
    default="pred_text1",
    type=str,
    help=""
)
@click.option(
    "--prefix",
    default="",
    type=str,
    help=""
)
def main(fname, path, input_col, pred_col, prefix):
    if not prefix:
        out = "pred_score_" + input_col + "_" + pred_col 
    else:
        out = "pred_score_" + prefix + "_" + input_col + "_" + pred_col  
    out = path + "/" + out
    fname = path + "/" + fname
    df = pd.read_table(fname)
    with open(f'{path}/labels.txt','w') as f:
        for label in ulist:
            print(label.strip(),' ', ulist.index(label), file=f)
    print(out, ":", len(df))
    with open(f'{path}/{input_col}.src', 'w') as src,  open(f'{path}/{input_col}.trg', 'w') as trg:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            inp = row[input_col]
            pred = row[pred_col]
            s = f'{inp} <mask> {pred}'
            print(s, file= src)
            l = ulist.index(row["prefix"])
            assert l in range(10), "Uknown lable"
            print(l, file= trg)

if __name__ == "__main__":
    main()
