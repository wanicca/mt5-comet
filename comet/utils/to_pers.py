from pathlib import Path
import pandas as pd
import click
from tqdm import tqdm
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
    "--col",
    default="natural_input_text_fa",
    type=str,
    help=""
)
def main(fname, path, col):
    df = pd.read_table(fname)
    rep = []
    for idx, row in tqdm(df.iterrows(), total = len(df)):
        per = row[col]
        per = per.replace("PersonX's", "خود")
        per = per.replace("PersonX", "او")
        per = per.replace("PersonY", "شخص دیگر")
        per = per.replace("PersonZ", "کس دیگری")
        rep.append(per)

    df[col+ "_clean"] = rep 
    df.to_csv(fname, sep="\t")


if __name__ == "__main__":
    main()

