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
    "--cols",
    default="",
    type=str,
    help=""
)
def main(fname, path, cols):
    df = pd.read_table(fname)
    df = df[cols.split("-")]
    print(df.head())

if __name__ == "__main__":
    main()

