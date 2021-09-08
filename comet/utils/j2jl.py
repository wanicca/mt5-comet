import json
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
    "--out",
    default=".txt",
    type=str,
    help=""
)
def main(fname, path, out):
    f = open(fname)
    data = json.load(f)
    entries = data["data"]
    if out == ".jsonl":
        outfile = open(Path(path + "/" + fname).stem + out, 'w', encoding="utf8")
        for entry in tqdm(entries, total = len(entries)):
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')
        outfile.close()
    elif out == ".txt":
        en = open(Path(path + "/" + fname).stem.replace("fa","") + out, 'w', encoding="utf8")
        fa = open(Path(path + "/" + fname).stem.replace("en","") + out, 'w', encoding="utf8")
        for entry in tqdm(entries, total = len(entries)):
            item = entry["translation"]
            print(item["en"], file=en)
            print(item["fa"], file=fa)
        en.close()
        fa.close()


if __name__ == "__main__":
    main()


