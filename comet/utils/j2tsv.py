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
    default="",
    type=str,
    help=""
)
def main(fname, path, out):
    f = open(fname)
    entries = json.load(f)
    print("Number of entries:", len(entries))
    if out == ".jsonl":
        outfile = open(Path(path + "/" + fname).stem + out, 'w', encoding="utf8")
        for entry in tqdm(entries, total = len(entries)):
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')
        outfile.close()
    elif out == ".txt" or out == "":
        outname = Path(path + "/" + fname).stem
        inputs = open(outname + "_inputs" + out, 'w', encoding="utf8")
        targets = open(outname + "_targets" + out, 'w', encoding="utf8")
        preds = open(outname + "_predictions" + out, 'w', encoding="utf8")
        for item in tqdm(entries, total = len(entries)):
            head = item["head"].replace("<gen>","").replace("<xIntent>","").strip()
            gen = item["gens"][0]
            for target in item["tails"]:
                print(head, file=inputs)
                print(target, file=targets)
                print(gen, file=preds)
        inputs.close()
        preds.close()
        targets.close()


if __name__ == "__main__":
    main()


