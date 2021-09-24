import pandas as pd
from tqdm import tqdm
import glob
from pathlib import Path
import click
@click.command()
@click.argument("fname", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
    help="The current path (it is set by system)"
)
def main(fname, path):
    if fname.endswith("csv"):
        srcdf = pd.read_csv(fname)
    else:
        srcdf = pd.read_table(fname)
        
    inps = glob.glob(f"{path}/*inputs")
    input_file = inps[0]
    inpcol=[]
    f = open(input_file, "r")
    ncc = 0

    def persianReplace(text):
        text = text.strip().replace("PersonX's", "ش").replace("PersonY", "رضا")
        text = text.strip().replace("PersonX", "علی").replace("PersonZ", "حمید")
        return text

    for x in f:
        text = x
        if "fa" in Path(path).stem:
            text = text[2:-1]
            text = text.encode("raw_unicode_escape")
            text = text.decode("unicode_escape")
            text = text.encode("raw_unicode_escape")
            text = text.decode()
            text = persianReplace(text)
        text = text.strip()
        inpcol.append(text)

        
    if not "/" in fname:
        fname = path + "/" + fname
    if not Path(fname).is_file():
        print(f"A file with the name {fname} wasn't found")
        return
    inpcol = list(set(inpcol))
    if len(inpcol) == len(srcdf) + 1:
        inpcol = inpcol[1:]
    print("len preds:", len(inpcol))
    print("len srcdf:", len(srcdf))
    assert len(inpcol) == len(srcdf), "The length of source and prediction files must be equal"
    srcdf["input_text_prompted"] = inpcol
    srcdf.to_csv(fname, index=False, sep="\t")
    print("inputs were added")

if __name__ == "__main__":
    main()

