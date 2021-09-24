# +
import sacrebleu
import json
#from transformers import GPT2LMHeadModel, GPT2Tokenizer

#model_name = "outputs/models/last_model"
#tokenizer = GPT2Tokenizer.from_pretrained("outputs/models/last_tokenizer") #model_name)
#model = GPT2LMHeadModel.from_pretrained(model_name) # use_cdn=False)

from pathlib import Path
import pandas as pd
import re
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
    "--src_dfname",
    default="/drive3/pouramini/data/atomic/en_fa/xIntent_en_fa_de_train_no_dups.tsv",
    type=str,
    help=""
)
def main(fname, path, src_dfname):
    file_name = fname
    with open(file_name) as f:
        data = f.readlines()
        generations = [json.loads(elem) for elem in data]

    print(len(generations), " generations")

    preds = {}
    refs = {}
    inp = open(path + "/gpt_inputs", "w")
    targets = open(path + "/gpt_targets", "w")
    predicts = open(path + "/gpt_predictions", "w")
    data = []
    _max = 0
    for item in tqdm(generations, total = len(generations)):
        src = item["source"].replace("[GEN]", "").strip()
        if len(src) > _max:
            _max = len(src)
        ref = item["target"].replace("[EOS]","").strip()
        predlist= []
        for pred in item["generations"]:
            if not "[EOS]" in pred and not "[GEN]" in pred and not "[PAD]" in pred:
                predlist.append("long")
                continue

            if not "[EOS]" in pred:
                if "[PAD]" in pred:
                    result =re.search("\[GEN\](.*)\[PAD\]", pred)
                else:
                    result =re.search("\[GEN\](.*)", pred)
            elif not "[GEN]" in pred:
                result =re.search("\[EOS\](.*)\[EOS\]", pred)
            else:
                result =re.search("\[GEN\](.*)\[EOS\]", pred)
            if result is None:
                predlist.append("null")
                continue

            p = result.group(1)
            if not p.strip() in predlist:
                predlist.append(p.strip())

        cat = "xIntent" #src.split()[0]
        print(src, file=inp)
        print(ref, file=targets)
        preds_text = "<br />".join(predlist)
        print(preds_text, file=predicts)
        data.append([src, ref, preds_text])

    print("max input_text", _max)
    inp.close()
    targets.close()
    predicts.close()
    df = pd.DataFrame(data, columns=["input_text_prompted", "target_text_sel", "preds_text"])
    src_df = pd.read_table(src_dfname)
    src_df["target_text"] = src_df["target_text"].astype(str)
    fname = Path(path + "/" + fname).stem
    #df2 = src_df.groupby(['prefix','input_text'], as_index=False, sort=False)["target_text"].agg('<br />'.join)
    df2 = src_df.drop_duplicates(['prefix','input_text']).merge(src_df.groupby(['prefix','input_text'],as_index=False, sort=False)["target_text"].agg('<br />'.join), on=["prefix","input_text"], suffixes=("_uk", None))
    df2.to_csv(path + "/group_" + fname  + ".tsv", sep = "\t")
    print("max original:", df2.input_text.str.len().max())
    #df3 = pd.merge(df, df2) #, on=["prefix", "input_text"])
    df3 = pd.concat(
        [df2.reset_index(drop=True), df.reset_index(drop=True)], axis=1
    )
    df.to_csv(path + "/" + fname  + ".tsv", sep = "\t")
    df3.to_csv(path + "/merge_" + fname  + ".tsv", sep = "\t")
    print(len(df3))

#    for cat in refs:
#        print(cat)
#        print(preds[cat])
#        print(refs[cat])
#        bleu_score = sacrebleu.sentence_bleu('happy', refs[cat], use_effective_order=True )
#        print(bleu_score.score)



if __name__ == "__main__":
    main()

