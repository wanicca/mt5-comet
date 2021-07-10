from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
import pandas as pd
from tqdm import tqdm


def calc(df, before, after, comp2):
  if after > 0:
    df = df.truncate(before=before, after=after)
  else:
    df = df.truncate(before=before)
  
  pred1 = df['pred_text1'].astype(str).tolist()
  if comp2 and "pred_text2" in df:
      pred2 = df['pred_text2'].astype(str).tolist()
  else:
      comp2 = False
  target = df[target_col].astype(str).tolist()
  
  pred1_emb = model.encode(pred1, convert_to_tensor=True)
  target_emb = model.encode(target, convert_to_tensor=True)
  if comp2:
      pred2_emb = model.encode(pred2, convert_to_tensor=True)
  
  #Compute cosine-similarits
  pred1_cosine_scores = util.pytorch_cos_sim(pred1_emb, target_emb)
  if comp2:
      preds_cosine_scores = util.pytorch_cos_sim(pred1_emb, pred2_emb)
      pred2_cosine_scores = util.pytorch_cos_sim(pred2_emb, target_emb)
  
  #Output the pairs with their score
  preds_scores = []
  pred1_scores = []
  pred2_scores = []
  for i in tqdm(range(len(df)), total = len(df)):
      pred1_scores.append("{:.4f}".format(pred1_cosine_scores[i][i]))
      if comp2:
          preds_scores.append("{:.4f}".format(preds_cosine_scores[i][i]))
          pred2_scores.append("{:.4f}".format(pred2_cosine_scores[i][i]))
  
  print(len(preds_scores))
  df['pred1_score'] = pred1_scores
  if comp2:
      df['pred2_score'] = pred2_scores
      df['preds_score'] = preds_scores
  return df


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
@click.option(
    "--step",
    default=10000,
    type=int,
    help=""
)
@click.option(
    "--comp2",
    "-",
    is_flag=True,
    help=""
)
def main(fname, path, step, comp2):
    before = 0
    after = step
    if fname.endswith("csv"):
        orig_df = pd.read_csv(fname)
    else:
        orig_df = pd.read_table(fname)
    out = path + "/scored_" + fname.replace(".csv", ".tsv")
    if after < 0:
        print(out)
        calc(orig_df, before, -1, comp2)
    else:
        #Path(out).mkdir(exist_ok=True, parents=True)
        df_old = None
        while True:
          print(before, "-", after)
          df = calc(orig_df, before, after, comp2)
          if df_old is not None:
              df = pd.concat([df, df_old], ignore_index=True)
          df_old = df
          before = after
          after += step
          if after >= len(orig_df):
              after = -1
          if before == -1:
              break
    df.to_csv(out, sep="\t")

if __name__ == "__main__":
    main()
