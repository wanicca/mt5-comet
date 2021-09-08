from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
import pandas as pd
from tqdm import tqdm
import glob
import torch

def calc(df, before, after, col1, col2, score_col, cpu):
  if after > 0:
    df = df.truncate(before=before, after=after)
  else:
    df = df.truncate(before=before)
  
  df["target_text"] = df["target_text"].astype(str)
  device = torch.device("cuda")
  if cpu:
      device = torch.device("cpu")
  for idx, row in tqdm(df.iterrows(), total=len(df)):
        sents1 = row["target_text"].split("<br />")
        sents2 = row["preds_text"].split("<br />")

        print("+++++++++++++++++++++++++++++++++++++")
        print(sents1)
        print("===================================")
        print(sents2)
        print("===================================")

        #Compute embeddings
        embeddings1 = model.encode(sents1, device=device, convert_to_tensor=True)
        embeddings2 = model.encode(sents2, device=device, convert_to_tensor=True)

        #Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        #print(cosine_scores)

        #Find the pairs with the highest cosine similarity scores
        pairs = []
        rows = cosine_scores.shape[0]
        cols = cosine_scores.shape[1]
        for i in range(rows):
            for j in range(cols):
                pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
                #print({'index': [i, j], 'score': cosine_scores[i][j]})

        #Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

        top = pairs[0]
        df.loc[idx, "top"] = sents2[top["index"][0]]
        df.loc[idx, "pred_text1"] = sents2[top["index"][1]]
        df.loc[idx, "pred1_score"] = "{:.4f}".format(top["score"])

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
    "--col1",
    default="preds_text",
    type=str,
    help=""
)
@click.option(
    "--col2",
    default="target_text",
    type=str,
    help=""
)
@click.option(
    "--score_col",
    default="pred1_score",
    type=str,
    help=""
)
@click.option(
    "--cpu",
    "-",
    is_flag=True,
    help=""
)
def main(fname, path, step, col1, col2, score_col, cpu):
    if fname.endswith("csv"):
        srcdf = pd.read_csv(fname)
    else:
        srcdf = pd.read_table(fname)
        
    before = 0
    after = step
    if after < 0:
        calc(srcdf, before, -1, col1, col2, score_col, cpu)
    else:
        #Path(out).mkdir(exist_ok=True, parents=True)
        df_old = None
        while True:
          print(before, "-", after)
          df = calc(srcdf, before, after, col1, col2, score_col, cpu)
          if df_old is not None:
              df = pd.concat([df_old, df], ignore_index=True)
          df_old = df
          before = after + 1
          after += step
          if after >= len(srcdf):
              after = -1
          if before == 0:
              break
    
    out = path + "/scored_" + Path(fname).stem  + ".tsv" 
    print(out)
    print(len(df))
    df.to_csv(out, sep="\t", index=False)

if __name__ == "__main__":
    main()
