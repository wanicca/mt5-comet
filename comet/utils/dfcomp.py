import pandas as pd
from pathlib import Path
target_col = 'natural_target_text'
def make_link(fid, dpath, caption):
    return f'<a cmd="replace_df" href="/" class="df-link" tag="{fid}" path="{dpath}">Browse</a>'

total_items = 1
unique_items = 1
row1={}
row2={}
row3={}
row4={}

def fill_rows(label, df, fid, dpath):
    global row1, row2, row3, row4

    df.to_csv(dpath, sep="\t")
    c = len(df)
    row1[label] = c 
    row2[label] = "{:.2f}".format(c / total_items) 
    row3[label] = "{:.2f}".format(c / unique_items)
    row4[label] =  make_link(fid, dpath, c)

def make_report(path, fid, df):
    global total_items, unique_items, row1, row2, row3

    nones = ["none", "هیچ یک"]
    dpath = f"{path}/all_{fid}.tsv"

    df.to_csv(dpath, sep="\t")
    total_items = len(df)
    unique_items = df['input_text'].nunique()

    fill_rows("Total rows", df, fid, dpath) 
    tdf = df.groupby('input_text').first()
    dpath = f"{path}/unique_{fid}.tsv"
    fill_rows("Unique events", tdf, fid, dpath) 

    tdf = df[df.duplicated(subset=['input_text','target_text'], keep=False)]
    dpath = f"{path}/dups_en_{fid}.tsv"
    fill_rows("English Duplicates", tdf, fid, dpath)

    tdf = df[df.duplicated(subset=['input_text_fa','target_text'], keep=False)]
    dpath = f"{path}/dups_fa_{fid}.tsv"
    fill_rows("Persian Duplicates", tdf, fid, dpath)

    tdf = (df[df['pred_text1'] == df['target_text']])
    dpath = f"{path}/match_en_{fid}.tsv"
    fill_rows("English Matches", tdf, fid, dpath) 

    tdf = (df[df['pred_text2'] == df['target_text']])
    dpath = f"{path}/match_fa_{fid}.tsv"
    fill_rows("Persian Matches", tdf, fid, dpath) 

    tdf = (df[(df['pred_text2'] == df['target_text']) 
             & (df['pred_text1'] != df['target_text'])
             & ~df['target_text'].isin(nones)])
    dpath = f"{path}/b2_{fid}.tsv"
    fill_rows("Persian is Better",tdf, fid, dpath) 

    tdf = (df[(df['pred_text1'] == df['target_text']) 
             & (df['pred_text2'] != df['target_text'])
             & ~df['target_text'].isin(nones)])
    dpath = f"{path}/b1_{fid}.tsv"
    fill_rows("English is Better", tdf, fid, dpath) 

    pdf = (df[(df['pred_text1'] == df['target_text']) & df['target_text'].isin(nones)])
    fill_rows("English None Matches",pdf, fid, dpath=f"{path}/none_matches_en_{fid}.tsv")
 
    pdf = (df[(df['pred_text2'] == df['target_text']) & df['target_text'].isin(nones)])
    fill_rows("Persian None Matches",pdf, fid, dpath=f"{path}/none_matches_fa_{fid}.tsv")

    qdf = (df[(df['pred_text1'] == df['target_text']) & ~df['target_text'].isin(nones)])
    fill_rows("English Not None Matches",qdf, fid, dpath=f"{path}/none_matches_en_{fid}.tsv")

    qdf = (df[(df['pred_text2'] == df['target_text']) & ~df['target_text'].isin(nones)])
    fill_rows("Persian Not None Matches", qdf, fid, dpath=f"{path}/none_matches_fa_{fid}.tsv")

    tdf = (df[df['pred_text1'] == 'none'])
    fill_rows("English None Predictions",tdf, fid, dpath=f"{path}/none_preds_en_{fid}.tsv")

    tdf = (df[df['pred_text2'] == 'none'])
    fill_rows("Persian None Predictions",tdf, fid, dpath=f"{path}/none_preds_fa_{fid}.tsv")

    tdf = (df[df['target_text'] == 'none'])
    fill_rows("None Targets",tdf, fid, dpath=f"{path}/none_targets_en_{fid}.tsv")

    tdf = df[(df['pred_text1'] == df['pred_text2']) & ~df['pred_text1'].isin(nones)]
    tdf = tdf.drop_duplicates(subset=['input_text', 'pred_text1'])
    fill_rows("Not None Common Predictions",tdf, fid, dpath=f"{path}/nn_commom_{fid}.tsv")
    tdf = (df[(df['pred_text1'] == df['pred_text2']) & df['pred_text1'].isin(nones)])
    fill_rows("None Common Predictions",tdf, fid, dpath=f"{path}/none_commom_{fid}.tsv")

    tdf = df[(df['pred_text1'] == df['pred_text2']) & 
            (df['pred_text1'] == df['target_text']) & 
            ~df['pred_text1'].isin(nones)]
    fill_rows("Common Matches", tdf, fid, dpath=f"{path}/all_commom_{fid}.tsv")

    match_df = pd.DataFrame(columns=row1.keys())
    match_df = match_df.append(row1, ignore_index=True)
    match_df = match_df.append(row2, ignore_index=True)
    match_df = match_df.append(row3, ignore_index=True)
    match_df = match_df.append(row4, ignore_index=True)
    retdf = match_df.T
    retdf.columns = ["Items", "Per Total", "Per Unique Events", ""]
    return retdf
