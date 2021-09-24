
from pathlib import Path
import pandas as pd
import click
from tqdm import tqdm
import torch


@click.command()
@click.argument("mt", type=str)
@click.option(
        "--path",
        envvar="PWD",
        #    multiple=True,
        type=click.Path(),
        help="The current path (it is set by system)"
        )
@click.option(
    "--text",
    default="This is a test.",
    type=str,
    help=""
)
def main(mt, path, text):
    device = 'cuda'
    if mt == "gpt":
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        #model_id = 'gpt2-large'
        model_id = 'gpt2'
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    else:
        from transformers import AutoTokenizer, AutoModelWithLMHead
        tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/gpt2-fa")
        model = AutoModelWithLMHead.from_pretrained(
                "HooshvareLab/gpt2-fa").to(device)

    # from datasets import load_dataset
    # test = load_dataset("persiannlp/parsinlu_translation_en_fa", split="validation")
    items = text.split(".")
    print(items)
    encodings = tokenizer('\n\n'.join(items), return_tensors='pt')

    max_length = model.config.n_positions
    stride = 512

    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        print(log_likelihood)
        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    print("ppl:", ppl)
    print("end loc:", end_loc)

if __name__ == "__main__":
    main()
