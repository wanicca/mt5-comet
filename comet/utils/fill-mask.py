from transformers import pipeline
import click

@click.command()
@click.option(
    "--mt",
    default="parsbert",
    type=str,
    help=""
)
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
    model_path="gpt2"
    if mt == "parsbert":
        model_path = "/drive2/pretrained/bert/parsbert/bert-base-parsbert-uncased/"
    elif mt == "xlm":
        model_path = "/drive2/pretrained/roberta/xlm-roberta-base"

    from transformers import AutoTokenizer, AutoModelWithLMHead
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelWithLMHead.from_pretrained(model_path)

    nlp = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    print(nlp(f"این یک  {nlp.tokenizer.mask_token}  است"))
    print("Enter a text")
    text = input("Text:")
    while text != "end":
        print(nlp(text.replace("mask", nlp.tokenizer.mask_token)))
        text = input("Text:")


if __name__ == "__main__":
    main()

