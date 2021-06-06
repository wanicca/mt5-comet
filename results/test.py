import click
@click.command()
# @click.argument("results_dir", type=str)
@click.option(
    "--path",
    envvar="PWD",
    #    multiple=True,
    type=click.Path(),
)
@click.option(
    "--opt",
    is_flag=True,
    help=""
)
def test(path, opt):
    print(opt)

if __name__ == "__main__":
    test()

