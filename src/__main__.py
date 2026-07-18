import fire

from src.cli import Cli


def main() -> None:
    fire.Fire(Cli)


if __name__ == "__main__":
    main()
