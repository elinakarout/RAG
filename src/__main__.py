def main() -> None:
    """Expose Cli's methods as Fire-generated CLI subcommands."""
    fire.Fire(Cli)


if __name__ == "__main__":
    try:
        import fire
        from src.cli import Cli

        main()
    except KeyboardInterrupt:
        print("Program Exited.")
    except Exception:
        print("Error.")
    finally:
        print("Thank you for checking my project :)")
