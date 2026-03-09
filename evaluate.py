import sys

from src.main import main


if __name__ == "__main__":
    if "--mode" not in sys.argv:
        sys.argv.extend(["--mode", "evaluate"])
    main()
