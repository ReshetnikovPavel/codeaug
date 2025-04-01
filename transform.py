import argparse

import codeaug
from utils import eprint


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform code augmentations on a dataset"
    )

    parser.add_argument(
        "-i",
        "--input",
        type=argparse.FileType("r", encoding="utf-8"),
        required=True,
        help="Input file path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w", encoding="utf-8"),
        help="Output file path. If not provided prints to stdout",
    )
    parser.add_argument(
        "techniques",
        nargs="*",
        type=str,
        help="A list of code augmentations techniques applied in given order."
        " Possible values are: remove-comments, invert-ifs",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    with open(args.input.name, "rb") as input_file:
        content = input_file.read()

        result = content
        for technique in args.techniques:
            match technique:
                case "remove-comments":
                    result = codeaug.remove_comments(result)
                case "invert-ifs":
                    result = codeaug.invert_if_statements(result)
                case _:
                    eprint(f"Technique `{technique}` is not supported")
        if args.output:
            with open(args.output.name, "wb") as output_file:
                output_file.write(result)
        else:
            print(result.decode("utf-8"))
