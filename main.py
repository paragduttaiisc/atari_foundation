import argparse
import numpy as np


def main(args: argparse.Namespace):
    print("Hello, World!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input file path")
    args = parser.parse_args()
    main(args)
