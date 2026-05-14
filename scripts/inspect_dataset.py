#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path

from src.data.inspect_oxford import inspect_hic_root


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hic-root", type=Path, required=True)
    args = parser.parse_args()
    inspect_hic_root(args.hic_root)
