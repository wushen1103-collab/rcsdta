#!/usr/bin/env python
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def split_train_entry_argv(argv: list[str]) -> tuple[Path, list[str]]:
    train_script: Path | None = None
    passthrough_args: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            passthrough_args.extend(argv[index + 1 :])
            break
        if token == "--train-script":
            if index + 1 >= len(argv):
                raise ValueError("--train-script requires a value")
            train_script = Path(argv[index + 1]).resolve()
            index += 2
            continue
        passthrough_args.extend(argv[index:])
        break
    if train_script is None:
        raise ValueError("--train-script is required")
    return train_script, passthrough_args


def main() -> int:
    train_script, train_args = split_train_entry_argv(sys.argv[1:])
    if not train_script.exists():
        raise FileNotFoundError(f"AdaMBind train script not found: {train_script}")

    import torch

    original_torch_load = torch.load

    def patched_torch_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = patched_torch_load

    sys.path.insert(0, str(train_script.parent))
    sys.argv = [str(train_script)] + train_args
    runpy.run_path(str(train_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

