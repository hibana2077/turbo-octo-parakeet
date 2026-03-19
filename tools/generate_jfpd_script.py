#!/usr/bin/env python3
import argparse
import re
import shlex
from collections import OrderedDict
from pathlib import Path


FLAG_ARGS = {
    "--use_cp",
    "--use_im",
    "--use_jfpd",
    "--no_pretrained",
    "--fp16",
    "--is_test",
}


def normalize_spacing(line: str) -> str:
    # Fix known typo patterns like: "--gpu_id 1--use_cp"
    return re.sub(r"(?<!\s)(--[A-Za-z0-9_]+)", r" \1", line)


def parse_command(line: str):
    parts = shlex.split(normalize_spacing(line))
    if len(parts) < 2:
        return None
    if parts[0] not in {"python", "python3", ".venv/bin/python3"}:
        return None
    if parts[1] != "main.py":
        return None

    args = OrderedDict()
    i = 2
    while i < len(parts):
        token = parts[i]
        if not token.startswith("--"):
            i += 1
            continue
        if token in FLAG_ARGS:
            args[token] = True
            i += 1
            continue
        if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
            args[token] = parts[i + 1]
            i += 2
        else:
            args[token] = None
            i += 1

    return args


def rewrite_args(
    args: OrderedDict,
    beta: str,
    gamma: str,
    theta: str,
    jfpd_lambda: str,
    jfpd_alpha: str,
    jfpd_mode: str,
    name_suffix: str,
):
    args.pop("--use_im", None)
    args.pop("--perturbationRatio", None)

    args["--beta"] = beta
    args["--gamma"] = gamma
    args["--theta"] = theta

    args["--use_jfpd"] = True
    args["--jfpd_lambda"] = jfpd_lambda
    args["--jfpd_alpha"] = jfpd_alpha
    args["--jfpd_mode"] = jfpd_mode

    if name_suffix and "--name" in args and args["--name"]:
        args["--name"] = f"{args['--name']}{name_suffix}"

    return args


def build_command(args: OrderedDict, python_bin: str) -> str:
    cmd = [python_bin, "main.py"]
    for key, value in args.items():
        if value is True:
            cmd.append(key)
        elif value is None:
            continue
        else:
            cmd.extend([key, str(value)])
    return " ".join(cmd)


def main():
    parser = argparse.ArgumentParser(description="Generate JFPD-only commands from script.txt")
    parser.add_argument("--input", default="script.txt", help="Input command file.")
    parser.add_argument("--output", default="jfpd_script.txt", help="Output command file.")
    parser.add_argument("--python-bin", default=".venv/bin/python3", help="Python binary to use in generated commands.")
    parser.add_argument("--beta", default="0.0")
    parser.add_argument("--gamma", default="0.0")
    parser.add_argument("--theta", default="0.0")
    parser.add_argument("--jfpd_lambda", default="0.5")
    parser.add_argument("--jfpd_alpha", default="0.5")
    parser.add_argument("--jfpd_mode", default="jfpd", choices=["jfpd", "fgpd", "pgfd"])
    parser.add_argument("--name-suffix", default="", help="Optional suffix appended to --name.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_lines = []
    total_cmd = 0
    converted_cmd = 0
    for raw in input_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        parsed = parse_command(stripped)
        if not parsed:
            out_lines.append(raw)
            continue

        total_cmd += 1
        new_args = rewrite_args(
            parsed,
            beta=args.beta,
            gamma=args.gamma,
            theta=args.theta,
            jfpd_lambda=args.jfpd_lambda,
            jfpd_alpha=args.jfpd_alpha,
            jfpd_mode=args.jfpd_mode,
            name_suffix=args.name_suffix,
        )
        out_lines.append(build_command(new_args, args.python_bin))
        converted_cmd += 1

    header = [
        "# Auto-generated from script.txt",
        "# Objective: keep only loss_clc + jfpd (set beta/gamma/theta to 0, remove --use_im)",
        f"# Converted commands: {converted_cmd}/{total_cmd}",
        "",
    ]
    output_path.write_text("\n".join(header + out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
