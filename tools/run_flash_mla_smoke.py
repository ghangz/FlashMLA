#!/usr/bin/env python3
import argparse
import random
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_flash_mla import test_flash_mla  # noqa: E402


def _dtype(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    raise argparse.ArgumentTypeError("dtype must be bf16 or fp16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small FlashMLA correctness smoke test on one MACA device."
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device to run on.")
    parser.add_argument("--dtype", type=_dtype, default=torch.bfloat16, help="bf16 or fp16.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--s-q", type=int, default=1)
    parser.add_argument("--mean-sk", type=int, default=4096)
    parser.add_argument("--h-q", type=int, default=16)
    parser.add_argument("--h-kv", type=int, default=1)
    parser.add_argument("--d", type=int, default=576)
    parser.add_argument("--dv", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--non-causal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("FlashMLA smoke test requires a CUDA-compatible MACA device.")

    torch.set_default_dtype(args.dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    print(f"torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name(device)}")
    print(f"dtype={args.dtype}")

    test_flash_mla(
        b=args.batch_size,
        s_q=args.s_q,
        mean_sk=args.mean_sk,
        h_q=args.h_q,
        h_kv=args.h_kv,
        d=args.d,
        dv=args.dv,
        causal=not args.non_causal,
        varlen=args.varlen,
        block_size=args.block_size,
    )
    print("flash_mla_smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
