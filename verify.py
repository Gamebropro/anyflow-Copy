from __future__ import annotations

import argparse
import json

from .verification import run_anyflow_verification


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ANYFLOW environment and model readiness checks.")
    parser.add_argument("--device", default=None, help="Device for smoke checks, for example cpu or cuda.")
    parser.add_argument("--dim", type=int, default=16, help="Small model width for smoke checks.")
    parser.add_argument("--vocab-size", type=int, default=32, help="Synthetic vocabulary size for smoke checks.")
    parser.add_argument("--strict-versions", action="store_true", help="Require PyTorch version prefix 2.10.0.")
    parser.add_argument("--require-sm75", action="store_true", help="Require compute capability sm_75.")
    parser.add_argument("--require-tilelang", action="store_true", help="Require TileLang version prefix 0.1.10.")
    args = parser.parse_args()
    report = run_anyflow_verification(
        strict_versions=args.strict_versions,
        require_sm75=args.require_sm75,
        require_tilelang=args.require_tilelang,
        device=args.device,
        dim=args.dim,
        vocab_size=args.vocab_size,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.passed_required else 1


if __name__ == "__main__":
    raise SystemExit(main())
