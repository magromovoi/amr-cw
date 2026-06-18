import os
import sys
import json
import hashlib
import argparse

from amr_cw.generation.templates import TEMPLATES

FROZEN_PATH = "data/output/synthetic_corpus/frozen_params.json"


def template_hash():
    payload = json.dumps(TEMPLATES, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=float, required=True)
    ap.add_argument("--min-support-abs", type=int, default=None)
    ap.add_argument("--generator-model", default="claude-opus-4-8")
    ap.add_argument("--annotator-model", default="gpt-4.1-2025-04-14")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stability", type=int, default=5)
    ap.add_argument("--out", default=FROZEN_PATH)
    args = ap.parse_args()

    if os.path.exists(args.out):
        sys.exit(1)

    record = {
        "min_support": args.min_support,
        "min_support_abs": args.min_support_abs,
        "stability_threshold": args.stability,
        "template_hash": template_hash(),
        "template_set": {c: [t["id"] for t in ts] for c, ts in TEMPLATES.items()},
        "generator": {"model": args.generator_model, "temperature": None, "top_p": None},
        "annotator": {"model": args.annotator_model},
        "seed": args.seed,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
