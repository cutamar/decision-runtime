"""Minimal customer integration: the application owns the routing action."""

import argparse
from pathlib import Path

from decision_runtime import DecisionModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--trusted-public-key", type=Path)
    args = parser.parse_args()
    key = args.trusted_public_key.read_bytes() if args.trusted_public_key else None
    model = DecisionModel.load(args.bundle, trusted_public_key=key,
                               allow_unsigned=key is None)
    for text in ("I was billed twice", "I cannot sign in"):
        result = model.predict(text)
        if result.status == "accepted":
            print({"route_to": result.choice, "model_version": result.model_version})
        else:
            print({"route_to": "manual_review", "reason": result.abstention_reason})


if __name__ == "__main__":
    main()
