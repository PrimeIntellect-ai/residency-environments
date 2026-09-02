"""Validate and rebuild the clean Eleusis rule dataset.

The output has exactly two splits and four columns (rule_id, label, family,
code). Validation is structural and dataset-agnostic: schema, non-empty
fields, restricted-AST compilation, a valid starter, test-set behavioral
uniqueness, no train/test behavioral overlap, test-family coverage in train,
and matched train/test difficulty distributions. Repeated behavior within train
is allowed so the training split can contain functional variations.

By default the script reads the published dataset, validates it, and rewrites
``outputs/rules_dataset``. Pass ``--push`` to publish the validated result.
"""

from __future__ import annotations

import argparse
import ast
import statistics
from collections import Counter
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset
from eleusis.cards import deck, parse_card
from eleusis.rules import (
    DEFAULT_RULE_DATASET,
    compile_python_rule,
    representative_mainlines,
    rule_extension_signature,
)

COLUMNS = ("rule_id", "label", "family", "code")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rows(split) -> list[dict[str, str]]:
    missing = set(COLUMNS) - set(split.column_names)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return [{column: str(row[column]) for column in COLUMNS} for row in split]


def _acceptance_rate(code: str) -> float:
    rule = compile_python_rule(code)
    outcomes = [
        bool(rule(parse_card(symbol), mainline))
        for mainline in representative_mainlines()
        if mainline
        for symbol in deck()
    ]
    return statistics.mean(outcomes)


def _node_count(code: str) -> int:
    """Count semantic nodes without train-only inert markers and wrappers."""
    body = "\n".join(f"    {line}" for line in code.splitlines())
    wrapper = ast.parse(f"def rule(card, mainline):\n{body}").body[0]
    assert isinstance(wrapper, ast.FunctionDef)
    semantic_body = list(wrapper.body)
    if (
        semantic_body
        and isinstance(semantic_body[0], ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "train_variant_marker" for target in semantic_body[0].targets
        )
    ):
        semantic_body.pop(0)
    base_rule = next(
        (node for node in semantic_body if isinstance(node, ast.FunctionDef) and node.name == "train_base_rule"),
        None,
    )
    if base_rule is not None:
        semantic_body = list(base_rule.body)
        complement_nodes = 1
    else:
        complement_nodes = 0
    tree = ast.Module(body=semantic_body, type_ignores=[])
    return sum(1 for _ in ast.walk(tree)) + complement_nodes


def _assert_similar_difficulty(test: list[dict], train: list[dict]) -> None:
    test_acceptance = [_acceptance_rate(row["code"]) for row in test]
    train_acceptance = [_acceptance_rate(row["code"]) for row in train]
    _require(
        abs(statistics.mean(test_acceptance) - statistics.mean(train_acceptance)) < 0.08,
        "Train/test mean acceptance rates differ too much.",
    )
    for test_q, train_q in zip(
        statistics.quantiles(test_acceptance, n=4),
        statistics.quantiles(train_acceptance, n=4),
    ):
        _require(
            abs(test_q - train_q) < 0.15,
            "Train/test acceptance-rate quartiles differ too much.",
        )

    test_nodes = [_node_count(row["code"]) for row in test]
    train_nodes = [_node_count(row["code"]) for row in train]
    _require(
        abs(statistics.mean(test_nodes) - statistics.mean(train_nodes)) < 8,
        "Train/test mean rule complexities differ too much.",
    )
    _require(
        abs(statistics.median(test_nodes) - statistics.median(train_nodes)) < 8,
        "Train/test median rule complexities differ too much.",
    )


def validate_dataset(dataset: DatasetDict) -> None:
    """Fail unless the dataset has the intended schema and distribution."""
    _require(set(dataset) == {"train", "test"}, "Expected only train and test splits.")
    _require(
        dataset["train"].column_names == list(COLUMNS),
        f"Train columns must be {list(COLUMNS)}.",
    )
    _require(
        dataset["test"].column_names == list(COLUMNS),
        f"Test columns must be {list(COLUMNS)}.",
    )

    train, test = list(dataset["train"]), list(dataset["test"])

    test_families = Counter(row["family"] for row in test)
    train_families = Counter(row["family"] for row in train)
    uncovered = set(test_families) - set(train_families)
    _require(not uncovered, f"Train does not cover test families: {sorted(uncovered)}.")

    test_signatures: dict[tuple[bool, ...], str] = {}
    for split_name, rows in (("test", test), ("train", train)):
        for row in rows:
            _require(
                all(row[column].strip() for column in COLUMNS),
                f"{split_name}/{row['rule_id']} has an empty field.",
            )
            rule = compile_python_rule(row["code"])
            _require(
                any(rule(parse_card(symbol), []) for symbol in deck()),
                f"{split_name}/{row['rule_id']} accepts no starter.",
            )
            signature = rule_extension_signature(row["code"])
            _require(
                signature is not None,
                f"{split_name}/{row['rule_id']} has no valid signature.",
            )
            assert signature is not None
            if split_name == "test":
                _require(
                    signature not in test_signatures,
                    f"test/{row['rule_id']} duplicates {test_signatures.get(signature, 'another test rule')}.",
                )
                test_signatures[signature] = f"test/{row['rule_id']}"
            else:
                _require(
                    signature not in test_signatures,
                    f"train/{row['rule_id']} duplicates {test_signatures.get(signature, 'a test rule')}.",
                )

    _assert_similar_difficulty(test, train)


def build_dataset(source: str) -> DatasetDict:
    source_dataset = load_dataset(source)
    if "train" not in source_dataset or "test" not in source_dataset:
        raise ValueError("Source dataset must contain train and test splits.")
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(_rows(source_dataset["train"])),
            "test": Dataset.from_list(_rows(source_dataset["test"])),
        }
    )
    validate_dataset(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_RULE_DATASET)
    parser.add_argument("--target", default=DEFAULT_RULE_DATASET)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "rules_dataset",
    )
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    dataset = build_dataset(args.source)
    dataset.save_to_disk(str(args.output))
    print(f"validated train={len(dataset['train'])}, test={len(dataset['test'])}; saved to {args.output}")
    if args.push:
        dataset.push_to_hub(args.target)
        print(f"pushed to {args.target}")


if __name__ == "__main__":
    main()
