"""Index-level evaluation of the files in output/.

Scores the recommendations a configuration actually delivered against the
sections the developer edited, and reports End-to-end Success over all positive
pull requests and Index Recall, MRR and Hierarchical Recall at levels 1 to 4
over the delivered ones.

The README structure is rebuilt from the prepared dataset, so the dataset has
to be downloaded first, see the Dataset section of the top-level README.

Usage:
    python index_level.py --dataset /path/to/prepared-data-augmented.jsonl
    python index_level.py --dataset ... --model qwen
"""

import argparse
import json
import os
from collections import defaultdict

import hierarchical_eval
from metrics import (index_recall, parse_generation_to_json, prediction_to_set,
                     reciprocal_rank)
from readme_parsing import build_tree_with_paragraph_mapping

CONFIGS = [
    ("gemma-3-27b-it, static", "static-gemma"),
    ("gemma-3-27b-it, agentic", "agentic-gemma"),
    ("Qwen3-32B, static", "static-qwen"),
    ("Qwen3-32B, agentic", "agentic-qwen"),
    ("Llama-3.3-70B-Instruct-AWQ, static", "static-llama"),
    ("Llama-3.3-70B-Instruct-AWQ, agentic", "agentic-llama"),
    ("Qwen3-32B, baseline prompt", "baseline-Qwen3-32B"),
]

DECISION_FIELDS = ("final_recommendation", "baseline_decision")


def delivered(entry):
    for field in DECISION_FIELDS:
        if field in entry:
            value = entry[field]
            if isinstance(value, bool):
                return value
            return str(value).strip().upper().startswith("YES")
    return False


def predicted_sections(entry):
    """The section ids a configuration returned, or None if unparseable."""
    if entry.get("generated_output") is not None:
        try:
            parsed = parse_generation_to_json(entry["generated_output"])
        except Exception:
            return None
        if parsed is None:
            return None
        return parsed.get("section_ids")
    return entry.get("section_ids")


def score(path, index_dictionary, tree_cache):
    recalls, ranks = [], []
    per_level = defaultdict(list)
    scored = unparseable = complete = 0
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if not delivered(entry):
                continue
            ground_truth = entry.get("ground_truth_idx")
            if not isinstance(ground_truth, list):
                continue
            prediction = predicted_sections(entry)
            predicted = prediction_to_set(prediction) if prediction is not None else None
            if predicted is None:
                unparseable += 1
                prediction, predicted = [], set()
            scored += 1
            recalls.append(index_recall(ground_truth, predicted))
            ranks.append(reciprocal_rank(ground_truth, predicted))
            if ground_truth and set(ground_truth) <= predicted:
                complete += 1

            key = (entry["repo_name"], entry["pr_number"])
            if key not in index_dictionary:
                continue
            if key not in tree_cache:
                try:
                    tree_cache[key] = build_tree_with_paragraph_mapping(index_dictionary[key])
                except Exception:
                    tree_cache[key] = None
            root = tree_cache[key]
            if root is None:
                continue
            try:
                result = hierarchical_eval.tree_level_eval(root, ground_truth, prediction)
            except Exception:
                continue
            gt_counts = result.get("gt_path_count_by_level", {})
            matches = result.get("match_count_by_level", {})
            for level, count in gt_counts.items():
                if level in (1, 2, 3, 4):
                    per_level[level].append(matches.get(level, 0) / count if count else 0)

    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "n": scored,
        "unparseable": unparseable,
        "complete": complete,
        "ir": mean(recalls),
        "mrr": mean(ranks),
        "hr": [mean(per_level[level]) for level in (1, 2, 3, 4)],
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="path to prepared-data-augmented.jsonl")
    parser.add_argument("--output_dir", default=os.path.join(here, "..", "output"))
    parser.add_argument("--model", default=None,
                        help="substring of a configuration prefix, e.g. qwen")
    args = parser.parse_args()

    index_dictionary = hierarchical_eval.load_index_dictionary(args.dataset)
    tree_cache = {}

    paths = {prefix: os.path.join(args.output_dir, prefix + "-positive.jsonl")
             for _, prefix in CONFIGS}
    positives = max(sum(1 for line in open(path) if line.strip())
                    for path in paths.values() if os.path.exists(path))

    print("%-38s %7s %7s %7s %7s  %s" % ("Configuration", "n", "E2E", "IR", "MRR",
                                         "HR (L1, L2, L3, L4)"))
    for label, prefix in CONFIGS:
        if args.model and args.model.lower() not in prefix.lower():
            continue
        path = paths[prefix]
        if not os.path.exists(path):
            print("%-38s %s" % (label, "file not found"))
            continue
        result = score(path, index_dictionary, tree_cache)
        hr = ", ".join("%.2f" % value for value in result["hr"])
        print("%-38s %7d %7.4f %7.4f %7.4f  (%s)" % (
            label, result["n"], result["complete"] / positives, result["ir"],
            result["mrr"], hr))


if __name__ == "__main__":
    main()
