"""Shared metric helpers and LLM-output parsing.

All metric logic is ported unchanged from the original evaluation scripts
(~/eval/agentic-folder/location-eval.py) so the paper numbers reproduce
exactly. See README for the mapping between functions and paper equations.
"""

import ast
import csv
import json
import re


def reciprocal_rank(g, prediction_set):
    """Reciprocal rank as computed in the original evaluation.

    NOTE: this iterates the *ground truth* list in order and returns 1/(i+1)
    for the first ground-truth index found among the predictions. Kept
    verbatim from the original code so results match the paper.
    """
    for i, item in enumerate(g):
        if item in prediction_set:
            return 1 / (i + 1)
    return 0


def index_recall(g, prediction_set):
    """|G ∩ P| / |G| (paper: Index Recall)."""
    return len(prediction_set.intersection(g)) / len(g) if len(g) > 0 else 0


def user_facing_accuracy(entry_recall, entry_specificity, prevalence=0.01):
    """Paper Eq. 3: simulated precision under a 1:99 positive:negative ratio."""
    pos = prevalence * 100
    neg = 100 - pos
    denom = entry_recall * pos + (1 - entry_specificity) * neg
    return entry_recall * pos / denom if denom > 0 else 0


def parse_generation_to_json(generation):
    """Robustly extract a JSON object from an LLM response.

    Returns None for None/non-string input; raises ValueError when no JSON
    object can be found. (Final version from location-eval.py.)
    """
    if generation is None:
        return None
    if not isinstance(generation, str):
        return None

    s = generation.strip()

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = s[start:end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Found braces but failed to parse: {e}") from e

    raise ValueError(f"No JSON object found. Content length: {len(s)}")


def clean_and_parse_list_string(s):
    """Parse a raw generated list like "[1, 2, 3]" leniently."""
    s = re.sub(r"[^\d,\[\]]", "", s)
    return ast.literal_eval(s)


def load_pair_set(csv_path):
    """Load a set of (repo_name, pr_number) pairs from a headered CSV."""
    pairs = set()
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            pairs.add((row[0], int(row[1])))
    return pairs


def prediction_to_set(p):
    """Normalise a parsed prediction (int or list/tuple) to a set, or None."""
    if isinstance(p, int):
        return {p}
    if isinstance(p, (list, tuple)):
        try:
            return set(p)
        except Exception:
            return None
    return None


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
