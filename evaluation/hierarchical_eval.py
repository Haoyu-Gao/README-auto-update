"""Hierarchical recall (paper Section V, Eq. 6, "Hierarchical Recall" L1–L4).

The README is parsed into a header tree (levels 1–4). A predicted paragraph
index counts as a level-i match if it falls under the same level-i node as a
ground-truth index. Logic ported unchanged from
~/eval/task2_hierarchical_eval/hierarchical_eval.py.
"""

from collections import defaultdict

import numpy as np

from readme_parsing import build_tree_with_paragraph_mapping, split_readme_sections_vanilla
from metrics import iter_jsonl, parse_generation_to_json, clean_and_parse_list_string

_INDEX_CACHE = {}


def load_index_dictionary(default_file):
    """Map (repo, pr) -> paragraph texts of the pre-update README."""
    if default_file in _INDEX_CACHE:
        return _INDEX_CACHE[default_file]
    index_dictionary = {}
    for data_entry in iter_jsonl(default_file):
        repo_name = data_entry["repo_name"]
        pr_number = data_entry["pr_number"]
        paragraph_texts, _ = split_readme_sections_vanilla(data_entry)
        index_dictionary[(repo_name, pr_number)] = paragraph_texts
    _INDEX_CACHE[default_file] = index_dictionary
    return index_dictionary


def hierarchical_recall(evaluation_file, eval_set, default_file, invalid_entries,
                        parse_mode="json", verbose=True):
    """Compute hierarchical recall for entries of `evaluation_file` in `eval_set`.

    parse_mode: "json" -> generated_output is JSON with a "section_ids" field
                "list" -> generated_output is a raw list string like "[1, 2]"
    Returns dict {level: mean recall} and prints the same summary the original
    script printed.
    """
    ground_truth = []
    predictions = []
    invalid_generation_count = 0
    repo_pr = []

    level_1, level_2, level_3, level_4 = [], [], [], []

    for data_entry in iter_jsonl(evaluation_file):
        repo_name = data_entry["repo_name"]
        pr_number = data_entry["pr_number"]
        if (repo_name, pr_number) in invalid_entries:
            continue
        if eval_set is not None and (repo_name, pr_number) not in eval_set:
            continue

        generated_output = data_entry["generated_output"]
        try:
            if parse_mode == "json":
                predictions.append(parse_generation_to_json(generated_output)["section_ids"])
            else:
                predictions.append(clean_and_parse_list_string(generated_output))
            ground_truth.append(data_entry["ground_truth_idx"])
            repo_pr.append((repo_name, pr_number))
        except Exception:
            invalid_generation_count += 1
            continue

    index_dictionary = load_index_dictionary(default_file)

    ground_truth_levels = []
    predictions_levels = []

    for idx, (repo_name, pr_number) in enumerate(repo_pr):
        try:
            if (repo_name, pr_number) not in index_dictionary:
                continue

            paragraph_texts = index_dictionary[(repo_name, pr_number)]
            root = build_tree_with_paragraph_mapping(paragraph_texts)
            calculate_levels(root, level_1, level_2, level_3, level_4)

            if type(ground_truth[idx]) is not list or type(predictions[idx]) is not list:
                continue

            result = tree_level_eval(root, ground_truth[idx], predictions[idx])
            ground_truth_levels.append(result["gt_path_count_by_level"])
            predictions_levels.append(result["match_count_by_level"])
        except Exception:
            continue

    level_recall = tree_level_eval_stats(ground_truth_levels, predictions_levels, verbose=verbose)
    if verbose:
        print("Instances evaluated:", len(ground_truth_levels),
              "| invalid generations:", invalid_generation_count)
    return level_recall


def calculate_levels(root, level_1, level_2, level_3, level_4):
    counts = [0, 0, 0, 0]

    def traverse(node):
        if 1 <= node.level <= 4:
            counts[node.level - 1] += 1
        for child in node.children:
            traverse(child)

    traverse(root)
    level_1.append(counts[0])
    level_2.append(counts[1])
    level_3.append(counts[2])
    level_4.append(counts[3])


def tree_level_eval_stats(ground_truth_levels, predictions_levels, verbose=True):
    per_level = {1: [], 2: [], 3: [], 4: []}
    for gt_level, pred_level in zip(ground_truth_levels, predictions_levels):
        for level, count in gt_level.items():
            pred_count = pred_level.get(level, 0)
            if level in per_level:
                per_level[level].append(pred_count / count if count > 0 else 0)

    result = {}
    for level in (1, 2, 3, 4):
        result[level] = float(np.mean(per_level[level])) if per_level[level] else 0
        if verbose:
            print(f"Level {level} Recall: {result[level]:.4f}")
    return result


def tree_level_eval(root, ground_truth, prediction):
    if type(ground_truth) is not list or type(prediction) is not list:
        return {}

    ground_truth_set = get_node_set(root, ground_truth)
    prediction_set = get_node_set(root, prediction)

    gt_paths = collapse_paths(get_paths_from_nodes(ground_truth_set))
    pred_paths = collapse_paths(get_paths_from_nodes(prediction_set))

    gt_grouped = group_paths_by_level(gt_paths)
    match_counts = count_matches_by_level(gt_grouped, pred_paths)

    return {
        "gt_path_count_by_level": {level: len(paths) for level, paths in gt_grouped.items()},
        "match_count_by_level": dict(match_counts),
    }


def get_node_set(root, para_indices):
    node_set = set()
    for idx in para_indices:
        node = find_node_for_paragraph(root, idx)
        if node:
            node_set.add(node)
    return node_set


def find_node_for_paragraph(node, paragraph_idx):
    if node.interval[0] <= paragraph_idx <= node.interval[1]:
        for child in node.children:
            result = find_node_for_paragraph(child, paragraph_idx)
            if result:
                return result
        return node
    return None


def get_paths_from_nodes(nodes):
    paths = []
    for node in nodes:
        path = []
        while node:
            path.append(node)
            node = node.parent
        paths.append(list(reversed(path)))
    return paths


def collapse_paths(paths):
    collapsed = []
    seen = set()
    for path in paths:
        for node in path:
            if node not in seen:
                collapsed.append(node)
                seen.add(node)
    for node in collapsed:
        if node.level == 0:
            collapsed.remove(node)
    return collapsed


def group_paths_by_level(paths):
    grouped = defaultdict(set)
    for path in paths:
        grouped[path.level].add(path)
    return grouped


def count_matches_by_level(gt_grouped, pred_paths):
    matches = defaultdict(int)
    pred_grouped = group_paths_by_level(pred_paths)

    for level in gt_grouped.keys():
        gt_level_paths = gt_grouped[level]
        pred_level_paths = pred_grouped.get(level, None)
        if pred_level_paths is None:
            continue
        for gt_path in gt_level_paths:
            if gt_path in pred_level_paths:
                matches[level] += 1
    return matches
