from vllm import SamplingParams
from typing import Union, List, Dict

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import json
import numpy as np
import os


os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""

cache_dir = ""
hf_token = os.getenv("HF_TOKEN")

inspection_set = set()


sbert_model = None

def stream_json_file(file_path):
    with open(file_path) as f:
        for idx, line in enumerate(f.readlines()):
            if not line.strip():
                continue
                
            data = json.loads(line)

            if (data['repo_name'], int(data['pr_number'])) not in inspection_set:
                continue
            try:
                pr_description = data['pr_title'] + " " + data['pr_description']
            except:
                pr_description = data['pr_description'] or data['pr_title'] or ""

            commit_message = data.get('commit_comments', "") or ""
            changed_files = data.get('files_changed', []) or []
            if "README.md" in changed_files:
                changed_files.remove("README.md")
            readme_content = data.get('readme_base', "")
            lines_changed = data.get('line_positions', [])
            changed_file_patches = data.get("files_patch", "")

            sections, changed_section_idx = split_readme_sections_vanilla({"readme_base": readme_content, "line_positions": lines_changed})
            repo = data['repo_name'] 
            pr_id = data['pr_number']



            with open("") as f1:
                # find the line with the same repo_name and pr_number
                for line1 in f1.readlines():
                    data1 = json.loads(line1)
                    if data1["repo_name"] == data["repo_name"] and data1["pr_number"] == data["pr_number"]:
                        # add the commit_data to original data
                        data["commit_data"] = data1["commit_data"]
      
            
            result_dict = run_semantic_reranking_search(
                pr_desc=pr_description,
                readme_sections=sections,
                file_patches=changed_file_patches,
                attempt_number=1
            )   

            filename = result_dict["filename"] if result_dict is not None else ""
            content = result_dict["content"] if result_dict is not None else ""

            yield {
                "repo_name": repo,
                "pr_number": pr_id,
                "selected_filename": filename,
                "selected_content": content,
                "changed_section_idx": changed_section_idx,
            }

def split_readme_sections_vanilla(data_entry):
    md_text = data_entry["readme_base"]
    line_positions = data_entry["line_positions"]
    sections = []
    section_lines = []
    current_section = []
    current_section_lines = []
    for line_pos, line in enumerate(md_text.split("\n")):
        line_pos += 1
        if line.strip():
            current_section.append(line.strip())
            current_section_lines.append(line_pos)
        else:
            sections.append("\n".join(current_section))
            current_section_lines.append(line_pos)
            section_lines.append(current_section_lines)
            current_section = []
            current_section_lines = []
    if current_section:
        sections.append("\n".join(current_section))
        current_section_lines.append(max(current_section_lines) + 1)
        section_lines.append(current_section_lines)
    changed_sections_idx = []
    if len(line_positions) == 1 and line_positions[0] == 0:
        changed_sections_idx = [0]
    for line in line_positions:
        for section_idx, section_line in enumerate(section_lines):
            if line in section_line:
                changed_sections_idx.append(section_idx)
                break
    changed_sections_idx = sorted(set(i + 1 for i in changed_sections_idx))
    return sections, changed_sections_idx





def run_semantic_reranking_search(
    pr_desc: str, 
    readme_sections: List[str], 
    file_patches: Dict[str, str], # <-- CHANGED: Now a Dict
    attempt_number: int = 1
) -> Union[Dict[str, str], None]:
    """
    Finds the file patch that is the "centre of change," i.e.,
    most relevant to BOTH the PR and the existing README.
    
    Args:
        pr_desc: The PR description text.
        readme_sections: A list of strings, each being a README section.
        file_patches: A DICT of {filename: patch_content}
        top_k: The number of candidates to re-rank.
    
    Returns:
        The content of the winning file patch, or None if no patches.
    """
    
    # --- Edge Case Handling ---
    if not file_patches:
        print("   > No file patches to search. Skipping.")
        return None
    
    # --- Unpack the dictionary into parallel lists ---
    patch_names = list(file_patches.keys())
    patch_contents = list(file_patches.values())

    # --- 1. Get All Embeddings ---
    # print(f"   > Encoding {1+len(readme_sections)+len(patch_contents)} texts...")
    
    pr_vec = sbert_model.encode([pr_desc])
    section_vecs = sbert_model.encode(readme_sections)
    patch_vecs = sbert_model.encode(patch_contents) # Encode contents

    # --- 2. Step 1: Candidate Generation (Find Top K) ---
    pr_to_patch_sim = cosine_similarity(pr_vec, patch_vecs)[0]
    
    final_scores = []

    for patch_idx in range(len(patch_contents)):
        patch_vec = patch_vecs[patch_idx:patch_idx+1]
        sim_with_pr = pr_to_patch_sim[patch_idx]
        patch_to_section_sim = cosine_similarity(patch_vec, section_vecs)[0]
        sim_with_readme_best_match = np.max(patch_to_section_sim)
        
        final_score = sim_with_pr + sim_with_readme_best_match
        final_scores.append(final_score)


    sorted_indices_desc = np.argsort(final_scores)[::-1]
    
    # Safety check:
    # If this is Attempt 3, but there are only 2 patches.
    # We'll just return the last (worst) one.
    if attempt_number > len(sorted_indices_desc):
        # print(f"   > Warning: Attempt {attempt_number} requested, but only {len(sorted_indices_desc)} patches. Returning last.")
        attempt_number = len(sorted_indices_desc)

    # Select the N-th best index (N-1 because of 0-indexing)
    winning_index = sorted_indices_desc[attempt_number - 1]
    
    # print(f"   > Returning Top-{attempt_number} result: (File: {patch_names[winning_index]})")
    
    return {
        "filename": patch_names[winning_index],
        "content": patch_contents[winning_index]
    }





def pipeline(model, input_file, output_file):
    logger.add("../logs/Negative_predict_vllm.log", format="{time} {level} {message}", level="INFO")
    for returned_result in stream_json_file(input_file):
        print(f"Processing {returned_result['repo_name']} {returned_result['pr_number']}")
    
        with open(output_file, "a") as f:
            f.write(json.dumps({
                "repo_name": returned_result['repo_name'],
                "pr_number": returned_result['pr_number'],
                "generated_output": returned_result['selected_filename'],
                "ground_truth_idx": returned_result['changed_section_idx'],
            }) + "\n")







def main(model, input_file_path, output_file_paths):
    pipeline(model, input_file_path, output_file_paths[0])

if __name__ == "__main__":
 
 
    # parser = argparse.ArgumentParser()
  
  
    models_list = [
        "google/gemma-3-27b-it",
        "Qwen/Qwen3-32B",
        "kosbu/Llama-3.3-70B-Instruct-AWQ",
    ]

    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)

    except Exception as e:
        print(f"Error loading VLLM or SBERT model: {e}")
        exit()




    with open("") as f:
        for line in f.readlines():
            data = json.loads(line)

            inspection_set.add((data['repo_name'], int(data['pr_number'])))


    print(len(inspection_set))
    
    # input_file_path_positive = "../data/prepared-data-augmented.jsonl"

    input_file_paths = [""]


    output_file_path_positive = ""
    # tokenizer, model = load_model(model_name)
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    print("Starting Main Function...")
    main(model, input_file_paths[0], [output_file_path_positive])
    