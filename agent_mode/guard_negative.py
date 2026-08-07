import operator
import torch
from typing import TypedDict, Union, Literal, List, Dict
import os
from langgraph.graph import StateGraph, END
import prompts  # We assume this now contains prompts for A1 and A5
import json
import random
import time
import csv
import itertools
import re

from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. NATIVE VLLM IMPORTS ---
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from loguru import logger


# --- Environment Setup ---
os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token

cache_dir = ""

llm = None
tokenizer = None
sbert_model = None


# --- 2. Define the Full AGENTIC SYSTEM State ---
# This is the "memory" for your full agentic system.

class WorkflowState(TypedDict):
    """
    Represents the state of our full workflow.
    """
    # --- Core Data (from original) ---
    pull_request: dict
    pr_description: str
    readme_content: str
    readme_sections: list
    file_patches: dict
    commit_data: str
    commit_messages: list
    changed_file_names: list

    # --- Inner Loop State (A2/A3) (from original) ---
    a3_attempts: int           # Inner loop counter for A2/A3 sufficiency
    is_sufficient: bool
    additional_file: Union[Dict[str, str], None]
    first_retrieval: Union[Dict[str, str], None]
    
    # --- Generation Output (from original) ---
    generated_output: Union[str, None]

    # --- NEW: Outer Loop State (A1/A5) ---
    classifier_decision: str  # "YES" or "NO" from A1
    examiner_critique: dict   # {"status": "SUCCESS" | "GENERIC" | "HALLUCINATION", "reason": "..."}
    k_attempt: int            # K=1, K=2... for A3's semantic search (outer loop)
    max_k_retries: int        # Max K value (e.g., 3)


# --- 3. Reused Utility Functions ---

def split_readme_sections_vanilla(data_entry):
    # (This function is unchanged from your original code)
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


# --- 4. Define Native vLLM Call Functions ---

# --- NEW: A1 Classifier Call ---
def call_vllm_classifier(pr_desc: str, readme: str, commit_messages: list, changed_file_names: list) -> str:
    """A1: Decides if a README update is needed at all."""
    print("--- A1: Classifying update necessity (Calling native vLLM)... ---")
    

    formatted_commits = "\n".join(
        f"{id}: {commit_message}"
        for id, commit_message in enumerate(commit_messages)
    )

    formatted_files = "\n".join(changed_file_names)
    # This prompt needs to be a simple, fast Yes/No
    prompt = prompts.agent_update_classification_prompt.format(README_content=readme,
                                                               PR_description=pr_desc,
                                                               commit_message=formatted_commits,
                                                               changed_file_names=formatted_files)
    prompt_instance = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    if len(tokenizer.encode(prompt_instance)) > 32768:
        tokens = tokenizer(prompt_instance, return_tensors="pt", truncation=True, max_length=32768-64)
        prompt_instance = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)

    outputs = llm.generate([prompt_instance], classification_params)
    response = outputs[0].outputs[0].text.strip().upper()
    
    if "YES" in response or "Y" in response:
        # print("   > vLLM Response: 'YES'")
        return "YES"
    else:
        # print("   > vLLM Response: 'NO'")
        return "NO"


# --- REUSED: A2 Sufficiency Call ---
def call_vllm_sufficiency_native(pr_desc: str, readme: str, additional_file: Union[Dict[str, str], None]) -> bool:
    """A2: Decides if PR info is sufficient."""
    print("--- A2: Assessing sufficiency (Calling native vLLM)... ---")

    if additional_file:
       
        files_str = ""
        for idx, f in enumerate(additional_file):
            fname = f.get('filename', 'unknown')
            content = f.get('content', '')
            files_str += f"File {idx+1}: {fname}\nContent:\n{content}\n\n"
        

        # additional_file_name = additional_file["filename"]
        # additional_file_content = additional_file["content"]
        context_block = f"""
        The PR description alone was not enough. I have retrieved the following file(s):
        {files_str}
        """
    else:
        print("   > (Assessing with PR description only)")
        context_block = "(You are assessing the PR description only)"


    prompt = prompts.agent_sufficient_prompt.format(README_content=readme, PR_description=pr_desc, context_block=context_block)

    prompt_instance = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    if len(tokenizer.encode(prompt_instance)) > 32768:
        tokens = tokenizer(prompt_instance, return_tensors="pt", truncation=True, max_length=32768-64)
        prompt_instance = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)

    outputs = llm.generate([prompt_instance], decision_params)
    response = outputs[0].outputs[0].text.strip()
    print(f"   > vLLM Response: '{response}'")
    return "YES" in response.upper()


# --- REUSED: A4 Generation Call ---
def call_vllm_generate_native(pr_desc: str, readme: str, file_data: Union[List[Dict], Dict, None], commit_msgs: list) -> str:
    """A4: Generates the final localisation and rationale."""
    # print("--- A4: Generating rationale (Calling native vLLM)... ---")
    formatted_all_commits = "\n".join(f"{i}: {msg}" for i, msg in enumerate(commit_msgs))
    


    if file_data:
        # Normalise input to a list to handle both single Dict and List[Dict]
        files_to_process = file_data if isinstance(file_data, list) else [file_data]
        
        files_context = ""
        for f in files_to_process:
            fname = f.get('filename', 'unknown')
            content = f.get('content', '')
            files_context += f"## Patch content of relevant file ({fname}):\n{content}\n\n"

        context_block = f"Commit message: {formatted_all_commits}\n{files_context}"
    else:
        context_block = f"Commit Messages: {formatted_all_commits}\n(No additional file patch was retrieved.)"

    prompt = prompts.agentic_localise_justify_prompt.format(
        README_content=readme,
        PR_description=pr_desc,
        context_block=context_block
    )

    prompt_instance = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    if len(tokenizer.encode(prompt_instance)) > 32768:
        tokens = tokenizer(prompt_instance, return_tensors="pt", truncation=True, max_length=32768-64)
        prompt_instance = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)

    outputs = llm.generate([prompt_instance], generation_params)
    response = outputs[0].outputs[0].text.strip()
    # print("   > vLLM Rationale Generated.")
    return response

# --- NEW: A5 Examiner Call ---
def call_vllm_examiner(readme: str, generated_output: str, pr_description: str, commit_messages) -> dict:
    """A5: Critiques the output from A4."""
    print("--- A5: Examining rationale (Calling native vLLM)... ---")
    
    # This prompt is the new, advanced one we discussed.
    # It must return a structured JSON.

    parsed_output = parse_generation_to_json(generated_output)


    prompt = prompts.agent_justification_prompt.format(
        formatted_readme=readme,
        predicted_section_ids=parsed_output.get("section_ids", []),
        justification=parsed_output.get("justification", ""),
        pr_description=pr_description,
        commit_message="\n".join(commit_messages)
    )

    prompt_instance = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    if len(tokenizer.encode(prompt_instance)) > 32768:
        tokens = tokenizer(prompt_instance, return_tensors="pt", truncation=True, max_length=32768-64)
        prompt_instance = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)

    outputs = llm.generate([prompt_instance], generation_params) # Use generation_params for JSON
    response_text = outputs[0].outputs[0].text.strip()

    try:
        # We expect A5 to return a JSON string
        critique_json = clean_and_parse_json(response_text)

        status = critique_json.get("status", "FAIL").upper()
        reason = critique_json.get("reason", "No reason provided.")
        
        if status not in ["YES", "GENERIC", "HALLUCINATE"]:
            status = "FAIL" # Default to a safe failure mode

        print(f"   > vLLM Critique: {status} - {reason}")
        return {"status": status, "reason": reason}

    except Exception as e:
        print(f"   > Error parsing A5 JSON: {e}. Defaulting to FAIL.")
        return {"status": "FAIL", "reason": "Failed to parse critique JSON."}


# --- 5. Define Graph Nodes (Agents) ---

# --- NEW: A1 Node ---
def agent_a1_classify(state: WorkflowState) -> dict:
    """Node for A1: Classifies if update is needed."""
    decision = call_vllm_classifier(state["pr_description"], state["readme_content"], state["commit_messages"], state["changed_file_names"])
    
    # Initialize the k_attempt for the outer loop
    return {
        "classifier_decision": decision,
        "k_attempt": 1 # Start the outer loop at K=1
    }

# --- REUSED: A2 Node ---
def agent_a2_assess(state: WorkflowState) -> dict:
    """Node for A2: Decides if information is sufficient."""
    pr_desc = state["pr_description"]
    readme = state["readme_content"]
    file = state["additional_file"] 

    sufficient = call_vllm_sufficiency_native(pr_desc, readme, file)
    
    # This counter is for the *inner* A2/A3 loop
    attempts = state["a3_attempts"] + 1

    return {"is_sufficient": sufficient, "a3_attempts": attempts}

# --- MODIFIED: A3 Node ---
def agent_a3_retrieve(state: WorkflowState) -> dict:
    """
    Node for A3: Tries to retrieve a file using advanced search.
    
    *** NOTE: This is the one modified function. ***
    It now reads `k_attempt` from the state to control the
    semantic search's `attempt_number`.
    """
    pr_desc = state["pr_description"]
    readme_sections = state["readme_sections"]
    readme_content = state["readme_content"]
    commit_data = state["commit_data"]
    patches = state["file_patches"]
    attempt_idx = state["a3_attempts"]
    
    # --- THIS IS THE NEW LOGIC ---
    # Read K from the state for the outer loop
    k = state["k_attempt"]
    # print(f"--- A3: Retrieving file (Outer Loop K={k})... ---")
    # --- END OF NEW LOGIC ---

    all_ranked_candidates = run_semantic_reranking_search(
            pr_desc, 
            readme_sections, 
            patches
            # Note: We do NOT pass top_k here anymore, we want the full list
        )
    
    num_candidates = len(all_ranked_candidates)
    if num_candidates == 0:   
        retrieved_data = []
    elif k > num_candidates:
            print(f"   > Warning: K={k} exceeds available files ({num_candidates}). Returning all {num_candidates} files.")
            retrieved_data = all_ranked_candidates
            
    else:
        # Normal case: Generate combinations
        combos = itertools.combinations(all_ranked_candidates, k)
        
        try:
            # Efficiently skip to the target combination based on attempt_idx
            # islice skips the first (attempt_idx - 1) items
            target_combo = next(itertools.islice(combos, attempt_idx - 1, None))
            retrieved_data = list(target_combo)
        except StopIteration:
            print(f"   > Warning: Ran out of combinations for K={k}, Attempt={attempt_idx}. Wrapping to first combo.")
            # Fallback: restart the iterator and take the first valid combination
            combos_retry = itertools.combinations(all_ranked_candidates, k)
            try:
                retrieved_data = list(next(combos_retry))
            except StopIteration:
                # Safety net if even the retry fails (e.g. list became empty)
                retrieved_data = []
    
    # # 2. Generate Combinations of size K
    # # Order is preserved: (Rank1, Rank2), (Rank1, Rank3)... (Rank2, Rank3)...
    # combos = itertools.combinations(all_ranked_candidates, k)
    
    # # 3. Select the specific combination based on attempt number
    # # 'attempt_idx' comes from A2 (starts at 1), so we need index (attempt_idx - 1)
    # target_combo = None
    # try:
    #     # Efficiently skip to the target combination
    #     target_combo = next(itertools.islice(combos, attempt_idx - 1, None))
    #     retrieved_data = list(target_combo)
    # except StopIteration:
    #     # print(f"   > Warning: Ran out of combinations for K={k}, Attempt={attempt_idx}. Wrapping to first combo.")
    #     # Fallback: If we run out of combos, just return the first one (best one)
    #     combos_retry = itertools.combinations(all_ranked_candidates, k)
    #     retrieved_data = list(next(combos_retry))
    
    # Store first retrieval for global fallback if needed
    first_retrieval = state["first_retrieval"]
    if state["a3_attempts"] == 1: 
        state["first_retrieval"] = retrieved_data
        first_retrieval = retrieved_data

    return {
        "additional_file": retrieved_data, 
        "first_retrieval": first_retrieval
    }


    

# --- REUSED: A4 Node ---
def agent_a4_generate(state: WorkflowState) -> dict:
    """Node for A4: Generates the final rationale."""
    pr_desc = state["pull_request"]["description"]
    readme = state["readme_content"]

    if state['a3_attempts'] == 4 and state['is_sufficient'] == False:
        file = state["first_retrieval"]
    else:
        file = state["additional_file"]

    generated_output = call_vllm_generate_native(pr_desc, readme, file, commit_msgs=state["commit_messages"])
    
    generated_output = generated_output.strip()

    return {"generated_output": generated_output}



def parse_generation_to_json(generation: str):
    s = generation.strip()

    # Opening fence: ```json (case-insensitive), optional spaces, then newline
    s = re.sub(r'^\s*```(?:json)?\s*(?:\r?\n)', '', s, flags=re.IGNORECASE)

    # Closing fence: newline, optional spaces, then ```
    s = re.sub(r'(?:\r?\n)\s*```\s*$', '', s)

    s = s.strip()

    # Fallback: if there’s still surrounding text, grab the JSON object/array
    if not (s.startswith('{') or s.startswith('[')):
        start, end = s.find('{'), s.rfind('}')
        if start != -1 and end != -1 and end > start:
            s = s[start:end+1]
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        # raise ValueError(f"Failed to parse JSON: {e.msg} (at line {e.lineno}, column {e.colno})") from e
        return {"section_ids": [], "justification": s}


def clean_and_parse_json(text: str):
    """
    Robustly extracts and parses JSON from a string that might contain 
    markdown fences (```json), extra text, or be messy.
    """
    s = text.strip()
    
    # 1. Remove markdown code fences (case insensitive) for json or just ```
    s = re.sub(r'^\s*```(?:json)?\s*(?:\r?\n)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'(?:\r?\n)\s*```\s*$', '', s)
    s = s.strip()
    
    # 2. Try to find the outer-most JSON object/array
    # This helps if the model writes "Here is the JSON: { ... }"
    start = s.find('{')
    end = s.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
        
    # 3. Parse
    return json.loads(s)


# --- NEW: A5 Node ---
def agent_a5_examine(state: WorkflowState) -> dict:
    """Node for A5: Critiques the A4 output."""
    critique = call_vllm_examiner(
        state["readme_content"],
        state["generated_output"],
        pr_description=state["pr_description"],
        commit_messages="\n".join(state["commit_messages"])
    )
    return {"examiner_critique": critique}

# --- NEW: A6 Node ---
def agent_a6_update_k(state: WorkflowState) -> dict:
    """Node for A6: Increments K for the outer retry loop."""
    k_new = state["k_attempt"] + 1
    print(f"...Coordinator: Generic rationale. Retrying with K={k_new}...")
    return {"k_attempt": k_new,
            "a3_attempts": 1,      # Reset attempt count
            "is_sufficient": False # Reset sufficiency status
            }



def run_semantic_reranking_search(
    pr_desc: str, 
    readme_sections: List[str], 
    file_patches: Dict[str, str]
) -> List[Dict[str, str]]:
    """
    *** MODIFIED: Returns ALL ranked candidates sorted by score ***
    No top_k slicing here anymore.
    """
    # print(f"--- Semantic Search: Ranking All Candidates ---")
    
    if not file_patches:
        return []
    
    patch_names = list(file_patches.keys())
    patch_contents = list(file_patches.values())
    
    pr_vec = sbert_model.encode([pr_desc])
    section_vecs = sbert_model.encode(readme_sections)
    patch_vecs = sbert_model.encode(patch_contents)

    pr_to_patch_sim = cosine_similarity(pr_vec, patch_vecs)[0]
    
    final_scores = []
    for patch_idx in range(len(patch_contents)):
        patch_vec = patch_vecs[patch_idx:patch_idx+1]
        sim_with_pr = pr_to_patch_sim[patch_idx]
        patch_to_section_sim = cosine_similarity(patch_vec, section_vecs)[0]
        sim_with_readme_best_match = np.max(patch_to_section_sim)
        final_score = sim_with_pr + sim_with_readme_best_match
        final_scores.append(final_score)

    # Sort descending
    sorted_indices_desc = np.argsort(final_scores)[::-1]
    
    results = []
    for idx in sorted_indices_desc:
        results.append({
            "filename": patch_names[idx],
            "content": patch_contents[idx]
        })
    
    return results


# --- 7. Define the Conditional Edges ---

# --- NEW: Edge after A1 ---
def decide_after_a1(state: WorkflowState) -> Literal["run_a2", "discard"]:
    """Checks the A1 classifier decision."""
    print("...Checking A1 decision...")
    if state["classifier_decision"] == "YES":
        print("...A1: Update needed. Proceeding to A2...")
        return "run_a2"
    else:
        print("...A1: No update needed. Discarding.")
        return "discard"

# --- REUSED: Edge after A2 ---
def decide_after_a2(state: WorkflowState) -> Literal["run_a3", "run_a4"]:
    """
    This is the main router for the *inner* graph.
    """
    print("...Checking A2 decision...")
    is_sufficient = state["is_sufficient"]
    attempts = state["a3_attempts"]
    
    if is_sufficient:
        print(f"...A2 sufficient after {attempts} attempts. Routing to A4...")
        return "run_a4"
    
    print(f"...A2 NOT sufficient. Attempt {attempts} failed.")
    
    if attempts >= 4:
        print("...Max A2/A3 attempts reached. Forcing generation. Routing to A4...")
        return "run_a4" 
    else:
        print("...Routing to A3 for more info...")
        return "run_a3"

# --- NEW: Edge after A5 ---
def decide_after_a5(state: WorkflowState) -> Literal["accept", "discard", "retry"]:
    """
    This is the main router for the *outer* graph (Coordinator logic).
    """
    print("...Checking A5 (Examiner) critique...")
    status = state["examiner_critique"].get("status", "FAIL")
    
    if status == "YES":
        print("...A5: Rationale is GOOD. Accepting.")
        return "accept"
    
    if status == "HALLUCINATE":
        print("...A5: Rationale is HALLUCINATE. Discarding.")
        return "discard"

    # If we are here, status is "GENERIC" or "FAIL"
    print("...A5: Rationale is GENERIC or FAILED.")
    
    # Check if we have retry attempts left for the outer loop
    if state["k_attempt"] < state["max_k_retries"]:
        print(f"...A5: Retries left (K={state['k_attempt']}). Routing to A6 to retry.")
        return "retry"
    else:
        print(f"...A5: Max K-retries ({state['max_k_retries']}) reached. Discarding.")
        return "discard"


# --- 8. Experiment Runner ---

def run_experiment():
    # (This function is unchanged, but the initial_input is updated)
    sufficient_entries = set()
    insufficient_entries = set()

    with open("") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            sufficient_entries.add((row[0], int(row[1])))
    with open("") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            insufficient_entries.add((row[0], int(row[1])))

    print(len(sufficient_entries), len(insufficient_entries))
    start_flag = False
    # logger.add("agentic-llama-log-full-loop-negative.log", rotation="10 MB")
    
    with open("") as f:
        for idx, line in enumerate(f.readlines()):
            start_time = time.time()
            print("...Processing line... ", idx)

            data = json.loads(line)
            repo_name = data["repo_name"]
            pr_number = data["pr_number"]

           
            try:
                pr_description = data['pr_title'] + " " + data['pr_description']
            except:
                pr_description = data['pr_description'] or data['pr_title'] or ""


            readme_content = data.get('readme_base', "")
            lines_changed = data.get('line_positions', [])

            with open("") as f1:
                    for line1 in f1.readlines():
                        data1 = json.loads(line1)
                        if data1["repo_name"] == repo_name and data1["pr_number"] == pr_number:
                            data["commit_data"] = data1["commit_data"]

            changed_file_patches = data.get("files_patch", "")
            commit_data = data.get('commit_data', [])

            sections, changed_section_idx = split_readme_sections_vanilla({"readme_base": readme_content, "line_positions": lines_changed})
            readme_template_content = ""
            for i, section in enumerate(sections):
                readme_template_content += f"Section ID {i + 1}: \n{section}\n"

            commit_messages = []
            changed_files_list = []
            changed_files_names = []
            for commit in commit_data:
                msg = commit.get('message', "")
                if "readme" in msg.lower():
                    continue
                changed_files = commit.get('files_changed', [])
                if "README.md" in changed_files:
                    changed_files.remove("README.md")
                commit_messages.append(msg)
                changed_files_list.append(changed_files)
                changed_files_names.extend(changed_files)
        
            formatted_commits_info = "\n".join(f"Commit {i}: {msg} \n Changed files: {', '.join(files)}" for i, (msg, files) in enumerate(zip(commit_messages, changed_files_list)))

            # --- Updated Initial Input for the new State ---
            initial_input = {
                "pull_request": {"id": pr_number, "description": pr_description},
                "pr_description": pr_description,
                "readme_content": readme_template_content,
                "readme_sections": sections,
                "file_patches": changed_file_patches,
                "commit_data": formatted_commits_info,
                "commit_messages": commit_messages,
                "changed_file_names": changed_files_names,
                "a3_attempts": 0,       # Inner loop counter
                "is_sufficient": False,
                "additional_file": None,
                "k_attempt": 1,         # Outer loop counter (starts at 1)
                "max_k_retries": 3,     # Max K value
                "classifier_decision": None,
                "examiner_critique": None,
                "first_retrieval": None,
                "generated_output": None,
            }
        
            inferring_justification = ""
            predicted_ids = []
            found_entry = False
            need_inference = False
            with open("") as justification_file:
                lines = justification_file.readlines()
                # find the corresponding repo and pr number
                for line in lines:
                    data_justification = json.loads(line)
                    
                    if data_justification["repo_name"] == repo_name and data_justification["pr_number"] == pr_number:
                        found_entry = True
    
                        if data_justification["examiner_critique"] is None or data_justification["examiner_critique"]["status"] != "YES":
                            continue
                        inferring_justification = data_justification["examiner_critique"]["reason"]
                        predicted_ids = parse_generation_to_json(data_justification.get("generated_output", []))['section_ids']
                        need_inference = True
                        break
                    if found_entry:
                        break
            if not need_inference:
                continue
            
            prompt = prompts.agent_guard_prompt.format(
                formatted_readme=readme_template_content,
                predicted_ids=predicted_ids,
                justification=inferring_justification,
                pr_description=pr_description,
                commit_message="\n".join(commit_messages),
            )
            prompt_instance = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            if len(tokenizer.encode(prompt_instance)) > 32768:
                tokens = tokenizer(prompt_instance, return_tensors="pt", truncation=True, max_length=32768-64)
                prompt_instance = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)
            
            outputs = llm.generate([prompt_instance], generation_params)
            response_text = outputs[0].outputs[0].text.strip()
            with open("agent-final-guard-llama-negative.jsonl", 'a') as guard_file:
                json.dump({
                    "repo_name": repo_name,
                    "pr_number": pr_number,
                    "guard_output": response_text
                }, guard_file)
                guard_file.write("\n")
            
        
 
        
            


# --- 9. Main Execution ---

if __name__ == "__main__":


    llm = LLM(
            model="kosbu/Llama-3.3-70B-Instruct-AWQ",
            tensor_parallel_size=torch.cuda.device_count(),
            trust_remote_code=True,
            download_dir=cache_dir,
            max_model_len=32768,
    )

    tokenizer = AutoTokenizer.from_pretrained("kosbu/Llama-3.3-70B-Instruct-AWQ", cache_dir=cache_dir, token=hf_token)
    guided_decoding_params = GuidedDecodingParams(choice=["Yes", "No"])
    decision_params = SamplingParams(temperature=0.0, max_tokens=10)

    classification_params = SamplingParams(
        temperature=0.0,
        logprobs=20,
        max_tokens=1,
        guided_decoding=guided_decoding_params
    )
        
        
    generation_params = SamplingParams(
        temperature=0.0,
        max_tokens=500 
    )
        
    run_experiment()