import torch
from typing import TypedDict, Union, Literal, List, Dict
import os
import argparse # NEW: Import argparse
from langgraph.graph import StateGraph, END
import prompts  # We assume this now contains prompts for A1 and A5
import json
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

# Globals to be set in __main__
llm = None
tokenizer = None
sbert_model = None
MODEL_ID = None  # To track which model is running for logic adaptation

# --- 2. Define the Full AGENTIC SYSTEM State ---
class WorkflowState(TypedDict):
    """
    Represents the state of our full workflow.
    """
    # --- Core Data ---
    pull_request: dict
    pr_description: str
    readme_content: str
    readme_sections: list
    file_patches: dict
    commit_data: str
    commit_messages: list
    changed_file_names: list

    # --- Inner Loop State ---
    a3_attempts: int
    is_sufficient: bool
    additional_file: Union[Dict[str, str], None]
    first_retrieval: Union[Dict[str, str], None]
    
    # --- Generation Output ---
    generated_output: Union[str, None]

    # --- Outer Loop State ---
    classifier_decision: str
    examiner_critique: dict
    k_attempt: int
    max_k_retries: int


# --- 3. Reused Utility Functions ---
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

# --- Helper for Chat Templates ---
def apply_template_safe(messages, add_generation_prompt=True):
    """
    Helper to apply chat templates. 
    Handles 'enable_thinking' specifically for Gemma-3, removes it for others.
    """
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt
    }
    
    # Check global MODEL_ID to see if we are using Gemma 3
    if MODEL_ID and "gemma-3" in MODEL_ID.lower():
        kwargs["enable_thinking"] = False

    return tokenizer.apply_chat_template(messages, **kwargs)


# --- 4. Define Native vLLM Call Functions ---

def call_vllm_classifier(pr_desc: str, readme: str, commit_messages: list, changed_file_names: list) -> str:
    """A1: Decides if a README update is needed at all."""
    print("--- A1: Classifying update necessity (Calling native vLLM)... ---")

    formatted_commits = "\n".join(
        f"{id}: {commit_message}"
        for id, commit_message in enumerate(commit_messages)
    )

    formatted_files = "\n".join(changed_file_names)
    prompt = prompts.agent_update_classification_prompt.format(README_content=readme,
                                                               PR_description=pr_desc,
                                                               commit_message=formatted_commits,
                                                               changed_file_names=formatted_files)
    
    # Use safe template helper
    prompt_instance = apply_template_safe([{"role": "user", "content": prompt}])

    if len(tokenizer.encode(prompt_instance)) > 32768:
        tokens = tokenizer(prompt_instance, return_tensors="pt", truncation=True, max_length=32768-64)
        prompt_instance = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)

    outputs = llm.generate([prompt_instance], classification_params)
    response = outputs[0].outputs[0].text.strip().upper()
    
    if "YES" in response or "Y" in response:
        return "YES"
    else:
        return "NO"


def call_vllm_sufficiency_native(pr_desc: str, readme: str, additional_file: Union[Dict[str, str], None]) -> bool:
    """A2: Decides if PR info is sufficient."""
    print("--- A2: Assessing sufficiency (Calling native vLLM)... ---")

    if additional_file:
        files_str = ""
        for idx, f in enumerate(additional_file):
            fname = f.get('filename', 'unknown')
            content = f.get('content', '')
            files_str += f"File {idx+1}: {fname}\nContent:\n{content}\n\n"
        
        context_block = f"""
        The PR description alone was not enough. I have retrieved the following file(s):
        {files_str}
        """
    else:
        print("   > (Assessing with PR description only)")
        context_block = "(You are assessing the PR description only)"

    prompt = prompts.agent_sufficient_prompt.format(README_content=readme, PR_description=pr_desc, context_block=context_block)

    prompt_instance = apply_template_safe([{"role": "user", "content": prompt}])

    if len(tokenizer.encode(prompt_instance)) > 32768:
        tokens = tokenizer(prompt_instance, return_tensors="pt", truncation=True, max_length=32768-64)
        prompt_instance = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)

    outputs = llm.generate([prompt_instance], decision_params)
    response = outputs[0].outputs[0].text.strip()
    print(f"   > vLLM Response: '{response}'")
    return "YES" in response.upper()


def call_vllm_generate_native(pr_desc: str, readme: str, file_data: Union[List[Dict], Dict, None], commit_msgs: list) -> str:
    """A4: Generates the final localisation and rationale."""
    formatted_all_commits = "\n".join(f"{i}: {msg}" for i, msg in enumerate(commit_msgs))

    if file_data:
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

    prompt_instance = apply_template_safe([{"role": "user", "content": prompt}])

    if len(tokenizer.encode(prompt_instance)) > 32768:
        tokens = tokenizer(prompt_instance, return_tensors="pt", truncation=True, max_length=32768-64)
        prompt_instance = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)

    outputs = llm.generate([prompt_instance], generation_params)
    response = outputs[0].outputs[0].text.strip()
    return response

def call_vllm_examiner(readme: str, generated_output: str, pr_description: str, commit_messages) -> dict:
    """A5: Critiques the output from A4."""
    print("--- A5: Examining rationale (Calling native vLLM)... ---")
    
    parsed_output = parse_generation_to_json(generated_output)

    prompt = prompts.agent_justification_prompt.format(
        formatted_readme=readme,
        predicted_section_ids=parsed_output.get("section_ids", []),
        justification=parsed_output.get("justification", ""),
        pr_description=pr_description,
        commit_message="\n".join(commit_messages)
    )

    prompt_instance = apply_template_safe([{"role": "user", "content": prompt}])

    if len(tokenizer.encode(prompt_instance)) > 32768:
        tokens = tokenizer(prompt_instance, return_tensors="pt", truncation=True, max_length=32768-64)
        prompt_instance = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)

    outputs = llm.generate([prompt_instance], generation_params)
    response_text = outputs[0].outputs[0].text.strip()

    try:
        critique_json = clean_and_parse_json(response_text)
        status = critique_json.get("status", "FAIL").upper()
        reason = critique_json.get("reason", "No reason provided.")
        
        if status not in ["YES", "GENERIC", "HALLUCINATE"]:
            status = "FAIL"

        print(f"   > vLLM Critique: {status} - {reason}")
        return {"status": status, "reason": reason}

    except Exception as e:
        print(f"   > Error parsing A5 JSON: {e}. Defaulting to FAIL.")
        return {"status": "FAIL", "reason": "Failed to parse critique JSON."}


# --- 5. Define Graph Nodes (Agents) ---
def agent_a1_classify(state: WorkflowState) -> dict:
    decision = call_vllm_classifier(state["pr_description"], state["readme_content"], state["commit_messages"], state["changed_file_names"])
    return {
        "classifier_decision": decision,
        "k_attempt": 1
    }

def agent_a2_assess(state: WorkflowState) -> dict:
    pr_desc = state["pr_description"]
    readme = state["readme_content"]
    file = state["additional_file"] 
    sufficient = call_vllm_sufficiency_native(pr_desc, readme, file)
    attempts = state["a3_attempts"] + 1
    return {"is_sufficient": sufficient, "a3_attempts": attempts}

def agent_a3_retrieve(state: WorkflowState) -> dict:
    pr_desc = state["pr_description"]
    readme_sections = state["readme_sections"]
    patches = state["file_patches"]
    attempt_idx = state["a3_attempts"]
    k = state["k_attempt"]

    all_ranked_candidates = run_semantic_reranking_search(pr_desc, readme_sections, patches)
    
    num_candidates = len(all_ranked_candidates)
    if num_candidates == 0:   
        retrieved_data = []
    elif k > num_candidates:
            print(f"   > Warning: K={k} exceeds available files ({num_candidates}). Returning all.")
            retrieved_data = all_ranked_candidates    
    else:
        combos = itertools.combinations(all_ranked_candidates, k)
        try:
            target_combo = next(itertools.islice(combos, attempt_idx - 1, None))
            retrieved_data = list(target_combo)
        except StopIteration:
            print(f"   > Warning: Ran out of combinations for K={k}. Wrapping to first.")
            combos_retry = itertools.combinations(all_ranked_candidates, k)
            try:
                retrieved_data = list(next(combos_retry))
            except StopIteration:
                retrieved_data = []
    
    first_retrieval = state["first_retrieval"]
    if state["a3_attempts"] == 1: 
        state["first_retrieval"] = retrieved_data
        first_retrieval = retrieved_data

    return {
        "additional_file": retrieved_data, 
        "first_retrieval": first_retrieval
    }

def agent_a4_generate(state: WorkflowState) -> dict:
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
    s = re.sub(r'^\s*```(?:json)?\s*(?:\r?\n)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'(?:\r?\n)\s*```\s*$', '', s)
    s = s.strip()
    if not (s.startswith('{') or s.startswith('[')):
        start, end = s.find('{'), s.rfind('}')
        if start != -1 and end != -1 and end > start:
            s = s[start:end+1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return {"section_ids": [], "justification": s}

def clean_and_parse_json(text: str):
    s = text.strip()
    s = re.sub(r'^\s*```(?:json)?\s*(?:\r?\n)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'(?:\r?\n)\s*```\s*$', '', s)
    s = s.strip()
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1]
    return json.loads(s)

def agent_a5_examine(state: WorkflowState) -> dict:
    critique = call_vllm_examiner(
        state["readme_content"],
        state["generated_output"],
        pr_description=state["pr_description"],
        commit_messages="\n".join(state["commit_messages"])
    )
    return {"examiner_critique": critique}

def agent_a6_update_k(state: WorkflowState) -> dict:
    k_new = state["k_attempt"] + 1
    print(f"...Coordinator: Generic rationale. Retrying with K={k_new}...")
    return {"k_attempt": k_new,
            "a3_attempts": 1, 
            "is_sufficient": False 
            }

def run_semantic_reranking_search(pr_desc: str, readme_sections: List[str], file_patches: Dict[str, str]) -> List[Dict[str, str]]:
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

    sorted_indices_desc = np.argsort(final_scores)[::-1]
    
    results = []
    for idx in sorted_indices_desc:
        results.append({
            "filename": patch_names[idx],
            "content": patch_contents[idx]
        })
    return results

# --- 7. Define the Conditional Edges ---
def decide_after_a1(state: WorkflowState) -> Literal["run_a2", "discard"]:
    print("...Checking A1 decision...")
    if state["classifier_decision"] == "YES":
        print("...A1: Update needed. Proceeding to A2...")
        return "run_a2"
    else:
        print("...A1: No update needed. Discarding.")
        return "discard"

def decide_after_a2(state: WorkflowState) -> Literal["run_a3", "run_a4"]:
    print("...Checking A2 decision...")
    is_sufficient = state["is_sufficient"]
    attempts = state["a3_attempts"]
    if is_sufficient:
        return "run_a4"
    if attempts >= 4:
        print("...Max A2/A3 attempts reached. Routing to A4...")
        return "run_a4" 
    else:
        return "run_a3"

def decide_after_a5(state: WorkflowState) -> Literal["accept", "discard", "retry"]:
    print("...Checking A5 (Examiner) critique...")
    status = state["examiner_critique"].get("status", "FAIL")
    if status == "YES":
        return "accept"
    if status == "HALLUCINATE":
        return "discard"
    if state["k_attempt"] < state["max_k_retries"]:
        return "retry"
    else:
        print(f"...A5: Max K-retries ({state['max_k_retries']}) reached. Discarding.")
        return "discard"

# --- 8. Experiment Runner ---
def run_experiment(model_sanitized_name):
    # Load input data
    sufficient_entries = set()
    insufficient_entries = set()

    # Wrapped in try-except for robustness to file paths
    try:
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
    except FileNotFoundError:
        print("Warning: CSV filters (sufficient/insufficient.csv) not found in current directory. Processing all data.")

    # Dynamic Log Name for Negative Samples
    logger.add(f"agentic-{model_sanitized_name}-log-negative.log", rotation="10 MB")
    
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

            # Load commit data (Mocked path)
            try:
                with open("") as f1:
                    for line1 in f1.readlines():
                        data1 = json.loads(line1)
                        if data1["repo_name"] == repo_name and data1["pr_number"] == pr_number:
                            data["commit_data"] = data1["commit_data"]
            except FileNotFoundError:
                 pass # Graceful fail if aux file missing

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
        
            final_state = app.invoke(initial_input, config={"recursion_limit": 100})
            
            # Dynamic Output Filename for Negative Samples
            output_filename = f"agentic-{model_sanitized_name}-results-negative.jsonl"
            
            with open(output_filename, "a") as out_f:
                result_entry = {
                    "repo_name": repo_name,
                    "pr_number": pr_number,
                    "classifier_decision": final_state["classifier_decision"],
                    "a3_inner_attempts": final_state["a3_attempts"],
                    "a5_outer_k_attempts": final_state["k_attempt"],
                    "examiner_critique": final_state["examiner_critique"],
                    "generated_output": final_state["generated_output"],
                    "ground_truth_idx": changed_section_idx,
                }
                out_f.write(json.dumps(result_entry) + "\n")
            
            end_time = time.time()
            logger.info(f"Completed PR #{repo_name}: {pr_number} in {end_time - start_time:.2f} seconds.")


# --- 9. Main Execution ---

if __name__ == "__main__":
    # --- ARGUMENT PARSING ---
    parser = argparse.ArgumentParser(description="Run Agentic PR Reviewer (Negative Samples) with specific LLM")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-27b-it", help="HuggingFace Model ID")
    parser.add_argument("--tensor_parallel", type=int, default=None, help="Tensor parallel size (default: GPU count)")
    args = parser.parse_args()
    
    # Set Global Model ID
    MODEL_ID = args.model_id
    sanitized_name = MODEL_ID.split("/")[-1] # e.g. "gemma-3-27b-it"

    print(f"--- Initializing Agentic System (Negative Samples) with Model: {MODEL_ID} ---")

    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_dir)
        
        tp_size = args.tensor_parallel if args.tensor_parallel else torch.cuda.device_count()

        llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            download_dir=cache_dir,
            max_model_len=32768,
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache_dir, token=hf_token)
        
        # Guided Decoding: Ensure capitalization matches logic
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
        
    except Exception as e:
        print(f"Error loading VLLM model: {e}")
        exit()

    # --- 10. Build the NEW Agentic Graph ---
    workflow = StateGraph(WorkflowState)

    # Add all nodes (new and old)
    workflow.add_node("A1", agent_a1_classify)
    workflow.add_node("A2", agent_a2_assess)
    workflow.add_node("A3", agent_a3_retrieve)
    workflow.add_node("A4", agent_a4_generate)
    workflow.add_node("A5", agent_a5_examine)
    workflow.add_node("A6", agent_a6_update_k)

    # Set the entry point
    workflow.set_entry_point("A1")

    # Add conditional edges
    workflow.add_conditional_edges("A1", decide_after_a1, {"run_a2": "A2", "discard": END})
    workflow.add_conditional_edges("A2", decide_after_a2, {"run_a3": "A3", "run_a4": "A4"})
    workflow.add_conditional_edges("A5", decide_after_a5, {"accept": END, "discard": END, "retry": "A6"})

    # Add final edges
    workflow.add_edge("A3", "A2") 
    workflow.add_edge("A4", "A5") 
    workflow.add_edge("A6", "A3") 

    # Compile the graph
    app = workflow.compile()

    # --- 11. Run the System ---
    run_experiment(sanitized_name)