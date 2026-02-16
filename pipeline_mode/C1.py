from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
import torch
import json
from loguru import logger
import numpy as np

import argparse
import prompts
import os
os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""

cache_dir = ""
hf_token = os.getenv("HF_TOKEN")

os.environ["HF_TOKEN"] = hf_token



def load_model(model_name):
    # tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)

    # Initialize vLLM LLM instance
    model = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        # tokenizer=model_name,
        # tokenizer_mode="slow",  # To match HF tokenizer behavior
        max_model_len=32768,
        # max_model_len=8192,
        trust_remote_code=True,
        download_dir=cache_dir,
        dtype="auto"
    )

    return tokenizer, model



def stream_json_file(file_path, prompt_template, batch_size=1):
    batch = []
    batch_meta = []

    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            try:
                pr_description = data['pr_title'] + " " + data['pr_description']
            except:
                pr_description = data['pr_description'] or data['pr_title'] or ""
            commit_message = data.get('commit_comments', "") or ""
            changed_files = data.get('files_changed', []) or []
            if "README.md" in changed_files:
                changed_files.remove("README.md")
            readme_content = data.get('readme_base', "")
            lines_changed = data.get('lines_positions', [])
            changed_file_patches = data.get("files_patch", "")
            commit_change_file_combined = data.get('files_changed_commits', "")

            repo = data['repo_name']
            pr_id = data['pr_number']

            prompt_instance = fit_content_to_prompt(prompt_template, readme_content, pr_description, commit_message, changed_files, lines_changed, changed_file_patches, commit_change_file_combined)

            if prompt_instance is not None:
                batch.append(prompt_instance)
                batch_meta.append({"repo_name": repo, "pr_number": pr_id})

            if len(batch) == batch_size:
                yield batch, batch_meta
                batch, batch_meta = [], []

        if batch:
            yield batch, batch_meta


def fit_content_to_prompt(prompt_template, readme_content, pr_description, commit_messages, changed_files, lines_changed, changed_file_patches, commit_change_file_combined):
    split_readme = readme_content.split(" ")
    if len(split_readme) > 3325 or len(commit_messages) > 23 or len(changed_files) > 145 or len(lines_changed) > 11:
        return None

    formatted_commits = "\n".join(
        f"{id}: {commit_message}"
        for id, commit_message in enumerate(commit_messages)
    )

    formatted_files = "\n".join(changed_files)

    if changed_file_patches == "":
        formatted_patches = formatted_files
    else:
        formatted_patches = "\n".join(
            f"{file_name}: {file_patch}"
            for file_name, file_patch in changed_file_patches.items()  ## README has been removed when preparing the data
        )

    valid_commit_messages = []
    for commit_message in commit_messages:
        if "readme" not in commit_message.lower():
            valid_commit_messages.append(commit_message)
    formatted_commits = "\n".join(
        f"{id}: {commit_message}"
        for id, commit_message in enumerate(valid_commit_messages)
    )

    prompt_instance = prompt_template.format(
        README_content=readme_content,
        PR_description=pr_description,
        commit_message=formatted_commits,
        changed_file_names=formatted_files,
        changed_file_patches=formatted_patches
    )

    tokens = tokenizer(prompt_instance, return_tensors="pt", truncation=True, max_length=32768 - 1)
    truncated_prompt = tokenizer.decode(tokens.input_ids[0], skip_special_tokens=True)

    return truncated_prompt


def pipeline(model, prompt_template, input_file, output_file):
    logger.add("", format="{time} {level} {message}", level="INFO")


    for batch, batch_meta in stream_json_file(input_file, prompt_template):
        logger.info(f"Processing {batch_meta[0]['repo_name']} {batch_meta[0]['pr_number']}")
        torch.cuda.empty_cache()
 
        outputs = model.generate(batch, sampling_params)

        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            token_output = output.outputs[0]
            token = token_output.token_ids[0]
           
            # Get logprobs for the top token generation step
            token_logprobs = token_output.logprobs[0]  # list of TopLogprobs
            
            logprob_dict = {
                obj.decoded_token.strip(): obj.logprob
                for _, obj in token_logprobs.items()
            }

            # Retrieve logprobs of "Yes" and "No"
            logprob_yes = logprob_dict.get("Yes", float('-inf'))
            logprob_no  = logprob_dict.get("No",  float('-inf'))
            logprob_y = logprob_dict.get("Y", float('-inf'))
            logprob_n = logprob_dict.get("N", float('-inf'))

            p_yes = np.exp(logprob_yes)
            p_no = np.exp(logprob_no)
            score = p_yes / (p_yes + p_no)  # this is your PRAUC score

            generated_text = generated_text.split("Answer:\nassistant")[-1]
            with open(output_file, "a") as f:
                f.write(json.dumps({
                    "repo_name": batch_meta[i]['repo_name'],
                    "pr_number": batch_meta[i]['pr_number'],
                    "generated_output": generated_text,
                    "logprob_yes": logprob_yes,
                    "logprob_no": logprob_no,
                    "logprob_y": logprob_y,
                    "logprob_n": logprob_n,
                    "score": score
                }) + "\n")


def main(model, prompt_template, input_file_paths, output_file_paths):
    pipeline(model, prompt_template, input_file_paths[0], output_file_paths[0])
    pipeline(model, prompt_template, input_file_paths[1], output_file_paths[1])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=int)
    parser.add_argument("--prompt_id", type=int)
    
    args = parser.parse_args()
    model_id = args.model_id
    prompt_id = args.prompt_id

    models_list = ["google/gemma-3-27b-it",
                   "Qwen/Qwen3-32B",
                   "kosbu/Llama-3.3-70B-Instruct-AWQ" ]
    prompts_list = prompts.prompt_list_new


    model_name = models_list[model_id]
    prompt = prompts_list[prompt_id]

    model_log_name = model_name.split("/")[-1]

    input_file_path_positive = ""
    input_file_path_negative = ""

    output_file_path_positive = f""
    output_file_path_negative = f""
    

    tokenizer, model = load_model(model_name)
    guided_decoding_params = GuidedDecodingParams(choice=["Yes", "No"])
    sampling_params = SamplingParams(
        temperature=0.0,
        logprobs=20,
        max_tokens=1,
        guided_decoding=guided_decoding_params
    )

    main(model, prompt, [input_file_path_positive, input_file_path_negative], [output_file_path_positive, output_file_path_negative])
