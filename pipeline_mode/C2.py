from vllm import LLM, SamplingParams

import re

from transformers import AutoTokenizer
import torch
import json
from loguru import logger
import argparse
import os
import prompts

os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""

cache_dir = ""
hf_token = os.getenv("HF_TOKEN")




insufficient_set = set()



def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
    model = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=32768,
        # quantization="awq",
        trust_remote_code=True,
        download_dir=cache_dir,
        dtype="auto",
    )
    return tokenizer, model

def stream_json_file(file_path, prompt_template, model_name, batch_size=1):
    batch = []
    batch_meta = []
    with open(file_path) as f:
        for idx, line in enumerate(f.readlines()):
            if not line.strip():
                continue
            data = json.loads(line)

            repo = data['repo_name']
            pr_id = int(data['pr_number'])
            if (repo, pr_id) in insufficient_set:
                continue

            ########### update this when get back home


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
            prompt_instance = fit_content_to_prompt(prompt_template, readme_content, pr_description, commit_message, changed_files, lines_changed, changed_file_patches, sections, model_name)
            if prompt_instance is not None:
                batch.append(prompt_instance)
                batch_meta.append({"repo_name": repo, "pr_number": pr_id, "changed_section_idx": changed_section_idx, "number_of_sections": len(sections)})
            if len(batch) == batch_size:
                yield batch, batch_meta
                batch, batch_meta = [], []
        if batch:
            yield batch, batch_meta

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

def fit_content_to_prompt(prompt_template, readme_content, pr_description, commit_messages, changed_files, lines_changed, changed_file_patches, sections, model_name):
    if len(readme_content.split(" ")) > 3325 or len(commit_messages) > 23 or len(changed_files) > 145 or len(lines_changed) > 11:
        return None
    readme_template_content = ""
    for idx, section in enumerate(sections):
        readme_template_content += f"Section ID {idx + 1}: \n{section}\n"
    valid_commit_messages = [msg for msg in commit_messages if "readme" not in msg.lower()]
    formatted_commits = "\n".join(f"{i}: {msg}" for i, msg in enumerate(valid_commit_messages))
    formatted_files = "\n".join(changed_files)
    formatted_patches = formatted_files if changed_file_patches == "" else "\n".join(f"{fn}: {fp}" for fn, fp in changed_file_patches.items())
    prompt_instance = prompt_template.format(
        README_content=readme_template_content,
        PR_description=pr_description,
        commit_message=formatted_commits,
        changed_file_names=formatted_files,
        changed_file_patches=formatted_patches
    )
    
    prompt_instance = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_instance}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    if len(prompt_instance) > 32768:
        prompt_instance = prompt_instance[:32768]

    return prompt_instance

def pipeline(model, prompt_template, input_file, output_file, model_name):
    logger.add("", format="{time} {level} {message}", level="INFO")
    for batch, batch_meta in stream_json_file(input_file, prompt_template, model_name):
        logger.info(f"Processing {batch_meta[0]['repo_name']} {batch_meta[0]['pr_number']}")
        torch.cuda.empty_cache()
        outputs = model.generate(batch, sampling_params)
        for i, output in enumerate(outputs):
            number_of_sections = batch_meta[i]["number_of_sections"]

            generated_text = output.outputs[0].text.strip().split("Answer:\nassistant")[-1]
            
            with open(output_file, "a") as f:
                f.write(json.dumps({
                    "repo_name": batch_meta[i]['repo_name'],
                    "pr_number": batch_meta[i]['pr_number'],
                    "generated_output": generated_text,
                    "ground_truth_idx": batch_meta[i]['changed_section_idx'],
                }) + "\n")







def main(model, prompt_template, input_file_paths, output_file_paths, model_name):
    pipeline(model, prompt_template, input_file_paths[0], output_file_paths[0], model_name)

if __name__ == "__main__":
 
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int)
    parser.add_argument("--prompt_id", type=int)
    args = parser.parse_args()
    model_id = args.model_id
    # prompt_id = args.prompt_id
    models_list = [
        "google/gemma-3-27b-it",
        "Qwen/Qwen3-32B",
        "kosbu/Llama-3.3-70B-Instruct-AWQ",
    ]

    prompt = prompts.C2_sufficiency_C4
    model_name = models_list[model_id]

    
    # newly added, comment out when running other experiments.
    file_list = [None,
                 "",
                 None,
                 "",
                 None,
                 "",
                 ]

    file = file_list[model_id]
    with open(file) as f:
        for line in f.readlines():
            entry = json.loads(line)
            repo = entry["repo_name"]
            pr_id = int(entry["pr_number"])
            insufficient_set.add((repo, pr_id))
    
    ###################################################
    
    model_log_name = model_name.split("/")[-1]
    input_file_path_positive = ""
    output_file_path_positive = ""
    tokenizer, model = load_model(model_name)
    
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    main(model, prompt, [input_file_path_positive], [output_file_path_positive], model_name)
    