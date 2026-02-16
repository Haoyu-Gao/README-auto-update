from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
import torch
import json
from loguru import logger
from sentence_transformers import SentenceTransformer, models
import numpy as np

import prompts
import re
import argparse

import os
os.environ["HF_HOME"] = ""
os.environ["TRANSFORMERS_CACHE"] = ""

cache_dir = ""
hf_token = os.getenv("HF_TOKEN")

# Step 1: Load CodeBERT as a HuggingFace transformer
word_embedding_model = models.Transformer("microsoft/codebert-base", max_seq_length=512)

# Step 2: Apply mean pooling to get sentence-level embeddings
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,  # This is good
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False
)

# Step 3: Combine into a SentenceTransformer pipeline
codebert_embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])

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
        raise ValueError(f"Failed to parse JSON: {e.msg} (at line {e.lineno}, column {e.colno})") from e

def stream_json_file(file_path, input_reference_file, batch_size=1):
    batch = []
    batch_meta = []
    with open(file_path) as f:
        for idx, line in enumerate(f.readlines()):
            if not line.strip():
                continue
            
            # if idx < 5824:
            #     continue

            try:
                data = json.loads(line)
                generated_output = parse_generation_to_json(data["generated_output"])
                # if generated_output["status"] == "NEED_MORE":
                #     continue
                justification = generated_output["justification"]
                predicted_idx = generated_output["section_ids"]
                readme = ""
                with open(input_reference_file) as ref_f:
                    for ref_line in ref_f.readlines():
                        ref_data = json.loads(ref_line)
                        if ref_data["repo_name"] == data["repo_name"] and ref_data["pr_number"] == data["pr_number"]:
                            readme = ref_data["readme_base"]
                            try:
                                pr_description = ref_data['pr_title'] + " " + ref_data['pr_description']
                            except:
                                pr_description = ref_data['pr_description'] or ref_data['pr_title'] or ""
                            commit_messages = data.get('commit_comments', "") or ""
                            formatted_commits = "\n".join(f"{i}: {msg}" for i, msg in enumerate(commit_messages))
                            break
            except Exception as e:
                print(f"Error processing line {idx}: {e}")
                continue

            sections, _ = split_readme_sections_vanilla({"readme_base": readme, "line_positions": []})
            readme_template_content = ""
            for idx, section in enumerate(sections):
                readme_template_content += f"Section ID {idx + 1}: \n{section}\n"
            
            prompt_instance = prompts.C5_prompt.format(
                formatted_readme=readme_template_content,
                predicted_section_ids=predicted_idx,
                justification=justification,
                pr_description=pr_description,
                commit_message=formatted_commits
            )
            prompt_instance = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_instance}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            if len(prompt_instance) > 32768:
                prompt_instance = prompt_instance[:32768]
            batch.append(prompt_instance)
            batch_meta.append({"repo_name": data['repo_name'], "pr_number": data['pr_number']})

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


def pipeline(model, input_file, output_file, input_reference_file):
    logger.add("", format="{time} {level} {message}", level="INFO")
    for batch, batch_meta in stream_json_file(input_file, input_reference_file):
        logger.info(f"Processing {batch_meta[0]['repo_name']} {batch_meta[0]['pr_number']}")
        torch.cuda.empty_cache()
        outputs = model.generate(batch, sampling_params)
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip().split("Answer:\nassistant")[-1]
            
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



def main(model, input_file_paths, output_file_paths, input_reference_file):
    pipeline(model, input_file_paths, output_file_paths, input_reference_file)
    # pipeline(model, prompt_template, input_file_paths, output_file_paths)


if __name__ == "__main__":
    # model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # parse arges
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=int)
    # parser.add_argument("--prompt_id", type=int)
    
    args = parser.parse_args()
    model_id = args.model_id
    # prompt_id = args.prompt_id

    models_list = ["google/gemma-3-27b-it",
                   "Qwen/Qwen3-32B",
                   "kosbu/Llama-3.3-70B-Instruct-AWQ"]


    model_name = models_list[model_id]

    model_log_name = model_name.split("/")[-1]

    input_reference_file = ""


    input_file_path_sufficient_positive =  ["",
                                            "",
                                            "",
                                            ]

    input_file_path_insufficient_positive = ["",
                                            "",
                                            ""
                                            ]

    input_file_path_sufficient_negative = ["",
                                            "",
                                            ""
                                            ]
    
    input_file_path_insufficient_negative = ["",
                                            "",
                                            ""
                                            ]

    output_file_path_insufficient = ""

    output_file_paths_sufficient_positive = ["",
                                                "",
                                                ""]

    output_file_paths_insufficient_positive = ["",
                                                "",
                                                ""]
    
    output_file_paths_sufficient_negative = ["",
                                                "",
                                                ""]
    
    output_file_paths_insufficient_negative = ["",
                                                "",
                                                ""]


    tokenizer, model = load_model(model_name)
    guided_decoding_params = GuidedDecodingParams(choice=["Yes", "No"])
    sampling_params = SamplingParams(
        temperature=0.0,
        logprobs=20,
        max_tokens=1,
        guided_decoding=guided_decoding_params
    )

    main(model, input_file_path_insufficient_positive[model_id], output_file_paths_insufficient_positive[model_id], input_reference_file)