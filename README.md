# README-auto-update

This is the replication for paper titled "Does My README File Need To Be Updated? Exploring LLM-Based README Maintenance".

## Folder walkthrough

`agent_mode` includes the agent implementation, while `pipeline_mode` includes the static version encompassing each component. Please refer to the `prompts.py` under each folder for the design. The final guard that decides whether a recommendation is delivered is in `agent_mode/guard_positive.py` and `agent_mode/guard_negative.py`, and its prompt is `agent_guard_prompt` in `agent_mode/prompts.py`. In folder `failure_analysis` we attached an excel spreadsheet for our qualitative analysis.

In folder `output` we release the model outputs behind the end-to-end results. There is one file per configuration and per set, and the agentic files cover the full evaluation set of 10,452 positive and 14,559 negative pull requests. The 250 positive and 250 negative pull requests we used during prompt development are not part of the evaluation set and are left out. Each line is one pull request and carries the decision of every stage it went through, from `classifier_decision` through `examiner_critique` to `guard_output`, together with the generated section ids in `generated_output` and the ground truth in `ground_truth_idx`. The field `final_recommendation` is what the pipeline would have shown to a developer.

`evaluation` contains the code that scores the recommendations against the sections the developer edited.

## How to run

The evaluation reports End-to-end Success over all positive pull requests, and Index Recall, MRR and Hierarchical Recall over the recommendations a configuration delivered. A delivered recommendation whose output cannot be parsed counts as a localisation failure. It rebuilds the README structure from the dataset, so download the dataset first and point the script at it:

```
cd evaluation
python index_level.py --dataset /path/to/prepared-data-augmented.jsonl
```

Add `--model qwen` to restrict it to one configuration. The entry-level numbers are not computed here: `final_recommendation` in the output files is the decision, so counting it against the two sets gives them directly.

Producing the outputs themselves needs the GPUs. The agentic workflow runs in two steps, the agent first and then the guard:

```
cd agent_mode
python agent_positive.py --model_id Qwen/Qwen3-32B
python agent_negative.py --model_id Qwen/Qwen3-32B
python guard_positive.py
python guard_negative.py
```

`--model_id` takes any HuggingFace id. We used `google/gemma-3-27b-it`, `Qwen/Qwen3-32B` and `kosbu/Llama-3.3-70B-Instruct-AWQ`. `--tensor_parallel` defaults to the number of visible GPUs. The two guard scripts have no arguments and read the model from the `LLM(...)` call at the bottom of the file, so set it there to the same model as the agent run.

The static pipeline runs its components in order:

```
cd pipeline_mode
python C1.py --model_id 1 --prompt_id 3
python C2.py --model_id 1
python C3.py
python C4.py --model_id 1
python C5.py --model_id 1
```

Here `--model_id` is an index into the model list at the bottom of each script, where 0 is gemma-3-27b-it, 1 is Qwen3-32B and 2 is Llama-3.3-70B-Instruct-AWQ. `--prompt_id` indexes `C1_prompt_list` in `prompts.py`, and 3 is the description, commit message and file name configuration we report in the paper. C3 takes no arguments.

The input and output paths are left empty in these scripts. Set them in the `__main__` block of each script before running.

## Environment

`environment.yml` is the conda environment we used. Create it with `conda env create -f environment.yml`. It covers both the GPU inference and the evaluation, and the evaluation on its own only needs numpy, beautifulsoup4 and markdown.

The models are pulled from HuggingFace at run time. `google/gemma-3-27b-it` is gated, so accept its license on the model page and export `HF_TOKEN` before running. For Llama we used the quantised weights `kosbu/Llama-3.3-70B-Instruct-AWQ` rather than the official release.

## Hardware

We served the models on two NVIDIA A100 80GB GPUs.

## Dataset

Dataset is uploaded in zenodo [here](https://zenodo.org/records/18654598).
