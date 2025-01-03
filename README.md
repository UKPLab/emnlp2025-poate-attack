# POATE Attack
[![Arxiv](https://img.shields.io/badge/Arxiv-YYMM.NNNNN-red?style=flat-square&logo=arxiv&logoColor=white)](https://put-here-your-paper.com)
[![License](https://img.shields.io/github/license/UKPLab/POATE-attack)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/UKPLab/POATE-attack/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/POATE-attack/actions/workflows/main.yml)

![PDF Image](./figures/potee_framework.pdf)

This repository contains the code for our paper "Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions".
We provide the code for the following tasks:

* Generating jailbreak prompts using our proposed method.
* Implementation of baseline attack and defense methods for LLMs.
* Metric for evaluating the safety of LLMs on jailbreak prompts.


> **Abstract:**
Despite significant efforts to align large language models with human values and ethical guidelines, these models remain susceptible to sophisticated jailbreak attacks that exploit their reasoning capabilities. 
Traditional safety mechanisms often focus on detecting explicit malicious intent, leaving deeper vulnerabilities unaddressed.
We propose a jailbreak technique, POATE (Polar Opposite query generation, Adversarial Template construction and Elaboration), which leverages contrastive reasoning to elicit unethical responses. 
POATE generates prompts with semantically opposite intents and combines them with adversarial templates to subtly direct models toward producing harmful outputs. 
We conduct extensive evaluations across six diverse language model families of varying parameter sizes, including LLaMA3, Gemma2, Phi3, and GPT-4, to demonstrate the robustness of the attack, achieving significantly higher attack success rates (44%) compared to existing methods. 
We evaluate our proposed attack against seven safety defenses, revealing their limitations in addressing reasoning-based vulnerabilities. To counteract this, we propose a defense strategy that improves reasoning robustness through chain-of-thought prompting and reverse thinking, mitigating reasoning-driven adversarial exploits. 

---
Contact person: [Rachneet Sachdeva](mailto:rachneet.sachdeva@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.


---

### :rocket: Getting Started :rocket:
```bash
# create a virtual environment (e.g. conda)
conda create -n llm-safety python=3.10
conda activate llm-safety

# install the requirements
pip install -r requirements.txt
```

---

### Generate Jailbreak prompts

1. Polar opposite generation
```bash
CUDA_LAUNCH_BLOCKING=1 python src/attacks/jailbreak/potee/polar_opposite_generation.py 
```

2. Template generation
```bash
CUDA_LAUNCH_BLOCKING=1 python src/attacks/jailbreak/potee/attack.py \
--dataset "advbench" \
--target_model "Mistral_7b_instruct"   # we use Mistral for template generation
```

---

### Attack methods

1. GCG Attack

```bash


### gcg Attack
CUDA_LAUNCH_BLOCKING=1 python src/attacks/jailbreak/gcg/nano_gcg_hf.py \
--dataset "advbench" \
--target_model "gemma2_9b_it"

```

2. DeepInception and POATE Attack

```bash
### DeepInception and POATE Attack

MODELS=(
"gemma2_9b_it"
"Llama_2_7b_chat_hf"
"Llama_3.1_8b_instruct"
"phi_3_mini_4k"
)


DATASETS=(
"advbench"
"xstest"
"malicious_instruct"
)
for model in "${MODELS[@]}"
do
  for dataset in "${DATASETS[@]}"
  do
    python "${BASE_PATH}src/attacks/jailbreak/base.py" \
    --target_model ${model} \
    --exp_name main \
    --defense 'none' \
    --attack potee \
    --dataset ${dataset} \
    --sample
  done
done
```

3. Generation Exploitation Attack

```bash
MODELS=(
"gemma2_9b_it"
"Llama_2_7b_chat_hf"
"Llama_3.1_8b_instruct"
"phi_3_mini_4k"
)  

DATASETS=(
"advbench"
"xstest"
"malicious_instruct"
)


for model in "${MODELS[@]}"
do
  for dataset in "${DATASETS[@]}"
  do
    python src/attacks/jailbreak/generation_exploitation/gen_exploitation_optim.py \
    --model $model \
    --tune_temp \
    --tune_topp \
    --tune_topk \
    --n_sample 1 \
    --dataset $dataset
  done
done
```

4. Puzzler Attack

```bash
MODELS=(
"gemma2_9b_it"
"Llama_2_7b_chat_hf"
"Llama_3.1_8b_instruct"
"phi_3_mini_4k"
)

DATASETS=(
"advbench"
"xstest"
"malicious_instruct"
)
for model in "${MODELS[@]}"
do
  for dataset in "${DATASETS[@]}"
  do
    python "${BASE_PATH}src/attacks/jailbreak/puzzler/main.py" \
    --target_model ${model} \
    --exp_name main \
    --defense 'none' \
    --attack puzzler \
    --dataset ${dataset}
  done
done
```

---

### Defense methods

1. Perplexity-based defense

```bash
MODELS=(
"gemma2_9b_it"
"Llama_2_70b_chat_hf"
"Llama_3.1_8b_instruct"
"phi_3_mini_4k"
)

DATASETS=(
"advbench"
"xstest"
"malicious_instruct"
)
for model in "${MODELS[@]}"
do
  for dataset in "${DATASETS[@]}"
  do
    python "${BASE_PATH}src/defenses/ppl_calculator.py" \
    --model_name ${model} \
    --dataset ${dataset}
  done
done
```

2. Self-refinement, In-context defense, Paraphrase, and System prompt

```bash
MODELS=(
"gemma2_9b_it"
"Llama_2_7b_chat_hf"
"Llama_3.1_8b_instruct"
"phi_3_mini_4k"
)


DATASETS=(
"advbench"
"xstest"
"malicious_instruct"
)
for model in "${MODELS[@]}"
do
  for dataset in "${DATASETS[@]}"
  do
    python "${BASE_PATH}src/attacks/jailbreak/base.py" \
    --target_model ${model} \
    --exp_name main \
    --defense 'paraphrase' \  # sr or ic or sys_prompt or paraphrase or none
    --attack potee \
    --dataset ${dataset} \
    --sample
  done
done
```

3. Safe-decoding defense

```bash
python src/defenses/safedecoding/main.py
```

4. SmoothLLM defense

```bash
MODELS=(
"gemma2_9b_it"
"Llama_2_7b_chat_hf"
"Llama_3.1_8b_instruct"
"phi_3_mini_4k"
)

DATASETS=(
"advbench"
"xstest"
"malicious_instruct"
)
for model in "${MODELS[@]}"
do
  for dataset in "${DATASETS[@]}"
  do
    python "${BASE_PATH}src/defenses/SmoothLLM/main.py" \
    --results_dir ./results \
    --target_model ${model} \
    --attack Potee \
    --dataset ${dataset} \
    --attack_logfile "./data/auto_potee_attack_harmbench_classifier_${model}_${dataset}_sampled_outputs.csv" \
    --smoothllm_pert_type RandomSwapPerturbation \
    --smoothllm_pert_pct 10 \
    --smoothllm_num_copies 10
  done
done
```

5. Chain-of-thought defenses

```bash
MODELS=(
"gemma2_9b_it"
"Llama_2_70b_chat_hf"
"Llama_3.1_8b_instruct"
"phi_3_mini_4k"
)


DATASETS=(
"advbench"
"xstest"
"malicious_instruct"
)
for model in "${MODELS[@]}"
do
  for dataset in "${DATASETS[@]}"
  do
    python "${BASE_PATH}src/attacks/jailbreak/base.py" \
    --target_model ${model} \
    --exp_name main \
    --defense 'reverse_thinking_cot' \     # reverse_thinking_cot or intent_alignment_prompt
    --attack puzzler \
    --dataset ${dataset} \
    --sample
  done
done
```

---

### ASR evaluation

```bash
python src/attacks/evaluators/harmbench_classifier.py
```

## Cite

Please use the following citation:

```
@InProceedings{smith:20xx:CONFERENCE_TITLE,
  author    = {Smith, John},
  title     = {My Paper Title},
  booktitle = {Proceedings of the 20XX Conference on XXXX},
  month     = mmm,
  year      = {20xx},
  address   = {Gotham City, USA},
  publisher = {Association for XXX},
  pages     = {XXXX--XXXX},
  url       = {http://xxxx.xxx}
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
