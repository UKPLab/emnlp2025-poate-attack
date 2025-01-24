# POATE Attack

[![Arxiv](https://img.shields.io/badge/Arxiv-2501.01872-red?style=flat-square&logo=arxiv&logoColor=white)](https://www.arxiv.org/abs/2501.01872)
[![License](https://img.shields.io/github/license/UKPLab/POATE-attack)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/Python-3.10-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://github.com/UKPLab/POATE-attack/actions/workflows/main.yml/badge.svg)](https://github.com/UKPLab/POATE-attack/actions/workflows/main.yml)

<p align="center">
  <img src="./figures/poate_logo.png" width="200" />
</p>

This repository contains the code for our paper "Turning Logic Against Itself: Probing Model Defenses Through Contrastive Questions".
We provide the code for the following tasks:

* Generating jailbreak prompts using our proposed method.
* Implementation of baseline attack and defense methods for LLMs.
* Metric for evaluating the safety of LLMs on jailbreak prompts.


> **Abstract:**
Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. 
> Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. 
> In this work, we introduce POATE (<ins>P</ins>olar <ins>O</ins>pposite query generation, <ins>A</ins>dversarial <ins>T</ins>emplate construction, and <ins>E</ins>laboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. 
> POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. 
> We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. 
> To counter this, we propose *Intent-Aware CoT* and *Reverse Thinking CoT*, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. 
> These methods enhance reasoning robustness and strengthen the model’s defense against adversarial exploits.
---
Contact person: [Rachneet Sachdeva](mailto:rachneet.sachdeva@tu-darmstadt.de) 

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

---

### Our Framework
![POATE Framework](./figures/potee_framework.jpg)

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
python poate_attack/attacks/jailbreak/potee/polar_opposite_generation.py 
```

2. Template generation
```bash
python poate_attack/attacks/jailbreak/potee/attack.py \
--dataset "advbench" \
--target_model "Mistral_7b_instruct"   # we use Mistral for template generation
```

---

### Attack methods

1. GCG Attack

```bash


### gcg Attack
python poate_attack/attacks/jailbreak/gcg/nano_gcg_hf.py \
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
    python "${BASE_PATH}poate_attack/attacks/jailbreak/base.py" \
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
    python poate_attack/attacks/jailbreak/generation_exploitation/gen_exploitation_optim.py \
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
    python "${BASE_PATH}poate_attack/attacks/jailbreak/puzzler/main.py" \
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
    python "${BASE_PATH}poate_attack/defenses/ppl_calculator.py" \
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
    python "${BASE_PATH}poate_attack/attacks/jailbreak/base.py" \
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
python poate_attack/defenses/safedecoding/main.py
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
    python "${BASE_PATH}poate_attack/defenses/SmoothLLM/main.py" \
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
    python "${BASE_PATH}poate_attack/attacks/jailbreak/base.py" \
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
python poate_attack/attacks/evaluators/harmbench_classifier.py
```

## Cite

Please use the following citation:

```
@misc{sachdeva2025turninglogicprobing,
      title={Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions}, 
      author={Rachneet Sachdeva and Rima Hazra and Iryna Gurevych},
      year={2025},
      eprint={2501.01872},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.01872}, 
}
```

## Disclaimer

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 
