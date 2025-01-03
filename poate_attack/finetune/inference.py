from transformers import set_seed
import torch
from vllm import LLM, SamplingParams
import pandas as pd

from poate_attack import prompts
from poate_attack.finetune.train import SYS_PROMPT

set_seed(42)
torch.manual_seed(42)

model_name = "Llama-3.1-8b-instruct-hf-jailbreak/potee_attack_safe_question_2024-10-15/"
llm = LLM(
    model_name,
    dtype="bfloat16",
    tensor_parallel_size=1,
    trust_remote_code=True,
    enforce_eager=True,
    disable_custom_all_reduce=True,
    gpu_memory_utilization=0.95,
    disable_sliding_window=True if model_name.lower().__contains__("phi") else False,
)

# load dataset
dataset = "harmbench"
df = pd.read_csv(f"data/od_test_sets/{dataset}.csv")

full_prompts = []
for p in df["prompt"].tolist():
    prompt = prompts.create_llama3_prompt(
        "fastchat", system_prompt=SYS_PROMPT, prompt=p
    )
    full_prompts.append(prompt)

params = SamplingParams(
    max_tokens=100,
    temperature=0.2,
    repetition_penalty=1.4,
)
predictions = llm.generate(full_prompts, params)
answers = [pred.outputs[0].text for pred in predictions]

# clean the answers
cleaned_answers = [ans.replace("QUESTION:", "").strip() for ans in answers]

# save the answers to a csv
df["jailbreak_prompt"] = cleaned_answers
df.to_csv(f"data/od_test_sets/{dataset}_potee_temp_0.2.csv", index=False)




