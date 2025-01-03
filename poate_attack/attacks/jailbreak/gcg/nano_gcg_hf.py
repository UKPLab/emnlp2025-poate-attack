"""
Simple script that runs GCG with the default settings
"""

import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import nanogcg
from nanogcg import GCGConfig

from poate_attack import prompts
from poate_attack.config import ModelPath, BASE_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", type=str, default="advbench")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # read the behaviors
    df = pd.read_csv(f"{BASE_PATH}/data/id_test_sets/{args.dataset}_gcg_errors.csv")
    # read only last 2 rows
    df = df.tail(2)
    print(df.head())

    # Function to chunk the DataFrame
    def chunk_dataframe(df, chunk_size=100):
        # Use list comprehension to generate chunks
        return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    chunked_df = chunk_dataframe(df, chunk_size=110)

    config = GCGConfig(
        num_steps=250,
        seed=42,
        verbosity="WARNING",
        use_prefix_cache=True,
    )

    model_id = ModelPath.get_path(args.target_model)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, args.dtype)).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    outputs: list = []
    for i, behavior in tqdm(chunked_df[0].iterrows()):
        prompt = behavior["prompt"]
        target = f'{behavior["target"]}:\n\n'

        messages = [{"role": "user", "content": prompt}]

        result = nanogcg.run(model, tokenizer, messages, target, config)

        messages[-1]["content"] = messages[-1]["content"] + " " + result.best_string

        input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(args.device)
        output = model.generate(input, do_sample=False, max_new_tokens=512)

        outputs.append({
            "prompt": prompt,
            "target": target,
            "jailbreak_prompt": messages[-1]['content'],
            "loss": result.best_loss,
            "response": tokenizer.batch_decode(output[:, input.shape[1]:], skip_special_tokens=True)[0]
        })
        # print(outputs)
        # break

    # Save the outputs
    output_df = pd.DataFrame(outputs)
    output_df.to_csv(f"{BASE_PATH}/data/gcg_attack_{args.target_model}_{args.dataset}_id_errors_1_outputs.csv", index=False)


if __name__ == "__main__":
    main()
