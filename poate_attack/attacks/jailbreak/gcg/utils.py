import torch
import os
import json
import pandas as pd
import string

########## GCG Utils ##########
def sample_control(control_toks, grad, search_width, topk=256, temp=1, not_allowed_tokens=None):
    if not_allowed_tokens is not None:
        # grad[:, not_allowed_tokens.to(grad.device)] = np.infty
        grad = grad.clone()
        grad[:, not_allowed_tokens.to(grad.device)] = grad.max() + 1

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(search_width, 1)
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / search_width,
        device=grad.device
    ).type(torch.int64)

    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,
        torch.randint(0, topk, (search_width, 1),
                      device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


def get_nonascii_toks(tokenizer, device='cpu'):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    if "Baichuan2" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(101, 1000)]

    return torch.tensor(ascii_toks, device=device)


def process_outputs():
    harmful_behaviors_path = f"./data/harmful_behaviors_custom.csv"
    harm_df = pd.read_csv(harmful_behaviors_path)
    harm_df['id'] = harm_df['id'].astype(int)

    gcg_attack_path = f"./data/gcg_custom_500"
    test_cases_path = gcg_attack_path + "/test_cases_individual_behaviors"
    files = os.listdir(test_cases_path)

    results = {"id": [], "prompt": []}

    for file in files:
        path = os.path.join(test_cases_path, file, "test_cases.json")
        with open(path, "r") as f:
            data = json.load(f)
            # print(data)
            for k, v in data.items():
                results["id"].append(k)
                results["prompt"].append(v[0])

    results_df = pd.DataFrame.from_dict(results)
    results_df['id'] = results_df['id'].astype(int)

    merged_df = pd.merge(harm_df, results_df, on="id")
    merged_df["source"] = "GCG"
    merged_df["target-model"] = "llama2"

    merged_df.to_csv(gcg_attack_path + "/gcg_attackers_500_steps.csv", index=False)


def merge_and_process_outputs(file_paths: list, output_path: str):
    dfs = []
    # for file_path in file_paths:
    #     df = pd.read_csv(file_path)
    #     dfs.append(df)
    # merged_df = pd.concat(dfs)
    # merged_df.to_csv(output_path, index=False, quoting=1)

    df = pd.read_csv(output_path)
    # df['jailbreak_prompt'] = df['jailbreak_prompt'].apply(lambda x: f'"{x.}"')
    df['jailbreak_prompt'] = df.apply(lambda x: f"{x['prompt']} {x['jailbreak_prompt']}", axis=1)

    df.to_csv("./data/test_datasets/advbench_gcg.csv", index=False, quoting=1)


def check_duplicates(file_path):
    df = pd.read_csv(file_path)
    print(df.duplicated().sum())
    print(df.shape)

    def clean_text(text):
        # Lowercase the text
        text = text.lower()

        # Check if last character is punctuation
        if text[-1] in string.punctuation:
            text = text[:-1]  # Remove last character if it is punctuation

        return text

    # print ids of duplicates after removing punctuation and lowercase
    df['prompt'] = df['prompt'].apply(clean_text)

    #load xstest
    data_path = f"./data/xstest_v2_prompts.csv"
    xs_df = pd.read_csv(data_path)

    safe_categories = ["homonyms", "figurative_language", "safe_targets", "safe_contexts", "definitions",
                       "historical_events", "privacy_public", "privacy_fictional", "real_group_nons_discr",
                       "nons_group_real_discr"]

    unsafe_df = xs_df[~xs_df['type'].isin(safe_categories)]

    unsafe_df['prompt'] = unsafe_df['prompt'].apply(clean_text)
    print(unsafe_df.head())

    # check the different prompts in the two datasets
    print(df['prompt'].isin(unsafe_df['prompt']).sum())

    # print the non dups
    print(df[~df['prompt'].isin(unsafe_df['prompt'])].head())

    # df.drop_duplicates(inplace=True)
    # print(df.shape)
    # df.to_csv(file_path, index=False)


if __name__ == '__main__':
    # merge_and_process_outputs(
    #     file_paths=[
    #         "./data/gcg_advbench_outputs_1.csv",
    #         "./data/gcg_advbench_outputs_2.csv",
    #     ],
    #     output_path="./data/gcg_advbench_outputs.csv"
    # )
    check_duplicates("./data/gcg_attack_gemma2_9b_it_xstest_outputs.csv")