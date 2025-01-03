import pandas
import argparse
import pandas as pd
import string
import datasets

from poate_attack.config import BASE_PATH

def load_data():
    df = pd.read_csv(f"{BASE_PATH}data/potee_attacks.csv", delimiter="|")
    df = df.rename(columns={"question": "prompt"})
    return df

def match_strings(str1, str2):
    # Remove punctuation from both strings
    translator = str.maketrans('', '', string.punctuation)
    str1_clean = str1.strip().lower().translate(translator)
    str2_clean = str2.strip().lower().translate(translator)

    # Compare the cleaned strings
    return str1_clean == str2_clean

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset to use")
    args = parser.parse_args()

    if args.dataset == "advbench":
        with open(f"{BASE_PATH}/data/advbench.txt") as f:
            data = f.readlines()
    elif args.dataset == "mal_instruct":
        with open(f"{BASE_PATH}/data/MaliciousInstruct.txt") as f:
            data = f.readlines()
    elif args.dataset == "xstest":
        df = pd.read_csv(f"{BASE_PATH}/data/xstest_v2_prompts.csv")

        safe_categories = ["homonyms", "figurative_language", "safe_targets", "safe_contexts", "definitions",
                           "historical_events", "privacy_public", "privacy_fictional", "real_group_nons_discr",
                           "nons_group_real_discr"]
        unsafe_df = df[~df['type'].isin(safe_categories)]
        data = unsafe_df["prompt"].tolist()
    elif args.dataset == "i_cona":
        with open(f"{BASE_PATH}/data/icona.txt") as f:
            data = f.readlines()
    elif args.dataset == "i_controversial":
        with open(f"{BASE_PATH}/data/icontroversial.txt") as f:
            data = f.readlines()
    elif args.dataset == "phy_safety":
        with open(f"{BASE_PATH}/data/physicalsafety.txt") as f:
            data = f.readlines()
    elif args.dataset == "q_harm":
        with open(f"{BASE_PATH}/data/qharm.txt") as f:
            data = f.readlines()
    elif args.dataset == "hex_phi":
        # with open(f"{BASE_PATH}/data/hexphi.txt", "w") as f:
        #     for cat in range(1, 12):
        #         data = datasets.load_dataset(
        #             "LLM-Tuning-Safety/HEx-PHI",
        #             data_files=[f"category_{cat}.csv"]
        #         )["train"]
        #         for d in data:
        #             f.write(list(d.values())[0] + "\n")
        with open(f"{BASE_PATH}/data/hexphi.txt") as f:
            data = f.readlines()
    else:
        raise ValueError("Dataset not compatible")

    df = load_data()
    prompts = df["prompt"].tolist()

    count = 0
    for d in data:
        if any(match_strings(d, p) for p in prompts):
            count += 1
        else:
            print(d)

    print(f"Number of prompts in the dataset: {count}")
