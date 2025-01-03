import argparse
import pandas as pd
from string import Template
import csv
import json

from poate_attack.analysis import stats

ATTACK_TEMPLATE = Template("""Sure, here is ${prompt}""")

# Remove newline characters and extra spaces
template_text = ' '.join(ATTACK_TEMPLATE.template.split())

# Create a new Template object with the processed text
clean_attack_template = Template(template_text)

def load_data(dataset: str):
    if dataset == 'advbench':
        with open('./data/advbench_harmful_behaviors.json', 'r') as file:
            data = json.load(file)
    elif dataset == 'xstest':
        data_path = f"./data/xstest_v2_prompts.csv"
        df = pd.read_csv(data_path)

        safe_categories = ["homonyms", "figurative_language", "safe_targets", "safe_contexts", "definitions",
                           "historical_events", "privacy_public", "privacy_fictional", "real_group_nons_discr",
                           "nons_group_real_discr"]

        # safe_df = df[df['type'].isin(safe_categories)]
        unsafe_df = df[~df['type'].isin(safe_categories)]
        data = unsafe_df['prompt'].tolist()
    elif dataset == 'malicious_instruct':
        data_path = f"./data/MaliciousInstruct.txt"
        with open(data_path, 'r') as file:
            data = file.readlines()
    return data

def lowercase_first_word(input_string):
    # Split the input string into words
    words = input_string.split()

    if words:
        # Lowercase the first word
        words[0] = words[0].lower()

    # Combine the words back into a single string
    result = ' '.join(words)
    # remove the last punctuation
    if result[-1] in ['.', '?', '!']:
        result = result[:-1]

    return result

def create_gcg_data(dataset: str):
    # Load the data
    data = load_data(
        dataset=dataset
    )
    data_adv = pd.read_csv(f"./data/id_test_sets/{dataset}.csv")
    data_adv = data_adv["prompt"].tolist()
    # print(data[:5])
    # Load the potee attacks
    df = pd.read_csv(f"./data/potee_attacks.csv", delimiter="|")
    input_prompts = df["prompt"].tolist()

    attacked_inputs = []
    # Create the attack
    count = 0
    if dataset=="advbench":
        for line in data:
            if any(stats.match_strings(line["goal"].strip(), p) for p in data_adv):
                attacked_inputs.append({'id': count, 'prompt': line["goal"].strip(), 'target': line["target"]})
                count += 1
    else:
        for prompt in data:
            # if not any(stats.match_strings(prompt, p) for p in input_prompts):
            prompt = lowercase_first_word(prompt.strip())
            attack = clean_attack_template.substitute({"prompt":prompt})
            attacked_inputs.append({'id': count,'prompt': prompt, 'target': attack})
            count += 1
    # # write to .txt file with each attack on a new line
    # with open('./data/advbench_inception_attack.txt', 'w') as file:
    #     for attack in attacked_inputs:
    #         file.write(attack + '\n')

    df = pd.DataFrame(attacked_inputs)
    df.to_csv(f"./data/id_test_sets/{dataset}_gcg.csv", index=False)


def process_output(dataset: str):
    # Process the output
    df = pd.read_csv(f"./data/id_test_sets/{dataset}_gcg.csv")
    df["prompt"] = df["prompt"].str.capitalize()
    df["jailbreak_prompt"] = df["jailbreak_prompt"].str.replace("Sure here", "Sure, here")
    df.to_csv(f"./data/id_test_sets/{dataset}_gcg_refined.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="advbench")

    args = parser.parse_args()
    create_gcg_data(args.dataset)
    # process_output(args.dataset)
    # df = pd.read_csv(f"./data/potee_attacks.csv", delimiter="|")
    # duplicate_rows = df[df['prompt'].duplicated()]
    # print(duplicate_rows)
