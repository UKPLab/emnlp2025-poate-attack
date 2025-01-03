import pandas as pd
import string


def remove_punctuation(text):
    # Create a translation table that maps punctuation to None
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def generate_train_data():
    # read the dataset
    adv_df = pd.read_csv("data/potee_attack_Llama_3.1_8b_instruct_advbench_outputs_mod.csv")
    adv_df_sampled = pd.read_csv("data/test_datasets/advbench_sampled_data.csv")

    xstest_df = pd.read_csv("data/potee_attack_Llama_3.1_8b_instruct_xstest_outputs.csv")
    xstest_df_sampled = pd.read_csv("data/test_datasets/xstest_sampled_data.csv")

    mi_df = pd.read_csv("data/potee_attack_Llama_3.1_8b_instruct_malicious_instruct_outputs_mod.csv")
    mi_df_sampled = pd.read_csv("data/test_datasets/malicious_instruct_sampled_data.csv")

    results = []

    # get the prompts
    for i, row in adv_df.iterrows():
        prompt = row["prompt"]
        jb_prompt = row["jailbreak_prompt"]
        # check prompt not in adv_df_sampled
        if prompt not in adv_df_sampled["prompt"].tolist():
            # print(prompt)
            # save the prompt to a new df
            rest_prompt = jb_prompt.split(".")[1].strip()
            results.append({
                "prompt": prompt,
                "potee_prompt": jb_prompt,
                "template_prompt": rest_prompt,
                "dataset": "advbench"
            })

    for i, row in xstest_df.iterrows():
        prompt = row["prompt"]
        jb_prompt = row["jailbreak_prompt"]
        # check prompt not in adv_df_sampled
        if prompt not in xstest_df_sampled["prompt"].tolist():
            # print(prompt)
            # save the prompt to a new df
            rest_prompt = jb_prompt.split(".")[1].strip()
            results.append({
                "prompt": prompt,
                "potee_prompt": jb_prompt,
                "template_prompt": rest_prompt,
                "dataset": "xstest"
            })

    for i, row in mi_df.iterrows():
        prompt = row["prompt"]
        jb_prompt = row["jailbreak_prompt"]
        # check prompt not in adv_df_sampled
        if prompt not in mi_df_sampled["prompt"].tolist():
            # print(prompt)
            # save the prompt to a new df
            rest_prompt = jb_prompt.split(".")[1].strip()
            results.append({
                "prompt": prompt,
                "potee_prompt": jb_prompt,
                "template_prompt": rest_prompt,
                "dataset": "mi"
            })

    # save the new df
    data_df = pd.DataFrame(results)
    # remove duplicate prompts
    data_df = data_df.drop_duplicates(subset=["prompt"])
    data_df.to_csv("data/test_datasets/train_data_template.csv", index=False)


def generate_test_sets(dataset):
    train_data = pd.read_csv("data/test_datasets/potee_complete.csv")
    # take subset where dataset col in df == dataset
    train_data = train_data[train_data["dataset"] == dataset]
    train_prompts = train_data["prompt"].tolist()
    print(len(train_prompts))

    # read the dataset
    if dataset == "advbench":
        df = pd.read_csv("data/advbench.txt", delimiter="\t", header=None)
        df.columns = ["prompt"]
        # add a punctuation to the prompt
        df["prompt"] = df["prompt"] + "."
    elif dataset == "xstest":
        data_path = "data/xstest_v2_prompts.csv"
        df = pd.read_csv(data_path)

        safe_categories = ["homonyms", "figurative_language", "safe_targets", "safe_contexts", "definitions",
                           "historical_events", "privacy_public", "privacy_fictional", "real_group_nons_discr",
                           "nons_group_real_discr"]

        df = df[~df['type'].isin(safe_categories)]

    elif dataset == "malicious_instruct":
        df = pd.read_csv("data/MaliciousInstruct.txt", delimiter="\t", header=None)
        df.columns = ["prompt"]

    all_prompts = df["prompt"].tolist()
    # get the non-overlapping prompts (do lowercase and check)
    train_prompts_cleaned = [(remove_punctuation(prompt.lower()), prompt) for prompt in train_prompts]
    all_prompts_cleaned = [(remove_punctuation(prompt.lower()), prompt) for prompt in all_prompts]
    cleaned_train_prompts_set = {cleaned for cleaned, original in train_prompts_cleaned}

    # Find non-overlapping prompts and keep their original forms
    non_overlapping_prompts = [
        original for cleaned, original in all_prompts_cleaned if cleaned not in cleaned_train_prompts_set
    ]
    print(len(non_overlapping_prompts))
    # print(non_overlapping_prompts)

    # save the non-overlapping prompts to a df
    non_overlapping_prompts_df = pd.DataFrame(non_overlapping_prompts, columns=["prompt"])
    non_overlapping_prompts_df.to_csv(f"data/id_test_sets/{dataset}.csv", index=False)



if __name__ == "__main__":
    # generate_test_sets(
    #     dataset="malicious_instruct"
    # )
    generate_train_data()