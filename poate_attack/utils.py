import pandas as pd
import json
import re


def remove_punctuation_regex(text):
    return re.sub(r'[^\w\s]', '', text)

def format_openai_response(dataset: str, model_name: str,  attack: str):

    # file_name = f"./data/auto_{attack}_attack_{model_name}_{dataset}_test_mistral_32_shot_with_clues"
    file_name = f'./data/auto_{attack}_attack_gemma2_9b_it_{dataset}_test_mistral_32_shot_paraphrased'
    with open(f'{file_name}.json', 'r') as f:
        data = json.load(f)

    # Initialize a list to hold rows for the DataFrame
    rows = []

    # Loop through each entry in the JSON file
    for entry in data:
        # print(entry)
        # Check if the entry contains an 'error' field
        if 'error' in entry: # or 'success' not in entry:
            # Handle error case (you can skip or log these entries, here we're skipping)
            print(f"Skipping error entry: {entry.get('prompt')}")

            flattened_entry = {
                'prompt': entry.get('prompt'),
                'input': entry.get('input'),
                'response': "Filtered by Azure policy!",
                'score': 5,
                'success': entry.get('success'),
                'completion_tokens': None,
                'prompt_tokens': None,
                'total_tokens': None,
                'model_name': None,
                'system_fingerprint': None,
                'finish_reason': None,
                'logprobs': None,
                'content_filter_results': None
            }

            # Add the flattened entry to the list of rows
            rows.append(flattened_entry)
            continue

        # extract the score
        score_match = re.search(r"#thescore:\s*(\d+)", entry.get('response'))
        score = int(score_match.group(1)) if score_match else None

        # Flatten regular entry structure
        flattened_entry = {
            'prompt': entry.get('prompt'),
            'input': entry.get('input'),
            'input_prompt_paraphrase': entry.get('input_prompt_paraphrase'),
            'response': entry.get('response'),
            'score': score,
            'success': entry.get('success'),
            'completion_tokens': entry.get('metadata', {}).get('token_usage', {}).get('completion_tokens'),
            'prompt_tokens': entry.get('metadata', {}).get('token_usage', {}).get('prompt_tokens'),
            'total_tokens': entry.get('metadata', {}).get('token_usage', {}).get('total_tokens'),
            'model_name': entry.get('metadata', {}).get('model_name'),
            'system_fingerprint': entry.get('metadata', {}).get('system_fingerprint'),
            'finish_reason': entry.get('metadata', {}).get('finish_reason'),
            'logprobs': entry.get('metadata', {}).get('logprobs'),
            'content_filter_results': entry.get('metadata', {}).get('content_filter_results')
        }

        # Add the flattened entry to the list of rows
        rows.append(flattened_entry)

    # Create DataFrame from the collected rows
    df = pd.DataFrame(rows)
    print(df["score"].mean())

    # count entries with  score >=3
    # print(df[df["score"] >= 3].shape[0])
    # print(df[df["score"] >= 3].shape[0]/df.shape[0] * 100)


    # Save the DataFrame to a CSV file
    df.to_csv(f'{file_name}.csv', index=False)

    print("CSV file has been created successfully, with error entries skipped.")


def format_harmbench_response(file_name: str):
    # file_name = f"./data/{attack}_attack_harmbench_classifier_{model_name}_{dataset}_sampled_outputs"
    with open(f'{file_name}.json', 'r') as f:
        data = json.load(f)

    # Initialize a list to hold rows for the DataFrame
    rows = []

    # Loop through each entry in the JSON file
    for entry in data:
        # print(entry)
        # Check if the entry contains an 'error' field
        if entry.get('response').lower().__contains__("the response was filtered"):  # or 'success' not in entry:
            # Handle error case (you can skip or log these entries, here we're skipping)
            print(f"Skipping error entry: {entry.get('prompt')}")

            flattened_entry = {
                'prompt': entry.get('prompt'),
                'jailbreak_prompt': entry.get('jailbreak_prompt'),
                'response': "Filtered by Azure policy!",
                'jailbreak_success': '',
                'success': entry.get('success'),
                'completion_tokens': None,
                'prompt_tokens': None,
                'total_tokens': None,
                'model_name': None,
                'system_fingerprint': None,
                'finish_reason': None,
                'logprobs': None,
                'content_filter_results': None
            }

            # Add the flattened entry to the list of rows
            rows.append(flattened_entry)
            continue

        # extract the score
        jb_success_match = re.search(r'\b(Yes|No)\b(?=\s*[\.\n])', entry.get('response'), re.IGNORECASE)
        jb_success = jb_success_match.group(0) if jb_success_match else None

        # Flatten regular entry structure
        flattened_entry = {
            'prompt': entry.get('prompt'),
            'jailbreak_prompt': entry.get('jailbreak_prompt'),
            'response': entry.get('response'),
            'jailbreak_success': jb_success,
            'success': entry.get('success'),
            'completion_tokens': entry.get('metadata', {}).get('token_usage', {}).get('completion_tokens'),
            'prompt_tokens': entry.get('metadata', {}).get('token_usage', {}).get('prompt_tokens'),
            'total_tokens': entry.get('metadata', {}).get('token_usage', {}).get('total_tokens'),
            'model_name': entry.get('metadata', {}).get('model_name'),
            'system_fingerprint': entry.get('metadata', {}).get('system_fingerprint'),
            'finish_reason': entry.get('metadata', {}).get('finish_reason'),
            'logprobs': entry.get('metadata', {}).get('logprobs'),
            'content_filter_results': entry.get('metadata', {}).get('content_filter_results')
        }

        # Add the flattened entry to the list of rows
        rows.append(flattened_entry)

    # Create DataFrame from the collected rows
    df = pd.DataFrame(rows)
    df.to_csv(f'{file_name}.csv', index=False)

    print("CSV file has been created successfully, with error entries skipped.")

def calculate_openai_cost():
    file_path = './data/openai_cost/cost.txt'
    # with open(file_path, 'r') as f:
    #     data = f.read()
    # print(data)
    # read comma separated values into a dataframe
    df = pd.read_csv(file_path)
    # col names
    names = ["date_time", "cost"]
    df.columns = names
    # filter for a particular month
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df[df['date_time'].dt.month == 12]
    total_cost = df['cost'].sum()
    print(f"Total cost: {total_cost}")


def extract_sampled_data(dataset: str, model_name: str, attack: str):
    df = pd.read_csv(f"./data/{attack}_attack_{model_name}_{dataset}_outputs.csv")
    print(df.head())
    # full_inputs = df['prompt'].tolist()
    sampled_df = pd.read_csv(f"./data/test_datasets/{dataset}_sampled_data.csv")
    sampled_inputs = [remove_punctuation_regex(p.lower().strip()) for p in sampled_df['prompt'].tolist()]

    results = []
    for i, row in df.iterrows():
        cleaned_input = remove_punctuation_regex(row['prompt'].lower().strip())
        if cleaned_input in sampled_inputs:
            # add to df
            results.append(row)

    # save the results to a df
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"./data/{attack}_attack_{model_name}_{dataset}_sampled_outputs.csv")


def select_gen_exploit_samples(model_name: str, dataset: str):
    df = pd.read_csv(f"./merged_eval_results/exploited/gen_exploit_{model_name}_{dataset}_sampled_outputs.csv")
    print(df.head())
    # full_inputs = df['prompt'].tolist()
    sampled_df = pd.read_csv(f"./data/test_datasets/{dataset}_sampled_data.csv")
    sampled_inputs = [remove_punctuation_regex(p.lower().strip()) for p in sampled_df['prompt'].tolist()]

    results = []
    for i, row in df.iterrows():
        cleaned_input = remove_punctuation_regex(row['prompt'].lower().strip())
        if cleaned_input in sampled_inputs:
            # add to df
            results.append(row)

    # save the results to a df
    results_df = pd.DataFrame(results)
    print(results_df.head())
    results_df.to_csv(f"./data/gen_exploit_{model_name}_{dataset}_sampled_outputs.csv")


def check_duplicates(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # prompts_list1 = df1['prompt'].tolist()
    # prompts_list2 = df2['prompt'].tolist()
    #
    # prompts_list1 = [remove_punctuation_regex(p.lower().strip()) for p in prompts_list1]
    # prompts_list2 = [remove_punctuation_regex(p.lower().strip()) for p in prompts_list2]
    #
    # # check prompts in file2 that are not in file1
    # for p in prompts_list2:
    #     if p not in prompts_list1:
    #         print(p)

    df2['prompt'] = df2['prompt'].apply(lambda x: remove_punctuation_regex(x.lower().strip()))

    # for prompt in df1, if prompt in df2, add the target value
    for i, row in df1.iterrows():
        prompt = remove_punctuation_regex(row['prompt'].lower().strip())
        # print(prompt)
        if prompt in df2['prompt'].tolist():
            print(df2[df2['prompt'] == prompt])
            target = df2[df2['prompt'] == prompt]['target'].values
            print(target)

            df1.loc[i, 'target'] = target
            # else:
            print(f"Prompt: {prompt} not found in df2")
    df1.to_csv("./data/test_datasets/xstest_sampled_data_1.csv", index=False)


def category_wise_asr(model_name: str):

    base_combined_df = pd.DataFrame()
    combined_df = pd.DataFrame()
    for dataset in ["advbench", "xstest", "malicious_instruct"]:
        file_name = f"./data/auto_potee_attack_harmbench_classifier_{model_name}_{dataset}_sampled_outputs.csv"
        df = pd.read_csv(file_name)
        combined_df = pd.concat([combined_df, df])

        base_df = pd.read_csv(f"./data/test_datasets/{dataset}_sampled_data.csv")
        base_combined_df = pd.concat([base_combined_df, base_df])

    cat_counts = dict(base_combined_df['category'].value_counts())
    print(cat_counts)
    # filter df rows where success is True and jailbreak_success is Yes
    combined_df = combined_df[(combined_df['success'] == True) & (combined_df['jailbreak_success'] == 'Yes')]
    print(combined_df.category.tolist())
    # print(df.head())
    # calculate category wise ASR by category counts by dividing by total number of samples in that category
    category_counts = combined_df['category'].value_counts()
    for cat, count in category_counts.items():
        asr = round((count / cat_counts[cat])*100, 2)
        print(f"{cat}: {asr}")


def extract_questions(file_name: str):
    df = pd.read_csv(file_name)
    jb_prompts = df['jailbreak_prompt'].tolist()
    polar_opposite_prompts = df['polar_opposite_prompt'].tolist()

    opposite_prompts = []
    rest_prompts = []
    auto_jb_prompts = []
    for prompt, polar_opp_prompt in zip(jb_prompts, polar_opposite_prompts):
        # extract first sentence from prompt
        opposite_prompts.append(prompt.split(".")[0] + ".")
        rest_prompts.append(".".join(prompt.split(".")[1:]).strip())
        auto_jb_prompts.append(polar_opp_prompt + ".".join(prompt.split(".")[1:]))

    df['polar_opposite_prompt_human'] = opposite_prompts
    df['rest_prompt'] = rest_prompts
    df['auto_jailbreak_prompt'] = auto_jb_prompts
    df.to_csv(file_name, index=False)


def check_asr(dataset: str, model: str):
    # prompt_with_clues = f"./data/auto_potee_attack_none_defense_{model}_{dataset}_sampled_outputs_mistral_32_shot_with_clues.csv"
    # prompt_wo_clues = f"./data/auto_potee_attack_none_defense_{model}_{dataset}_sampled_outputs_mistral_32_shot_without_clues.csv"
    prompt_with_clues = f"./data/auto_potee_attack_{model}_{dataset}_test_mistral_32_shot_with_clues.csv"
    prompt_wo_clues = f"./data/auto_potee_attack_{model}_{dataset}_test_mistral_32_shot_without_clues.csv"

    df_with_clues = pd.read_csv(prompt_with_clues)
    df_wo_clues = pd.read_csv(prompt_wo_clues)

    # get the rows from dfs where success is True
    df_with_clues_false = df_with_clues[df_with_clues['success'] == False]
    df_with_clues = df_with_clues[df_with_clues['success'] == True]
    df_wo_clues = df_wo_clues[df_wo_clues['success'] == True]

    # remove the rows from df_wo_clues that are in df_with_clues
    df_wo_clues = df_wo_clues[~df_wo_clues['prompt'].isin(df_with_clues['prompt'])]

    print(df_with_clues.shape)
    print(df_wo_clues.shape)

    # calculate % of samples in df with and wo clues out of total samples
    print(f"Samples with clues: {df_with_clues.shape[0]/(df_with_clues.shape[0] + df_wo_clues.shape[0]) * 100}")
    print(f"Samples without clues: {df_wo_clues.shape[0]/(df_with_clues.shape[0] + df_wo_clues.shape[0]) * 100}")

    # # df clues false keep only samples not in df wo clues
    # df_with_clues_false = df_with_clues_false[~df_with_clues_false['prompt'].isin(df_wo_clues['prompt'])]
    #
    # # merge the two dfs
    # df = pd.concat([df_with_clues, df_wo_clues, df_with_clues_false])
    # print(df.shape)
    # print(df['prompt'].nunique())
    #
    # df.to_csv(f"./data/auto_potee_attack_none_defense_{model}_{dataset}_sampled_outputs_mistral_32_shot_jailbroken_complete.csv", index=False)



def add_categories_to_prompts(dataset: str, model_name: str):
    """
    Add categories to the prompts in the dataset
    :param dataset:
    :param model_name:
    :return:
    """

    df = pd.read_csv(f"./data/test_datasets/{dataset}_sampled_data.csv")

    # load the response files
    response_file = f"./data/auto_potee_attack_harmbench_classifier_{model_name}_{dataset}_sampled_outputs.csv"

    response_df = pd.read_csv(response_file)

    # for prompt in response_df, if prompt in df, add the category
    for i, row in response_df.iterrows():
        prompt = row['prompt']
        # print(prompt)
        if prompt in df['prompt'].tolist():
            category = df[df['prompt'] == prompt]['category'].values
            # print(category)

            response_df.loc[i, 'category'] = category
            # else:
            print(f"Prompt: {prompt} not found in df")

    response_df.to_csv(f"./data/auto_potee_attack_harmbench_classifier_{model_name}_{dataset}_sampled_outputs.csv", index=False)


if __name__ == '__main__':
    attack = "gen_exploit"
    datasets = ["advbench", "xstest", "malicious_instruct"]
    model = "gpt-4o"
    for d in datasets:
        check_asr(d, model)
    #     ## add_categories_to_prompts(dataset=d, model_name=model)
    #     format_openai_response(dataset=d, model_name=model, attack=attack)
    #     format_openai_response(dataset=d, model_name=model, attack=attack)
    #     format_harmbench_response(f"./data/{attack}_attack_harmbench_classifier_{model}_{d}_sampled_outputs")


    # calculate_openai_cost()
    # extract_sampled_data(dataset=dataset, model_name=model, attack=attack)

    # file_name = f"./data/{attack}_attack_openai_judge_{model}_{dataset}_sampled_outputs_human_mod_2"
    # df = pd.read_csv(f"{file_name}.csv")
    # print(df["score"].mean())

    # select_gen_exploit_samples(model_name=model, dataset=dataset)

    # check_duplicates("./data/test_datasets/malicious_instruct_sampled_data.csv",
    #                  "./data/test_datasets/malicious_instruct_gcg.csv")

    # category_wise_asr("gpt-4o")

    # extract_questions("./data/potee_attack_polar_opp_gen_gpt3_5_34_shot_template_2_xstest_sampled_outputs.csv")

    # check_asr("malicious_instruct", "gemma2_9b_it")
