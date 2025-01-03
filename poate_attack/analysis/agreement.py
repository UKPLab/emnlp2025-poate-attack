import os
import pandas as pd
import numpy as np
import krippendorff
from sklearn.metrics import cohen_kappa_score



def calculate_agreement(file_path):
    df = pd.read_csv(file_path)
    # print(df.head())
    human_responses = df['success'].values.tolist()
    model_responses = df['jailbreak_success'].values.tolist()

    # print("="*50)
    # print(human_responses)
    # print(model_responses)

    human_responses = [1 if response is True else 0 for response in human_responses]
    model_responses = [1 if response == "Yes" else 0 for response in model_responses]

    raw_agreement = sum(r1 == r2 for r1, r2 in zip(human_responses, model_responses)) / len(human_responses)
    # print(f"Raw Agreement: {raw_agreement:.2%}")
    # print("=" * 50)

    return raw_agreement


if __name__ == '__main__':
    dataset = ["advbench", "xstest", "malicious_instruct"]
    models = ["Llama_2_7b_chat_hf", "Llama_3.1_8b_instruct", "gemma2_9b_it", "phi_3_mini_4k", "gpt_4o"]
    attack = "gen_exploit"

    if attack == "gcg":
        models = models[:-1]
    elif attack == "puzzler":
        models = ["Llama_2_7b_chat_hf", "gemma2_9b_it", "phi_3_mini_4k", "gpt-4o"]

    final_avg_agreement = 0
    for model in models:
        final_agreement = 0
        for d in dataset:

            file_path = f"./data/{attack}_attack_harmbench_classifier_{model}_{d}_sampled_outputs.csv"

            if os.path.exists(file_path):
                agreement = calculate_agreement(file_path)
                final_agreement += agreement
                # print(f"{model} {d} Agreement: {agreement:.2%}")
            else:
                print(f"{model} {d} Agreement: File not found")
                argreement = 100
                final_agreement += agreement

        print(f"Final Agreement: {final_agreement / len(dataset):.2%}")
        final_avg_agreement += final_agreement/len(dataset)

    print(f"Final Average Agreement: {final_avg_agreement / len(models):.2%}")