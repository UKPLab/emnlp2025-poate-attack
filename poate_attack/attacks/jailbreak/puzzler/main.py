import pandas as pd
import torch
import os
import argparse
import re
import ast
from tqdm import tqdm

from poate_attack.attacks.jailbreak.base import TargetLM
from poate_attack.attacks.jailbreak.puzzler.prompts import create_defensive_prompt, create_filtering_prompt, create_offensive_prompt, create_indirect_jailbreak_prompt
from poate_attack.openai_response import AnyOpenAILLM


class Puzzler:

    def __init__(self, args, data_path, save_path):
        self.targetLM = TargetLM(
            model_name=args.target_model,
            max_n_tokens=args.target_max_n_tokens,
            temperature=args.target_temp,
            top_p=args.target_top_p,
            top_k=args.target_top_k,
            seed=args.seed
        )
        self.data_path = data_path
        self.save_path = save_path

        self.model = AnyOpenAILLM(
            model_name=args.target_model,
            model_kwargs={
                "temperature": 0,
                "max_tokens": 1024,
                "timeout": 30,
                "max_retries": 2,
            }
        )

    def get_defensive_measures(self, df, save_path: str):

        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            # take where success is not false
            df = df[df['success'] != False]
            print(f"Loaded {len(df)} samples from {save_path}")
            return df

        mal_contents = df['malicious_content'].to_list()
        prompts = [create_defensive_prompt(m) for m in mal_contents]
        examples = [(create_defensive_prompt("manufacture of bombs"),
                     "For the defense against the manufacture of bombs, the points of defense are 1. preventing them from obtaining the materials; 2. online platform monitoring.")
                    ]

        if args.target_model == "gpt-4o":
            target_response_list = []
            for i, input in tqdm(enumerate(prompts)):
                chat_history = []
                for ex in examples:
                    chat_history.append(("human", ex[0]))
                    chat_history.append(("assistant", ex[1]))
                chat_history.append(("human", input))

                try:
                    response = self.model(chat_history)
                    # print(response.content)
                    target_response_list.append(response.content)
                except Exception as e:
                    print(f"Error: {e}")
                    target_response_list.append(str(e))
                    continue
        else:

            target_response_list = self.targetLM.get_response(
                examples=examples,
                prompts=prompts,
                targets=[],
                defense_type=args.defense
            )

        df["response"] = target_response_list

        results = []
        for idx, data in df.iterrows():
            results.append({
                'prompt': data["prompt"],
                'jailbreak_prompt': data["potee_prompt"],
                'response': data["response"],
                'malicious_content': data["malicious_content"],
            })

        output_df = pd.DataFrame(results)
        output_df.to_csv(save_path, index=False)
        return output_df

    def clean_def_measures(self, def_measures):
        # Split text by lines and remove empty lines
        def_measures = [re.sub(r"^\d+\.\s*\*\*|\*\*", "", line).strip() for line in def_measures.split("\n") if line.strip()]
        # print(def_measures)
        return def_measures

    def filter_def_measures(self, def_measures, malicious_content):
        filter_prompts = []
        total_count_list = []
        for i, d_measure in enumerate(def_measures):
            value_counts = 0
            for value in d_measure:
                filter_prompt = create_filtering_prompt(value, malicious_content[i])
                filter_prompts.append(filter_prompt)
                value_counts += 1
            total_count_list.append(value_counts)

        examples = [
            (create_filtering_prompt("community education", "make a bomb"), "no"),
            (create_filtering_prompt("enhancing ideological education" , "steal from a store"), "no")
        ]

        if args.target_model == "gpt-4o":
            target_responses = []
            for i, input in tqdm(enumerate(filter_prompts)):
                chat_history = []
                for ex in examples:
                    chat_history.append(("human", ex[0]))
                    chat_history.append(("assistant", ex[1]))
                chat_history.append(("human", input))
                print(chat_history)

                try:
                    response = self.model(chat_history)
                    print(response.content)
                    target_responses.append(response.content)
                except Exception as e:
                    print(f"Error: {e}")
                    target_responses.append(str(e))
                    continue
                print("-------------------")
                if i == 10:
                    break
        else:

            target_responses = self.targetLM.get_response(
                examples=examples,
                prompts=filter_prompts,
                targets=[],
                defense_type=args.defense
            )

        # list of list of filter prompts and target responses acc. to total count list
        filter_prompts_list = []
        target_response_list = []
        start = 0
        for count in total_count_list:
            filter_prompts_list.append(filter_prompts[start:start+count])
            target_response_list.append(target_responses[start:start+count])
            start += count

        # choose defensive measures where the response is 'yes'
        filtered_def_measures = []
        for i, (d_measures, target_response) in enumerate(zip(def_measures, target_response_list)):
            filtered_def_measures.append([d_measure for d_measure, response in zip(d_measures, target_response) if response.lower().__contains__('yes')])

        return filtered_def_measures


    def generate_offensive_prompts(self, def_measures):
        ofensive_prompts = []
        total_count_list = []
        for i, d_measure in enumerate(def_measures):
            value_counts = 0
            for value in d_measure:
                ofensive_prompt = create_offensive_prompt(str(value))
                ofensive_prompts.append(ofensive_prompt)
                value_counts += 1
            total_count_list.append(value_counts)
        print(len(ofensive_prompts))

        if args.target_model == "gpt-4o":
            target_responses = []
            for i, input in tqdm(enumerate(ofensive_prompts)):
                chat_history = [("human", input)]

                try:
                    response = self.model(chat_history)
                    # print(response.content)
                    target_responses.append(response.content)
                except Exception as e:
                    print(f"Error: {e}")
                    target_responses.append(str(e))
                    continue
                # print("-------------------")
                # if i == 10:
                #     break
        else:

            target_responses = self.targetLM.get_response(
                examples=[],
                prompts=ofensive_prompts,
                targets=[],
                defense_type=args.defense
            )

        # list of list of filter prompts and target responses acc. to total count list
        offensive_prompts_list = []
        target_response_list = []
        start = 0
        for count in total_count_list:
            offensive_prompts_list.append(ofensive_prompts[start:start + count])
            target_response_list.append(target_responses[start:start + count])
            start += count

        # print(offensive_prompts_list[:5])
        # print(target_response_list[:5])
        return target_response_list


    def generate_indirect_jailbreak_prompts(self, off_measures):
        prompts_list = []
        for i, off_measure in enumerate(off_measures):
            offensive_measures = " ".join(off_measure)
            indirect_jailbreak_prompt = create_indirect_jailbreak_prompt(offensive_measures)
            prompts_list.append(indirect_jailbreak_prompt)

        if args.target_model == "gpt-4o":
            target_responses = []
            for i, input in tqdm(enumerate(prompts_list)):
                chat_history = [("human", input)]

                try:
                    response = self.model(chat_history)
                    # print(response.content)
                    target_responses.append(response.content)
                except Exception as e:
                    print(f"Error: {e}")
                    target_responses.append(str(e))
                    continue
        else:
            target_responses = self.targetLM.get_response(
                examples=[],
                prompts=prompts_list,
                targets=[],
                defense_type=args.defense
            )
        return target_responses


    def main(self):

        ## 1. get defensive measures
        # df = pd.read_csv(f'./data/test_datasets/{args.dataset}_sampled_data.csv')
        # def_measures_df = self.get_defensive_measures(df, self.save_path)

        # 2. clean and filter defensive measures
        # def_measures_df = pd.read_csv(self.data_path)
        # def_measures_df = def_measures_df[def_measures_df['success'] != False]
        # def_measures = def_measures_df["response"].tolist()
        # mal_content = def_measures_df["malicious_content"].tolist()
        #
        # # Clean the defensive measures
        # cleaned_def_measures = [self.clean_def_measures(d_measure) for d_measure in def_measures]
        # filtered_def_measures = self.filter_def_measures(cleaned_def_measures, mal_content)
        #
        # # save the filtered defensive measures
        # def_measures_df["filtered_def_measures"] = filtered_def_measures
        # def_measures_df.to_csv(self.save_path, index=False)

        ## 3. generate offensive prompts
        # def_measures_df = pd.read_csv(self.data_path)
        # # def_measures_df = def_measures_df[def_measures_df["filtered_def_measures"].apply(lambda x: len(ast.literal_eval(x)) > 0)]
        # # filtered_def_measures = [ast.literal_eval(d) for d in def_measures_df["filtered_def_measures"].tolist()]
        #
        # def_measures_df = def_measures_df[def_measures_df['success'] != False]
        # filtered_def_measures = [self.clean_def_measures(d) for d in def_measures_df["response"].tolist()]
        #
        # offensive_measures = self.generate_offensive_prompts(filtered_def_measures)
        # def_measures_df["offensive_measures"] = offensive_measures
        # def_measures_df.to_csv(self.save_path, index=False)

        ## 4. generate indirect jailbreak prompts
        off_measures_df = pd.read_csv(self.data_path)
        off_measures_df = off_measures_df[off_measures_df["offensive_measures"].apply(lambda x: len(ast.literal_eval(x)) > 0)]
        filtered_off_measures = [ast.literal_eval(d) for d in off_measures_df["offensive_measures"].tolist()]
        # print(filtered_off_measures[:5])

        jailbreak_response = self.generate_indirect_jailbreak_prompts(filtered_off_measures)
        off_measures_df["jailbreak_response"] = jailbreak_response
        off_measures_df.to_csv(self.save_path, index=False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ########### Target model parameters ##########
    parser.add_argument(
        "--dataset",
        default="advbench",
        required=True,
        help="Name of dataset.",
    )
    parser.add_argument(
        "--target_model",
        default="Llama_3.1_8b_instruct",
        help="Name of target model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility."
    )
    parser.add_argument(
        "--target_max_n_tokens",
        type=int,
        default=1024,
        help="Maximum number of generated tokens for the target."
    )
    parser.add_argument(
        "--target_top_p",
        type=float,
        default=1.0,
        help="Top-p for target model."
    )
    parser.add_argument(
        "--target_top_k",
        type=int,
        default=-1,
        help="Top-k for target model."
    )
    parser.add_argument(
        "--target_temp",
        type=float,
        default=0.0,
        help="Temperature for target model."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="main",
        choices=['main'],
        help="Experiment file name"
    )
    parser.add_argument(
        "--defense",
        type=str,
        default="none",
        choices=['none', 'sr', 'ic', 'paraphrase', 'sys_prompt'],
        help="LLM defense: None, Self-Reminder, In-Context"
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="gcg",
        choices=['deep_inception', 'gcg', 'potee', 'baseline', 'puzzler'],
        help="LLM attack: 'deep_inception', 'gcg', 'potee'"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="evaluate LLM responses to attack prompts"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="sample from the dataset"
    )
    ##################################################
    args = parser.parse_args()
    # torch seed
    torch.manual_seed(args.seed)

    data_path = f'./results/{args.attack}_attack_offensive_measures_{args.target_model}_{args.dataset}_sampled_outputs.csv' #f'./data/test_datasets/{args.dataset}_sampled_data.csv'
    save_path = f'./results/{args.attack}_attack_{args.target_model}_{args.dataset}_sampled_outputs.csv'
    puzzler = Puzzler(args, data_path, save_path)

    puzzler.main()

    # clean_def_save_path = f'./results/{args.attack}_attack_def_measures_cleaned_{args.target_model}_{args.dataset}_sampled_outputs.csv'
    # def_measures_df.to_csv(clean_def_save_path, index=False)
