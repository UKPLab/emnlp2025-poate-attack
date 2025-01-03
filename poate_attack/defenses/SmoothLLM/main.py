import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import argparse

import poate_attack.defenses.SmoothLLM.perturbations as perturbations
import poate_attack.defenses.SmoothLLM.defenses as defenses
import poate_attack.defenses.SmoothLLM.attacks as attacks
import poate_attack.defenses.SmoothLLM.language_models as language_models
# import poate_attack.defenses.SmoothLLM.model_configs as model_configs

from poate_attack.config import ModelPath, ConvTemplates


def main(args):
    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)

    # Instantiate the targeted LLM
    model_path = ModelPath.get_path(args.target_model)
    print("Model name:", model_path)
    conv = ConvTemplates()
    conv_template_name = conv.get_template_name(args.target_model)

    target_model = language_models.LLM(
        model_path=model_path,
        tokenizer_path=model_path,
        conv_template_name=conv_template_name,
        device='cuda:0'
    )

    # Create SmoothLLM instance
    defense = defenses.SmoothLLM(
        target_model=target_model,
        pert_type=args.smoothllm_pert_type,
        pert_pct=args.smoothllm_pert_pct,
        num_copies=args.smoothllm_num_copies
    )

    # Create attack instance, used to create prompts
    attack = vars(attacks)[args.attack](
        logfile=args.attack_logfile,
        target_model=target_model
    )

    # print("Attack prompts", attack.prompts)
    df = pd.read_csv(args.attack_logfile)
    results_dict = {}
    for i, prompt in tqdm(enumerate(attack.prompts)):
        all_inputs, all_outputs, output = defense(prompt)
        jb = defense.is_jailbroken(output)

        results_dict[i] = {
            'prompt': df['prompt'].to_list()[i],
            'jailbreak_prompt': df['jailbreak_prompt'].to_list()[i],
            'perturbed_inputs': all_inputs,
            'perturbed_outputs': all_outputs,
            'response': output,
            'jailbroken': jb
        }
        # break

    summary_df = pd.DataFrame.from_dict(results_dict, orient='index')
    summary_df.to_csv(os.path.join(
        args.results_dir, f'auto_potee_attack_smoothllm_defense_{args.target_model}_{args.dataset}_sampled_outputs.csv'
    ))
    # print(summary_df)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results'
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=0
    )

    # Targeted LLM
    parser.add_argument(
        '--target_model',
        type=str,
        default='vicuna',
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='advbench',
    )

    # Attacking LLM
    parser.add_argument(
        '--attack',
        type=str,
        default='GCG',
        choices=['GCG', 'PAIR', 'Potee']
    )
    parser.add_argument(
        '--attack_logfile',
        type=str,
        default=""
    )

    # SmoothLLM
    parser.add_argument(
        '--smoothllm_num_copies',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--smoothllm_pert_pct',
        type=int,
        default=10
    )
    parser.add_argument(
        '--smoothllm_pert_type',
        type=str,
        default='RandomSwapPerturbation',
        choices=[
            'RandomSwapPerturbation',
            'RandomPatchPerturbation',
            'RandomInsertPerturbation'
        ]
    )

    args = parser.parse_args()
    main(args)