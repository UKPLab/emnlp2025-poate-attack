import argparse
import os
import json
import csv

def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--method_name", type=str, default='GCG',
                        help="The name of the red teaming method")
    parser.add_argument("--experiment_name", type=str,
                        help="The name of the experiment (options are in configs/methods_config/{{method_name}}_config.yaml)")
    parser.add_argument("--method_config_file", type=str, default=None,
                        help="The path to the config file with method hyperparameters. This will default to configs/method_configs/{{method_name}}_config.yaml")
    parser.add_argument("--models_config_file", type=str, default='./configs/model_configs/models.yaml',
                        help="The path to the config file with model hyperparameters")
    parser.add_argument("--behaviors_path", type=str,
                        default='./data/behavior_datasets/harmbench_behaviors_text_all.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--save_dir", type=str, default='./test_cases',
                        help="The directory used for saving test cases")
    parser.add_argument("--behavior_start_idx", type=int, default=None,
                        help="Start index for behaviors_path data (inclusive)")
    parser.add_argument("--behavior_end_idx", type=int, default=None,
                        help="End index for behaviors_path data (exclusive)")
    parser.add_argument("--behavior_ids_subset", type=str, default='',
                        help="An optional comma-separated list of behavior IDs, or a path to a newline-separated list of behavior IDs. If provided, this will override behavior_start_idx and behavior_end_idx for selecting a subset of behaviors.")
    parser.add_argument("--run_id", type=str, default=None,
                        help="An optional run id for output files, will be appended to output filenames. Must be an integer.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Whether to overwrite existing saved test cases")
    parser.add_argument("--verbose", action="store_true",
                        help="Whether to print out intermediate results")

    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    method_name = args.method_name
    experiment_name = args.experiment_name
    save_dir = os.path.abspath(args.save_dir)

    # ========== generate test cases ========== #
    # load behaviors
    if args.behaviors_path.endswith('.csv'):
        with open(args.behaviors_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            behaviors = [entry for entry in reader]
    elif args.behaviors_path.endswith('.json'):
        with open(args.behaviors_path, 'r', encoding='utf-8') as f:
            reader = json.load(f)
            behaviors = [entry for entry in reader]

    # just select a subset of behaviors
    behaviors = behaviors[args.behavior_start_idx:args.behavior_end_idx]
    # print(behaviors)
    #
    # # ==== Init Method ====
    from src.attacks.jailbreak.gcg.gcg import GCG

    target_model = dict(
        model_name_or_path= "meta-llama/Meta-Llama-3.1-8B-Instruct",
        num_gpus= 1,
        chat_template= "llama-3",
    )

    method_config = dict(
        target_model=target_model,  # Target model
        num_steps=500,
        adv_string_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        search_width=512,
        eval_steps=10,
        eval_with_check_refusal=True,
        check_refusal_min_loss=0.3, # only start to check refusal if loss <= check_refusal_min_loss, as generate can take a lot of time
        starting_search_batch_size=None,  # auto preset search_batch_size which will auto go half each time go OOM
    )

    method = GCG(**method_config)

    # ====== Run attack ======
    test_case, logs = method.generate_test_cases(behaviors, verbose=True)
    # print(test_cases)
    # ====== Save test cases and logs ======
    print("Saving outputs to: ", save_dir)
    method.save_test_cases(save_dir, test_case, logs, method_config=method_config, run_id=args.run_id)
    print("Finished")


if __name__ == "__main__":
    main()
