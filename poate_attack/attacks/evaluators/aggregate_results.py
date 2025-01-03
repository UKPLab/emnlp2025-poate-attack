import json
import pandas as pd


models = ['Llama_3.1_8b_instruct']
dataset = 'advbench'

dfs = []
config_dfs = []
for model in models:
    # vanilla = json.load(open(f'./merged_eval_results/greedy/{model}_summary.json', 'r'))
    data = json.load(open(f'./merged_eval_results/exploited/{model}_{dataset}_matching_only_summary.json', 'r'))
    data_cp = {'model': model} # 'vanilla': vanilla['greedy']}
    config_cp = {'model': model}
    for (k,v) in data.items():
        if k != 'best_attack_config':
            data_cp[k] = v
        else:
            for kk, vv in v.items():
                config_cp[kk] = (list(vv.keys())[0].split('_')[1], list(vv.values())[0])
    dfs.append(pd.DataFrame([data_cp]))
    config_dfs.append(pd.DataFrame([config_cp]))


concat_dfs = pd.concat(dfs)
print(concat_dfs.head())