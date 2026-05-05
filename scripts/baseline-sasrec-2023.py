from recbole.quick_start import run_recbole

parameter_dict = {
    'load_col': {
        'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
        'item': ['item_id', 'genre']
    },
    'train_neg_sample_args': None,
    'loss_type': 'CE',
    'MAX_ITEM_LIST_LENGTH': 50,
    'topk': [20],  # Paper uses @20 instead of default @10
    'valid_metric': 'NDCG@20',  # Changed from MRR@10 to NDCG@20
    'metrics': ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision',
                'ItemCoverage', 'ShannonEntropy', 'GiniIndex',
                'AveragePopularity', 'TailPercentage'],
    'tail_ratio': 0.1,  # Added for TailPercentage metric
    'seed': 2023,
    'stopping_step': 300,
    'user_inter_num_interval': "[0,inf)",
    'item_inter_num_interval': "[0,inf)",
    'epochs': 300,
}

run_recbole(model='SASRec', dataset='amazon-office-products', config_dict=parameter_dict)

# CUDA_VISIBLE_DEVICES=1 nohup python baseline-sasrec-2023.py > logs/movielens/baseline/filtered/sasrec_baseline_seed_2023.log 2>&1 &