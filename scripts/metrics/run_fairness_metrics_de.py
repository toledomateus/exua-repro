from fairness_metrics import evaluate_popularity_metrics
from sasrec_de_no_dis import load_ensemble

checkpoint_paths = [
   "saved/ensemble/amazon/movielens/filter/a1/posthoc/SASRec_DE_posthoc_member1.pth",
   "saved/ensemble/amazon/movielens/filter/a1/posthoc/SASRec_DE_posthoc_member2.pth",
   "saved/ensemble/amazon/movielens/filter/a1/posthoc/SASRec_DE_posthoc_member3.pth",
   "saved/ensemble/amazon/movielens/filter/a1/posthoc/SASRec_DE_posthoc_member4.pth",
   "saved/ensemble/amazon/movielens/filter/a1/posthoc/SASRec_DE_posthoc_member5.pth"
]

models, config, dataset, train_data, valid_data, test_data = load_ensemble(
    checkpoint_paths, device='cuda'
)

device = config['device']

results = evaluate_popularity_metrics(
    model            = models,
    test_dataloader  = test_data,
    dataset          = train_data, 
    topk             = 20,
    short_head_ratio = 0.2,
    device           = device,
)