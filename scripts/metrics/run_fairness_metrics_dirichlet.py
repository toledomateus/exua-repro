from fairness_metrics import evaluate_popularity_metrics
from sasrec_dirichlet import load_dirichlet_model   # your existing file

checkpoint = "saved/SASRec_dirichlet_ml-1m_seed_2020_edl_no_dis_posthoc_filter.pth"
dirichlet_model, config, dataset, train_data, valid_data, test_data = \
    load_dirichlet_model(checkpoint)

device = config['device']
dirichlet_model = dirichlet_model.to(device)

results = evaluate_popularity_metrics(
    model            = dirichlet_model,
    test_dataloader  = test_data,
    dataset          = train_data,
    topk             = 20,
    short_head_ratio = 0.2,   # top 20% by rating count = short-head
    device           = device,
)