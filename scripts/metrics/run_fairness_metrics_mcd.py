from fairness_metrics import evaluate_popularity_metrics
from recbole.quick_start import load_data_and_model

checkpoint = "saved/SASRec-Mar-24-2026_07-23-07.pth"
config, model, dataset, train_data, valid_data, test_data = \
    load_data_and_model(model_file=checkpoint)

device = config['device']
model = model.to(device)

results = evaluate_popularity_metrics(
    model            = model,
    test_dataloader  = test_data,
    dataset          = train_data,
    topk             = 20,
    short_head_ratio = 0.2,
    device           = device,
)