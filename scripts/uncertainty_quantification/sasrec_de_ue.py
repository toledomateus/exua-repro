import torch
import torch.nn.functional as F
from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils import get_trainer
import numpy as np
import os
import copy

# ─────────────────────────────────────────────
# 1.  TRAIN CONFIG
# ─────────────────────────────────────────────

N_ENSEMBLE = 5
BASE_SEEDS =[2020, 2021, 2022, 2023, 2024]
TEMPERATURE = 0.01

parameter_dict = {
    'load_col': {
        'inter':['user_id', 'item_id', 'timestamp'],
        'item':['item_id', 'item_category'],
    },
    'train_neg_sample_args': None,
    'loss_type': 'CE',
    'MAX_ITEM_LIST_LENGTH': 50,
    'topk': [20],
    'metrics':[
        'Recall', 'MRR', 'NDCG', 'Hit', 'Precision',
        'ItemCoverage', 'ShannonEntropy', 'GiniIndex',
        'AveragePopularity', 'TailPercentage'
    ],
    'valid_metric': 'NDCG@20',
    'stopping_step': 10,  # Adjusted to a standard patience value
    'attn_dropout_prob': 0.2,
    'hidden_dropout_prob': 0.2,
}


# ─────────────────────────────────────────────
# 2.  HELPERS
# ─────────────────────────────────────────────

def sasrec_compat_forward(model, item_emb, item_seq, item_seq_len):
    """
    Replicates RecBole SASRec.forward exactly, but accepts external item_emb
    so gradients can be tracked w.r.t. the embeddings for ExUA.
    """
    position_ids = torch.arange(
        item_seq.size(1), dtype=torch.long, device=item_seq.device
    ).unsqueeze(0).expand_as(item_seq)
    position_emb = model.position_embedding(position_ids)

    input_emb = item_emb + position_emb
    input_emb = model.LayerNorm(input_emb)
    input_emb = model.dropout(input_emb)

    attn_mask = model.get_attention_mask(item_seq, bidirectional=False)
    output = model.trm_encoder(input_emb, attn_mask, output_all_encoded_layers=True)[-1]
    seq_out = model.gather_indexes(output, item_seq_len - 1)

    return seq_out @ model.item_embedding.weight.T


def evaluate(model, data, config, dataset=None):
    """Run RecBole's evaluator, injecting dataset stats for advanced metrics."""
    trainer = get_trainer(config['MODEL_TYPE'], config['loss_type'])(config, model)

    if dataset:
        collector = trainer.eval_collector
        original_get_struct = collector.get_data_struct

        def patched_get_data_struct():
            struct = original_get_struct()
            struct.set('data.num_items', dataset.item_num)
            if hasattr(dataset, 'item_counter'):
                struct.set('data.count_items', dataset.item_counter)
            elif hasattr(dataset, 'inter_feat'):
                from collections import Counter
                counts = Counter(dataset.inter_feat[dataset.iid_field].numpy())
                struct.set('data.count_items', counts)
            return struct

        collector.get_data_struct = patched_get_data_struct

    return trainer.evaluate(data, load_best_model=False, show_progress=False)


def print_full_comparison(res_before, res_after):
    all_keys = sorted(list(set(res_before.keys()) | set(res_after.keys())))
    priority =['ndcg', 'hit', 'mrr', 'recall', 'precision']

    def sort_key(k):
        k_lower = k.lower()
        for i, p in enumerate(priority):
            if p in k_lower:
                return (i, k)
        return (len(priority), k)

    all_keys.sort(key=sort_key)
    print(f"\n{'METRIC':<25} | {'BEFORE':<10} | {'AFTER':<10} | {'DIFF':<10}")
    print("─" * 65)
    for k in all_keys:
        val_b = res_before.get(k, None)
        val_a = res_after.get(k, None)
        str_b = f"{val_b:.4f}" if isinstance(val_b, (float, int)) else str(val_b)
        str_a = f"{val_a:.4f}" if isinstance(val_a, (float, int)) else str(val_a)
        diff_str = "-"
        if isinstance(val_b, (float, int)) and isinstance(val_a, (float, int)):
            diff = val_a - val_b
            diff_str = f"{diff:+.4f}"
            if "gini" in k.lower():
                if diff < 0: diff_str += " (Better)"
            elif diff > 0:
                diff_str += " (Better)"
        print(f"{k:<25} | {str_b:<10} | {str_a:<10} | {diff_str:<10}")
    print("─" * 65 + "\n")


def print_scores(label, result):
    h    = result.get('hit@20',  result.get('Hit@20',  '?'))
    ndcg = result.get('ndcg@20', result.get('NDCG@20', '?'))
    print(f"  {label:40s}  H@20={h}  NDCG@20={ndcg}")


# ─────────────────────────────────────────────
# 3.  ENSEMBLE LOADING / TRAINING
# ─────────────────────────────────────────────

def train_ensemble(dataset: str = 'ml-1m',
                   save_dir: str = 'saved/ensemble',
                   n_models: int = N_ENSEMBLE,
                   seeds: list = BASE_SEEDS) -> list:
    """
    Trains N independent SASRec models, each with a different random seed.
    Returns a list of checkpoint paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_paths = []

    for i, seed in enumerate(seeds[:n_models]):
        print(f"\n──────────────────────────────────────────────────")
        print(f"  Training ensemble member {i+1}/{n_models}  (seed={seed})")
        print(f"──────────────────────────────────────────────────")
        cfg = dict(parameter_dict)
        cfg['seed'] = seed
        run_recbole(model='SASRec', dataset=dataset, config_dict=cfg)

        saved_files = sorted([f for f in os.listdir('saved') if f.startswith('SASRec') and f.endswith('.pth')],
            key=lambda f: os.path.getmtime(os.path.join('saved', f))
        )
        if saved_files:
            src = os.path.join('saved', saved_files[-1])
            dst = os.path.join(save_dir, f'SASRec_member_{i+1}_seed{seed}.pth')
            os.rename(src, dst)
            checkpoint_paths.append(dst)
            print(f"  Saved to: {dst}")

    return checkpoint_paths


def load_ensemble(checkpoint_paths: list, device: str = 'cuda'):
    """
    Loads N SASRec models from checkpoint paths.
    All models share the same dataset split (loaded from the first checkpoint).
    """
    models =[]
    config = dataset = train_data = valid_data = test_data = None

    for i, path in enumerate(checkpoint_paths):
        print(f"  Loading ensemble member {i+1}/{len(checkpoint_paths)}: {path}")
        cfg, mdl, ds, tr, vl, te = load_data_and_model(model_file=path)

        if i == 0:
            config, dataset, train_data, valid_data, test_data = cfg, ds, tr, vl, te

        if i > 0:
            mdl_fresh = copy.deepcopy(models[0])
            mdl_fresh.load_state_dict(mdl.state_dict())
            mdl = mdl_fresh

        mdl = mdl.to(device)
        mdl.eval()
        models.append(mdl)

    config['metrics'] = parameter_dict['metrics']
    return models, config, dataset, train_data, valid_data, test_data


# ─────────────────────────────────────────────
# 4.  DEEP ENSEMBLES UNCERTAINTY ESTIMATION
# ─────────────────────────────────────────────

@torch.no_grad()
def de_uncertainty(models: list, interaction):
    """
    Computes uncertainty metrics (Total, Aleatoric, Epistemic) using Deep Ensembles.
    Models are in eval() mode — randomness comes from parameter diversity.
    """
    ref_device = next(models[0].parameters()).device
    all_logits =[]
    for model in models:
        model.eval()
        device = next(model.parameters()).device
        all_logits.append(model.full_sort_predict(interaction.to(device)).to(ref_device))

    # probs_stack:[N_ensemble, batch_size, n_items]
    probs_stack = F.softmax(torch.stack(all_logits, dim=0), dim=-1)
    probs_mean  = probs_stack.mean(0)  # Posterior predictive probability
    eps = 1e-10

    # Total uncertainty: entropy of the averaged predictive distribution
    U_T = -(probs_mean * (probs_mean + eps).log()).sum(-1)
    # Aleatoric: expected entropy of each member's predictive distribution
    U_A = -(probs_stack * (probs_stack + eps).log()).sum(-1).mean(0)
    # Epistemic: mutual information = U_T - U_A
    U_E = U_T - U_A

    return probs_mean, U_T, U_A, U_E


# ─────────────────────────────────────────────
# 5.  EXPLAINABLE UNCERTAINTY ATTRIBUTION (ExUA)
# ─────────────────────────────────────────────

def compute_exua_ensemble(models: list, interaction, tau: float = TEMPERATURE):
    """
    Computes the ExUA attribution map for a Deep Ensemble based on Epistemic
    Uncertainty (U_E = mutual information = U_T - U_A).

    Using U_E focuses the saliency signal on sequence positions that drive
    *disagreement between ensemble members* rather than overall predictive
    entropy, which may be dominated by the inherent difficulty of the task.

    Gradients are enabled only for the forward+backward pass to compute saliency.
    Post-backward aggregation runs under no_grad to prevent building a useless
    graph from the still-live requires_grad=True leaf tensors (item_embs).
    """
    ref_device   = next(models[0].parameters()).device
    item_seq     = interaction['item_id_list']
    item_seq_len = interaction['item_length']

    item_embs  = []
    probs_list = []

    with torch.enable_grad():
        for model in models:
            model.eval()
            device = next(model.parameters()).device

            item_seq_d     = item_seq.to(device)
            item_seq_len_d = item_seq_len.to(device)

            item_emb = model.item_embedding(item_seq_d).detach()
            item_emb.requires_grad = True
            item_embs.append((model, item_emb))

            logits = sasrec_compat_forward(model, item_emb, item_seq_d, item_seq_len_d)
            probs  = F.softmax(logits, dim=-1)
            probs_list.append(probs.to(ref_device))

        # ── Compute U_E (epistemic / mutual information) ──────────────────────
        probs_stack = torch.stack(probs_list, dim=0)   # [N, B, V]
        probs_mean  = probs_stack.mean(dim=0)           # [B, V]
        eps = 1e-10

        # U_T: entropy of the mean predictive distribution
        U_T = -(probs_mean * (probs_mean + eps).log()).sum(dim=-1)   # [B]
        # U_A: expected entropy of each member's predictive distribution
        U_A = -(probs_stack * (probs_stack + eps).log()).sum(dim=-1).mean(dim=0)  # [B]
        # U_E: mutual information = disagreement signal between members
        U_E = U_T - U_A                                               # [B]

        # Backpropagate through U_E so that gradients reflect which positions
        # caused the *ensemble members to disagree* most with each other.
        loss = U_E.sum()
        loss.backward()

    # Post-backward aggregation: no_grad quarantine prevents the still-live
    # requires_grad=True leaf tensors (item_embs) from building a useless
    # graph through the saliency math. A is returned as a plain tensor.
    with torch.no_grad():
        M_plus_accum  = None
        M_minus_accum = None

        for model, item_emb in item_embs:
            grad = item_emb.grad
            grad_x_emb = (grad * item_emb).sum(dim=-1)

            m_plus  = F.relu(grad_x_emb)
            m_minus = F.relu(-grad_x_emb)

            m_plus_soft  = F.softmax(m_plus  / tau, dim=-1).to(ref_device)
            m_minus_soft = F.softmax(m_minus / tau, dim=-1).to(ref_device)

            if M_plus_accum is None:
                M_plus_accum  = m_plus_soft
                M_minus_accum = m_minus_soft
            else:
                M_plus_accum  += m_plus_soft
                M_minus_accum += m_minus_soft

            model.zero_grad()

        n = len(models)
        M_max = M_plus_accum  / n
        M_min = M_minus_accum / n
        A = 0.5 * (1.0 - M_max + M_min)

    return M_max, M_min, A

# ─────────────────────────────────────────────
# 6.  ENSEMBLE-LEVEL EVALUATE WRAPPER
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_ensemble(models: list, data, config, dataset=None):
    """
    Evaluates the ensemble by scoring with the averaged posterior predictive.
    Averages the *probabilities* (not logits) to mathematically maintain the Bayesian format.
    """
    primary = models[0]
    original_predicts =[m.full_sort_predict for m in models]

    def ensemble_predict(interaction):
        prob_sum = None
        for predict_fn, model in zip(original_predicts, models):
            device = next(model.parameters()).device
            logits = predict_fn(interaction.to(device))
            
            # Apply softmax to aggregate probabilities, establishing the true expected posterior
            probs = F.softmax(logits, dim=-1)
            probs = probs.to(next(primary.parameters()).device)
            
            prob_sum = probs if prob_sum is None else prob_sum + probs
            
        return prob_sum / len(models)

    # Temporarily patch only the primary model's predict method
    primary.full_sort_predict = ensemble_predict
    try:
        result = evaluate(primary, data, config, dataset)
    finally:
        primary.full_sort_predict = original_predicts[0]  # Always restore

    return result


# ─────────────────────────────────────────────
# 7.  UNCERTAINTY-AWARE POST-HOC TRAINING 
# ─────────────────────────────────────────────

def uncertainty_aware_training_ensemble(
        models: list,
        train_dataloader, valid_data, test_data,
        config, optimizers: list, dataset,
        alpha: float = 0.2, tau: float = 0.001,
        max_epochs: int = 30, device: str = 'cuda',
        save_dir: str = 'saved/ensemble_posthoc'):
    """
    Post-hoc uncertainty-aware training for Deep Ensembles.
    Uses U_E (epistemic uncertainty / mutual information) as the ExUA signal,
    directing attention toward sequence positions that cause ensemble disagreement.
    """
    os.makedirs(save_dir, exist_ok=True)
    best_valid_ndcg = -1.0
    best_states = [None] * len(models)

    for epoch in range(max_epochs):
        for model in models:
            model.train()
        total_loss = 0.0

        for batch_idx, interaction in enumerate(train_dataloader):
            interaction  = interaction.to(device)
            item_seq     = interaction['item_id_list']
            item_seq_len = interaction['item_length']
            pos_items    = interaction['item_id']

            # Compute shared ExUA attention vector derived from U_E.
            # Function internally manages its own gradient context.
            _, _, A = compute_exua_ensemble(models, interaction, tau=tau)

            # Fine-tune each ensemble member with the shared attention vector
            batch_loss = 0.0
            for model, optimizer in zip(models, optimizers):
                model.train()
                model_device = next(model.parameters()).device

                item_seq_d     = item_seq.to(model_device)
                item_seq_len_d = item_seq_len.to(model_device)
                pos_items_d    = pos_items.to(model_device)
                A_d            = A.to(model_device)

                item_emb     = model.item_embedding(item_seq_d)
                
                # Apply the ExUA attention modifications (Paper section 3.1)
                item_emb_hat = (1.0 + alpha * A_d.unsqueeze(-1)) * item_emb

                # "Proceed with inference as described in Sec. 2" -> Forward Pass using hat embedding 
                logits = sasrec_compat_forward(model, item_emb_hat, item_seq_d, item_seq_len_d)
                loss   = F.cross_entropy(logits, pos_items_d)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss += loss.item()

            total_loss += batch_loss / len(models)

        avg_loss = total_loss / max(1, batch_idx + 1)

        valid_result = evaluate_ensemble(models, valid_data, config, dataset)
        valid_ndcg   = valid_result.get('ndcg@20', valid_result.get('NDCG@20', 0.0))

        is_best = valid_ndcg > best_valid_ndcg
        print(f"Epoch[{epoch+1:02d}/{max_epochs}]  "
              f"avg_loss={avg_loss:.4f}  "
              f"valid NDCG@20={valid_ndcg:.4f}"
              + (" ← best" if is_best else ""))

        if is_best:
            best_valid_ndcg = valid_ndcg
            best_states =[
                {k: v.cpu().clone() for k, v in m.state_dict().items()}
                for m in models
            ]

    if any(s is not None for s in best_states):
        print(f"\nBest valid NDCG@20: {best_valid_ndcg:.4f}")
        for i, (model, state) in enumerate(zip(models, best_states)):
            model.load_state_dict({k: v.to(device) for k, v in state.items()})
            path = os.path.join(save_dir, f'SASRec_DE_posthoc_ue_member{i+1}.pth')
            torch.save({
                'config':          config,
                'state_dict':      model.state_dict(),
                'best_valid_ndcg': best_valid_ndcg,
                'member_index':    i,
            }, path)
            print(f"  Saved member {i+1} → {path}")
    else:
        print("No improvement found during post-hoc training.")

    return models


# ─────────────────────────────────────────────
# 8.  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    from datetime import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoints', type=str, nargs='+', default=None,
        help='Paths to N trained SASRec checkpoints (one per ensemble member). '
             'If not provided, trains the ensemble from scratch.'
    )
    parser.add_argument(
        '--mode', type=str, default='uncertainty',
        choices=['uncertainty', 'posthoc'],
        help='"uncertainty" for analysis only, "posthoc" for post-hoc training.'
    )
    parser.add_argument('--dataset',   type=str, default='ml-1m')
    parser.add_argument('--save_dir',  type=str, default='saved/ensemble/amazon/office/posthoc')
    parser.add_argument('--n_models',  type=int, default=N_ENSEMBLE)
    args = parser.parse_args()

    # ── Step 1: Obtain ensemble checkpoints ───────────────────────────────────
    if args.checkpoints is None:
        print(f"No checkpoints supplied — training {args.n_models} SASRec models from scratch.")
        checkpoint_paths = train_ensemble(
            dataset=args.dataset,
            save_dir='saved/ensemble/amazon/office/posthoc',
            n_models=args.n_models,
            seeds=BASE_SEEDS[:args.n_models],
        )
        print(f"\nTraining complete. Re-run with --checkpoints {' '.join(checkpoint_paths)} "
              f"to skip retraining.")
    else:
        checkpoint_paths = args.checkpoints
        if len(checkpoint_paths) != args.n_models:
            print(f"Warning: received {len(checkpoint_paths)} checkpoints "
                  f"but --n_models={args.n_models}. Adjusting n_models.")
            args.n_models = len(checkpoint_paths)

    # ── Step 2: Load ensemble ─────────────────────────────────────────────────
    print(f"\nLoading ensemble ({args.n_models} members)...")
    models, config, dataset, train_data, valid_data, test_data = load_ensemble(
        checkpoint_paths, device='cuda'
    )
    device = 'cuda'

    # ── Step 3: Uncertainty stats on first test batch ─────────────────────────
    try:
        batch = next(iter(test_data))
        interaction = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)

        _, U_T, U_A, U_E = de_uncertainty(models, interaction)
        print(f"\n── Deep Ensemble Uncertainty (N={len(models)}, first test batch) ──")
        print(f"  Total      U_T : mean={U_T.mean():.4f}  std={U_T.std():.4f}")
        print(f"  Aleatoric  U_A : mean={U_A.mean():.4f}  std={U_A.std():.4f}")
        print(f"  Epistemic  U_E : mean={U_E.mean():.4f}  std={U_E.std():.4f}")
    except Exception as e:
        print(f"Could not run uncertainty stats on batch: {e}")

    # ── Step 4: Mode-specific logic ───────────────────────────────────────────
    if args.mode == 'posthoc':
        # Baseline: ensemble evaluation before post-hoc training
        print("\n── Evaluating ensemble baseline (Table 1 comparison) ──")
        test_before = evaluate_ensemble(models, test_data, config, dataset)
        print_scores("SASRec + DeepEnsembles (before)", test_before)

        # Post-hoc training
        print("\n── Starting uncertainty-aware post-hoc training (ExUA via U_E) ──")
        optimizers =[torch.optim.Adam(m.parameters(), lr=1e-4) for m in models]

        models = uncertainty_aware_training_ensemble(
            models, train_data, valid_data, test_data,
            config, optimizers, dataset,
            alpha=0.2, tau=0.001, max_epochs=30,
            device=device, save_dir=args.save_dir,
        )

        # Final evaluation after post-hoc training
        print("\n── Final evaluation (Table 3 comparison) ──")
        test_after = evaluate_ensemble(models, test_data, config, dataset)
        print_scores("ExUA-DE / U_E (after)", test_after)

        print("\n── Final Impact Analysis ────────────────────────────────────────")
        print_full_comparison(test_before, test_after)

    else:
        # Uncertainty-only mode: just print ensemble scores
        print("\n── Ensemble evaluation (Table 1 comparison) ──")
        test_result = evaluate_ensemble(models, test_data, config, dataset)
        print_scores("SASRec + DeepEnsembles", test_result)