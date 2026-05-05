import torch
import torch.nn.functional as F
from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils import get_trainer
import numpy as np

# ─────────────────────────────────────────────
# 1.  TRAIN CONFIG 
# ─────────────────────────────────────────────

parameter_dict = {
    'load_col': {
        'inter':['user_id', 'item_id', 'timestamp'],
        'item': ['item_id', 'item_category'],
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
    'stopping_step': 10,
    'seed': 2020,
    'attn_dropout_prob': 0.2,
    'hidden_dropout_prob': 0.2,
}


# ─────────────────────────────────────────────
# 2.  DEVICE SELECTION
# ─────────────────────────────────────────────

def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[Device] CUDA available — using {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[Device] CUDA not available — using CPU")
    return device


# ─────────────────────────────────────────────
# 3.  HELPERS
# ─────────────────────────────────────────────

MCD_SAMPLES = 10
TEMPERATURE  = 0.01


def enable_dropout(model: torch.nn.Module) -> None:
    """Forces Dropout layers to be in training mode (for MCD)."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def sasrec_compat_forward(model, item_emb, item_seq, item_seq_len):
    """
    Replicates RecBole SASRec.forward exactly, but accepts external item_emb
    so gradients can be tracked w.r.t. the embeddings for ExUA.
    """
    position_ids = torch.arange(item_seq.size(1), dtype=torch.long,
                                device=item_seq.device).unsqueeze(0).expand_as(item_seq)
    position_emb = model.position_embedding(position_ids)

    input_emb = item_emb + position_emb
    input_emb = model.LayerNorm(input_emb)
    input_emb = model.dropout(input_emb)

    attn_mask = model.get_attention_mask(item_seq, bidirectional=False)

    output = model.trm_encoder(input_emb, attn_mask,
                               output_all_encoded_layers=True)[-1]

    seq_out = model.gather_indexes(output, item_seq_len - 1)

    return seq_out @ model.item_embedding.weight.T


# ─────────────────────────────────────────────
# 4.  MC DROPOUT UNCERTAINTY ESTIMATION
# ─────────────────────────────────────────────

@torch.no_grad()
def mcd_uncertainty(model, interaction, n_samples: int = MCD_SAMPLES):
    """Computes uncertainty metrics (Total, Aleatoric, Epistemic) using MCD."""
    model.eval()
    enable_dropout(model)

    all_logits =[]
    for _ in range(n_samples):
        all_logits.append(model.full_sort_predict(interaction))

    probs_stack = F.softmax(torch.stack(all_logits, dim=0), dim=-1)
    probs_mean  = probs_stack.mean(0)
    eps = 1e-10

    U_T = -(probs_mean * (probs_mean + eps).log()).sum(-1)
    U_A = -(probs_stack * (probs_stack + eps).log()).sum(-1).mean(0)
    U_E = U_T - U_A

    return probs_mean, U_T, U_A, U_E


# ─────────────────────────────────────────────
# 5.  EXPLAINABLE UNCERTAINTY ATTRIBUTION (ExUA)
#     Driven by Epistemic Uncertainty (U_E)
# ─────────────────────────────────────────────

def compute_exua(model, interaction, n_samples: int = MCD_SAMPLES, tau: float = TEMPERATURE):
    """
    Computes the attribution map (Attention Vector A) via ∇U_E ≈ ∇U_T - ∇U_A
    
    Theoretical Note on the ∇U_T Approximation:
    Perfect calculation of ∇U_T requires backpropagating through the ensemble mean of N stochastic 
    forward passes. This requires holding N computation graphs in VRAM, causing immediate OOM errors.
    Instead, we substitute the ensemble mean E[p] with the deterministic proxy p_det (model.eval()).
    While H(p_det) <= H(E[p]) due to softmax non-linearities (acting as a biased estimator), it 
    provides a highly correlated directional gradient while dropping memory complexity to O(1).
    """
    model.eval()
    enable_dropout(model)

    item_seq     = interaction['item_id_list']
    item_seq_len = interaction['item_length']
    eps = 1e-10

    # Accumulator for ∇U_A
    grad_x_emb_A_accum = None

    # ── Pass 1: accumulate ∇U_A (averaged per-sample entropy gradient) ────
    for _ in range(n_samples):
        model.zero_grad()

        with torch.enable_grad():
            item_emb = model.item_embedding(item_seq).detach()
            item_emb.requires_grad = True

            logits = sasrec_compat_forward(model, item_emb, item_seq, item_seq_len)
            probs  = F.softmax(logits, dim=-1)

            # Per-sample entropy H(p_t) — aleatoric component
            per_sample_entropy = -(probs * (probs + eps).log()).sum(dim=-1)
            loss_A = per_sample_entropy.sum()

        loss_A.backward()
        
        # Input x Gradient Saliency
        grad_x_emb_t = (item_emb.grad * item_emb).sum(dim=-1)

        if grad_x_emb_A_accum is None:
            grad_x_emb_A_accum = grad_x_emb_t
        else:
            grad_x_emb_A_accum += grad_x_emb_t

    # Average over samples
    grad_x_emb_A = grad_x_emb_A_accum / n_samples

    # ── Pass 2: compute ∇U_T using the deterministic proxy ────────────────
    model.zero_grad()

    with torch.enable_grad():
        # Re-attach a fresh embedding for conceptual cleanliness
        item_emb_for_ut = model.item_embedding(item_seq).detach()
        item_emb_for_ut.requires_grad = True

        # Use the deterministic pass (dropout disabled) as the proxy for E[p]
        model.eval()   
        logits_ut = sasrec_compat_forward(model, item_emb_for_ut, item_seq, item_seq_len)
        probs_mean_proxy = F.softmax(logits_ut, dim=-1)

        # H(E[p]) proxy — total uncertainty
        entropy_T = -(probs_mean_proxy * (probs_mean_proxy + eps).log()).sum(dim=-1)
        loss_T    = entropy_T.sum()

    loss_T.backward()
    
    # Restore dropout for subsequent training/batch iterations
    enable_dropout(model)   

    grad_x_emb_T = (item_emb_for_ut.grad * item_emb_for_ut).sum(dim=-1)

    # ── Epistemic gradient: ∇U_E = ∇U_T - ∇U_A ────────────────────────────
    grad_x_emb_E = grad_x_emb_T - grad_x_emb_A

    # ── Positive / negative saliency → temperature softmax ───────────────
    m_plus  = F.relu( grad_x_emb_E)
    m_minus = F.relu(-grad_x_emb_E)

    M_max = F.softmax(m_plus  / tau, dim=-1)
    M_min = F.softmax(m_minus / tau, dim=-1)

    A = 0.5 * (1.0 - M_max + M_min)

    return M_max, M_min, A


# ─────────────────────────────────────────────
# 6.  EVALUATION HELPERS
# ─────────────────────────────────────────────

def evaluate(model, data, config, dataset=None):
    """Run RecBole's evaluator with manual injection of dataset stats for advanced metrics."""
    trainer = get_trainer(config['MODEL_TYPE'], config['loss_type'])(config, model)

    if dataset:
        collector = trainer.eval_collector
        original_get_struct = collector.get_data_struct

        def patched_get_data_struct():
            struct = original_get_struct()
            struct.set('data.num_items', dataset.item_num)

            if hasattr(dataset, 'item_counter'):
                struct.set('data.count_items', dataset.item_counter)
            else:
                from collections import Counter
                if hasattr(dataset, 'inter_feat'):
                    counts = Counter(dataset.inter_feat[dataset.iid_field].numpy())
                    struct.set('data.count_items', counts)

            return struct

        collector.get_data_struct = patched_get_data_struct

    return trainer.evaluate(data, load_best_model=False, show_progress=False)


def print_full_comparison(res_before, res_after):
    """Prints a formatted table comparing two RecBole result dictionaries."""
    all_keys = sorted(list(set(res_before.keys()) | set(res_after.keys())))

    priority =['ndcg', 'hit', 'mrr', 'recall', 'precision']
    def sort_key(k):
        k_lower = k.lower()
        for i, p in enumerate(priority):
            if p in k_lower: return (i, k)
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
    print(f"  {label:30s}  H@20={h}  NDCG@20={ndcg}")


# ─────────────────────────────────────────────
# 7.  UNCERTAINTY-AWARE POST-HOC TRAINING
# ─────────────────────────────────────────────

def uncertainty_aware_training(model, train_dataloader, valid_data, test_data,
                                config, optimizer, dataset,
                                alpha: float = 0.2, tau: float = 0.001,
                                max_epochs: int = 30, device='cpu',
                                save_path: str = None):
    best_valid_ndcg = -1.0
    best_state      = None

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, interaction in enumerate(train_dataloader):
            interaction  = interaction.to(device)
            item_seq     = interaction['item_id_list']
            item_seq_len = interaction['item_length']
            pos_items    = interaction['item_id']

            with torch.no_grad():
                _, _, A = compute_exua(model, interaction, tau=tau)

            model.train()

            item_emb     = model.item_embedding(item_seq)
            item_emb_hat = (1.0 + alpha * A.unsqueeze(-1)) * item_emb

            logits = sasrec_compat_forward(model, item_emb_hat, item_seq, item_seq_len)
            loss   = F.cross_entropy(logits, pos_items)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, batch_idx + 1)

        valid_result = evaluate(model, valid_data, config, dataset)
        valid_ndcg   = valid_result.get('ndcg@20', valid_result.get('NDCG@20', 0.0))

        print(f"Epoch[{epoch+1:02d}/{max_epochs}]  "
              f"avg_loss={avg_loss:.4f}  "
              f"valid NDCG@20={valid_ndcg:.4f}"
              + (" ← best" if valid_ndcg > best_valid_ndcg else ""))

        if valid_ndcg > best_valid_ndcg:
            best_valid_ndcg = valid_ndcg
            best_state      = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        print(f"\nBest valid NDCG@20: {best_valid_ndcg:.4f}  →  saving to {save_path}")
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        torch.save({
            'config':          config,
            'state_dict':      model.state_dict(),
            'best_valid_ndcg': best_valid_ndcg,
        }, save_path)
    else:
        print("No improvement found.")

    return model


# ─────────────────────────────────────────────
# 8.  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a trained RecBole SASRec checkpoint')
    parser.add_argument('--mode', type=str, default='uncertainty',
                        choices=['uncertainty', 'posthoc'],
                        help='Choose "uncertainty" for analysis or "posthoc" for training')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the post-hoc trained model')
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed for reproducibility')
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        help='Dataset name (e.g., ml-1m, beauty)')
    parser.add_argument('--prep', type=str, default='', help='Preprocessing identifier')
    
    args = parser.parse_args()

    # ── Auto device selection ─────────────────────────────────────────────
    device = get_device()

    timestamp = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")

    if args.save_path is None:
        args.save_path = f'saved/sasrec_mcd_ue{timestamp}.pth'

    if args.checkpoint is None:
        print("No checkpoint supplied – training SASRec from scratch first.")
        run_recbole(model='SASRec', dataset='ml-1m', config_dict=parameter_dict)
        print("\nTraining done. Re-run with --checkpoint <path> to use ExUA.")

    else:
        print(f"Loading checkpoint: {args.checkpoint}")
        config, model, dataset, train_data, valid_data, test_data = \
            load_data_and_model(model_file=args.checkpoint)

        config['metrics'] = parameter_dict['metrics']
        config['device']  = device
        model = model.to(device)

        # ── Uncertainty stats on first test batch ─────────────────────────
        try:
            batch = next(iter(test_data))
            interaction = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)

            _, U_T, U_A, U_E = mcd_uncertainty(model, interaction)
            print(f"\n── MC Dropout Uncertainty (S={MCD_SAMPLES}, first test batch) ──")
            print(f"  Total      U_T : mean={U_T.mean():.4f}  std={U_T.std():.4f}")
            print(f"  Aleatoric  U_A : mean={U_A.mean():.4f}  std={U_A.std():.4f}")
            print(f"  Epistemic  U_E : mean={U_E.mean():.4f}  std={U_E.std():.4f}")
        except Exception as e:
            print(f"Could not run uncertainty stats on batch: {e}")

        if args.mode == 'posthoc':
            # ── Baseline scores BEFORE post-hoc training ──────────────────
            print("\n── Evaluating baseline (Table 1 comparison) ──")
            test_before = evaluate(model, test_data, config, dataset)
            print_scores("SASRec + MCDropout (before)", test_before)

            # ── Post-hoc training ─────────────────────────────────────────
            print("\n── Starting uncertainty-aware post-hoc training (U_E signal) ──")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            model = uncertainty_aware_training(
                model, train_data, valid_data, test_data, config,
                optimizer, dataset,
                alpha=0.2, tau=0.001, max_epochs=30,
                device=device, save_path=args.save_path,
            )

            # ── Final scores AFTER post-hoc training ──────────────────────
            print("\n── Final evaluation (Table 3 comparison) ──")
            test_after = evaluate(model, test_data, config, dataset)
            print_scores("ExUA-MCD U_E (after)", test_after)

            # ── Side-by-side summary ───────────────────────────────────────
            print("\n── Final Impact Analysis ────────────────────────────────────────")
            print_full_comparison(test_before, test_after)