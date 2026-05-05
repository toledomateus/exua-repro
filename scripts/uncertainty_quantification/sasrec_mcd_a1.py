import torch
import torch.nn.functional as F
from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils import get_trainer
import numpy as np

# ─────────────────────────────────────────────
# 1.  TRAIN CONFIG (used if no checkpoint given)
# ─────────────────────────────────────────────

parameter_dict = {
    'load_col': {
        'inter': ['user_id', 'item_id', 'timestamp'],
        'item': ['item_id', 'item_category'],
    },
    'train_neg_sample_args': None,
    'loss_type': 'CE',
    'MAX_ITEM_LIST_LENGTH': 50,
    'topk': [20],
    
    # Updated metrics list including diversity/tail metrics
    'metrics': [
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
# 2.  HELPERS
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
    Includes LayerNorm, dropout (correctly named), and RecBole masking.
    """
    # 1. Position embedding
    position_ids = torch.arange(item_seq.size(1), dtype=torch.long,
                                device=item_seq.device).unsqueeze(0).expand_as(item_seq)
    position_emb = model.position_embedding(position_ids)

    # 2. Combine, LayerNorm, Dropout
    # Note: RecBole names the layer 'dropout', not 'emb_dropout'
    input_emb = item_emb + position_emb
    input_emb = model.LayerNorm(input_emb)
    input_emb = model.dropout(input_emb)

    # 3. RecBole's own attention mask
    attn_mask = model.get_attention_mask(item_seq, bidirectional=False)

    # 4. Transformer encoder
    output = model.trm_encoder(input_emb, attn_mask,
                               output_all_encoded_layers=True)[-1]

    # 5. Gather indices
    seq_out = model.gather_indexes(output, item_seq_len - 1)

    # 6. Final projection
    return seq_out @ model.item_embedding.weight.T


# ─────────────────────────────────────────────
# 3.  MC DROPOUT UNCERTAINTY ESTIMATION
# ─────────────────────────────────────────────

@torch.no_grad()
def mcd_uncertainty(model, interaction, n_samples: int = MCD_SAMPLES):
    """Computes uncertainty metrics (Total, Aleatoric, Epistemic) using MCD."""
    model.eval()
    enable_dropout(model)

    all_logits = []
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
# 4.  EXPLAINABLE UNCERTAINTY ATTRIBUTION (ExUA)
# ─────────────────────────────────────────────

def compute_exua(model, interaction, n_samples: int = MCD_SAMPLES,
                 tau: float = TEMPERATURE):
    """Computes the attribution map (Attention Vector A)."""
    model.eval()
    enable_dropout(model)

    item_seq     = interaction['item_id_list']
    item_seq_len = interaction['item_length']

    M_plus_accum = None
    M_minus_accum = None

    for _ in range(n_samples):
        # 1. Clean gradients from previous loop
        model.zero_grad()
        
        # 2. Force gradient tracking even if called inside no_grad
        with torch.enable_grad():
            item_emb = model.item_embedding(item_seq).detach()
            item_emb.requires_grad = True
            
            logits = sasrec_compat_forward(model, item_emb, item_seq, item_seq_len)
            probs  = F.softmax(logits, dim=-1)
            
            eps = 1e-10
            entropy = -(probs * (probs + eps).log()).sum(dim=-1)
            
            # Backprop sum of entropies
            loss = entropy.sum()
        
        loss.backward()

        # 3. Compute Saliency
        grad = item_emb.grad
        grad_x_emb = (grad * item_emb).sum(dim=-1)
        
        m_plus     = F.relu(grad_x_emb)
        m_minus    = F.relu(-grad_x_emb)

        m_plus_soft  = F.softmax(m_plus  / tau, dim=-1)
        m_minus_soft = F.softmax(m_minus / tau, dim=-1)

        if M_plus_accum is None:
            M_plus_accum  = m_plus_soft
            M_minus_accum = m_minus_soft
        else:
            M_plus_accum  += m_plus_soft
            M_minus_accum += m_minus_soft

    M_max = M_plus_accum  / n_samples
    M_min = M_minus_accum / n_samples
    
    A = 0.5 * (1.0 - M_max + M_min)

    return M_max, M_min, A


# ─────────────────────────────────────────────
# 5.  EVALUATION HELPERS
# ─────────────────────────────────────────────

def evaluate(model, data, config, dataset=None):
    """Run RecBole's evaluator with manual injection of dataset stats for advanced metrics."""
    # 1. Create a fresh trainer for evaluation
    trainer = get_trainer(config['MODEL_TYPE'], config['loss_type'])(config, model)
    
    # 2. Patch the Collector to inject missing global stats (ItemCoverage, Gini, etc.)
    if dataset:
        collector = trainer.eval_collector
        original_get_struct = collector.get_data_struct

        def patched_get_data_struct():
            # Get the standard struct (contains model scores/rankings)
            struct = original_get_struct()
            
            # Inject 'data.num_items' (Required for ItemCoverage)
            # RecBole uses .set() method on DataStruct
            struct.set('data.num_items', dataset.item_num)
            
            # Inject 'data.count_items' (Required for AveragePopularity / TailPercentage)
            if hasattr(dataset, 'item_counter'):
                struct.set('data.count_items', dataset.item_counter)
            else:
                # Fallback: calculate item frequencies on the fly if counter is missing
                from collections import Counter
                # dataset.inter_feat is the interaction dataframe (user, item, ...)
                if hasattr(dataset, 'inter_feat'):
                    counts = Counter(dataset.inter_feat[dataset.iid_field].numpy())
                    struct.set('data.count_items', counts)
            
            return struct

        # Apply the patch to the trainer instance
        collector.get_data_struct = patched_get_data_struct

    # 3. Run standard evaluation
    return trainer.evaluate(data, load_best_model=False, show_progress=False)


def print_full_comparison(res_before, res_after):
    """Prints a formatted table comparing two RecBole result dictionaries."""
    all_keys = sorted(list(set(res_before.keys()) | set(res_after.keys())))
    
    # Priority sort: NDCG, Hit first
    priority = ['ndcg', 'hit', 'mrr', 'recall', 'precision']
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
            
            # Simple visual indicator
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
# 6.  UNCERTAINTY-AWARE POST-HOC TRAINING
# ─────────────────────────────────────────────

def uncertainty_aware_training(model, train_dataloader, valid_data, test_data,
                                config, optimizer, dataset, # <--- Added dataset arg
                                alpha: float = 0.2, tau: float = 0.001,
                                max_epochs: int = 30, device='cuda',
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

            # with torch.no_grad():
            #     # compute_exua handles gradient enabling internally now
            #     _, _, A = compute_exua(model, interaction, tau=tau)
            
            model.train()

            item_emb     = model.item_embedding(item_seq)
            item_emb_hat = (1.0 + alpha) * item_emb

            logits = sasrec_compat_forward(model, item_emb_hat, item_seq, item_seq_len)
            loss   = F.cross_entropy(logits, pos_items)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, batch_idx + 1)

        # Pass dataset to evaluate so it can calculate advanced metrics
        valid_result = evaluate(model, valid_data, config, dataset)
        valid_ndcg   = valid_result.get('ndcg@20', valid_result.get('NDCG@20', 0.0))

        print(f"Epoch [{epoch+1:02d}/{max_epochs}]  "
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
            'config':           config,
            'state_dict':       model.state_dict(),
            'best_valid_ndcg':  best_valid_ndcg,
        }, save_path)
    else:
        print("No improvement found.")

    return model


# ─────────────────────────────────────────────
# 7.  ENTRY POINT
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
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the post-hoc trained model')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed for reproducibility')
    parser.add_argument('--prep', type=str, default='', help='Preprocessing identifier')
    parser.add_argument('--dataset', type=str, default='ml-1m', help='Dataset name (e.g., ml-1m, beauty)')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")


    if args.save_path is None:
        args.save_path = f'saved/sasrec_mcd_{timestamp}.pth'
    if args.checkpoint is None:
        print("No checkpoint supplied – training SASRec from scratch first.")
        run_recbole(model='SASRec', dataset=args.dataset, config_dict=parameter_dict)
        print("\nTraining done. Re-run with --checkpoint <path> to use ExUA.")

    else:
        print(f"Loading checkpoint: {args.checkpoint}")
        config, model, dataset, train_data, valid_data, test_data = \
            load_data_and_model(model_file=args.checkpoint)
        
        # Ensure the config includes the new metrics (overwriting loaded config metrics)
        config['metrics'] = parameter_dict['metrics']

        device = config['device']
        model  = model.to(device)

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
            # Pass dataset to fix IndexError
            test_before = evaluate(model, test_data, config, dataset)
            print_scores("SASRec + MCDropout (before)", test_before)

            # ── Post-hoc training ─────────────────────────────────────────
            print("\n── Starting uncertainty-aware post-hoc training ──")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
            
            model = uncertainty_aware_training(
                model, train_data, valid_data, test_data, config,
                optimizer, dataset,  # <--- Pass dataset here
                alpha=0.2, tau=0.001, max_epochs=30,
                device=device, save_path=args.save_path,
            )

            # ── Final scores AFTER post-hoc training ──────────────────────
            print("\n── Final evaluation (Table 3 comparison) ──")
            test_after = evaluate(model, test_data, config, dataset)
            print_scores("ExUA-MCD (after)", test_after)

            # ── Side-by-side summary ───────────────────────────────────────
            print("\n── Final Impact Analysis ────────────────────────────────────────")
            print_full_comparison(test_before, test_after)