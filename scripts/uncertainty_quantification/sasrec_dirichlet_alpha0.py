import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.quick_start import run_recbole, load_data_and_model
from recbole.utils import get_trainer
import numpy as np
import os

# ─────────────────────────────────────────────
# 1.  TRAIN CONFIG
# ─────────────────────────────────────────────

parameter_dict = {
    'load_col': {
        'inter': ['user_id', 'item_id', 'timestamp'],
        'item':  ['item_id', 'item_category'],
    },
    'train_neg_sample_args': None,
    'loss_type': 'CE',
    'MAX_ITEM_LIST_LENGTH': 50,
    'topk': [20],
    'metrics': [
        'Recall', 'MRR', 'NDCG', 'Hit', 'Precision',
        'ItemCoverage', 'ShannonEntropy', 'GiniIndex',
        'AveragePopularity', 'TailPercentage',
    ],
    'valid_metric': 'NDCG@20',
    'stopping_step': 150,
    'seed': 2020,
    'attn_dropout_prob': 0.2,
    'hidden_dropout_prob': 0.2,
}

# KL regularisation weight (same for both BM and EDL in practice; tune if needed)
KL_WEIGHT = 1e-3
TEMPERATURE = 0.01          # ExUA softmax temperature


# ─────────────────────────────────────────────
# 2.  DIRICHLET WRAPPER MODULE
# ─────────────────────────────────────────────

class DirichletSASRec(nn.Module):
    """
    Wraps a pre-trained RecBole SASRec model and adds a Dirichlet output head.

    The head is a single linear layer (hidden_size → n_items) followed by
    softplus, which guarantees α > 0 everywhere (required by the Dirichlet).

    During inference the predicted relevance scores exposed to RecBole's
    evaluator are  α - 1, keeping argmax behaviour consistent with the
    standard dot-product scores while remaining compatible with the
    full_sort_predict interface.
    """

    def __init__(self, sasrec_model: nn.Module):
        super().__init__()
        self.sasrec = sasrec_model

        hidden_size = sasrec_model.hidden_size
        n_items     = sasrec_model.n_items          # includes padding item 0

        # Projection: seq_out ∈ R^d  →  α ∈ R^M  (all positive via softplus)
        self.evidence_head = nn.Linear(hidden_size, n_items, bias=True)

        # Initialise close to the identity mapping so training starts from a
        # sensible point (optional but helps convergence).
        nn.init.normal_(self.evidence_head.weight, std=0.01)
        nn.init.zeros_(self.evidence_head.bias)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_seq_out(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor):
        """Replicates SASRec.forward up to the final sequence embedding."""
        model = self.sasrec
        position_ids = (
            torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
            .unsqueeze(0)
            .expand_as(item_seq)
        )
        position_emb = model.position_embedding(position_ids)
        item_emb     = model.item_embedding(item_seq)

        input_emb = item_emb + position_emb
        input_emb = model.LayerNorm(input_emb)
        input_emb = model.dropout(input_emb)

        attn_mask = model.get_attention_mask(item_seq, bidirectional=False)
        output    = model.trm_encoder(input_emb, attn_mask, output_all_encoded_layers=True)[-1]
        seq_out   = model.gather_indexes(output, item_seq_len - 1)
        return seq_out

    def _get_seq_out_from_emb(self, item_emb: torch.Tensor,
                            item_seq: torch.Tensor,
                            item_seq_len: torch.Tensor):
        """
        Same as _get_seq_out but accepts external item_emb so we can track
        gradients w.r.t. the embedding for ExUA.
        """
        model = self.sasrec
        position_ids = (
            torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
            .unsqueeze(0)
            .expand_as(item_seq)
        )
        position_emb = model.position_embedding(position_ids)

        input_emb = item_emb + position_emb
        input_emb = model.LayerNorm(input_emb)
        input_emb = model.dropout(input_emb)

        attn_mask = model.get_attention_mask(item_seq, bidirectional=False)
        output    = model.trm_encoder(input_emb, attn_mask, output_all_encoded_layers=True)[-1]
        seq_out   = model.gather_indexes(output, item_seq_len - 1)
        return seq_out

    def get_alpha(self, item_seq: torch.Tensor,
                item_seq_len: torch.Tensor) -> torch.Tensor:
        """Returns Dirichlet concentration parameters α, shape [B, M]."""
        seq_out = self._get_seq_out(item_seq, item_seq_len)
        alpha   = F.softplus(self.evidence_head(seq_out)) + 1.0   # α > 1 keeps mode well-defined
        return alpha

    def get_alpha_from_emb(self, item_emb: torch.Tensor,
                            item_seq: torch.Tensor,
                            item_seq_len: torch.Tensor) -> torch.Tensor:
        """Same as get_alpha but from an external embedding (for ExUA gradients)."""
        seq_out = self._get_seq_out_from_emb(item_emb, item_seq, item_seq_len)
        alpha   = F.softplus(self.evidence_head(seq_out)) + 1.0
        return alpha

    # ── RecBole interface ─────────────────────────────────────────────────────

    def full_sort_predict(self, interaction) -> torch.Tensor:
        """Returns item scores compatible with RecBole's evaluator."""
        item_seq     = interaction['item_id_list']
        item_seq_len = interaction['item_length']
        alpha = self.get_alpha(item_seq, item_seq_len)
        # Scores = α - 1  (equivalent to standard logits; keeps ranking identical)
        return alpha - 1.0

    def predict(self, interaction) -> torch.Tensor:
        """Pointwise score for a single target item (used in some RecBole trainers)."""
        item_seq     = interaction['item_id_list']
        item_seq_len = interaction['item_length']
        test_item    = interaction['item_id']
        alpha        = self.get_alpha(item_seq, item_seq_len)
        return (alpha - 1.0).gather(1, test_item.unsqueeze(1)).squeeze(1)

    def forward(self, interaction):
        """Alias so RecBole trainers that call model(batch) also work."""
        return self.full_sort_predict(interaction)


# ─────────────────────────────────────────────
# 3.  DIRICHLET UNCERTAINTY (ANALYTICAL)
# ─────────────────────────────────────────────

def dirichlet_uncertainty(alpha, eps=1e-8):
    S0    = alpha.sum(dim=-1, keepdim=True)          # α₀
    p_bar = alpha / S0                                # posterior mean

    # U_T — entropy of posterior mean
    U_T = -(p_bar * (p_bar + eps).log()).sum(dim=-1)

    # U_A — expected entropy (aleatoric), Ulmer Eq. 12
    U_A = (torch.digamma(S0.squeeze(-1) + 1.0)
        - (p_bar * torch.digamma(alpha + 1.0)).sum(dim=-1))


    #before: U_E = U_T - U_A
    # U_E = U_T - U_A
    # U_E — direct analytical MI, Eq. 14 (preferred over U_T - U_A)
    
    U_E = -(p_bar * (
        (alpha / S0 + eps).log()
        - torch.digamma(alpha + 1.0)
        + torch.digamma(S0 + 1.0)
    )).sum(dim=-1)

    return p_bar, U_T, U_A, U_E



# ─────────────────────────────────────────────
# 4.  LOSS FUNCTIONS
# ─────────────────────────────────────────────

def kl_dirichlet(alpha: torch.Tensor, target_alpha: torch.Tensor,
                eps: float = 1e-10) -> torch.Tensor:
    """
    KL(Dir(alpha) || Dir(target_alpha)), computed per sample.
    Returns shape [B].
    """
    S_alpha  = alpha.sum(dim=-1, keepdim=True)
    S_target = target_alpha.sum(dim=-1, keepdim=True)

    kl = (
        torch.lgamma(S_alpha)
        - torch.lgamma(S_target)
        - torch.lgamma(alpha + eps).sum(dim=-1, keepdim=True)
        + torch.lgamma(target_alpha + eps).sum(dim=-1, keepdim=True)
        + ((alpha - target_alpha) * (torch.digamma(alpha + eps)
                                    - torch.digamma(S_alpha + eps))).sum(dim=-1, keepdim=True)
    )
    return kl.squeeze(-1)   # [B]


def bm_loss(alpha: torch.Tensor, targets: torch.Tensor,
            kl_weight: float = KL_WEIGHT) -> torch.Tensor:
    """
    Bayesian Matching loss (Joo et al., ICML 2020).

    L = CE(p̄, y)  +  λ · KL( Dir(ᾱ) || Dir(1) )

    where ᾱ = α with the target class zeroed out (so the KL only penalises
    non-target evidence, forcing the model to be uncertain about wrong items).

    Args:
        alpha:   [B, M]  concentration params.
        targets: [B]     integer class labels.
        kl_weight: λ scaling for the KL term.
    """
    eps     = 1e-10
    S0      = alpha.sum(dim=-1, keepdim=True)
    p_bar   = alpha / S0

    # Standard cross-entropy on the posterior mean
    # ce_loss = F.cross_entropy(p_bar.log() + eps, targets)
    # Joo et al. Eq. 10 requires ψ(αᵧ) − ψ(α₀)
    S0_sq = alpha.sum(dim=-1)
    ce_loss = -(torch.digamma(alpha[range(alpha.size(0)), targets]) - torch.digamma(S0_sq)).mean()

    # KL regularisation: remove target-class evidence → Dir(ᾱ) vs Dir(1)
    alpha_hat = alpha.clone()
    alpha_hat.scatter_(1, targets.unsqueeze(1), 1.0)   # set target α_k → 1
    prior_alpha = torch.ones_like(alpha)                # uniform Dirichlet
    kl = kl_dirichlet(alpha_hat, prior_alpha)           # [B]

    return ce_loss + kl_weight * kl.mean()


def edl_loss(alpha: torch.Tensor, targets: torch.Tensor,
            epoch: int = 0, max_epochs: int = 30,
            kl_weight: float = KL_WEIGHT) -> torch.Tensor:
    """
    Evidential Deep Learning loss (Sensoy et al., NeurIPS 2018).

    L = NLL(α, y) + λ(t) · KL( Dir(ᾱ) || Dir(1) )

    where NLL is the type-II maximum likelihood (expected log-likelihood
    under the Dirichlet prior), and λ(t) is an annealed KL weight.

    Args:
        alpha:      [B, M]  concentration params.
        targets:    [B]     integer class labels.
        epoch:      current epoch (for annealing).
        max_epochs: total epochs (for annealing).
        kl_weight:  base KL weight (annealed further by epoch ratio).
    """
    eps = 1e-10
    S0  = alpha.sum(dim=-1, keepdim=True)   # [B, 1] — keepdim so it broadcasts against [B, M]

    # One-hot encoding
    y_oh = torch.zeros_like(alpha).scatter_(1, targets.unsqueeze(1), 1.0)   # [B, M]

    # Type-II maximum likelihood (NLL under Dir prior)
    # = Ψ(S₀) - Ψ(α_y)  for the target class
    nll = (torch.digamma(S0 + eps) - torch.digamma(alpha + eps)) * y_oh
    nll = nll.sum(dim=-1).mean()   # scalar

    # KL annealing: weight increases linearly over training
    # annealed_kl_w = kl_weight * min(1.0, epoch / max(1, max_epochs))
    annealed_kl_w = kl_weight * min(1.0, epoch / max(1, max_epochs))

    # Remove target evidence before KL (same as BM)
    alpha_hat = alpha.clone()
    alpha_hat.scatter_(1, targets.unsqueeze(1), 1.0)
    prior_alpha = torch.ones_like(alpha)
    kl = kl_dirichlet(alpha_hat, prior_alpha).mean()   # scalar

    return nll + annealed_kl_w * kl


# ─────────────────────────────────────────────
# 5.  ExUA — DIRICHLET VERSION  (S = 1)
# ─────────────────────────────────────────────

def compute_exua_dirichlet(dirichlet_model: DirichletSASRec,
                            interaction,
                            tau: float = TEMPERATURE):
    """
    ExUA attribution map for Dirichlet models.

    Because uncertainty is deterministic for Dirichlet networks (no dropout,
    no ensemble), S = 1 and we perform a single gradient pass (see paper, Sec. 3.1).
    The total uncertainty U_T (entropy of the posterior predictive) is used
    as the scalar to backpropagate, matching RQ2 motivation in the paper.

    Returns:
        M_max: [B, T]  uncertainty-increasing saliency (softmax-normalised).
        M_min: [B, T]  uncertainty-decreasing saliency.
        A:     [B, T]  attention vector in [0, 1].
    """
    dirichlet_model.eval()

    item_seq     = interaction['item_id_list']
    item_seq_len = interaction['item_length']
    device       = next(dirichlet_model.parameters()).device

    item_seq_d     = item_seq.to(device)
    item_seq_len_d = item_seq_len.to(device)

    with torch.enable_grad():
        item_emb = dirichlet_model.sasrec.item_embedding(item_seq_d).detach()
        item_emb.requires_grad = True

        alpha = dirichlet_model.get_alpha_from_emb(item_emb, item_seq_d, item_seq_len_d)
        _, U_T, _, _ = dirichlet_uncertainty(alpha)
        loss = U_T.sum()

    loss.backward()

    grad        = item_emb.grad                         # [B, T, d]
    grad_x_emb  = (grad * item_emb).sum(dim=-1)        # [B, T]

    m_plus  = F.relu( grad_x_emb)
    m_minus = F.relu(-grad_x_emb)

    M_max = F.softmax(m_plus  / tau, dim=-1)
    M_min = F.softmax(m_minus / tau, dim=-1)

    A = 1

    dirichlet_model.zero_grad()
    return M_max, M_min, A


# ─────────────────────────────────────────────
# 6.  EVALUATION HELPERS
# ─────────────────────────────────────────────

def evaluate(model, data, config, dataset=None):
    """Run RecBole's evaluator, injecting dataset stats for diversity metrics."""
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
    priority = ['ndcg', 'hit', 'mrr', 'recall', 'precision']

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
    print(f"  {label:45s}  H@20={h}  NDCG@20={ndcg}")


# ─────────────────────────────────────────────
# 7.  DIRICHLET MODEL LOADING
# ─────────────────────────────────────────────

def load_dirichlet_model(checkpoint_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Loads a checkpoint and returns a ready-to-use DirichletSASRec.

    Handles TWO checkpoint formats automatically:

    1. Plain SASRec checkpoint (saved by RecBole / deep_ensembles / MCD scripts)
    Keys look like: "item_embedding.weight", "trm_encoder.*", ...
    Action: load into SASRec, then wrap with DirichletSASRec (random head).
    Use-case: post-hoc fine-tuning starting from a pre-trained backbone.

    2. DirichletSASRec checkpoint (saved by dirichlet_train_table1.py)
    Keys look like: "sasrec.item_embedding.weight", "evidence_head.weight", ...
    Action: reconstruct the full DirichletSASRec and load all weights directly.
    Use-case: post-hoc ExUA fine-tuning starting from a fully trained
                Dirichlet model (Table 1 -> Table 3 pipeline).

    Returns:
        dirichlet_model, config, dataset, train_data, valid_data, test_data
    """
    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    # ── Detect checkpoint type by inspecting state_dict keys ────────────────
    is_dirichlet_ckpt = any(k.startswith('sasrec.') or k.startswith('evidence_head.')
                            for k in state_dict.keys())

    if is_dirichlet_ckpt:
        # ── Format 2: DirichletSASRec checkpoint from dirichlet_train_table1.py
        print("  Detected DirichletSASRec checkpoint — loading backbone + head.")

        # We still need RecBole's dataset/dataloaders, so load via load_data_and_model
        # using a plain SASRec checkpoint stored alongside, OR reconstruct from config.
        # Since dirichlet_train_table1.py saves config in the checkpoint, we can
        # rebuild the dataset directly.
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import init_seed
        from recbole.model.sequential_recommender import SASRec as RecBoleSASRec

        config = ckpt['config']
        config['metrics'] = parameter_dict['metrics']
        init_seed(config['seed'], config['reproducibility'])

        dataset                           = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # Rebuild DirichletSASRec with fresh weights, then load saved state
        sasrec_backbone = RecBoleSASRec(config, dataset).to(device)
        dirichlet_model = DirichletSASRec(sasrec_backbone).to(device)
        dirichlet_model.load_state_dict(
            {k: v.to(device) for k, v in state_dict.items()}
        )

    else:
        # ── Format 1: Plain SASRec checkpoint (RecBole native format)
        print("  Detected plain SASRec checkpoint — wrapping with DirichletSASRec (random head).")
        config, sasrec, dataset, train_data, valid_data, test_data = \
            load_data_and_model(model_file=checkpoint_path)
        config['metrics'] = parameter_dict['metrics']
        dirichlet_model = DirichletSASRec(sasrec).to(device)

    dirichlet_model.eval()
    return dirichlet_model, config, dataset, train_data, valid_data, test_data


# ─────────────────────────────────────────────
# 8.  POST-HOC UNCERTAINTY-AWARE TRAINING
# ─────────────────────────────────────────────

def uncertainty_aware_training_dirichlet(
        dirichlet_model: DirichletSASRec,
        train_dataloader, valid_data, test_data,
        config, optimizer, dataset,
        method: str = 'bm',          # 'bm' or 'edl'
        alpha_scale: float = 0.2,    # controls ExUA attention intensity (paper's α)
        tau: float = 0.001,          # ExUA temperature
        kl_weight: float = KL_WEIGHT,
        warmup_epochs: int = 10,
        max_epochs: int = 30,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_path: str = None,
        ):
    """
    Post-hoc fine-tuning using ExUA attention and a Dirichlet loss.

    WARMUP PHASE
    ------------
    Trains ONLY the evidence head (backbone frozen) for warmup_epochs epochs
    before ExUA starts. This ensures α has learned a meaningful distribution
    so ∂U_T/∂e carries real signal rather than near-zero gradients from a
    randomly initialised head.

    Set warmup_epochs=0 if using a checkpoint from dirichlet_train_table1.py
    (head already trained end-to-end).

    Workflow per batch (main training):
    1. Compute ExUA attention map A (deterministic, S=1).
    2. Scale item embeddings: ê = (1 + α_scale * A) ⊙ e.
    3. Forward through Dirichlet head → α concentrations.
    4. Compute BM or EDL loss and backprop.

    Note: only the Dirichlet evidence_head (and optionally the SASRec backbone)
    are optimised.  By default we optimise ALL parameters (head + backbone) to
    allow the backbone to adapt, consistent with the paper.
    """
    assert method in ('bm', 'edl'), f"Unknown method '{method}'; choose 'bm' or 'edl'."

    best_valid_ndcg = -1.0
    best_state      = None

    # ── WARMUP: evidence head only, backbone frozen ──────────────────────────
    if warmup_epochs > 0:
        print(f"\n── Warmup ({warmup_epochs} epochs, head only, backbone frozen) ──")
        for param in dirichlet_model.sasrec.parameters():
            param.requires_grad_(False)

        warmup_optimizer = torch.optim.Adam(
            dirichlet_model.evidence_head.parameters(), lr=1e-3
        )

        for epoch in range(warmup_epochs):
            dirichlet_model.train()
            total_loss = 0.0
            for batch_idx, interaction in enumerate(train_dataloader):
                interaction  = interaction.to(device)
                item_seq     = interaction['item_id_list']
                item_seq_len = interaction['item_length']
                pos_items    = interaction['item_id']
                alpha_out = dirichlet_model.get_alpha(item_seq, item_seq_len)
                if method == 'bm':
                    loss = bm_loss(alpha_out, pos_items, kl_weight=kl_weight)
                else:
                    loss = edl_loss(alpha_out, pos_items,
                                    epoch=epoch, max_epochs=warmup_epochs,
                                    kl_weight=kl_weight)
                warmup_optimizer.zero_grad()
                loss.backward()
                warmup_optimizer.step()
                total_loss += loss.item()
            avg_loss     = total_loss / max(1, batch_idx + 1)
            valid_result = evaluate(dirichlet_model, valid_data, config, dataset)
            valid_ndcg   = valid_result.get('ndcg@20', valid_result.get('NDCG@20', 0.0))
            print(f"  Warmup [{epoch+1:02d}/{warmup_epochs}]  "
                f"loss={avg_loss:.4f}  val_NDCG@20={valid_ndcg:.4f}")

        for param in dirichlet_model.sasrec.parameters():
            param.requires_grad_(True)
        print("── Warmup done — backbone unfrozen, starting ExUA training ──\n")

    # ── MAIN TRAINING: ExUA fine-tuning ─────────────────────────────────────
    for epoch in range(max_epochs):
        dirichlet_model.train()
        total_loss = 0.0

        for batch_idx, interaction in enumerate(train_dataloader):
            interaction  = interaction.to(device)
            item_seq     = interaction['item_id_list']
            item_seq_len = interaction['item_length']
            pos_items    = interaction['item_id']

            # ── Step 1: ExUA attention (no gradient through this pass) ──────
            # with torch.no_grad():
            #     _, _, A = compute_exua_dirichlet(dirichlet_model, interaction, tau=tau)

            dirichlet_model.train()

            # ── Step 2: Scale embeddings by ExUA attention ──────────────────
            item_emb     = dirichlet_model.sasrec.item_embedding(item_seq)
            item_emb_hat = 1.0 * item_emb

            # ── Step 3: Get concentration parameters from modified embedding ─
            alpha_out = dirichlet_model.get_alpha_from_emb(item_emb_hat, item_seq, item_seq_len)

            # ── Step 4: Dirichlet loss ───────────────────────────────────────
            if method == 'bm':
                loss = bm_loss(alpha_out, pos_items, kl_weight=kl_weight)
            else:  # edl
                loss = edl_loss(alpha_out, pos_items,
                                epoch=epoch, max_epochs=max_epochs,
                                kl_weight=kl_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, batch_idx + 1)

        valid_result = evaluate(dirichlet_model, valid_data, config, dataset)
        valid_ndcg   = valid_result.get('ndcg@20', valid_result.get('NDCG@20', 0.0))

        is_best = valid_ndcg > best_valid_ndcg
        print(f"Epoch [{epoch+1:02d}/{max_epochs}]  "
            f"avg_loss={avg_loss:.4f}  "
            f"valid NDCG@20={valid_ndcg:.4f}"
            + (" ← best" if is_best else ""))

        if is_best:
            best_valid_ndcg = valid_ndcg
            best_state = {k: v.cpu().clone() for k, v in dirichlet_model.state_dict().items()}

    if best_state:
        print(f"\nBest valid NDCG@20: {best_valid_ndcg:.4f}  →  saving to {save_path}")
        dirichlet_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        torch.save({
            'config':          config,
            'state_dict':      dirichlet_model.state_dict(),
            'best_valid_ndcg': best_valid_ndcg,
            'method':          method,
            'warmup_epochs':   warmup_epochs,
        }, save_path)
    else:
        print("No improvement found during post-hoc training.")

    return dirichlet_model


# ─────────────────────────────────────────────
# 9.  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    from datetime import datetime


    parser = argparse.ArgumentParser(
        description='Dirichlet UQ (BM / EDL) for SASRec — SIGIR 2024 ExUA'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to a trained RecBole SASRec checkpoint (.pth). '
            'If not provided, trains SASRec from scratch first.'
    )
    parser.add_argument(
        '--method', type=str, default='bm', choices=['bm', 'edl'],
        help='"bm" for Bayesian Matching, "edl" for Evidential Deep Learning.'
    )
    parser.add_argument(
        '--mode', type=str, default='uncertainty',
        choices=['uncertainty', 'posthoc'],
        help='"uncertainty" for analysis only, "posthoc" for post-hoc fine-tuning.'
    )
    parser.add_argument('--dataset',       type=str,   default='ml-1m')
    parser.add_argument('--save_path',     type=str,   default=None)
    parser.add_argument('--kl_weight',     type=float, default=KL_WEIGHT)
    parser.add_argument('--warmup_epochs', type=int,   default=10,
                        help='Epochs to train head only before ExUA. '
                            'Set 0 if checkpoint is from dirichlet_train_table1.py.')
    parser.add_argument('--max_epochs',    type=int,   default=30)
    parser.add_argument('--seed',          type=int,   default=2020)
    parser.add_argument('--prep',          type=str,   default='default')
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")

    if args.save_path is None:
        args.save_path = f'saved/sasrec_dirichlet_no_dis_{args.method}_{timestamp}.pth'

    # ── Step 1: Obtain a base SASRec checkpoint ──────────────────────────────
    if args.checkpoint is None:
        print("No checkpoint supplied — training SASRec from scratch first.")
        run_recbole(model='SASRec', dataset=args.dataset, config_dict=parameter_dict)
        print("\nTraining done. Re-run with --checkpoint <path> to use Dirichlet UQ.")
        raise SystemExit(0)

    # ── Step 2: Load & wrap ──────────────────────────────────────────────────
    print(f"\nLoading checkpoint and wrapping with DirichletSASRec ({args.method.upper()})...")
    dirichlet_model, config, dataset, train_data, valid_data, test_data = \
        load_dirichlet_model(args.checkpoint, device='cpu')

    device = config['device']
    dirichlet_model = dirichlet_model.to(device)

    # ── Step 3: Uncertainty stats on first test batch ────────────────────────
    try:
        batch = next(iter(test_data))
        interaction = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)

        item_seq     = interaction['item_id_list']
        item_seq_len = interaction['item_length']

        with torch.no_grad():
            alpha_out = dirichlet_model.get_alpha(item_seq, item_seq_len)
            _, U_T, U_A, U_E = dirichlet_uncertainty(alpha_out)

        method_label = args.method.upper()
        print(f"\n── Dirichlet {method_label} Uncertainty (first test batch) ──")
        print(f"  Total      U_T : mean={U_T.mean():.4f}  std={U_T.std():.4f}")
        print(f"  Aleatoric  U_A : mean={U_A.mean():.4f}  std={U_A.std():.4f}")
        print(f"  Epistemic  U_E : mean={U_E.mean():.4f}  std={U_E.std():.4f}")
    except Exception as e:
        print(f"Could not run uncertainty stats on batch: {e}")

    # ── Step 4: Mode-specific logic ──────────────────────────────────────────
    if args.mode == 'posthoc':

        # Baseline scores BEFORE post-hoc training
        print(f"\n── Evaluating SASRec+Dirichlet{args.method.upper()} baseline (Table 1) ──")
        test_before = evaluate(dirichlet_model, test_data, config, dataset)
        print_scores(f"SASRec + Dirichlet{args.method.upper()} (before)", test_before)

        # Post-hoc fine-tuning
        print(f"\n── Starting ExUA-{args.method.upper()} post-hoc training "
            f"(warmup={args.warmup_epochs}) ──")
        optimizer = torch.optim.Adam([
            {'params': dirichlet_model.evidence_head.parameters(), 'lr': 1e-3},
            {'params': dirichlet_model.sasrec.parameters(),         'lr': 1e-4},
        ])

        dirichlet_model = uncertainty_aware_training_dirichlet(
            dirichlet_model,
            train_data, valid_data, test_data,
            config, optimizer, dataset,
            method        = args.method,
            alpha_scale   = 0.2,
            tau           = 0.001,
            kl_weight     = args.kl_weight,
            warmup_epochs = args.warmup_epochs,
            max_epochs    = args.max_epochs,
            device        = device,
            save_path     = args.save_path,
        )

        # Final evaluation AFTER post-hoc training
        print(f"\n── Final evaluation — ExUA-{args.method.upper()} (Table 3) ──")
        test_after = evaluate(dirichlet_model, test_data, config, dataset)
        print_scores(f"ExUA-{args.method.upper()} (after)", test_after)

        print("\n── Final Impact Analysis ────────────────────────────────────────")
        print_full_comparison(test_before, test_after)

    else:
        # Uncertainty-only mode: just evaluate and print scores
        print(f"\n── Evaluating SASRec + Dirichlet{args.method.upper()} (Table 1) ──")
        test_result = evaluate(dirichlet_model, test_data, config, dataset)
        print_scores(f"SASRec + Dirichlet {args.method.upper()}", test_result)