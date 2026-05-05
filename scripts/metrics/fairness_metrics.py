import torch
import numpy as np
from collections import Counter
from recbole.utils import get_trainer


# ─────────────────────────────────────────────
# 1.  BUILD ITEM POPULARITY FROM TRAINING DATA
# ─────────────────────────────────────────────

def build_item_popularity(dataset):
    """
    Returns a dict {item_id (int): interaction_count (int)}
    built from the full interaction feature table of the dataset.
    Item 0 is the padding token and is excluded.
    """
    if hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    
    iid_field = dataset.iid_field
    item_ids = dataset.inter_feat[iid_field].numpy()
    counter = Counter(item_ids.tolist())
    counter.pop(0, None)   # remove padding
    return counter          # {item_id: count}


def split_items_by_popularity(item_popularity, short_head_ratio=0.2):
    """
    Splits items into short-head (Φ) and long-tail (Γ) sets.

    Following Abdollahpouri et al. 2017 / 2019:
      short-head = top `short_head_ratio` fraction of items by rating count
      long-tail  = the remaining items

    Returns:
        short_head: set of item ids
        long_tail:  set of item ids
    """
    items_sorted = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
    n_short = max(1, int(len(items_sorted) * short_head_ratio))
    short_head = {iid for iid, _ in items_sorted[:n_short]}
    long_tail   = {iid for iid, _ in items_sorted[n_short:]}
    return short_head, long_tail


# ─────────────────────────────────────────────
# 2.  GENERATE TOP-K RECOMMENDATION LISTS
# ─────────────────────────────────────────────

@torch.no_grad()
def get_topk_lists(model, test_dataloader, topk=20, device='cpu'):
    """
    Runs model.full_sort_predict() for every test user and returns:
        user_recs  : dict {user_id (int): [item_id, ...]}  (length = topk)
        user_pos   : dict {user_id (int): set of positive item_ids in test set}
        user_history: dict {user_id (int): set of item_ids seen during training}

    Items the user interacted with during training are excluded from recs
    (RecBole's full_sort_predict already masks history, but we also record it
    for PopREO denominator calculation).

    `model` may also be a list of models (Deep Ensemble), in which case
    scores are averaged as probabilities across members.
    """
    # ── Ensemble support ──────────────────────────────────────────────────────
    is_ensemble = isinstance(model, (list, tuple))
    if is_ensemble:
        models = model
        for m in models:
            m.eval()
        ref_device = next(models[0].parameters()).device
    else:
        models = None
        model.eval()
        model.to(device)
        ref_device = device

    def predict(interaction):
        """Returns averaged-probability scores for single model or ensemble."""
        if is_ensemble:
            prob_sum = None
            for m in models:
                m_device = next(m.parameters()).device
                logits = m.full_sort_predict(interaction.to(m_device))
                probs  = torch.softmax(logits, dim=-1).to(ref_device)
                prob_sum = probs if prob_sum is None else prob_sum + probs
            return prob_sum / len(models)
        else:
            return model.full_sort_predict(interaction)

    user_recs    = {}
    user_pos     = {}
    user_history = {}

    for batch in test_dataloader:
        # RecBole test batches can be (interaction, history, ...) tuples
        if isinstance(batch, (list, tuple)):
            interaction = batch[0]
        else:
            interaction = batch

        interaction = interaction.to(ref_device)

        uid_field = test_dataloader.dataset.uid_field
        iid_field = test_dataloader.dataset.iid_field

        user_ids = interaction[uid_field].cpu().numpy()

        # Full-catalog scores — shape [B, n_items]
        scores = predict(interaction)   # [B, n_items]

        # Mask items seen during training (set score to -inf)
        # RecBole stores history in 'item_id_list' for sequential models;
        # for general models it may differ — adapt if needed.
        if 'item_id_list' in interaction:
            history = interaction['item_id_list'].cpu().numpy()   # [B, hist_len]
        else:
            history = None

        scores_np = scores.cpu().numpy()    # [B, n_items]

        for b_idx, uid in enumerate(user_ids):
            uid = int(uid)
            row_scores = scores_np[b_idx].copy()

            # Mask padding item 0
            row_scores[0] = -np.inf

            # Mask history
            if history is not None:
                for hitem in history[b_idx]:
                    if hitem > 0:
                        row_scores[int(hitem)] = -np.inf

            # Record history set for this user
            hist_set = set()
            if history is not None:
                hist_set = {int(x) for x in history[b_idx] if x > 0}
            user_history[uid] = hist_set

            # Top-k items
            topk_items = np.argpartition(row_scores, -topk)[-topk:]
            topk_items = topk_items[np.argsort(row_scores[topk_items])[::-1]]
            user_recs[uid] = topk_items.tolist()

        # Collect positive items from test set
        pos_items = interaction[iid_field].cpu().numpy()
        for b_idx, uid in enumerate(user_ids):
            uid = int(uid)
            pos = int(pos_items[b_idx])
            user_pos.setdefault(uid, set()).add(pos)

    return user_recs, user_pos, user_history


# ─────────────────────────────────────────────
# 3.  METRIC IMPLEMENTATIONS
# ─────────────────────────────────────────────

def compute_ARP(user_recs, item_popularity):
    """
    Average Recommendation Popularity (Yin et al., 2012).

    ARP = (1/|Ut|) * Σ_u [ (Σ_{i ∈ L_u} φ(i)) / |L_u| ]

    Lower ARP → recommender favours less-popular (long-tail) items.
    """
    total = 0.0
    for uid, rec_list in user_recs.items():
        if len(rec_list) == 0:
            continue
        avg_pop = np.mean([item_popularity.get(i, 0) for i in rec_list])
        total += avg_pop
    return total / max(1, len(user_recs))


def compute_APLT(user_recs, long_tail):
    """
    Average Percentage of Long-Tail items (Abdollahpouri et al., 2017).

    APLT = (1/|Ut|) * Σ_u [ |{i : i ∈ L_u ∩ Γ}| / |L_u| ]

    Higher → more long-tail items per recommendation list.
    """
    total = 0.0
    for uid, rec_list in user_recs.items():
        if len(rec_list) == 0:
            continue
        lt_count = sum(1 for i in rec_list if i in long_tail)
        total += lt_count / len(rec_list)
    return total / max(1, len(user_recs))


def compute_ACLT(user_recs, long_tail):
    """
    Average Coverage of Long-Tail items (Abdollahpouri et al., 2019).

    ACLT = (1/|Ut|) * Σ_u Σ_{i ∈ L_u} 1(i ∈ Γ)

    Counts total long-tail item appearances across all lists (not unique).
    Higher → greater long-tail exposure across the catalogue.
    """
    total = 0.0
    for uid, rec_list in user_recs.items():
        lt_count = sum(1 for i in rec_list if i in long_tail)
        total += lt_count
    return total / max(1, len(user_recs))


def compute_PopREO(user_recs, user_pos, user_history, short_head, long_tail,
                   dataset_items=None, topk=20):
    """
    Popularity-based Ranking Equal Opportunity (adapted from Zhu et al., SIGIR 2020).

    Groups items by popularity: short-head (Φ) vs long-tail (Γ).
    For each group g:
        P(R@k | g, y=1) = (Σ_u Σ_{i ∈ top-k} 1(i ∈ g) * 1((u,i) ∈ test))
                          / (Σ_u Σ_{i ∉ history_u} 1(i ∈ g) * 1((u,i) ∈ test))

    REO@k = std([P_short, P_long]) / mean([P_short, P_long])

    Lower → more balanced true-positive rates across popularity groups.
    """
    # Numerator: items in top-k that are in test set, per group
    tp_short = 0
    tp_long  = 0

    # Denominator: test-set positives per group (excluding training history)
    denom_short = 0
    denom_long  = 0

    for uid, rec_list in user_recs.items():
        pos = user_pos.get(uid, set())
        hist = user_history.get(uid, set())
        rec_set = set(rec_list[:topk])

        # Numerator
        for i in rec_set:
            if i in pos:
                if i in short_head:
                    tp_short += 1
                elif i in long_tail:
                    tp_long += 1

        # Denominator: test positives NOT in training history
        for i in pos:
            if i not in hist:
                if i in short_head:
                    denom_short += 1
                elif i in long_tail:
                    denom_long  += 1

    p_short = tp_short / max(1, denom_short)
    p_long  = tp_long  / max(1, denom_long)

    probs = np.array([p_short, p_long])
    mean_p = probs.mean()
    if mean_p < 1e-10:
        return 0.0, p_short, p_long

    reo = probs.std() / mean_p
    return reo, p_short, p_long

def compute_PopRSP(user_recs, user_history, short_head, long_tail,
                   all_items, topk=20):
    """
    Popularity-based Ranking-based Statistical Parity
    (adapted from Zhu et al., SIGIR 2020).

    Groups items by popularity: short-head (Φ) vs long-tail (Γ).

    For each group g:
        P(R@k | g) = (Σ_u Σ_{i ∈ top-k_u} 1(i ∈ g))
                     / (Σ_u |{i ∉ history_u : i ∈ g}|)

    RSP@k = std([P_short, P_long]) / mean([P_short, P_long])

    Lower → more balanced recommendation probability across popularity groups,
    regardless of whether the user actually likes the item.

    Unlike PopREO (which conditions on y=1 / test positives), PopRSP
    uses ALL un-interacted items in the denominator — it measures whether
    the *chance of being recommended at all* is equal across groups.

    Args:
        user_recs    : dict {uid: [item_id, ...]}  (ranked, length = topk)
        user_history : dict {uid: set of item_ids seen in training}
        short_head   : set of short-head item ids
        long_tail    : set of long-tail item ids
        all_items    : set / collection of ALL valid item ids (excl. padding 0)
        topk         : recommendation list length

    Returns:
        rsp          : scalar PopRSP@k  (lower is better)
        p_short      : P(R@k | short-head)
        p_long       : P(R@k | long-tail)
    """
    # ── Numerator: recommended items per group ───────────────────────────────
    num_short = 0
    num_long  = 0

    # ── Denominator: un-interacted items per group, summed over users ────────
    denom_short = 0
    denom_long  = 0

    for uid, rec_list in user_recs.items():
        hist = user_history.get(uid, set())
        rec_set = set(rec_list[:topk])

        # Numerator: how many top-k items fall in each group
        for i in rec_set:
            if i in short_head:
                num_short += 1
            elif i in long_tail:
                num_long += 1

        # Denominator: all items the user has NOT interacted with, per group
        # = |group ∩ (all_items \ history_u)|
        # Computed efficiently without iterating all items every time:
        #   |group \ history_u| = |group| - |group ∩ history_u|
        hist_short = sum(1 for i in hist if i in short_head)
        hist_long  = sum(1 for i in hist if i in long_tail)

        denom_short += len(short_head) - hist_short
        denom_long  += len(long_tail)  - hist_long

    p_short = num_short / max(1, denom_short)
    p_long  = num_long  / max(1, denom_long)

    probs  = np.array([p_short, p_long])
    mean_p = probs.mean()
    if mean_p < 1e-10:
        return 0.0, p_short, p_long

    rsp = probs.std() / mean_p
    return rsp, p_short, p_long

# ─────────────────────────────────────────────
# 4.  UNIFIED EVALUATION ENTRY POINT
# ─────────────────────────────────────────────
def evaluate_popularity_metrics(model, test_dataloader, dataset,
                                  topk=20,
                                  short_head_ratio=0.2,
                                  device='cpu'):
    """
    Convenience wrapper: runs all five popularity metrics in one call.

    Args:
        model           : trained RecBole / DirichletSASRec model
        test_dataloader : RecBole test dataloader
        dataset         : RecBole dataset (for item popularity counts)
        topk            : recommendation list length
        short_head_ratio: fraction of items considered short-head (default 0.2)
        device          : torch device string

    Returns:
        dict with keys: ARP, APLT, ACLT, PopRSP, PopREO and their sub-probabilities
    """
    print("  [PopMetrics] Building item popularity from training interactions...")
    item_pop = build_item_popularity(dataset)

    print(f"  [PopMetrics] Splitting items (short_head_ratio={short_head_ratio})...")
    short_head, long_tail = split_items_by_popularity(item_pop, short_head_ratio)
    print(f"               short-head: {len(short_head)} items | "
          f"long-tail: {len(long_tail)} items")

    # All valid items (exclude padding token 0)
    all_items = set(item_pop.keys())

    print(f"  [PopMetrics] Generating top-{topk} recommendation lists...")
    user_recs, user_pos, user_history = get_topk_lists(
        model, test_dataloader, topk=topk, device=device
    )
    print(f"               {len(user_recs)} users evaluated.")

    arp  = compute_ARP(user_recs, item_pop)
    aplt = compute_APLT(user_recs, long_tail)
    aclt = compute_ACLT(user_recs, long_tail)

    rsp, p_sh_rsp, p_lt_rsp = compute_PopRSP(
        user_recs, user_history, short_head, long_tail,
        all_items=all_items, topk=topk
    )
    reo, p_sh_reo, p_lt_reo = compute_PopREO(
        user_recs, user_pos, user_history, short_head, long_tail, topk=topk
    )

    results = {
        'ARP':                arp,
        'APLT':               aplt,
        'ACLT':               aclt,
        'PopRSP':             rsp,
        'PopRSP_p_shorthead': p_sh_rsp,
        'PopRSP_p_longtail':  p_lt_rsp,
        'PopREO':             reo,
        'PopREO_p_shorthead': p_sh_reo,
        'PopREO_p_longtail':  p_lt_reo,
    }

    print(f"\n── Popularity Bias Metrics (@{topk}) ─────────────────────────────────")
    print(f"  ARP    (↓ better, lower = less popular recs)  : {arp:.4f}")
    print(f"  APLT   (↑ better, more long-tail per list)    : {aplt:.4f}")
    print(f"  ACLT   (↑ better, more long-tail coverage)    : {aclt:.4f}")
    print(f"  PopRSP (↓ better, equal exposure probability) : {rsp:.4f}")
    print(f"    P(R@{topk}|short-head)       = {p_sh_rsp:.6f}")
    print(f"    P(R@{topk}|long-tail)        = {p_lt_rsp:.6f}")
    print(f"  PopREO (↓ better, equal TPR across groups)    : {reo:.4f}")
    print(f"    P(R@{topk}|short-head, y=1) = {p_sh_reo:.4f}")
    print(f"    P(R@{topk}|long-tail,  y=1) = {p_lt_reo:.4f}")
    print("──────────────────────────────────────────────────────────────────\n")

    return results