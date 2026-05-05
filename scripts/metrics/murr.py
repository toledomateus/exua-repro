import torch

def compute_universal_mURR(test_dataloader, exua_fn, uncertainty_fn, device, K=5):
    """
    Computes mURR for any model architecture by taking custom functions for ExUA and Uncertainty.
    
    Args:
        test_dataloader: RecBole test dataloader
        exua_fn: A function that takes (interaction) and returns (M_max, M_min, A)
        uncertainty_fn: A function that takes (interaction) and returns a scalar tensor U_T
        device: 'cuda' or 'cpu'
        K: Maximum number of items to mask
    """
    total_mURR = 0.0
    valid_samples = 0
    
    # print(f"\n── Computing mURR (max K={K}) ──")
    
    for batch_idx, batch in enumerate(test_dataloader):
        interaction = batch[0] if isinstance(batch, (tuple, list)) else batch
        interaction = interaction.to(device)
        
        # ── DEBUG BLOCK ──
        with torch.no_grad():
            u_baseline = uncertainty_fn(interaction)
        
        # print(f"U_T mean: {u_baseline.mean():.6f}, std: {u_baseline.std():.6f}, "
        #     f"min: {u_baseline.min():.6f}, max: {u_baseline.max():.6f}")
        # print(f"u_baseline shape: {u_baseline.shape}")
        # print(f"interaction keys: {list(interaction.interaction.keys())}")
        
        # print(f"item_id shape: {interaction['item_id'].shape}")
        # print(f"item_id sample: {interaction['item_id'][:5]}")
        # print(f"item_id_list shape: {interaction['item_id_list'].shape}")


        item_seq = interaction['item_id_list']
        batch_size = item_seq.size(0)
        
        # 1. Get ExUA attribution maps (M_max and M_min)
        M_max, M_min, _ = exua_fn(interaction)
        # print(f"M_max: min={M_max.min():.4f} max={M_max.max():.4f} std={M_max.std():.4f}")
        
        
        batch_mURR = torch.zeros(batch_size, device=device) - 999.0 
        
        # Keep a backup of the original sequence to restore later
        original_seq = item_seq.clone()
        
        for i in range(1, K + 1):
            # Get the i-th most attributed item (rank i, 0-indexed = i-1)
            # topk returns sorted descending, so index i-1 is the i-th largest
            top_k_max = torch.topk(M_max, i, dim=1).indices  # shape: (B, i)
            top_k_min = torch.topk(M_min, i, dim=1).indices  # shape: (B, i)

            # The i-th item is the LAST column (index i-1)
            idx_max = top_k_max[:, i-1].unsqueeze(1)  # shape: (B, 1)
            idx_min = top_k_min[:, i-1].unsqueeze(1)  # shape: (B, 1)

            # Mask ONLY the i-th ranked item
            seq_max_masked = original_seq.clone()
            seq_min_masked = original_seq.clone()

            seq_max_masked.scatter_(1, idx_max, 0)
            seq_min_masked.scatter_(1, idx_min, 0)
            # 4. Compute new uncertainties 
            with torch.no_grad():
                # Evaluate Max Masked
                interaction.interaction['item_id_list'] = seq_max_masked
                U_T_max = uncertainty_fn(interaction)
                
                # Evaluate Min Masked
                interaction.interaction['item_id_list'] = seq_min_masked
                U_T_min = uncertainty_fn(interaction)
            
            # 5. Compute URR(i) = 1 - (U_max / U_min)
            URR_i = 1.0 - (U_T_max / (U_T_min + 1e-8))
            
            # print(f"U_T_max sample: {U_T_max[:3]}")
            # print(f"U_T_min sample: {U_T_min[:3]}")
            # print(f"URR_i sample:   {URR_i[:3]}")
    
            # Keep max URR across all K iterations
            batch_mURR = torch.max(batch_mURR, URR_i)
            
        # Restore original sequence for the next iteration step (safety precaution)
        interaction.interaction['item_id_list'] = original_seq
            
        total_mURR += batch_mURR.sum().item()
        valid_samples += batch_size

    final_mURR = total_mURR / valid_samples
    print(f"mURR@{K}: {final_mURR:.4f}")
    return final_mURR