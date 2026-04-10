def compute_metrics(model, v_dir, a_dir, device, k_values=[1, 5, 10], batch_size=64):
    model.eval()
    v_ds = ValVisionDataset(v_dir, transform=vision_transform)
    a_ds = ValAudioDataset(a_dir)
    
    # --- DEBUG PRINTS ---
    print(f"\n[DEBUG] Testing Vision Path: {v_dir}")
    print(f"[DEBUG] Found {len(v_ds)} images for evaluation.")
    print(f"[DEBUG] Testing Audio Path: {a_dir}")
    print(f"[DEBUG] Found {len(a_ds)} audio files for evaluation.")
    
    if len(v_ds) == 0 or len(a_ds) == 0:
        print("!!! ERROR: Evaluation sets are empty. Check your folder structure or file extensions !!!")
        return {k: 0 for k in k_values}, {k: 0 for k in k_values}
    # --------------------

    v_loader = DataLoader(v_ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    a_loader = DataLoader(a_ds, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    v_base_feats, v_labels = [], []
    a_base_feats, a_labels = [], []
    
    with torch.no_grad():
        for imgs, lbls in tqdm(v_loader, leave=False, desc="Extracting Vision"):
            v_base_feats.append(model.vision_model(imgs.to(device), reconstruct=False).cpu())
            v_labels.extend(lbls.numpy())
            
        for auds, lbls in tqdm(a_loader, leave=False, desc="Extracting Audio"):
            a_base_feats.append(model.audio_model(auds.to(device)).cpu())
            a_labels.extend(lbls.numpy())
            
    v_base_feats = torch.cat(v_base_feats, dim=0)
    a_base_feats = torch.cat(a_base_feats, dim=0)
    v_labels = np.array(v_labels)
    a_labels = np.array(a_labels)
    
    N, M = v_base_feats.size(0), a_base_feats.size(0)
    sim_matrix = np.zeros((N, M))
    
    with torch.no_grad():
        for i in tqdm(range(N), leave=False, desc="Cross-Interacting"):
            v_row = v_base_feats[i].unsqueeze(0).to(device)
            for j in range(0, M, batch_size):
                a_chunk = a_base_feats[j:j+batch_size].to(device)
                v_chunk = v_row.expand(a_chunk.size(0), -1)
                
                v_final, a_final = model.iclm(v_chunk, a_chunk)
                sims = F.cosine_similarity(v_final, a_final, dim=1).cpu().numpy()
                sim_matrix[i, j:j+batch_size] = sims
            
    def calc_r(matrix, q_lbl, g_lbl):
        res = {k: 0 for k in k_values}
        for idx in range(len(q_lbl)):
            indices = np.argsort(matrix[idx])[::-1]
            top = g_lbl[indices[:10]]
            for k in k_values:
                if q_lbl[idx] in top[:k]: res[k] += 1
        return {k: (v/len(q_lbl))*100 for k,v in res.items()}

    i2a = calc_r(sim_matrix, v_labels, a_labels)
    a2i = calc_r(sim_matrix.T, a_labels, v_labels)
    return i2a, a2i