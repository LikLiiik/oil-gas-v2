#!/usr/bin/env python3
"""
Evaluate Seismic-Well CLIP model.

Metrics:
  1. Zero-shot retrieval: seismic ↔ well (top-1, top-5 accuracy)
  2. Embedding space visualization (t-SNE / UMAP)
  3. Cross-modal similarity heatmap
  4. Linear probe for well log property prediction from seismic
  5. Ablation: seismic-only vs seismic+well embedding
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import argparse
from sklearn.manifold import TSNE
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/data/yxjiang/datatest")
from clip_seismic_well.config import CLIPConfig
from clip_seismic_well.models import SeismicWellCLIP
from clip_seismic_well.data.dataset import SeismicWellDataset, collate_pairs


@torch.no_grad()
def compute_embeddings(model, dataloader, device):
    """Extract all seismic and well embeddings."""
    model.eval()
    s_embs, w_embs = [], []
    for batch in dataloader:
        seismic = batch["seismic"].to(device)
        well_logs = batch["well_logs"].to(device)
        s_embs.append(model.encode_seismic(seismic).cpu().numpy())
        w_embs.append(model.encode_well(well_logs).cpu().numpy())
    return np.concatenate(s_embs, axis=0), np.concatenate(w_embs, axis=0)


def evaluate_retrieval(s_embs, w_embs, top_k=(1, 5)):
    """
    Cross-modal retrieval evaluation.
    - s→w: given seismic, find correct well
    - w→s: given well, find correct seismic
    """
    n = s_embs.shape[0]
    sim = s_embs @ w_embs.T  # (n, n) cosine similarity
    labels = np.arange(n)

    results = {}
    for k in top_k:
        # Seismic → Well
        pred_s2w = np.argsort(-sim, axis=1)[:, :k]
        acc_s2w = np.any(pred_s2w == labels[:, None], axis=1).mean()

        # Well → Seismic
        pred_w2s = np.argsort(-sim.T, axis=1)[:, :k]
        acc_w2s = np.any(pred_w2s == labels[:, None], axis=1).mean()

        results[f"top{k}_s2w"] = acc_s2w
        results[f"top{k}_w2s"] = acc_w2s

    return results, sim


def evaluate_linear_probe(s_embs, well_data, well_names):
    """
    Linear probe: predict individual well log properties from seismic embeddings.
    This quantifies how well the seismic embedding captures petrophysical info.
    """
    scaler = StandardScaler()
    s_scaled = scaler.fit_transform(s_embs)

    # Train on 70%, test on 30%
    n = s_scaled.shape[0]
    n_train = int(0.7 * n)
    idx = np.random.RandomState(42).permutation(n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    r2_scores = {}
    for i, name in enumerate(well_names):
        y = well_data[:, i]
        model = Ridge(alpha=1.0)
        model.fit(s_scaled[train_idx], y[train_idx])
        y_pred = model.predict(s_scaled[test_idx])
        r2_scores[name] = r2_score(y[test_idx], y_pred)

    return r2_scores


def plot_similarity_matrix(sim, save_path, n_display=40):
    """Plot cross-modal similarity heatmap."""
    n = min(sim.shape[0], n_display)
    sim_sub = sim[:n, :n]

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(sim_sub, cmap='RdYlBu_r', aspect='auto',
                   vmin=-1, vmax=1)
    ax.set_xlabel('Well index')
    ax.set_ylabel('Seismic index')
    ax.set_title(f'Cross-Modal Similarity Matrix ({n}×{n})\n'
                 f'Diagonal → correct pairs')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Cosine Similarity')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_embedding_tsne(s_embs, w_embs, save_path):
    """t-SNE visualization of joint embedding space."""
    n = min(s_embs.shape[0], 100)
    combined = np.concatenate([s_embs[:n], w_embs[:n]], axis=0)
    labels = np.array([0] * n + [1] * n)  # 0=seismic, 1=well

    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n - 1))
    embedded = tsne.fit_transform(combined)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Color by modality
    colors = ['#2196F3', '#FF5722']
    labels_str = ['Seismic', 'Well Log']
    for i in range(2):
        mask = labels == i
        axes[0].scatter(embedded[mask, 0], embedded[mask, 1],
                        c=colors[i], label=labels_str[i], alpha=0.7, s=30)
    axes[0].set_title(f't-SNE: Seismic vs Well Embeddings (n={n})')
    axes[0].legend()
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')

    # Color by pair ID (show alignment)
    for i in range(n):
        axes[1].plot([embedded[i, 0], embedded[n + i, 0]],
                     [embedded[i, 1], embedded[n + i, 1]],
                     'gray', alpha=0.2, linewidth=0.5)
    axes[1].scatter(embedded[:n, 0], embedded[:n, 1], c='#2196F3',
                    label='Seismic', alpha=0.7, s=30)
    axes[1].scatter(embedded[n:, 0], embedded[n:, 1], c='#FF5722',
                    label='Well Log', alpha=0.7, s=30)
    axes[1].set_title(f't-SNE with Pair Connections')
    axes[1].legend()
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    plt.suptitle('Seismic-Well CLIP: Joint Embedding Space',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_retrieval_examples(model, dataloader, device, save_path, n_examples=4):
    """Show retrieval examples: query seismic + top-3 matching wells."""
    model.eval()
    batches = list(dataloader)
    batch = batches[0]
    seismic = batch["seismic"][:20].to(device)
    well_logs = batch["well_logs"][:20].to(device)

    s_emb = model.encode_seismic(seismic)
    w_emb = model.encode_well(well_logs)
    sim = (s_emb @ w_emb.T).cpu().numpy()

    indices = np.random.choice(len(seismic), n_examples, replace=False)

    fig, axes = plt.subplots(n_examples, 4, figsize=(18, 3 * n_examples))
    if n_examples == 1:
        axes = axes[None, :]

    LOG_NAMES = ["GR", "NPHI", "RHOB", "DT", "RT", "VEL"]
    LOG_COLORS = ['green', 'purple', 'red', 'blue', 'brown', 'orange']

    for row, idx in enumerate(indices):
        # Query seismic
        ax_s = axes[row, 0]
        ax_s.imshow(seismic[idx, 0].cpu().numpy(), cmap='seismic', aspect='auto')
        ax_s.set_title(f'Query Seismic #{idx}', fontsize=10)
        ax_s.axis('off')

        # Correct well
        ax_w = axes[row, 1]
        w_correct = well_logs[idx].cpu().numpy()
        for c in range(min(3, w_correct.shape[0])):
            ax_w.plot(w_correct[c], np.arange(256), color=LOG_COLORS[c],
                      linewidth=0.8)
        ax_w.set_title(f'Ground Truth Well #{idx}', fontsize=10)
        ax_w.invert_yaxis()

        # Top-1 retrieved well
        top1 = np.argsort(-sim[idx])[0]
        ax_t1 = axes[row, 2]
        w_t1 = well_logs[top1].cpu().numpy()
        for c in range(min(3, w_t1.shape[0])):
            ax_t1.plot(w_t1[c], np.arange(256), color=LOG_COLORS[c],
                       linewidth=0.8)
        ax_t1.set_title(f'Retrieved #{top1} (sim={sim[idx, top1]:.3f})',
                        fontsize=10, color='green' if top1 == idx else 'red')
        ax_t1.invert_yaxis()

        # Top-2 retrieved well
        top2 = np.argsort(-sim[idx])[1]
        ax_t2 = axes[row, 3]
        w_t2 = well_logs[top2].cpu().numpy()
        for c in range(min(3, w_t2.shape[0])):
            ax_t2.plot(w_t2[c], np.arange(256), color=LOG_COLORS[c],
                       linewidth=0.8)
        ax_t2.set_title(f'Retrieved #{top2} (sim={sim[idx, top2]:.3f})',
                        fontsize=10, color='green' if top2 == idx else 'red')
        ax_t2.invert_yaxis()

    fig.suptitle('Seismic-Well CLIP: Retrieval Examples\n'
                 'Green title = correct retrieval',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def plot_linear_probe_results(r2_dict, save_path):
    """Bar chart of linear probe R² scores."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = list(r2_dict.keys())
    scores = list(r2_dict.values())
    colors = plt.cm.RdYlGn(np.clip(np.array(scores), 0, 1))

    bars = ax.bar(names, scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('R² Score')
    ax.set_title('Linear Probe: Predicting Well Log Properties\n'
                 'from Seismic CLIP Embeddings')
    ax.set_ylim(max(-0.3, min(min(scores) - 0.1, -0.05)), min(1.05, max(max(scores) + 0.1, 0.3)))

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="./clip_checkpoints/best_model.pt")
    parser.add_argument("--dataset", type=str,
                        default="./openseisml_dataset_large.h5")
    parser.add_argument("--output-dir", type=str,
                        default="./clip_evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        cfg = ckpt.get("config", CLIPConfig())
        model = SeismicWellCLIP(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Epoch: {ckpt['epoch']} | Val Loss: {ckpt.get('val_metrics', {}).get('loss', 'N/A')}")
    else:
        # Fallback: untrained model
        print("No checkpoint found — using untrained model for baseline")
        cfg = CLIPConfig()
        cfg.dataset_path = args.dataset
        model = SeismicWellCLIP(cfg).to(device)

    # Data
    dataset = SeismicWellDataset(
        hdf5_path=cfg.dataset_path if os.path.exists(cfg.dataset_path) else args.dataset,
        augment=False, normalize=True,
    )

    # Try to use available dataset
    dataset.hdf5_path = args.dataset
    if not os.path.exists(dataset.hdf5_path):
        # Fallback to small dataset
        dataset.hdf5_path = "/data/yxjiang/datatest/openseisml_dataset.h5"
        print(f"Using fallback dataset: {dataset.hdf5_path}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=collate_pairs,
    )
    print(f"Dataset: {len(dataset)} pairs")

    # ── 1. Compute embeddings ──────────────────────────────
    print("\nComputing embeddings...")
    s_embs, w_embs = compute_embeddings(model, loader, device)
    print(f"  Seismic: {s_embs.shape}  |  Well: {w_embs.shape}")

    # ── 2. Retrieval evaluation ────────────────────────────
    print("\n─── Retrieval Metrics ───")
    retrieval_results, sim_matrix = evaluate_retrieval(s_embs, w_embs)
    for k, v in retrieval_results.items():
        print(f"  {k}: {v:.4f}")

    # ── 3. Linear probe ────────────────────────────────────
    print("\n─── Linear Probe (Seismic → Well Properties) ───")
    # Collect raw well data
    well_data_list = []
    for batch in loader:
        well_data_list.append(batch["well_logs"].numpy())
    all_well_data = np.concatenate(well_data_list, axis=0)  # (N, 6, 256)
    # Use mean per channel as target (simplified)
    well_channel_means = all_well_data.mean(axis=2)  # (N, 6)

    LOG_NAMES = ["GR", "NPHI", "RHOB", "DT", "RT", "VEL"]
    r2_results = evaluate_linear_probe(s_embs, well_channel_means, LOG_NAMES)
    for k, v in r2_results.items():
        print(f"  {k}: R² = {v:.4f}")

    # ── 4. Plots ───────────────────────────────────────────
    print("\nGenerating plots...")
    plot_similarity_matrix(
        sim_matrix,
        os.path.join(args.output_dir, "similarity_matrix.png"),
    )
    plot_embedding_tsne(
        s_embs, w_embs,
        os.path.join(args.output_dir, "embedding_tsne.png"),
    )
    plot_linear_probe_results(
        r2_results,
        os.path.join(args.output_dir, "linear_probe.png"),
    )
    plot_retrieval_examples(
        model, loader, device,
        os.path.join(args.output_dir, "retrieval_examples.png"),
    )

    # ── 5. Summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print("Cross-Modal Retrieval:")
    for k, v in retrieval_results.items():
        print(f"  {k}: {v:.4f}")
    print("Linear Probe (Seismic → Well Properties):")
    for k, v in r2_results.items():
        print(f"  {k}: R² = {v:.4f}")
    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
