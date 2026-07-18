"""
Evaluate Dense Seismic-Well CLIP for geological feature detection.

Measures:
  1. Dense retrieval: seismic↔well at each depth point
  2. Task prediction quality (fault, horizon, facies, fracture)
  3. Improvement from well-log fusion (seismic-only vs fused)
  4. Feature map visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os, sys, argparse

sys.path.insert(0, "/data/yxjiang/datatest")
from dense_clip_seismic.config import DenseCLIPConfig
from dense_clip_seismic.models.dense_clip import DenseSeismicWellCLIP
from dense_clip_seismic.data.dataset import DenseSeismicWellDataset, collate_dense


@torch.no_grad()
def evaluate_and_visualize(model, dataloader, device, output_dir: str):
    """Full evaluation with visualizations."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Collect one batch
    batch = next(iter(dataloader))
    seismic = batch["seismic"].to(device)
    well_logs = batch["well_logs"].to(device)
    well_x = batch["well_x"].to(device)
    labels = batch.get("labels", None)

    # Forward
    outputs = model(seismic, well_logs, well_x,
                    task_labels={k: v.to(device) for k, v in labels.items()}
                    if labels else None)

    B = seismic.shape[0]
    n_display = min(B, 4)

    # ═══════════════════════════════════════════════════════════
    # Figure 1: Task Predictions — Seismic-only vs Fused
    # ═══════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 5 * n_display))
    gs = GridSpec(n_display, 6, figure=fig, hspace=0.3, wspace=0.3)

    for b in range(n_display):
        # Seismic
        ax = fig.add_subplot(gs[b, 0])
        ax.imshow(seismic[b, 0].cpu().numpy(), cmap='seismic', aspect='auto')
        ax.axvline(x=well_x[b].item(), color='lime', linewidth=2, linestyle='--')
        ax.set_title(f'Seismic #{batch["index"][b].item()}')
        ax.axis('off')

        # Fault prediction (fused)
        ax = fig.add_subplot(gs[b, 1])
        fault_pred = torch.sigmoid(outputs["preds_fused"]["fault"][b, 0]).cpu()
        ax.imshow(fault_pred, cmap='hot', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Fault (fused)')
        ax.axis('off')

        # Fault — seismic only
        ax = fig.add_subplot(gs[b, 2])
        fault_s = torch.sigmoid(outputs["preds_seismic"]["fault"][b, 0]).cpu()
        ax.imshow(fault_s, cmap='hot', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Fault (seismic only)')
        ax.axis('off')

        # Facies prediction (fused)
        ax = fig.add_subplot(gs[b, 3])
        facies = outputs["preds_fused"]["facies"][b].argmax(0).cpu()
        ax.imshow(facies, cmap='tab10', aspect='auto', vmin=0, vmax=4)
        ax.set_title('Facies (fused)')
        ax.axis('off')

        # Horizon prediction (fused)
        ax = fig.add_subplot(gs[b, 4])
        hor = outputs["preds_fused"]["horizon"][b].argmax(0).cpu()
        ax.imshow(hor, cmap='tab20', aspect='auto')
        ax.set_title('Horizons (fused)')
        ax.axis('off')

        # Fracture prediction (fused)
        ax = fig.add_subplot(gs[b, 5])
        frac = torch.sigmoid(outputs["preds_fused"]["fracture"][b, 0]).cpu()
        ax.imshow(frac, cmap='hot', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Fracture (fused)')
        ax.axis('off')

    fig.suptitle('Dense Seismic-Well CLIP: Geological Feature Predictions\n'
                 'Well position: green dashed line | '
                 '"fused" = seismic + well features',
                 fontweight='bold')
    plt.savefig(os.path.join(output_dir, "01_predictions.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 01_predictions.png")

    # ═══════════════════════════════════════════════════════════
    # Figure 2: Cross-modal feature alignment at well column
    # ═══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, n_display, figsize=(5 * n_display, 8))

    for b in range(n_display):
        # Seismic features along well column
        s_feats = outputs["s_feats"][b]  # (C, H, W)
        wx = well_x[b].item()
        s_col = s_feats[:, :, wx].cpu().numpy()  # (C, H)

        # Well features
        w_feats = outputs["w_feats"][b].cpu().numpy()  # (C, L)

        # Similarity matrix
        sim = F.normalize(torch.tensor(s_col).T, dim=-1) @ \
              F.normalize(torch.tensor(w_feats).T, dim=-1).T

        ax = axes[0, b] if n_display > 1 else axes[0]
        im = ax.imshow(sim.numpy(), cmap='RdYlBu_r', aspect='auto',
                       vmin=-1, vmax=1)
        ax.set_title(f'Section #{batch["index"][b].item()}\n'
                     f'Seismic↔Well similarity')
        ax.set_xlabel('Well depth idx')
        ax.set_ylabel('Seismic depth idx')
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Diagonal profile
        ax = axes[1, b] if n_display > 1 else axes[1]
        diag = np.diag(sim.numpy())
        ax.plot(diag, 'b-', linewidth=1, label='Diagonal similarity')
        off_diag = sim.numpy().sum(0) - diag
        off_diag /= (sim.shape[0] - 1)
        ax.plot(off_diag, 'r--', linewidth=1, alpha=0.5,
                label='Mean off-diagonal')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.set_xlabel('Depth index')
        ax.set_ylabel('Cosine similarity')
        ax.legend(fontsize=7)
        ax.set_title('Diagonal vs Off-diagonal')

    fig.suptitle('Dense Cross-Modal Feature Alignment\n'
                 'Strong diagonal → depth-level alignment works',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_alignment.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_alignment.png")

    # ═══════════════════════════════════════════════════════════
    # Figure 3: Feature maps — before/after fusion
    # ═══════════════════════════════════════════════════════════
    fig, axes = plt.subplots(n_display, 3, figsize=(15, 4 * n_display))

    for b in range(n_display):
        s_f = outputs["s_feats"][b].mean(0).cpu().numpy()
        f_f = outputs["fused_feats"][b].mean(0).cpu().numpy()
        diff = f_f - s_f

        ax0 = axes[b, 0] if n_display > 1 else axes[0]
        ax1 = axes[b, 1] if n_display > 1 else axes[1]
        ax2 = axes[b, 2] if n_display > 1 else axes[2]

        ax0.imshow(s_f, cmap='viridis', aspect='auto')
        ax0.axvline(x=well_x[b].item(), color='red', linewidth=1)
        ax0.set_title(f'Seismic features #{b}')
        ax0.axis('off')

        ax1.imshow(f_f, cmap='viridis', aspect='auto')
        ax1.axvline(x=well_x[b].item(), color='lime', linewidth=1)
        ax1.set_title(f'Fused features #{b}')
        ax1.axis('off')

        ax2.imshow(diff, cmap='RdBu_r', aspect='auto',
                   vmin=-abs(diff).max(), vmax=abs(diff).max())
        ax2.set_title(f'Difference (well info injected)')
        ax2.axis('off')

    fig.suptitle('Feature Map Comparison: Before vs After Well Fusion\n'
                 'Red/green line = well position',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_feature_fusion.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 03_feature_fusion.png")

    # ═══════════════════════════════════════════════════════════
    # Quantitative metrics
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Quantitative Results")
    print(f"{'='*60}")

    # Dense retrieval accuracy
    s_feats_all = []
    w_feats_all = []
    for batch in dataloader:
        s = batch["seismic"].to(device)
        w = batch["well_logs"].to(device)
        wx = batch["well_x"].to(device)
        _, s_proj = model.encode_seismic(s)
        _, w_proj = model.encode_well(w)

        for b_idx in range(s.shape[0]):
            s_col = s_proj[b_idx, :, :, wx[b_idx]]
            w_col = w_proj[b_idx, :, :]
            s_feats_all.append(F.normalize(s_col, dim=0))
            w_feats_all.append(F.normalize(w_col, dim=0))

    # Aggregate metrics
    all_acc = []
    for i in range(min(len(s_feats_all), 20)):
        sim = s_feats_all[i].T @ w_feats_all[i]
        acc = (sim.argmax(dim=0) == torch.arange(sim.shape[1],
                device=sim.device)).float().mean()
        all_acc.append(acc.item())

    print(f"  Depth-level retrieval accuracy: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")

    # Compare seismic-only vs fused fault prediction difference
    fault_s = torch.sigmoid(outputs["preds_seismic"]["fault"])
    fault_f = torch.sigmoid(outputs["preds_fused"]["fault"])
    diff_norm = (fault_f - fault_s).abs().mean(dim=(1, 2, 3))
    print(f"  Mean |fault_fused - fault_seismic|: {diff_norm.mean().item():.6f}")
    print(f"  → Well log info changes {diff_norm.mean().item() / (fault_s.abs().mean().item() + 1e-8) * 100:.1f}% "
          f"of fault prediction values")

    print(f"\nAll figures saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="./dense_clip_checkpoints/best_s2_joint.pt")
    parser.add_argument("--dataset", type=str,
                        default="./openseisml_dataset_labeled.h5")
    parser.add_argument("--output-dir", type=str,
                        default="./dense_clip_evaluation")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    cfg = DenseCLIPConfig()
    model = DenseSeismicWellCLIP(cfg).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint found — using random weights")
        # Fallback: try stage 1 checkpoint
        s1_ckpt = args.checkpoint.replace("best_s2_joint", "best_s1_contrastive")
        if os.path.exists(s1_ckpt):
            ckpt = torch.load(s1_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded: {s1_ckpt}")
        else:
            # Try small dataset
            args.dataset = "/data/yxjiang/datatest/openseisml_dataset.h5"
            print(f"Using: {args.dataset}")

    # Data
    dataset = DenseSeismicWellDataset(
        hdf5_path=args.dataset if os.path.exists(args.dataset)
        else "/data/yxjiang/datatest/openseisml_dataset.h5",
        augment=False, normalize=True,
        has_labels=os.path.exists(args.dataset) and "labeled" in args.dataset,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False,
        num_workers=2, pin_memory=True,
        collate_fn=collate_dense,
    )

    evaluate_and_visualize(model, loader, device, args.output_dir)


if __name__ == "__main__":
    main()
