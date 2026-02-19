"""
Training script for the Hybrid Neural ODE model.

Usage:
  python experiments/train_hybrid.py [--epochs 100] [--lr 1e-3] [--augment_dim 4]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hybrid_ode import HybridNeuralODE
from data.tbi_dataset import TBITemporalDataset
from evaluation.trajectory_metrics import compute_trajectory_metrics
from evaluation.constraint_compliance import evaluate_constraint_compliance


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid Neural ODE for TBI")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--augment_dim", type=int, default=4)
    parser.add_argument("--constraint_weight", type=float, default=0.1)
    parser.add_argument("--data_path", type=str, default=None, help="Path to CSV data (uses synthetic if None)")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, dataset, optimizer, constraint_weight):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for sample in dataset:
        obs = sample["observations"]  # [T, D]
        times = sample["times"]  # [T]
        mask = sample["mask"]  # [T, D]

        if obs.shape[0] < 2:
            continue

        first_obs = obs[0:1]  # [1, D]

        result = model(first_obs, times)
        predictions = result["predictions"]  # [T, 1, D]
        raw_states = result["raw_states"]

        targets = obs.unsqueeze(1)  # [T, 1, D]
        mask_3d = mask.unsqueeze(1)  # [T, 1, D]

        losses = model.compute_loss(
            predictions, targets, mask_3d,
            constraint_weight=constraint_weight,
            raw_states=raw_states,
        )

        optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses["total_loss"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []
    all_states = []

    for sample in dataset:
        obs = sample["observations"]
        times = sample["times"]
        mask = sample["mask"]

        if obs.shape[0] < 2:
            continue

        first_obs = obs[0:1]
        result = model(first_obs, times)

        all_preds.append(result["predictions"].squeeze(1).numpy())
        all_targets.append(obs.numpy())
        all_masks.append(mask.numpy())
        all_states.append(result["states"].squeeze(1).numpy())

    if not all_preds:
        return {}

    # Pad to same length for metric computation
    max_t = max(p.shape[0] for p in all_preds)
    d = all_preds[0].shape[-1]

    def pad(arr, max_len, fill=0.0):
        padded = np.full((max_len, d), fill)
        padded[: arr.shape[0]] = arr
        return padded

    preds = np.stack([pad(p, max_t) for p in all_preds], axis=1)
    tgts = np.stack([pad(t, max_t) for t in all_targets], axis=1)
    masks = np.stack([pad(m, max_t, fill=0.0) for m in all_masks], axis=1)
    states = np.stack([pad(s, max_t) for s in all_states], axis=1)

    from ode_systems.multi_compartment import MultiCompartmentTBI

    traj_metrics = compute_trajectory_metrics(
        preds, tgts, masks,
        variable_names=MultiCompartmentTBI.STATE_NAMES[:d],
    )
    compliance = evaluate_constraint_compliance(
        states,
        variable_names=MultiCompartmentTBI.STATE_NAMES[:min(states.shape[-1], 7)],
    )

    return {
        "trajectory": traj_metrics,
        "constraint_compliance": compliance,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset (data_path={args.data_path})...")
    dataset = TBITemporalDataset(data_path=args.data_path)
    print(f"  {len(dataset)} patients loaded")

    model = HybridNeuralODE(
        obs_dim=7,
        augment_dim=args.augment_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, dataset, optimizer, args.constraint_weight)

        if epoch % 10 == 0 or epoch == 1:
            metrics = evaluate(model, dataset)
            compliance_rate = metrics.get("constraint_compliance", {}).get("overall_compliance_rate", 0)
            agg = metrics.get("trajectory", {}).get("aggregate", {})
            rmse = agg.get("rmse", float("nan"))
            print(
                f"  Epoch {epoch:4d} | loss={loss:.4f} | RMSE={rmse:.4f} | "
                f"constraint_compliance={compliance_rate:.4f}"
            )

    # Final evaluation
    final_metrics = evaluate(model, dataset)

    # Save results
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "final_metrics": final_metrics,
                "model_params": model.ode_func.get_param_summary(),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to {results_path}")

    # Save model checkpoint
    ckpt_path = output_dir / "hybrid_ode_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
        },
        ckpt_path,
    )
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
