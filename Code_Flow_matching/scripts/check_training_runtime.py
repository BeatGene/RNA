"""Bounded Lightning/DDP smoke test or real-PT throughput measurement.

Without --data-dir, generate disposable synthetic schema-v2 data; their
timings are NOT estimates of real RNA throughput. With --data-dir, read only
its train/val PT files and time a short run using the actual YAML model.
No checkpoint, WandB run, or persistent training data is written.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/RNA_test.yaml")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, help="Existing real PT root with train/val directories")
    parser.add_argument("--train-batches", type=int, default=12, help="Per-rank microbatches, not optimizer steps")
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, help="Override per-GPU graph batch size")
    parser.add_argument("--num-workers", type=int, help="Override workers per DDP process")
    parser.add_argument(
        "--largest-files",
        action="store_true",
        help="Probe the largest PT files (a conservative proxy for large RNA graphs)",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        help="Override gradient accumulation for the bounded run",
    )
    args = parser.parse_args()
    positive_overrides = (
        args.batch_size,
        args.num_workers,
        args.accumulate_grad_batches,
    )
    if (
        args.devices < 1
        or args.val_batches < 1
        or not 0 <= args.warmup_batches < args.train_batches
        or any(value is not None and value < 1 for value in positive_overrides)
    ):
        parser.error("Require devices/val-batches > 0 and 0 <= warmup-batches < train-batches")

    import torch
    import yaml
    from lightning.pytorch import Trainer, Callback, seed_everything
    from torch_geometric.loader import DataLoader
    from etflow.data.dataset import EuclideanDataset
    from etflow.models.model import BaseFlow
    from smoke_test_synthetic import make_synthetic_sample

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.devices:
        raise SystemExit(f"Need {args.devices} visible CUDA devices")
    seed_everything(config.get("seed", 42), workers=True)
    torch.set_float32_matmul_precision("high")
    synthetic = args.data_dir is None

    class RuntimeChecks(Callback):
        def __init__(self):
            self.state = {}

        def begin(self, name):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.state[name] = dict(
                start=time.perf_counter(),
                end=None,
                batches=0,
                graphs=0,
                max_allocated=0,
                max_reserved=0,
            )

        def finish_batch(self, name, batch):
            torch.cuda.synchronize()
            state = self.state[name]
            state["end"] = time.perf_counter()
            state["batches"] += 1
            state["graphs"] += batch.num_graphs
            state["max_allocated"] = max(
                state["max_allocated"], torch.cuda.max_memory_allocated()
            )
            state["max_reserved"] = max(
                state["max_reserved"], torch.cuda.max_memory_reserved()
            )

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if batch_idx == args.warmup_batches:
                self.begin("train")

        def on_after_backward(self, trainer, pl_module):
            if synthetic:
                gradients = [p.grad for p in pl_module.parameters() if p.grad is not None]
                assert gradients, "No parameter received gradients"
                assert all(bool(g.isfinite().all()) for g in gradients), "Nonfinite gradients"

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if outputs is not None:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                assert bool(torch.isfinite(loss).all()), "Nonfinite train loss"
            if batch_idx >= args.warmup_batches:
                self.finish_batch("train", batch)

        def on_validation_epoch_start(self, trainer, pl_module):
            self.begin("val")

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            assert bool(torch.isfinite(outputs).all()), "Nonfinite validation loss"
            self.finish_batch("val", batch)

        def on_fit_end(self, trainer, pl_module):
            result = {"synthetic": synthetic, "world_size": trainer.world_size,
                      "precision": str(trainer.precision),
                      "training_objective": pl_module.training_objective,
                      "batch_size_per_gpu": train_loader.batch_size,
                      "accumulate_grad_batches": trainer.accumulate_grad_batches,
                      "num_workers_per_rank": train_loader.num_workers,
                      "largest_files": args.largest_files,
                      "device_total_memory_gib": (
                          torch.cuda.get_device_properties(pl_module.device).total_memory / 2**30
                      ),
                      "warning": "Synthetic timings are not representative of real RNAs." if synthetic
                                 else "Extrapolate only if measured samples represent real RNA lengths and storage."}
            for name, state in self.state.items():
                if not state["batches"]:
                    continue
                elapsed = torch.tensor(state["end"] - state["start"], device=pl_module.device)
                graphs = torch.tensor(state["graphs"], device=pl_module.device)
                max_allocated = torch.tensor(state["max_allocated"], device=pl_module.device)
                max_reserved = torch.tensor(state["max_reserved"], device=pl_module.device)
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
                    torch.distributed.all_reduce(graphs, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(max_allocated, op=torch.distributed.ReduceOp.MAX)
                    torch.distributed.all_reduce(max_reserved, op=torch.distributed.ReduceOp.MAX)
                result[name] = dict(seconds=float(elapsed), microbatches=state["batches"],
                                    seconds_per_microbatch=float(elapsed) / state["batches"],
                                    global_samples_per_second=float(graphs / elapsed),
                                    max_allocated_gib=float(max_allocated) / 2**30,
                                    max_reserved_gib=float(max_reserved) / 2**30,
                                    max_reserved_fraction=(
                                        float(max_reserved)
                                        / torch.cuda.get_device_properties(pl_module.device).total_memory
                                    ))
            if trainer.is_global_zero:
                print("RUNTIME_RESULT " + json.dumps(result), flush=True)
                print("BOUNDED TRAINING RUNTIME CHECK PASSED", flush=True)

    # Each DDP subprocess creates the same deterministic fixtures in its own
    # temp directory; no shared-path races and no interaction with real data.
    with tempfile.TemporaryDirectory(prefix="etflow_runtime_") as temp:
        data_root = args.data_dir or Path(temp)
        loader_args = dict(config["datamodule_args"]["dataloader_args"])
        if args.batch_size is not None:
            loader_args["batch_size"] = args.batch_size
        if args.num_workers is not None:
            loader_args["num_workers"] = args.num_workers
        batch_size = loader_args.get("batch_size", 1)
        if synthetic:
            for split, batches in (("train", args.train_batches), ("val", args.val_batches)):
                split_dir = data_root / split
                split_dir.mkdir()
                for index in range(args.devices * batch_size * batches):
                    torch.save(make_synthetic_sample(index % 2), split_dir / f"{index:06d}.pt")
        train_data = EuclideanDataset(data_dir=data_root, split="train")
        val_data = EuclideanDataset(data_dir=data_root, split="val")
        if args.largest_files and not synthetic:
            requested_train = args.devices * batch_size * args.train_batches
            requested_val = args.devices * batch_size * args.val_batches
            train_data.data_files = sorted(
                train_data.data_files,
                key=lambda path: path.stat().st_size,
                reverse=True,
            )[:requested_train]
            val_data.data_files = sorted(
                val_data.data_files,
                key=lambda path: path.stat().st_size,
                reverse=True,
            )[:requested_val]
        if len(train_data) < args.devices * batch_size * args.train_batches:
            raise SystemExit("Not enough train PT files for requested batches; lower --train-batches/--warmup-batches")
        if len(val_data) < args.devices * batch_size * args.val_batches:
            raise SystemExit("Not enough val PT files for requested batches; lower --val-batches")
        train_loader = DataLoader(train_data, shuffle=True, **loader_args)
        val_loader = DataLoader(val_data, shuffle=False, **loader_args)
        trainer_args = dict(config.get("trainer_args", {}))
        if args.accumulate_grad_batches is not None:
            trainer_args["accumulate_grad_batches"] = args.accumulate_grad_batches
        trainer_args.update(devices=args.devices, accelerator="gpu", max_epochs=1,
                            strategy="ddp_find_unused_parameters_true" if args.devices > 1 else "auto",
                            limit_train_batches=args.train_batches, limit_val_batches=args.val_batches,
                            num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
                            enable_progress_bar=False, callbacks=[RuntimeChecks()])
        trainer = Trainer(**trainer_args)
        model = BaseFlow(**config["model_args"])
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
