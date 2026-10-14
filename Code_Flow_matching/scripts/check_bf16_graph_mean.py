"""Read-only graph-mean precision check; run with the server's actual PyG.

The local fallback tests Torch scatter_add mean arithmetic only. It does not
substitute for the installed PyG/GPU check on the laboratory server.
"""

import argparse
import ast
import importlib.util
from pathlib import Path

import torch

try:
    from torch_geometric.utils import scatter
    BACKEND = "installed torch_geometric.utils.scatter"
except ImportError:
    BACKEND = "local reference: Torch scatter_add mean (PyG unavailable)"

    def scatter(src, index, dim=0, reduce="mean"):
        assert dim == 0 and reduce == "mean"
        size = int(index.max()) + 1
        count = src.new_zeros(size)
        count.scatter_add_(0, index, src.new_ones(len(src)))
        out = src.new_zeros(size, src.shape[1])
        out.scatter_add_(0, index[:, None].expand_as(src), src)
        return out / count.clamp(min=1)[:, None]


def actual_center_function():
    source = Path(__file__).resolve().parents[1] / "etflow/models/utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "center_of_mass")
    namespace = {"torch": torch, "scatter": scatter}
    helper_path = source.parents[1] / "commons/reductions.py"
    if helper_path.exists():
        spec = importlib.util.spec_from_file_location("graph_reductions", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        namespace["center_by_graph"] = helper.center_by_graph
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["center_of_mass"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--assert-fixed", action="store_true", help="require correct production centering and gradient bounds")
    args = parser.parse_args()
    center = actual_center_function()
    print("TORCH", torch.__version__, "BACKEND", BACKEND, "DEVICE", args.device)
    print("backend_count checks the unmodified backend; production_* checks project centering.", flush=True)
    for n in (256, 576, 2048, 4096, 7984, 10629, 20000):
        batch = torch.zeros(n, dtype=torch.long, device=args.device)
        for dtype in (torch.float32, torch.bfloat16):
            ones = torch.ones(n, 1, dtype=dtype, device=args.device)
            mean_of_ones = float(scatter(ones, batch, dim=0, reduce="mean")[0, 0])
            count = torch.zeros(1, dtype=dtype, device=args.device)
            count.scatter_add_(0, batch, torch.ones(n, dtype=dtype, device=args.device))
            x = torch.full((n, 3), 0.001, dtype=dtype, device=args.device)
            centered = center(x, batch=batch)
            error = float(centered.float().abs().max())
            legacy_error = float((x - scatter(x, batch, dim=0, reduce="mean")[batch]).float().abs().max())
            torch.manual_seed(9)
            raw = torch.randn(n, 3, device=args.device).to(dtype) * 0.001
            raw = center(raw, batch=batch)
            gates = torch.linspace(0.1, 0.99, n, dtype=dtype, device=args.device)[:, None]
            gated = center(raw * gates, batch=batch)
            raw_rms = float(raw.float().square().sum(-1).mean().sqrt())
            gated_rms = float(gated.float().square().sum(-1).mean().sqrt())
            reference = raw.float() * gates.float()
            reference = reference - reference.mean(0, keepdim=True)
            deviation = float((gated.float() - reference).square().sum(-1).mean().sqrt())
            raw_leaf = raw.detach().requires_grad_(True)
            live_gated = center(raw_leaf * gates, batch=batch)
            target = torch.randn(n, 3, device=args.device) + torch.tensor([2.0, -1.0, 0.5], device=args.device)
            coordinate_loss = (live_gated.float() - target).norm(dim=-1).mean()
            gradient = torch.autograd.grad(coordinate_loss, raw_leaf)[0]
            # Correct scalar gates and orthogonal centering cannot increase
            # the 1/sqrt(N) upper bound of the mean Euclidean-loss gradient.
            gradient_bound_ratio = float(gradient.float().norm()) * n ** 0.5
            print(f"n={n} dtype={dtype} backend_count={float(count[0]):.1f} mean_ones={mean_of_ones:.6f} "
                  f"backend_constant_center_error={legacy_error:.8f} production_dtype={centered.dtype} "
                  f"production_constant_center_error={error:.8f} gate_RMS_ratio={gated_rms/raw_rms:.6f} "
                  f"vs_FP32_gate_RMS_difference={deviation:.8f} "
                  f"coordinate_gradient_bound_ratio={gradient_bound_ratio:.6f}", flush=True)
            if args.assert_fixed:
                assert centered.dtype == torch.float32, "low-precision graph projection must remain FP32"
                assert error < 5e-7, f"constant projection error: {error}"
                assert gated_rms / raw_rms <= 1.0001, "scalar gate plus correct centering must contract RMS"
                assert deviation < 5e-7, f"FP32 reference deviation: {deviation}"
                assert gradient_bound_ratio <= 1.001, "coordinate gradient exceeds projection bound"
    if args.assert_fixed:
        print("FP32_GRAPH_MEAN_CHECK_PASSED", flush=True)


if __name__ == "__main__":
    main()
