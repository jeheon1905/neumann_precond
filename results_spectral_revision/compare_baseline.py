"""Gate 0: compare the CPU re-run against the archived GPU baseline (numerics only, no timing)."""
import glob, sys, re
import torch

NEW = "results_spectral_revision/baseline/B12_o4_seed2_history.pt"
REF_DIR = ("20260225_to_hyunbin/result.neumann.fixed/history/B12/phase=fixed/pp=TM/cuda=1/"
           "thr=1/prec=neumann/outerorder=4/ec=-0.4/avgsum=1/weight=0.5/scell=1x1x1/pbc=0x0x0/"
           "nbands=auto/spacing=0.2/vf=1.2/diag_iter=1000/diag_tol=1e-05/nblock=2/lock=0/fill=0")
REF = f"{REF_DIR}/median/history.pt"          # median run == seed 2
REF_LOG = f"{REF_DIR}/median/stdout.log"

eN, rN = torch.load(NEW, map_location="cpu")
eR, rR = torch.load(REF, map_location="cpu")

print(f"reference (GPU, seed 2): {len(rR)} history entries")
print(f"new       (CPU, seed 2): {len(rN)} history entries")

n_ref_iter = sum(1 for l in open(REF_LOG) if "Time: Diag. Iter." in l)
print(f"\nDavidson iteration count  reference = {n_ref_iter}")

m = min(len(rN), len(rR))
print(f"\n{'it':>3} | {'max|g| ref':>12} | {'max|g| new':>12} | {'rel.diff':>10} | "
      f"{'max|de| (Ha)':>12}")
print("-" * 62)
for i in range(m):
    a, b = rR[i].double(), rN[i].double()
    k = min(a.numel(), b.numel())
    rel = float(((a[:k] - b[:k]).abs() / a[:k].abs().clamp_min(1e-300)).max())
    de = float((eR[i].double()[:k] - eN[i].double()[:k]).abs().max())
    print(f"{i:3d} | {float(a.max()):12.5e} | {float(b.max()):12.5e} | {rel:10.3e} | {de:12.5e}")

print("\nFinal residual norms:")
print(f"  reference max = {float(rR[-1].max()):.6e}   min = {float(rR[-1].min()):.6e}")
print(f"  new       max = {float(rN[-1].max()):.6e}   min = {float(rN[-1].min()):.6e}")
print("\nLowest 8 Ritz values (Hartree):")
print("  ref:", [f"{v:.10f}" for v in eR[-1].double()[:8].tolist()])
print("  new:", [f"{v:.10f}" for v in eN[-1].double()[:8].tolist()])
d = (eR[-1].double() - eN[-1].double()).abs()
print(f"  max |Delta eigenvalue| over all {d.numel()} states = {float(d.max()):.3e} Ha")
