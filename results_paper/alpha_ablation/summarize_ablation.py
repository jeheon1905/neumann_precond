"""Damping-weight ablation on B12: measured iteration counts against the mode-factor prediction.

Reads the SLURM stdout the ab_B12_a*.sh jobs write and prints two tables: the
iteration counts as they go into the SI, and the single-mode prediction that
says which orders should be the bad ones for each weight.

The prediction comes from the damped mode factor.  Averaging orders N-1 and N
with weight alpha multiplies each mode of E = I - PM by

    q_N(lambda) = (1 - alpha + alpha*lambda) * lambda**N = c(alpha) * lambda**N ,

and the preconditioned operator has mu_N = 1 - q_N on that mode.  Convergence
needs mu_N in (0, 2).  c(alpha) vanishes at the annihilator 1/(1-lambda), and
its SIGN is what decides whether the even or the odd orders are the bad ones --
which is why alpha on either side of the annihilator gives opposite parity.

    python summarize_ablation.py [--lam -1.1365]

Default lambda is lambda_min(Pi E Pi) for the B12 lowest state (report.md 13.3).
"""

import argparse
import collections
import glob
import os
import re

DONE_RE = re.compile(r"#+ B12 alpha=(\S+) order=(\d+) DONE (\d+) s #+")
HEAD_RE = re.compile(r"#+ B12 alpha=(\S+) order=(\d+) #+$")
ITER_RE = re.compile(r"Diag\. Iter\.\s+\|\s+\S+\s+\|\s+(\d+)")


def read_logs(jobs_dir):
    """alpha -> {order: (iterations, seconds)}; only orders that ran to DONE."""
    out = collections.defaultdict(dict)
    running = {}
    for path in sorted(glob.glob(os.path.join(jobs_dir, "B12_a*_*.txt"))):
        cur, iters = None, None
        for line in open(path):
            head = HEAD_RE.match(line.strip())
            if head:
                cur, iters = int(head.group(2)), None
                running[head.group(1)] = cur
                continue
            hit = ITER_RE.match(line)
            if hit and cur is not None:
                iters = int(hit.group(1))
                continue
            done = DONE_RE.match(line.strip())
            if done:
                a, n, secs = done.group(1), int(done.group(2)), int(done.group(3))
                if iters is not None:
                    out[a][n] = (iters, secs)
                running.pop(a, None)
                cur, iters = None, None
    return out, running


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=-1.1365,
                    help="lambda_min(Pi E Pi), B12 lowest state")
    ap.add_argument("--jobs", default=os.path.join(os.path.dirname(__file__), "jobs"))
    args = ap.parse_args()

    tab, running = read_logs(args.jobs)
    if not tab:
        raise SystemExit(f"no ablation logs under {args.jobs}")
    alphas = sorted(tab, key=float)
    orders = sorted({n for v in tab.values() for n in v})

    print("Davidson iterations (B12, fixed H, tol 1e-5, seed 0, CPU / 8 threads)")
    print("Separate sweep from the archived GPU benchmark -- do not merge the two.\n")
    print("  N  |" + "".join(f"  a={a}" for a in alphas))
    print("-----+" + "-------" * len(alphas))
    for n in orders:
        row = f"  {n:<2} |"
        for a in alphas:
            row += f"{tab[a][n][0]:>6} " if n in tab[a] else "     ."
        print(row)

    if running:
        print("\nstill running: " + ", ".join(
            f"alpha={a} order={n}" for a, n in sorted(running.items())))

    ann = 1.0 / (1.0 - args.lam)
    print(f"\n\nSingle-mode prediction at lambda = {args.lam}  (annihilator alpha = {ann:.3f})\n")
    print("  a    c(a)    N   q_N      mu_N    predicted        measured")
    print("-" * 62)
    for a in alphas:
        c = 1.0 - float(a) * (1.0 - args.lam)
        for n in orders:
            if n not in tab[a]:
                continue
            q = c * args.lam ** n
            mu = 1.0 - q
            if mu < 0:
                verdict = "sign reversal"
            elif mu > 2:
                verdict = "over-correction"
            else:
                verdict = "in range"
            it = tab[a][n][0]
            mark = "   <-- blow-up" if it > 40 else ""
            print(f" {a}  {c:+7.4f}  {n:>2} {q:+7.3f} {mu:+8.3f}   {verdict:<16} {it:>4}{mark}")
        print()

    print("A blow-up always sits on a sign reversal, but the converse fails: a sign\n"
          "reversal with small |c(a)| is absorbed by the Rayleigh-Ritz step.  The\n"
          "single mode predicts WHICH orders are exposed, not how much they cost.")


if __name__ == "__main__":
    main()
