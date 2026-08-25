# Existing (manuscript/SI) order-dependent benchmark data — no new timing runs

Source (already on disk, GPU runs from the original environment):
- Undamped NP  : `20260211_to_hyunbin/result.neumann_normal.fixed_speed/` (config `config.neumann_normal.fixed_speed.yaml`, `averaged_sum: false`)
- Damped-NP    : `20260225_to_hyunbin/result.neumann.fixed/`               (config `config.neumann.fixed.yaml`,        `averaged_sum: true`, `weight: 0.5`)

Both: fixed-Hamiltonian phase, pp=TM, spacing 0.2, virtual_factor 1.2,
diag_tol 1e-5, diag_iter 1000, nblock 2, locking=false, fill_block=false,
seeds 0/1/2 (median run quoted).  Counts are the Davidson iteration count
(`Diag. Iter.` Timer count / `Time: Diag. Iter.` line count).

## Davidson iterations, undamped NP
| order | water_cluster_128 | C60_4 | B12 | MAPbI3 |
|---:|---:|---:|---:|---:|
| 0 | 127 | 123 | 158 | 80 |
| 1 | 322 | 223 | **1000 (no conv.)** | 37 |
| 2 | 45 | 42 | 47 | 27 |
| 3 | 174 | 112 | **1000 (no conv.)** | 21 |
| 4 | 32 | 29 | 34 | 18 |
| 5 | 173 | 80 | **1000 (no conv.)** | 16 |
| 6 | 27 | 24 | 29 | 15 |
| 7 | 174 | 68 | **1000 (no conv.)** | 14 |
| 8 | 23 | 20 | 28 | 13 |
| 9 | 230 | 68 | **1000 (no conv.)** | 12 |
| 10 | 21 | 18 | 25 | 12 |

## Davidson iterations, Damped-NP (weight = 0.5)
| order | water_cluster_128 | C60_4 | B12 | MAPbI3 |
|---:|---:|---:|---:|---:|
| 0 | 127 | 123 | 158 | 80 |
| 1 | 58 | 57 | 73 | 51 |
| 2 | 34 | 33 | 42 | 31 |
| 3 | 27 | 26 | 33 | 24 |
| 4 | 22 | 21 | 27 | 20 |
| 5 | 20 | 18 | 23 | 17 |
| 6 | 19 | 17 | 21 | 15 |
| 7 | 19 | 16 | 20 | 14 |
| 8 | 19 | 16 | 19 | 13 |
| 9 | 19 | 16 | 19 | 12 |
| 10 | 18 | 16 | 19 | 12 |

## Observed odd-even oscillation strength (to be explained spectrally)
1. **B12**     — odd orders do not converge within 1000 iterations. Strongest.
2. **water_cluster_128** — odd orders 5-10x worse than the neighbouring even orders.
3. **C60_4**   — same pattern, weaker (odd/even ratio shrinks with order: 5.3x at N=1, 3.8x at N=9).
4. **MAPbI3**  — **no oscillation at all**; undamped NP is monotone in N and is even
   slightly *better* than Damped-NP at low order. This is the reviewer's "MAPbI3 exception".

Damping (weight 0.5) removes the oscillation in all four systems; for MAPbI3 it
costs a few iterations at low order because there was nothing to suppress.

## Falsifiable prediction for the spectral analysis
If the odd-even oscillation is caused by eigenvalues of E near lambda = -1, then the
most-negative eigenvalue should order as

    B12  <=  water_cluster_128  <  C60_4  <<  MAPbI3

with B12 at or below -1 (divergent odd orders), and MAPbI3 far from -1.
The damping factor |(1+lambda)/2| should be near zero exactly for the oscillating systems.

## Reviewer question on practical order (answered from this table, no new timing)
There is no universal optimal order: for Damped-NP the iteration count saturates
around N = 6-8 for water_cluster_128 / C60_4 / B12 but keeps improving to N = 9-10
for MAPbI3, and each extra order costs one extra H application plus one GAPP
application per preconditioner call.  The wall-time optimum therefore depends on the
system and on the solver condition, and is already documented by the existing
manuscript/SI timing results.
