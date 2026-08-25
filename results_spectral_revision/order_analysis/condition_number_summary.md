# Condition-number results

All values are for the DEFLATED operator (Pi = I - X X^H, X = Davidson subspace),
snapshot at Davidson iteration 15, Krylov dimension m = 80.

`mu = 1 - theta` are the eigenvalues of `Pi P M Pi`;  `kappa = mu_max / mu_min`.

## A. kappa of the bare GAPP preconditioner (N = 0)

| System | State | band | eps_t | mu_min | mu_max | **kappa** | res(mu_min) | PCG check |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B12 | lowest | 0 | -1.1601 | 0.07205 | 2.1365 | **29.7** | 1.1e-03 | 28.0 (49 its) |
| B12 | middle | 125 | -0.5798 | 0.06982 | 1.8971 | **27.2** | 6.7e-04 | 25.6 (46 its) |
| B12 | homo | 249 | -0.3394 | 0.06890 | 1.8225 | **26.5** | 6.5e-04 | 26.7 (47 its) |
| water_cluster_128 | lowest | 0 | -1.1590 | 0.07812 | 2.1010 | **26.9** | 2.5e-03 | 26.3 (47 its) |
| water_cluster_128 | middle | 256 | -0.5403 | 0.07577 | 1.8713 | **24.7** | 2.2e-03 | 23.0 (44 its) |
| water_cluster_128 | homo | 511 | -0.3723 | 0.07512 | 1.8133 | **24.1** | 2.2e-03 | 22.8 (44 its) |
| C60_4 | lowest | 0 | -1.0821 | 0.07794 | 2.0733 | **26.6** | 3.5e-03 | 24.3 (45 its) |
| C60_4 | middle | 240 | -0.6145 | 0.07616 | 1.8998 | **24.9** | 3.2e-03 | 22.7 (44 its) |
| C60_4 | homo | 479 | -0.3592 | 0.07476 | 1.8113 | **24.2** | 3.2e-03 | 23.8 (45 its) |
| MAPbI3 | lowest | 0 | -0.8233 | 0.19093 | 1.4995 | **7.9** | 1.3e-03 | 7.0 (24 its) |
| MAPbI3 | middle | 320 | -0.2867 | 0.15294 | 1.3690 | **9.0** | 7.0e-04 | 8.6 (26 its) |
| MAPbI3 | homo | 639 | -0.0127 | 0.07267 | 1.3042 | **17.9** | 1.4e-03 | 17.6 (38 its) |

## B. kappa(P_N M) versus Neumann order   **[RETRACTED]**

> **WITHDRAWN.**  The table below inserts the DEFLATED Ritz values into the closed form
> `P_N M = I - 0.5 E^N (I+E)`, which is an identity only in the FULL space.  Direct
> measurement (job 329741) shows the deflated version fails: at N = 8 the closed form
> requires mu_max = 1.00000 while the measured value is 2.05236.  Section A (N = 0) is
> unaffected, since it involves no closed form.  See report.md section 20.2.



### homo state

| N | B12 NP / DNP | water_cluster_128 NP / DNP | C60_4 NP / DNP | MAPbI3 NP / DNP |
|---:|---:|---:|---:|---:|
| 0 | 26.5 / 26.5 | 24.1 / 24.1 | 24.2 / 24.2 | 17.9 / 17.9 |
| 1 | 7.5 / 11.1 | 6.9 / 10.2 | 6.9 / 10.3 | 7.1 / 10.4 |
| 2 | 8.1 / 6.1 | 7.4 / 5.7 | 7.4 / 5.7 | 5.1 / 5.8 |
| 3 | 4.0 / 4.8 | 3.7 / 4.4 | 3.7 / 4.4 | 3.8 / 4.4 |
| 4 | 4.6 / 3.6 | 4.2 / 3.4 | 4.2 / 3.4 | 3.2 / 3.5 |
| 5 | 2.9 / 3.2 | 2.7 / 3.0 | 2.7 / 3.0 | 2.7 / 3.0 |
| 6 | 3.2 / 2.7 | 2.9 / 2.5 | 2.9 / 2.5 | 2.4 / 2.6 |
| 7 | 2.3 / 2.5 | 2.2 / 2.3 | 2.2 / 2.3 | 2.2 / 2.3 |
| 8 | 2.5 / 2.2 | 2.3 / 2.1 | 2.3 / 2.1 | 2.0 / 2.1 |
| 9 | 2.0 / 2.1 | 1.8 / 1.9 | 1.9 / 1.9 | 1.9 / 2.0 |
| 10 | 2.1 / 1.9 | 1.9 / 1.8 | 1.9 / 1.8 | 1.8 / 1.8 |
| 11 | 1.7 / 1.8 | 1.6 / 1.7 | 1.6 / 1.7 | 1.7 / 1.7 |

### lowest state

| N | B12 NP / DNP | water_cluster_128 NP / DNP | C60_4 NP / DNP | MAPbI3 NP / DNP |
|---:|---:|---:|---:|---:|
| 0 | 29.7 / 29.7 | 26.9 / 26.9 | 26.6 / 26.6 | 7.9 / 7.9 |
| 1 | **IND** / 10.7 | **IND** / 9.9 | **IND** / 9.9 | 2.9 / 4.2 |
| 2 | 12.3 / 6.4 | 10.8 / 5.8 | 10.4 / 5.7 | 2.4 / 2.5 |
| 3 | **IND** / 4.6 | **IND** / 4.3 | **IND** / 4.3 | 1.7 / 2.0 |
| 4 | 9.3 / 3.9 | 7.8 / 3.5 | 7.3 / 3.4 | 1.6 / 1.6 |
| 5 | **IND** / 3.1 | **IND** / 2.9 | **IND** / 2.9 | 1.4 / 1.5 |
| 6 | 8.5 / 3.0 | 6.8 / 2.7 | 6.1 / 2.6 | 1.3 / 1.3 |
| 7 | **IND** / 2.4 | **IND** / 2.2 | **IND** / 2.2 | 1.2 / 1.3 |
| 8 | 8.5 / 2.5 | 6.5 / 2.2 | 5.6 / 2.1 | 1.2 / 1.2 |
| 9 | **IND** / 2.0 | **IND** / 1.9 | **IND** / 1.9 | 1.1 / 1.2 |
| 10 | 9.1 / 2.3 | 6.6 / 2.0 | 5.4 / 1.9 | 1.1 / 1.1 |
| 11 | **IND** / 1.8 | **IND** / 1.7 | **IND** / 1.7 | 1.1 / 1.1 |

## C. What these kappa values imply

| kappa | fixed-point / Neumann rate (k-1)/(k+1) | CG rate (sqrt(k)-1)/(sqrt(k)+1) | CG its to 1e-8 |
|---:|---:|---:|---:|
| 26 | 0.926 | 0.672 | 46 |
| 24 | 0.920 | 0.661 | 44 |
| 18 | 0.895 | 0.619 | 38 |
| 8 | 0.778 | 0.478 | 25 |
| 5 | 0.667 | 0.382 | 19 |
| 3 | 0.500 | 0.268 | 14 |
| 2 | 0.333 | 0.172 | 10 |
| 1.5 | 0.200 | 0.101 | 8 |
