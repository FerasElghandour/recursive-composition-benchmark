# Seed-Based Recursive Intelligence: completed small experiment

Run date: 15 October 2026. Five seeds (0–4), CPU, NumPy float64. Values are mean ± sample standard deviation across seeds. All final checkpoints are reported; no seed was excluded.

## Finding

The balanced addition-chain rule did **not** improve this implementation. Left-to-right recursive composition performed better on held-out combinations in every seed. A standard modular model supplied with the correct composition algebra solved every evaluated program. These results do not establish emergence, novelty, or a new intelligence architecture.

## Accuracy

Exact accuracy requires the entire four-state map to be correct. The primary OOD split contains previously forbidden adjacent command pairs. IID means previously unseen programs from the permitted grammar. Long tests use permitted pairs at lengths 8 and 12.

| Model | Training % | IID % | Held-out pairs % | Long % |
|---|---:|---:|---:|---:|
| Seed + balanced stages | 40.6 ± 1.1 | 15.9 ± 2.0 | 4.4 ± 0.6 | 1.2 ± 0.4 |
| Seed + left stages | 98.8 ± 1.2 | 89.0 ± 5.9 | 42.1 ± 12.4 | 53.3 ± 12.9 |
| Seed + random stages | 42.0 ± 13.0 | 16.0 ± 9.1 | 6.6 ± 1.8 | 1.5 ± 0.4 |
| Shared balanced, joint | 65.2 ± 3.5 | 29.8 ± 6.7 | 9.6 ± 1.3 | 1.5 ± 0.2 |
| Shared left, joint | 94.3 ± 2.3 | 80.2 ± 4.5 | 34.9 ± 9.6 | 36.0 ± 6.3 |
| Independent, total params matched | 10.2 ± 1.3 | 2.1 ± 1.3 | 1.5 ± 0.4 | N/A |
| Independent, compute matched | 28.1 ± 3.3 | 6.0 ± 2.6 | 3.4 ± 1.4 | N/A |
| Independent, per-task width matched* | 47.0 ± 3.4 | 11.7 ± 3.0 | 4.4 ± 1.0 | N/A |
| Modular matrix (known algebra) | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 |

Independent models have no trained network for lengths 8/12; they are not assigned a zero score. *The per-task-width comparison is an explicitly post hoc sensitivity check; see PROTOCOL_AMENDMENT.md.

A predictor that always emits the most frequent training map scores 4.1 ± 0.3% on held-out pairs. This is an empirical reference, not a universal chance rate.

## Compute, time and memory

The neural topology comparison has identical width, training minibatches and dense MAC counts. Joint and staged curricula have the same number of examples, but different length distributions and therefore different computation. MACs count dense forward/backward matrix multiplications only; approximate FLOPs = 2 × MACs. Elementwise operations, Adam, embedding scatter and progress evaluation are excluded from MACs. Times include six progress evaluations (more for the extended independent run).

| Model | Parameters | Updates | Examples | Train G-MAC | Wall s | CPU s | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seed + balanced stages | 2,589 | 3,000 | 192,000 | 2.437 ± 0.024 | 1.117 ± 0.020 | 1.116 ± 0.020 | 117.03 ± 1.04 |
| Seed + left stages | 2,589 | 3,000 | 192,000 | 2.437 ± 0.024 | 1.114 ± 0.019 | 1.113 ± 0.019 | 117.17 ± 1.09 |
| Seed + random stages | 2,589 | 3,000 | 192,000 | 2.437 ± 0.024 | 1.093 ± 0.019 | 1.093 ± 0.018 | 117.17 ± 1.01 |
| Shared balanced, joint | 2,589 | 3,000 | 192,000 | 3.044 ± 0.043 | 1.263 ± 0.011 | 1.262 ± 0.010 | 95.81 ± 2.58 |
| Shared left, joint | 2,589 | 3,000 | 192,000 | 3.044 ± 0.043 | 1.261 ± 0.035 | 1.260 ± 0.034 | 94.93 ± 2.35 |
| Independent, total params matched | 2,556 | 3,000 | 192,000 | 0.379 ± 0.004 | 0.711 ± 0.010 | 0.710 ± 0.009 | 57.67 ± 1.60 |
| Independent, compute matched | 2,556 | 24,042 | 1,538,675 | 3.044 ± 0.043 | 5.758 ± 0.088 | 5.753 ± 0.087 | 59.10 ± 0.25 |
| Independent, per-task width matched* | 15,534 | 3,000 | 192,000 | 3.044 ± 0.043 | 1.272 ± 0.016 | 1.270 ± 0.015 | 90.29 ± 0.44 |
| Modular matrix (known algebra) | 64 | 3,000 | 192,000 | 0.129 ± 0.001 | 0.510 ± 0.003 | 0.509 ± 0.003 | 42.66 ± 0.12 |

Each run used a fresh serial worker process. Peak RSS includes Python, NumPy, generated data and evaluation, so it is not pure model or training memory. Short runs have noisy timing and startup/data-generation time is excluded from training seconds. The independent compute run reaches the shared joint balanced budget within at most one update; its smaller matrices can take longer despite equal dense MACs.

| Model | Weights KiB | Weights + gradients + Adam KiB | Inference on 512 programs ms | Cache payload KiB | Cached inference ms |
|---|---:|---:|---:|---:|---:|
| Seed + balanced stages | 20.23 | 80.91 | 1.226 ± 0.055 | 144.34 ± 0.00 | 6.796 ± 0.370 |
| Seed + left stages | 20.23 | 80.91 | 1.189 ± 0.055 | 284.04 ± 3.14 | 11.139 ± 0.478 |
| Seed + random stages | 20.23 | 80.91 | 1.202 ± 0.037 | 227.12 ± 41.51 | 9.520 ± 1.384 |
| Shared balanced, joint | 20.23 | 80.91 | 1.210 ± 0.044 | 144.34 ± 0.00 | 6.469 ± 0.310 |
| Shared left, joint | 20.23 | 80.91 | 1.215 ± 0.062 | 284.04 ± 3.14 | 10.803 ± 0.237 |
| Independent, total params matched | 19.97 | 79.88 | 0.671 ± 0.034 | 46.56 ± 0.00 | 4.889 ± 0.149 |
| Independent, compute matched | 19.97 | 79.88 | 0.684 ± 0.040 | 46.56 ± 0.00 | 4.976 ± 0.140 |
| Independent, per-task width matched* | 121.36 | 485.44 | 1.188 ± 0.047 | 144.34 ± 0.00 | 6.497 ± 0.193 |
| Modular matrix (known algebra) | 0.50 | 2.00 | 0.441 ± 0.010 | N/A | N/A |

Inference is on the 512 length-6 held-out-pair programs. Uncached timing is the mean of five batched calls; cached timing is one cold-cache, program-by-program traversal. These implementations have different batching overheads, so this does not isolate an algorithmic caching speedup. Cache payload excludes keys, dictionaries and Python object overhead. No cache survives a training parameter update. Working arrays and temporary optimizer allocations are additional to the persistent tensor counts.

## Paired comparisons

Held-out-pair accuracy differences in percentage points, pairing the same seed and split. With only five seeds these are descriptive results; no population significance or broad superiority is claimed.

| First minus second | Mean ± SD pp | Per-seed pp |
|---|---:|---|
| seed_balanced minus seed_left | -37.66 ± 12.30 | -27.07, -35.08, -26.43, -55.43, -44.29 |
| seed_balanced minus seed_random | -2.12 ± 1.32 | -2.95, -1.10, -2.03, -3.87, -0.64 |
| seed_balanced minus shared_balanced | -5.14 ± 1.59 | -5.16, -4.14, -6.08, -3.13, -7.18 |
| seed_left minus shared_left | 7.20 ± 13.80 | 0.09, 15.47, -4.88, 27.44, -2.12 |
| seed_balanced minus independent_compute | 0.99 ± 1.91 | 2.58, 0.09, 0.00, 3.41, -1.10 |
| seed_left minus independent_capacity | 37.73 ± 12.60 | 28.82, 34.99, 24.40, 55.62, 44.84 |

## Failures and retention

| Model | OOD state accuracy % | Invalid OOD map % | Final primitive accuracy % | Primitive change from first checkpoint pp |
|---|---:|---:|---:|---:|
| Seed + balanced stages | 36.7 ± 1.0 | 73.6 ± 1.0 | 100.0 ± 0.0 | 0.0 ± 0.0 |
| Seed + left stages | 73.9 ± 6.9 | 43.1 ± 7.9 | 100.0 ± 0.0 | 0.0 ± 0.0 |
| Seed + random stages | 38.5 ± 2.5 | 70.1 ± 4.1 | 100.0 ± 0.0 | 0.0 ± 0.0 |
| Shared balanced, joint | 44.4 ± 1.6 | 69.0 ± 2.2 | 100.0 ± 0.0 | 0.0 ± 0.0 |
| Shared left, joint | 70.8 ± 6.1 | 46.6 ± 6.4 | 100.0 ± 0.0 | 0.0 ± 0.0 |
| Independent, total params matched | 29.7 ± 1.0 | 78.5 ± 1.3 | 100.0 ± 0.0 | 5.0 ± 11.2 |
| Independent, compute matched | 34.5 ± 2.1 | 69.9 ± 5.7 | 100.0 ± 0.0 | 5.0 ± 11.2 |
| Independent, per-task width matched* | 38.7 ± 0.7 | 70.6 ± 2.3 | 100.0 ± 0.0 | 0.0 ± 0.0 |
| Modular matrix (known algebra) | 100.0 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 |

Invalid means the prediction repeats an output state and therefore cannot be a permutation. The neural decoder does not enforce validity. The modular baseline always predicts valid maps in the observed tests. Primitive retention covers four elementary commands only, with replay; it is not a broad continual-learning evaluation. Raw JSON contains per-length accuracy, training checkpoints and specific failures.

Examples from seed 0 (zero-based command/state IDs):

| Model | Program | Correct map | Predicted map |
|---|---|---|---|
| Seed + balanced stages | [0, 1] | [0, 2, 3, 1] | [2, 2, 0, 1] |
| Seed + left stages | [2, 3] | [3, 1, 2, 0] | [0, 1, 2, 1] |
| Seed + random stages | [2, 3] | [3, 1, 2, 0] | [1, 1, 2, 0] |
| Shared balanced, joint | [2, 3] | [3, 1, 2, 0] | [2, 1, 2, 0] |
| Shared left, joint | [2, 3] | [3, 1, 2, 0] | [0, 1, 2, 0] |
| Independent, total params matched | [2, 3] | [3, 1, 2, 0] | [1, 1, 2, 3] |
| Independent, compute matched | [2, 3] | [3, 1, 2, 0] | [1, 1, 2, 3] |
| Independent, per-task width matched* | [2, 3] | [3, 1, 2, 0] | [1, 1, 2, 3] |

## Interpretation and limits

1. **Balance did not help here.** Balanced assembly lost to left assembly under the same staged training and same dense computation. Their binary-tree node counts are identical. A balanced tree has shorter dependency depth, but this implementation executes tree nodes serially; it does not benchmark parallel hardware scheduling.
2. **Reusing learned components is viable, but the numerical rule is not established as useful.** Sharing plus an appropriate composition operator can generalize. The matrix model is advantaged by knowing associativity and matrix multiplication; it learns only primitive meanings. It is not parameter- or optimizer-matched and should not be treated as evidence that all modular models will solve harder tasks.
3. **Poor optimization constrains the negative result.** Balanced models fail to fit substantial portions of their training set; this cannot distinguish an inherently poor rule from an unsuitable tanh composer, width or optimizer budget. A plausible explanation for the left advantage is that the composer usually combines a running result with a primitive, while balanced assembly asks it to combine two larger latent programs. This mechanism was not directly tested.
4. **The seed is an encoder template, not a magical constant.** The four commands use one learned leaf-encoding table. Recursion reuses that table and the same composer. Representations become larger programs, while the number of trainable parameters remains fixed; this is not autonomous network or skill growth.
5. **Independent learning is defined by length.** Matching total parameters makes each independent task model narrower. The extended run matches approximate dense compute but sees many more repeated examples. The post hoc width check matches per-task capacity at six times aggregate parameters. No single comparison simultaneously matches total parameters, per-task width, data and runtime.
6. **The benchmark is small and structured.** Four states, four fixed permutations, two held-out directed pairs, one underlying algebra and short programs. A legal training program can share its output map with a forbidden test program; the split withholds syntax combinations, not output classes. Results do not establish performance on language, perception, real-world continual learning or other algebras.
7. **No data-efficiency curve was run.** Fixed-budget comparisons and repeated-example counts do not establish sample complexity or that a target accuracy can be reached with less data. There is no Transformer/Tree-LSTM comparison or tuned published system replication.
8. **Small seed count and timing scope.** Five seeds vary initialization, minibatches and program subsets together; they do not independently estimate each source of variation. CPU runtime is one environment and analytic MACs are only a proxy. All arithmetic is float64. Hardware/BLAS versions can affect times and low-order numeric results.

A useful next experiment would keep the same held-out split, vary training-set size and compute budgets, and compare stronger associative/neural composers on several independent transformation algebras. That would test whether these findings survive removal of this experiment’s main bottlenecks.

## Research context

Recursive neural composition predates this experiment; see [Socher et al., 2013](https://aclanthology.org/D13-1170/). Holding out systematic command combinations is motivated by the generalization question studied in [Lake & Baroni, 2018](https://proceedings.mlr.press/v80/lake18a.html). This benchmark is newly generated for this small test and is **not SCAN**. The implemented baselines are controlled elementary instances, not reproductions of those papers.

See README.md for the full data-generation description, executable commands and file manifest. PROTOCOL.md records the original fixed design; PROTOCOL_AMENDMENT.md records the post hoc capacity check.
