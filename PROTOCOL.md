# Locked small-experiment protocol

Written before inspecting the pilot's accuracy. No benchmark-specific accuracy targets,
architecture search, test-based early stopping or selective seed exclusion.

- Five seeds: 0–4. Each changes train/test sampling, initialization and minibatches.
- Fixed four noncommuting permutation commands over four states. Predict the complete
  four-state input/output map. Chance exact accuracy is not necessarily 1/24 because
  predicted outputs need not form a permutation; report a majority-map reference too.
- Train program lengths 1–6, with adjacent pairs (0,1) and (2,3) forbidden everywhere.
  Test novel pairs at lengths 2–6; separately unseen legal programs at lengths 3–6;
  and legal programs of unseen lengths 8 and 12. Train/test programs are disjoint.
- Train sets capped at 512 programs per length; evaluation uses up to 512 novel-pair
  and 256 IID programs per length, and 512 per long length. Test metrics pool examples
  within each split. Per-length results are also retained.
- Float64 NumPy models with analytic gradients and Adam. No pretrained weights.
  3,000 updates, batch 64, learning rate .003, gradient norm clip 5.
- Shared neural models: token embedding to width 31; one shared tanh affine binary
  composer; one shared decoder to 4×4 output logits. 2,589 parameters. No positional
  embeddings, hidden external labels, or intermediate ground-truth supervision.
- Seed balanced / left / random: same examples, optimizer, width and update count.
  Length stages 1,2,3,4,5,6, 500 updates each; stage selected 2/3 of the time with
  1/3 replay of uniformly sampled earlier lengths. First stage is all primitives.
  This creates more complex representations from the same learned seed encoder and
  composer; model parameter count does not increase. Random tree is fixed per length
  and seed. The growth rule concerns subtree size, not the arithmetic label to predict.
- Shared balanced / shared left: standard tree/sequence recursive neural models,
  same components; joint uniformly sampled lengths from update one. Isolates the
  staged curriculum from topology. Left tree is a tied binary-composition recurrence.
- Independent: six separate balanced tree models, one per length. Width 10 makes the
  aggregate 2,556 parameters (1.27% below shared); no transfer of embeddings/composer
  between tasks. Same joint sampling and update budget. No extrapolation to untrained
  lengths is defined. This operationalizes task independence by program length.
- Independent compute: same width and sampling stream, continuing until shared joint
  balanced's dense training-MAC budget is reached; extra examples are explicitly
  reported. This matches approximate dense computation, not data exposure.
- Modular matrix: standard strongly structured baseline, 64 learned primitive logits,
  row softmax then exact matrix multiplication. Known composition algebra is supplied,
  unlike the neural composer. Joint sampling, 3,000 updates, learning rate .03.
  Low parameter count and supplied algebra mean it is a reference, not a matched test
  of the growth rule. Symbolic truth interpreter has 100% by construction.
- Primary endpoint: exact complete-map accuracy on novel adjacent-pair compositions.
  Secondary: state accuracy, IID, length shift, primitives and retention.
- No significance claims from five seeds. Report sample SD, range and paired differences.
- Dense training compute proxy: three times forward matrix-multiply MACs (forward plus
  two backward multiplications), excludes elementwise work, scatter, Adam and evaluation.
  FLOPs proxy is twice MACs. Parameter/gradient/Adam bytes exact; isolated process peak
  RSS includes interpreter, data and evaluation. Time includes fixed progress checks.
- All neural methods evaluate optional inference memoization keyed by subprogram and
  tree shape. Training does not use stale cached representations across parameter updates.
  Cache bytes count array payload only. Verify cached predictions equal uncached.
- Fresh worker process per method and seed, serial CPU execution, BLAS thread variables
  set to one. Save data, weights, individual results, learning checkpoints and failures.
- Verify hand gradients by finite differences, target interpreter by an independent
  loop, split disjointness and forbidden-pair conditions, cache equivalence.

The hypothesis is operationalized narrowly. The experiment cannot establish emergence,
autonomous skill invention, broad intelligence, novelty, or superiority to published
state-of-the-art systems. Balanced/left/random are assembly topologies under the same
ascending-size curriculum; the rule does not specify a unique temporal curriculum.
