# Validation record

Release review: 17 October 2026.

The saved five-seed study reports **4.4%** held-out-pair accuracy for balanced staged composition versus **42.1%** for left staged composition; a known-algebra baseline reaches **100%**. Fresh audit passes regenerate all five datasets and exactly reproduce three representative seed-0 runs, including weights, training histories, and metrics.

The listed fresh checks exercise the included public edition. Historical full experiments, representative replays, and synthetic fixtures are identified separately. Test success does not certify commercial deployment.

```sh
python experiment.py --verify
```

The experiment rejects the balanced-growth hypothesis in this implementation. The algebraic baseline has an information advantage. Results concern the defined four-state task and do not establish general reasoning or training-efficiency claims.
