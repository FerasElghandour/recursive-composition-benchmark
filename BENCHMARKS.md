# Reproduction and benchmark scope

A fresh full run retrained all**nine methods × five seeds**. Train/IID/OOD/length-extrapolation accuracies match the retained45 runs exactly. The result preserves the original failed balanced-tree hypothesis.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Behavioral numerical regression / gradient and split verification, not a fresh dataset performance result | None |
| `full-benchmark` | All9conditions by5seeds,3000baseupdates; built-in generated permutation task | None |

Supply named paths with `--input name=/absolute/path`. Exact commands are versioned in [reproduction.json](reproduction.json); direct training commands and preparation contracts are below.

## Evidence boundary

The task is a controlled generated permutation benchmark. Timing and memory measurements depend on the runtime and hardware.

Keep input acquisition and source credits in [DATA.md](DATA.md). Historical results, new reference runs and small contract fixtures have separate evidential scope.

## Recorded command definitions

### checks

```text
{python} experiment.py --verify
```

### full-benchmark

```text
{python} experiment.py --out {output}/experiment
```

`{python}` is the current interpreter. `{output}` is the new run directory. `{input:name}` refers to a named absolute input path supplied to the driver. Install declared dependencies in a clean environment before executing. Never execute untrusted serialized models or notebooks.
