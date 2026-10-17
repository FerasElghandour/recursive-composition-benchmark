# Explicit post hoc sensitivity check

After inspecting seed 0 from the fixed main protocol, add `independent_capacity`:
six width-31 balanced networks, one per length, with the same joint sampling stream
and 3,000 updates. This matches each task network's capacity and the shared model's
dense computation, but uses six times the total parameters. It checks the alternative
explanation that the parameter-matched independent networks simply lack sufficient
per-task width. It is not a preregistered primary comparison. No hyperparameters or
main experimental conditions are changed. All five seeds will be rerun with the final
source so its recorded SHA256 matches every delivered result.

Also clarify the batch-stream statement in the executable: all joint methods use the
same sequence of length and example draws; all staged methods use another shared
sequence. The staged and joint curricula intentionally have different exposures.
