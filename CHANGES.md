# Validation update

* `models/vegformer.py`: MSM loss is now computed per sensor, only on the channels and timesteps where that
  sensor was masked (previously visible channels at masked steps were included, which dilutes the objective).
  Mask-token sizes follow `n_bands`/`n_climate_vars` instead of being hard-coded. Adds `heads_trained` flag.
  Retraining is needed for the loss values to be comparable with the old checkpoint.
* `evaluation/`: held-out reconstruction test and linear probe (see README, Validation).
* `README.md`, `configs/pretrain_config.yaml`, `notebooks/05_degradation_mapping.py`: claims aligned with what the
  code does. Notebook `.ipynb` copies need re-exporting from the `.py` files.
* `docs/CITATION_CHECK.md`: references to verify.
