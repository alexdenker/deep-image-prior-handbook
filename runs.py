import subprocess
from itertools import product
import random
import string

# Shared sweep parameters
base_sweep_config = {
    "denoise_strength": [1e-3, 1e-4, 1e-6],
    "num_steps": [20000],
    "model_inp": ["fbp", "adjoint", "random"]
}

# method-specific
methods = ["selfguided"]
method_sweeps = {
    "selfguided": {"num_noise_realisations": [5, 10]},
    "aseq": {"num_inner_steps": [5, 10]}
}

use_wandb = True
# Collect all sweep configs
all_runs = []

for method in methods:
    sweep_config = {
        **base_sweep_config,
        "method": [method],
        **method_sweeps[method]  # Add conditional sweeps
    }

    # Compute all combinations for this method
    keys = list(sweep_config.keys())
    combinations = list(product(*sweep_config.values()))

    for combo in combinations:
        args = dict(zip(keys, combo))
        all_runs.append(args)

# Run all
for args in all_runs:
    print("Running with config:", args)

    cli_args = ["python", "overeng_rundip.py"]
    for k, v in args.items():
        cli_args.extend([f"--{k}", str(v)])
    if use_wandb:    
        cli_args.append("--use_wandb")

    subprocess.run(cli_args)
