"""Start a hippocampal simulation from the project folder.

Model construction lives in each model's folder. This entry point loads the
selected parameter set and passes it to that model's run_simulation function.
"""
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["hpc_only", "pfc_hpc", "pfc_ca1", "pfc_ca3", "ec3_ca3_ca1"],
        default="hpc_only",
    )
    parser.add_argument("--out", help="New or empty output directory; defaults to results/<model>/baseline")
    parser.add_argument("--duration", type=float, default=60, help="Simulated seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=["numpy", "cpp_standalone"], default="cpp_standalone")
    parser.add_argument("--plot", action="store_true", help="Save and display simulation figures")
    args = parser.parse_args()

    # Reduced PFC models use the shared baseline parameters directly. The pulse
    # model adds its input timings; HPC-only uses its separate parameter set.

    if args.model == "hpc_only":
        from hpc_only import parameters
        from hpc_only.simulation import run_simulation

    elif args.model == "pfc_hpc":
        from pfc_hpc import parameters
        from pfc_hpc.simulation import run_simulation

    elif args.model == "pfc_ca1":
        from pfc_hpc import parameters
        from pfc_hpc.pfc_ca1.simulation import run_simulation

    elif args.model == "pfc_ca3":
        from pfc_hpc import parameters
        from pfc_hpc.pfc_ca3.simulation import run_simulation

    else:
        from pfc_hpc.ec3_ca3_ca1 import parameters
        from pfc_hpc.ec3_ca3_ca1.simulation import run_simulation

    params = parameters.local_params()

    output_directory = args.out or f"results/{args.model}/baseline"

    run_simulation(
        out=output_directory,
        params=params,
        duration=args.duration,
        seed_value=args.seed,
        backend=args.backend,
        plotting=args.plot,
    )
