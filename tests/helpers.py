"""Shared model names and compact connectivity fixtures."""
from importlib import import_module

MODELS = {
    "hpc_only": "hpc_only",
    "pfc_hpc": "pfc_hpc",
    "pfc_ca1": "pfc_hpc.pfc_ca1",
    "pfc_ca3": "pfc_hpc.pfc_ca3",
    "ec3_ca3_ca1": "pfc_hpc.ec3_ca3_ca1",
}


def parameters_for(name):
    package = name if name in ("hpc_only", "ec3_ca3_ca1") else "pfc_hpc"
    if package == "ec3_ca3_ca1":
        package = MODELS[package]
    return import_module(f"{package}.parameters").local_params()


def small_parameters(package):
    """Connectivity only: retain hippocampal 10:1 and cortical 4:1 geometry."""
    params = parameters_for(package)
    params.update(N_PY3=40, N_IN3=4, N_PY1=40, N_IN1=4)
    if package == "pfc_hpc":
        params.update(N_PYc=40, N_INc=10, Gc=50, N_PYh=80, N_INh=8, N_IN=18)
    return params
