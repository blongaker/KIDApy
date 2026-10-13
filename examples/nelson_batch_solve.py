"""
Batch point integration of the Nelson network: one independent solve per row.

Given

    states : (M, 14)  species abundances per H nucleus, columns in net.species order
    params : (M, 7)   environment, columns [nH, T, Tgrain, Av, uv1, uv3, uv4] (the layout of HydroData(three_uv_bands=True))

each row i is integrated for a time dt with A and B held fixed at params[i], starting from states[i].  The rows are independent, so they are solved in parallel with QuadraticSolver.solve_multiple.

  - Photorates use the three-band radiation field (Network(pd_fac=True), see pdfac.py), so the Av column is ignored and derived from the bands.
  - Charge conservation is enforced as in nelson_charge.py: x(e-) = sum of cation abundances throughout.  The e- column of states is reset to the cation sum before solving.

With no arguments, the script runs a demo on rows of datasets/grav_collapse/, every row starting from the network's initial abundances, for one hydro step.  Output is written to examples/data/nelson_batch_solve/.

Run from the repository root or from within examples/:

    python examples/nelson_batch_solve.py
    python examples/nelson_batch_solve.py --states states.npy --params params.npy --dt-kyr 0.25
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.hydro import HydroData
from examples.nelson_charge import charges, electron_row_matrix
from parser import Network, load_abundances
from solver import QuadraticSolver

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NETWORK_PATH    = REPO_ROOT / "networks" / "nelson" / "gas_reactions.in"
ABUNDANCES_PATH = REPO_ROOT / "networks" / "nelson" / "abundances.in"
TRACE_FILE      = REPO_ROOT / "datasets" / "grav_collapse" / "R0.1_M6_trace_cells.npy"
TIME_FILE       = REPO_ROOT / "datasets" / "grav_collapse" / "R0.1_M6_time.npy"
SAVE_DIR = HERE / "data" / "nelson_batch_solve"

YEAR = 3600.0 * 24.0 * 365.25

ATOL = 1e-20
RTOL = 1e-3

N_WORKERS  = 8
DEMO_ROWS  = 64       # rows of the dataset used by the demo
BAND_FLOOR = 1e-30    # pdfac takes log10 of the bands, so they must stay positive

# Environment columns, in the order HydroData(three_uv_bands=True) returns them
ENV_COLS = ["nH", "T", "Tgrain", "Av", "uv1", "uv3", "uv4"]

# ---------------------------------------------------------------------------
# Network, with charge conservation
# ---------------------------------------------------------------------------

net = Network(grains=True, self_shielding=True, pd_fac=True)
net.load_from_disk(str(NETWORK_PATH))
net.drop_passive_species()

# R replaces the e- row of A and B with the sum of the cation rows, so that dx(e-)/dt = q . dx/dt and x(e-) = q . x holds for the whole run.
R = electron_row_matrix(net)


def get_tensors(env):
    A, B = net.get_operators_vectorized(env)
    return R @ A, R @ B


def params_to_env(p):
    """One row of params -> the env dict the network expects."""
    env = dict(zip(ENV_COLS, map(float, p)))
    env["Tcap_2body"] = True
    for band in ("uv1", "uv3", "uv4"):
        env[band] = max(env[band], BAND_FLOOR)
    return env


def batch_solve(states, params, dt, n_workers=1, atol=ATOL, rtol=RTOL, verbose=False):
    """
    Integrate each row of states for a time dt at the fixed environment in the same row of params.

    Parameters
    ----------
    states : np.ndarray, shape (M, 14)
        Initial abundances per H nucleus, columns in ``net.species`` order.
    params : np.ndarray, shape (M, 7)
        Environment per row, columns ``ENV_COLS``.
    dt : float
        Integration time [s], the same for every row.
    n_workers : int
        Number of worker processes. ``1`` runs serially.

    Returns
    -------
    final : np.ndarray, shape (M, 14)
        Abundances after dt.  Rows whose solve failed are NaN.
    failures : list of (row, exception)
    """
    states = np.asarray(states, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    N = len(net.species)
    if states.ndim != 2 or states.shape[1] != N:
        raise ValueError(f"states must have shape (M, {N}); got {states.shape}")
    if params.ndim != 2 or params.shape[1] != len(ENV_COLS):
        raise ValueError(f"params must have shape (M, {len(ENV_COLS)}); got {params.shape}")
    if states.shape[0] != params.shape[0]:
        raise ValueError(f"states and params need the same number of rows; got {states.shape[0]} and {params.shape[0]}")

    # Start each row charge-neutral, which the electron-row fix then preserves.
    x0 = states.copy()
    x0[:, net.species_map["e-"]] = x0 @ charges(net)

    tensors = [get_tensors(params_to_env(p)) for p in params]

    results, failures = QuadraticSolver().solve_multiple(
        tensors,
        t_span=(0.0, dt),
        x0=x0,
        atol=atol,
        rtol=rtol,
        method="BDF",
        t_eval=[dt],
        n_workers=n_workers,
        verbose=verbose,
        return_failures=True,
    )

    final = np.full_like(x0, np.nan)
    for i, result in enumerate(results):
        if result is not None:
            final[i] = result[1][:, -1]
    return final, failures


def demo_inputs(m):
    """First m rows of the gravitational-collapse dataset, each starting from the network's initial abundances."""
    h = HydroData(three_uv_bands=True)
    h.load_hydro_data(str(TRACE_FILE), max_tracers=1, abundances=ABUNDANCES_PATH)
    params = h.params[:m]

    abund = load_abundances(str(ABUNDANCES_PATH))
    x0 = np.zeros(len(net.species))
    for name, val in abund.items():
        if name in net.species_map:
            x0[net.species_map[name]] = val
    states = np.tile(x0, (params.shape[0], 1))
    return states, params


if __name__ == "__main__":
    t_kyr = np.load(TIME_FILE)
    dt_hydro_kyr = t_kyr[1] - t_kyr[0]

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", type=Path, help=f"(M, {len(net.species)}) .npy of abundances")
    ap.add_argument("--params", type=Path, help=f"(M, {len(ENV_COLS)}) .npy of environments")
    ap.add_argument("--dt-kyr", type=float, default=dt_hydro_kyr, help=f"integration time per row [kyr] (default: one hydro step, {dt_hydro_kyr:g})")
    ap.add_argument("--n-workers", type=int, default=N_WORKERS)
    args = ap.parse_args()

    if (args.states is None) != (args.params is None):
        ap.error("--states and --params must be given together")

    if args.states is None:
        states, params = demo_inputs(DEMO_ROWS)
    else:
        states, params = np.load(args.states), np.load(args.params)

    dt = args.dt_kyr * 1e3 * YEAR
    M = states.shape[0]

    print("Batch solve")
    print(f"  species = {len(net.species)}")
    print(f"  rows    = {M}")
    print(f"  dt      = {args.dt_kyr:g} kyr")

    final, failures = batch_solve(states, params, dt, n_workers=args.n_workers, verbose=True)

    for i, exc in failures:
        print(f"  [WARN] row {i} failed: {exc}")
    print(f"  converged = {M - len(failures)}/{M}")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / "final_states.npy"
    np.save(out_path, final)
    with open(SAVE_DIR / "columns.txt", "w", encoding="ascii") as f:
        f.write("\n".join(net.species))
    print(f"  saved     = {out_path}  {final.shape}")
