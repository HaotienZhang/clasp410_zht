#!/usr/bin/env python3
"""
lab 4 - stochastic spread model (wildfire core) in a single file.

to reproduce the figures used in the report, run:

    python lab4.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# state constants (wildfire)
BARE = 1    # bare or consumed
FOREST = 2  # has fuel
BURN = 3    # burning

# colormap for plotting wildfire grids
COLORS = ListedColormap(["tan", "darkgreen", "crimson"])


@dataclass
class FireParams:
    """parameters for the wildfire simulation."""

    ny: int
    nx: int
    p_spread: float
    p_bare: float = 0.0
    ignite: str = "center"     # "center" or "random"
    p_ignite: float = 0.0      # only used if ignite == "random"
    seed: Optional[int] = 42
    steps: int = 2             # number of update steps (t=0 -> t=steps)


def _rng_from_seed(seed: Optional[int]) -> np.random.Generator:
    """
    create a numpy random generator from a seed.

    Parameters
    ----------
    seed : int or None
        integer seed value. if None, a default generator is used.

    Returns
    -------
    numpy.random.Generator
        random number generator instance.
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def init_fire_grid(
    params: FireParams,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    initialize a wildfire grid in the style of the lab manual starter code.

    Parameters
    ----------
    params : FireParams
        parameters describing grid size and probabilities.
    rng : numpy.random.Generator or None, optional
        random number generator. if None, a new generator is created
        from params.seed.

    Returns
    -------
    numpy.ndarray
        2d array of shape (ny, nx) with integer state values.
    """
    if rng is None:
        rng = _rng_from_seed(params.seed)

    ny, nx = params.ny, params.nx

    # create an initial grid with all cells as forest (state 2)
    forest = np.zeros((ny, nx), dtype=int) + FOREST

    # set bare spots using per-cell bernoulli draws with probability p_bare
    if params.p_bare > 0.0:
        for i in range(nx):
            for j in range(ny):
                if rng.random() < params.p_bare:
                    forest[j, i] = BARE

    # seed ignition in the center or randomly in non-bare cells
    if params.ignite == "center":
        cy, cx = ny // 2, nx // 2
        if forest[cy, cx] != BARE:
            forest[cy, cx] = BURN
    elif params.ignite == "random":
        if params.p_ignite > 0.0:
            for i in range(nx):
                for j in range(ny):
                    if forest[j, i] != BARE and rng.random() < params.p_ignite:
                        forest[j, i] = BURN
    else:
        raise ValueError(f"unknown ignite mode: {params.ignite!r}")

    return forest


def step_fire(
    grid: np.ndarray,
    p_spread: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    advance the wildfire model by one synchronous time step.

    Parameters
    ----------
    grid : numpy.ndarray
        2d array of current cell states.
    p_spread : float
        probability that fire spreads from a burning cell
        to a forest neighbor in one step.
    rng : numpy.random.Generator or None, optional
        random number generator. if None, a new generator is used.

    Returns
    -------
    numpy.ndarray
        updated grid after one time step.
    """
    if rng is None:
        rng = _rng_from_seed(None)

    ny, nx = grid.shape
    new_grid = grid.copy()

    # loop over all cells and spread fire from burning cells
    for i in range(nx):
        for j in range(ny):
            if grid[j, i] == BURN:
                # up neighbor (j+1, i)
                if j + 1 < ny and grid[j + 1, i] == FOREST:
                    if rng.random() < p_spread:
                        new_grid[j + 1, i] = BURN
                # down neighbor (j-1, i)
                if j - 1 >= 0 and grid[j - 1, i] == FOREST:
                    if rng.random() < p_spread:
                        new_grid[j - 1, i] = BURN
                # right neighbor (j, i+1)
                if i + 1 < nx and grid[j, i + 1] == FOREST:
                    if rng.random() < p_spread:
                        new_grid[j, i + 1] = BURN
                # left neighbor (j, i-1)
                if i - 1 >= 0 and grid[j, i - 1] == FOREST:
                    if rng.random() < p_spread:
                        new_grid[j, i - 1] = BURN

    # convert cells that were burning at the start to bare cells
    for i in range(nx):
        for j in range(ny):
            if grid[j, i] == BURN:
                new_grid[j, i] = BARE

    return new_grid


def simulate_fire(params: FireParams) -> List[np.ndarray]:
    """
    run the wildfire simulation for a fixed number of time steps.

    Parameters
    ----------
    params : FireParams
        parameters describing grid size, probabilities, and time steps.

    Returns
    -------
    list of numpy.ndarray
        list of grids at times t=0,1,...,steps.
    """
    rng = _rng_from_seed(params.seed)
    states: List[np.ndarray] = []

    grid = init_fire_grid(params, rng)
    states.append(grid.copy())

    for _ in range(params.steps):
        grid = step_fire(grid, params.p_spread, rng)
        states.append(grid.copy())

    return states

def _plot_states(
    states: List[np.ndarray],
    title: str,
    savepath: str,
    font_size: int,
) -> None:
    """
    plot a sequence of wildfire states side by side and save to file.

    Parameters
    ----------
    states : list of numpy.ndarray
        list of 2d state arrays at successive time steps.
    title : str
        figure title string.
    savepath : str
        output filename for the figure.
    font_size : int
        font size for in-cell text labels.

    Returns
    -------
    None
    """
    n_steps = len(states)

    fig, axes = plt.subplots(1, n_steps, figsize=(3.6 * n_steps, 3.6), constrained_layout=True)
    if n_steps == 1:
        axes = [axes]

    state_text = {BARE: "Bare", FOREST: "Forest", BURN: "FIRE!"}

    for i, (ax, s) in enumerate(zip(axes, states)):
        ny, nx = s.shape

        # use pcolor with custom colormap and fixed value range
        im = ax.pcolor(s, cmap=COLORS, vmin=1, vmax=3, edgecolors="k", linewidth=1.0)

        ax.set_xticks(np.arange(0, nx + 1))
        ax.set_yticks(np.arange(0, ny + 1))
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_title(f"Forest Status (iStep={i})", fontsize=12)

        # draw label text in the center of each cell
        for r in range(ny):
            for c in range(nx):
                label = state_text.get(int(s[r, c]), str(int(s[r, c])))
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    f"{label}\n i, j = {r}, {c}",
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="black",
                )

        ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_ticks([1.33, 2.0, 2.66])
    cbar.set_ticklabels(["Bare (1)", "Forest (2)", "Burn (3)"])

    fig.suptitle(title, fontsize=13)
    fig.savefig(savepath, dpi=220)
    plt.close(fig)


def validation() -> None:
    """
    run the two validation tests for the wildfire model.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    outdir = "results"

    # 3x3 grid with p_spread=1, p_bare=0, center ignition
    params_3x3 = FireParams(
        ny=3,
        nx=3,
        p_spread=1.0,
        p_bare=0.0,
        ignite="center",
        p_ignite=0.0,
        seed=0,
        steps=2,
    )
    states_3x3 = simulate_fire(params_3x3)
    _plot_states(
        states_3x3,
        title="Validation 1: 3x3 (t=0,1,2)",
        savepath=os.path.join(outdir, "validation_fire_3x3.png"),
        font_size=10,
    )

    # 3x7 grid with the same settings
    params_wide = FireParams(
        ny=3,
        nx=7,
        p_spread=1.0,
        p_bare=0.0,
        ignite="center",
        p_ignite=0.0,
        seed=0,
        steps=2,
    )
    states_wide = simulate_fire(params_wide)
    _plot_states(
        states_wide,
        title="Validation 2: 3x7 (t=0,1,2)",
        savepath=os.path.join(outdir, "validation_fire_3x7.png"),
        font_size=6,
    )


def run_experiment1() -> None:
    """
    run experiment 1 for rq1 using the wildfire model.

    this explores how spread depends on p_spread and p_bare
    and creates two multi-panel figures plus one time series.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    outdir = "results/exp1"

    def compute_metrics(states: List[np.ndarray]) -> tuple[float, int, int]:
        """
        compute burned fraction, duration, and peak burning.

        Parameters
        ----------
        states : list of numpy.ndarray
            list of grids at successive time steps.

        Returns
        -------
        tuple of (float, int, int)
            burned fraction, duration in steps, and peak burning count.
        """
        n_steps = len(states)
        ny, nx = states[0].shape
        n_cells = ny * nx

        final = states[-1]
        burned_fraction = np.sum(final == BARE) / n_cells

        duration = 0
        for t in range(n_steps):
            if np.any(states[t] == BURN):
                duration = t

        peak_burning = max(np.sum(s == BURN) for s in states)
        return burned_fraction, duration, peak_burning

    # sweep over p_spread
    p_spread_values = np.linspace(0.0, 1.0, 21)
    bf_ps, dur_ps, peak_ps = [], [], []

    for p_spread in p_spread_values:
        params = FireParams(
            ny=50,
            nx=50,
            p_spread=float(p_spread),
            p_bare=0.2,
            ignite="random",
            p_ignite=0.02,
            seed=0,
            steps=200,
        )
        states = simulate_fire(params)
        bf, dur, pk = compute_metrics(states)
        bf_ps.append(bf)
        dur_ps.append(dur)
        peak_ps.append(pk)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    axes[0].plot(p_spread_values, bf_ps)
    axes[0].set_title("Burned fraction vs Pspread")
    axes[0].set_xlabel("Pspread")
    axes[0].set_ylabel("Burned fraction")

    axes[1].plot(p_spread_values, dur_ps)
    axes[1].set_title("Duration vs Pspread")
    axes[1].set_xlabel("Pspread")
    axes[1].set_ylabel("Duration (steps)")

    axes[2].plot(p_spread_values, peak_ps)
    axes[2].set_title("Peak burning vs Pspread")
    axes[2].set_xlabel("Pspread")
    axes[2].set_ylabel("Peak burning")

    fig.suptitle("RQ1 — Parameter Sweep: Pspread", fontsize=14)
    fig.savefig(os.path.join(outdir, "pspread_sweep.png"), dpi=220)
    plt.close(fig)

    # sweep over p_bare
    p_bare_values = np.linspace(0.0, 1.0, 21)
    bf_pb, dur_pb, peak_pb = [], [], []

    for p_bare in p_bare_values:
        params = FireParams(
            ny=50,
            nx=50,
            p_spread=0.5,
            p_bare=float(p_bare),
            ignite="random",
            p_ignite=0.02,
            seed=1,
            steps=200,
        )
        states = simulate_fire(params)
        bf, dur, pk = compute_metrics(states)
        bf_pb.append(bf)
        dur_pb.append(dur)
        peak_pb.append(pk)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    axes[0].plot(p_bare_values, bf_pb)
    axes[0].set_title("Burned fraction vs Pbare")
    axes[0].set_xlabel("Pbare")
    axes[0].set_ylabel("Burned fraction")

    axes[1].plot(p_bare_values, dur_pb)
    axes[1].set_title("Duration vs Pbare")
    axes[1].set_xlabel("Pbare")
    axes[1].set_ylabel("Duration (steps)")

    axes[2].plot(p_bare_values, peak_pb)
    axes[2].set_title("Peak burning vs Pbare")
    axes[2].set_xlabel("Pbare")
    axes[2].set_ylabel("Peak burning")

    fig.suptitle("RQ1 — Parameter Sweep: Pbare", fontsize=14)
    fig.savefig(os.path.join(outdir, "pbare_sweep.png"), dpi=220)
    plt.close(fig)

    # representative time series
    params_ts = FireParams(
        ny=50,
        nx=50,
        p_spread=0.6,
        p_bare=0.2,
        ignite="random",
        p_ignite=0.02,
        seed=2,
        steps=200,
    )
    states_ts = simulate_fire(params_ts)
    burning_ts = [np.sum(s == BURN) for s in states_ts]

    plt.figure(figsize=(6, 4))
    plt.plot(burning_ts)
    plt.xlabel("Time step")
    plt.ylabel("Burning cells")
    plt.title("Wildfire Time Evolution (Representative Run)")
    plt.savefig(os.path.join(outdir, "time_series_example.png"), dpi=220)
    plt.close()


def run_experiment2() -> None:
    """
    run experiment 2 for rq2 using the disease model.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    outdir = "results/exp2"

    def simulate_disease(
        ny: int,
        nx: int,
        p_spread: float,
        p_bare: float,
        p_ignite: float,
        p_survive: float,
        steps: int,
        seed: int,
    ) -> tuple[List[np.ndarray], np.ndarray]:
        """
        run a single disease simulation on a grid.

        Parameters
        ----------
        ny : int
            number of grid cells in y direction.
        nx : int
            number of grid cells in x direction.
        p_spread : float
            probability that infection spreads to a neighbor.
        p_bare : float
            initial immune fraction (vaccine coverage).
        p_ignite : float
            probability that a susceptible cell is initially infected.
        p_survive : float
            probability that an infected cell survives and becomes immune.
        steps : int
            maximum number of time steps.
        seed : int
            seed for the random number generator.

        Returns
        -------
        tuple
            list of state grids over time and a boolean array for ever-infected cells.
        """
        rng = _rng_from_seed(seed)

        # start everyone as susceptible (state 2)
        grid = np.zeros((ny, nx), dtype=int) + 2

        # create initial immune population using p_bare
        for i in range(nx):
            for j in range(ny):
                if rng.random() < p_bare:
                    grid[j, i] = 1

        # seed infection among remaining susceptible cells
        for i in range(nx):
            for j in range(ny):
                if grid[j, i] == 2 and rng.random() < p_ignite:
                    grid[j, i] = 3

        states: List[np.ndarray] = [grid.copy()]
        ever_infected = grid == 3

        for _ in range(steps):
            ny_, nx_ = grid.shape
            new_grid = grid.copy()

            # spread infection from currently infected cells
            for i in range(nx_):
                for j in range(ny_):
                    if grid[j, i] == 3:
                        if j + 1 < ny_ and grid[j + 1, i] == 2:
                            if rng.random() < p_spread:
                                new_grid[j + 1, i] = 3
                        if j - 1 >= 0 and grid[j - 1, i] == 2:
                            if rng.random() < p_spread:
                                new_grid[j - 1, i] = 3
                        if i + 1 < nx_ and grid[j, i + 1] == 2:
                            if rng.random() < p_spread:
                                new_grid[j, i + 1] = 3
                        if i - 1 >= 0 and grid[j, i - 1] == 2:
                            if rng.random() < p_spread:
                                new_grid[j, i - 1] = 3

            # resolve infected cells to immune or dead
            for i in range(nx_):
                for j in range(ny_):
                    if grid[j, i] == 3:
                        if rng.random() < p_survive:
                            new_grid[j, i] = 1
                        else:
                            new_grid[j, i] = 0

            grid = new_grid
            ever_infected |= grid == 3
            states.append(grid.copy())

            # stop early if no infection remains
            if not np.any(grid == 3):
                break

        return states, ever_infected

    def compute_disease_metrics(
        states: List[np.ndarray],
        ever_infected: np.ndarray,
    ) -> tuple[float, float, int]:
        """
        compute summary metrics for a disease simulation.

        Parameters
        ----------
        states : list of numpy.ndarray
            list of grids over time.
        ever_infected : numpy.ndarray
            boolean array marking cells that were ever infected.

        Returns
        -------
        tuple
            fraction ever infected, fraction dead, and peak infected count.
        """
        ny, nx = states[0].shape
        n_cells = ny * nx

        frac_ever_infected = np.sum(ever_infected) / n_cells
        final = states[-1]
        frac_dead = np.sum(final == 0) / n_cells
        peak_infected = max(np.sum(s == 3) for s in states)

        return frac_ever_infected, frac_dead, peak_infected

    ny = nx = 50
    steps = 200
    p_spread = 0.6
    p_ignite = 0.02

    # sweep over survival probability with fixed vaccine coverage
    p_survive_values = np.linspace(0.0, 1.0, 21)
    p_bare_fixed = 0.2

    frac_inf_ps, frac_dead_ps, peak_inf_ps = [], [], []

    for idx, p_survive in enumerate(p_survive_values):
        states, ever_inf = simulate_disease(
            ny=ny,
            nx=nx,
            p_spread=p_spread,
            p_bare=p_bare_fixed,
            p_ignite=p_ignite,
            p_survive=float(p_survive),
            steps=steps,
            seed=idx,
        )
        fi, fd, pk = compute_disease_metrics(states, ever_inf)
        frac_inf_ps.append(fi)
        frac_dead_ps.append(fd)
        peak_inf_ps.append(pk)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    axes[0].plot(p_survive_values, frac_inf_ps)
    axes[0].set_title("Fraction ever infected vs $P_{\\text{survive}}$")
    axes[0].set_xlabel("$P_{\\text{survive}}$")
    axes[0].set_ylabel("Fraction ever infected")

    axes[1].plot(p_survive_values, frac_dead_ps)
    axes[1].set_title("Fraction dead vs $P_{\\text{survive}}$")
    axes[1].set_xlabel("$P_{\\text{survive}}$")
    axes[1].set_ylabel("Fraction dead")

    axes[2].plot(p_survive_values, peak_inf_ps)
    axes[2].set_title("Peak infected vs $P_{\\text{survive}}$")
    axes[2].set_xlabel("$P_{\\text{survive}}$")
    axes[2].set_ylabel("Peak infected")

    fig.suptitle("RQ2 — Parameter Sweep: $P_{\\text{survive}}$", fontsize=14)
    fig.savefig(os.path.join(outdir, "disease_psurvive_sweep.png"), dpi=220)
    plt.close(fig)

    # sweep over vaccine coverage with fixed survival probability
    p_bare_values = np.linspace(0.0, 1.0, 21)
    p_survive_fixed = 0.7

    frac_inf_pb, frac_dead_pb, peak_inf_pb = [], [], []

    for idx, p_bare in enumerate(p_bare_values):
        states, ever_inf = simulate_disease(
            ny=ny,
            nx=nx,
            p_spread=p_spread,
            p_bare=float(p_bare),
            p_ignite=p_ignite,
            p_survive=p_survive_fixed,
            steps=steps,
            seed=100 + idx,
        )
        fi, fd, pk = compute_disease_metrics(states, ever_inf)
        frac_inf_pb.append(fi)
        frac_dead_pb.append(fd)
        peak_inf_pb.append(pk)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    axes[0].plot(p_bare_values, frac_inf_pb)
    axes[0].set_title("Fraction ever infected vs $P_{\\text{bare}}$")
    axes[0].set_xlabel("$P_{\\text{bare}}$")
    axes[0].set_ylabel("Fraction ever infected")

    axes[1].plot(p_bare_values, frac_dead_pb)
    axes[1].set_title("Fraction dead vs $P_{\\text{bare}}$")
    axes[1].set_xlabel("$P_{\\text{bare}}$")
    axes[1].set_ylabel("Fraction dead")

    axes[2].plot(p_bare_values, peak_inf_pb)
    axes[2].set_title("Peak infected vs $P_{\\text{bare}}$")
    axes[2].set_xlabel("$P_{\\text{bare}}$")
    axes[2].set_ylabel("Peak infected")

    fig.suptitle("RQ2 — Parameter Sweep: $P_{\\text{bare}}$ (vaccine coverage)", fontsize=14)
    fig.savefig(os.path.join(outdir, "disease_pbare_sweep.png"), dpi=220)
    plt.close(fig)

    # representative time series for the disease model
    params_ts = {
        "ny": ny,
        "nx": nx,
        "p_spread": p_spread,
        "p_bare": 0.2,
        "p_ignite": p_ignite,
        "p_survive": p_survive_fixed,
        "steps": steps,
        "seed": 999,
    }
    states_ts, _ = simulate_disease(**params_ts)
    infected_ts = [np.sum(s == 3) for s in states_ts]

    plt.figure(figsize=(6, 4))
    plt.plot(infected_ts)
    plt.xlabel("Time step")
    plt.ylabel("Infected cells")
    plt.title("Disease Time Evolution (Representative Run)")
    plt.savefig(os.path.join(outdir, "disease_time_series_example.png"), dpi=220)
    plt.close()


if __name__ == "__main__":
    validation()
    run_experiment1()
    run_experiment2()
