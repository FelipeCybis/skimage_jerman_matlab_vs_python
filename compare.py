# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
"""Compare the MATLAB Jerman output with the two scikit-image variants.

Run the three implementation scripts first, then::

    uv run compare.py

Writes ``fig_maps.png`` and ``fig_stats.png`` next to this file.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = Path(__file__).parent
matlab = np.load(HERE / "retina_jerman_original_implementation.npy")
variants = {
    "scikit-image (Gaussian derivatives)": np.load(
        HERE / "retina_jerman_skimage_implementation.npy"
    ),
    "scikit-image (finite differences)": np.load(
        HERE / "retina_jerman_skimage_finite_differences.npy"
    ),
}

# Colors: one hue for magnitude, two hues + neutral midpoint for signed difference,
# fixed hue per variant in the histogram.
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#dedcd5"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f7f4", BLUE])
DIV = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efe9", ORANGE])
SERIES = [BLUE, ORANGE]

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.titlecolor": MUTED,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
    }
)


def summarize(out):
    d = np.abs(out - matlab)
    return {
        "mean |diff|": d.mean(),
        "99th pct |diff|": np.percentile(d, 99),
        "|diff| > 0.05": np.mean(d > 0.05),
        "|diff| > 0.20": np.mean(d > 0.20),
        "corr": np.corrcoef(matlab.ravel(), out.ravel())[0, 1],
    }


def bare(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)


def colorbar(fig, im, ax):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=MUTED)


stats = {name: summarize(out) for name, out in variants.items()}
print(f"{'':36s}" + "".join(f"{k:>16s}" for k in next(iter(stats.values()))))
for name, st in stats.items():
    print(f"{name:36s}" + "".join(f"{v:16.5f}" for v in st.values()))

# --- figure 1: maps on a vessel-rich crop ----------------------------------
# Top row: MATLAB and the two variants. Bottom row: each variant minus MATLAB on a
# shared diverging scale, so the two difference maps are directly comparable.
crop = (slice(300, 600), slice(600, 900))
vmax_diff = max(
    np.percentile(np.abs(out[crop] - matlab[crop]), 99.9) for out in variants.values()
)

fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
im = axes[0, 0].imshow(matlab[crop], cmap=SEQ, vmin=0, vmax=1)
bare(axes[0, 0], "MATLAB (Jerman)")
colorbar(fig, im, axes[0, 0])
bare(axes[1, 0])
axes[1, 0].text(
    0.0,
    0.95,
    "\n\n".join(
        f"{name}\n"
        f"  mean |diff| = {st['mean |diff|']:.4f}\n"
        f"  |diff| > 0.05: {st['|diff| > 0.05']:.2%} of pixels\n"
        f"  corr = {st['corr']:.5f}"
        for name, st in stats.items()
    ),
    transform=axes[1, 0].transAxes,
    ha="left",
    va="top",
    fontsize=8,
    color=INK,
)
for col, (name, out) in enumerate(variants.items(), start=1):
    im = axes[0, col].imshow(out[crop], cmap=SEQ, vmin=0, vmax=1)
    bare(axes[0, col], name)
    colorbar(fig, im, axes[0, col])
    im = axes[1, col].imshow(
        out[crop] - matlab[crop], cmap=DIV, vmin=-vmax_diff, vmax=vmax_diff
    )
    bare(axes[1, col], f"{name} − MATLAB")
    colorbar(fig, im, axes[1, col])
fig.suptitle(
    f"retina, sigmas 0.5:0.5:2.5, tau=1, crop rows {crop[0].start}:{crop[0].stop}, "
    f"cols {crop[1].start}:{crop[1].stop}",
    color=MUTED,
    fontsize=9,
)
fig.tight_layout()
fig.savefig(HERE / "fig_maps.png", bbox_inches="tight")

# --- figure 2: distributions -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
ax_hist, ax_joints = axes[0], axes[1:]

# Histogram of signed differences, log counts so the tails are visible.
bins = np.linspace(-1, 1, 201)
for (name, out), color in zip(variants.items(), SERIES):
    ax_hist.hist(
        (out - matlab).ravel(), bins=bins, histtype="step", color=color, lw=1.5,
        label=name,
    )
ax_hist.set_yscale("log")
ax_hist.set_xlabel("variant − MATLAB")
ax_hist.set_ylabel("pixels")
ax_hist.set_title("signed difference per pixel")
ax_hist.axvline(0, color=GRID, lw=1, zorder=0)
ax_hist.legend(frameon=False, fontsize=8, loc="upper right")

# Joint histograms: where along the response range do they disagree?
for ax, (name, out) in zip(ax_joints, variants.items()):
    h, xe, ye = np.histogram2d(
        matlab.ravel(), out.ravel(), bins=100, range=[[0, 1], [0, 1]]
    )
    im = ax.pcolormesh(xe, ye, h.T, cmap=SEQ, norm=LogNorm(vmin=1, vmax=h.max()))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, zorder=3)
    ax.set_xlabel("MATLAB response")
    ax.set_ylabel(name)
    ax.set_title(f"joint distribution, corr = {stats[name]['corr']:.5f}")
    ax.set_aspect("equal")
    colorbar(fig, im, ax)

for ax in axes:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig(HERE / "fig_stats.png", bbox_inches="tight")
print("wrote fig_maps.png and fig_stats.png")
