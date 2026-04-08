"""
Generate all figures for the cross-modal refusal report from step7_summary_statistics.json.

Outputs (saved to figures/):
  1. refusal_rates.pdf        — Bar chart of refusal rates
  2. layer_dot_products.pdf    — Mean dot product per layer with std error bars
  3. layer_dot_heatmap.pdf     — Single-row heatmap of mean dot products
  4. kl_summary.pdf            — Horizontal bar summary of KL divergence stats
  5. experiment_pipeline.pdf   — Schematic of the 7-step pipeline
  6. layer_dot_range.pdf       — Min/mean/max range per layer
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ──────────────────────────────────────────────────────────────────
JSON_PATH = os.path.join("..", "main-experiment", "step7_summary_statistics.json")
OUT_DIR = os.path.join("..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Consistent neutral style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

with open(JSON_PATH, "r") as f:
    data = json.load(f)

# Neutral grayscale palette
BLACK = "#1a1a1a"
DARK = "#4a4a4a"
MID = "#7a7a7a"
LIGHT = "#aaaaaa"
VLIGHT = "#d0d0d0"
FILL = "#e8e8e8"


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Refusal Rates
# ═══════════════════════════════════════════════════════════════════════════
def fig_refusal_rates():
    rr = data["refusal_rates"]
    sc = data["sample_counts"]

    labels = ["USU\nbaseline", "SUU\nbaseline", "SUU\nablated", "SSS\nablated"]
    values = [
        rr["usu_refusal_rate"],
        rr["suu_baseline_refusal_rate"],
        rr["suu_ablated_refusal_rate"],
        rr["sss_ablated_refusal_rate"],
    ]
    counts = [
        f"{rr['usu_num_refused']}/{sc['usu_total_inferred']}",
        f"{rr['suu_num_refused']}/{sc['suu_total_inferred']}",
        f"{int(rr['suu_ablated_refusal_rate'] * sc['suu_ablated_inferred'])}/{sc['suu_ablated_inferred']}",
        f"0/{sc['sss_ablated_inferred']}",
    ]
    colors = [VLIGHT, DARK, MID, LIGHT]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar(labels, values, color=colors, edgecolor=BLACK, linewidth=0.5, width=0.55)

    for bar, count, val in zip(bars, counts, values):
        y_pos = max(bar.get_height() + 0.03, 0.05)
        ax.text(
            bar.get_x() + bar.get_width() / 2, y_pos,
            f"{val:.1%}  ({count})",
            ha="center", va="bottom", fontsize=7.5,
        )

    # Jailbreak delta bracket
    x1, x2 = 1, 2
    y1, y2 = rr["suu_baseline_refusal_rate"], rr["suu_ablated_refusal_rate"]
    bracket_x = 3.45
    ax.plot([x1 + 0.3, bracket_x], [y1, y1], color=BLACK, lw=0.6, ls=":")
    ax.plot([x2 + 0.3, bracket_x], [y2, y2], color=BLACK, lw=0.6, ls=":")
    ax.annotate(
        "", xy=(bracket_x, y2), xytext=(bracket_x, y1),
        arrowprops=dict(arrowstyle="<->", color=BLACK, lw=0.8),
    )
    mid_y = (y1 + y2) / 2
    ax.text(bracket_x + 0.08, mid_y, f"$\\Delta$ = {rr['jailbreak_success']:.0%}",
            ha="left", va="center", fontsize=8)

    ax.set_ylabel("Refusal Rate")
    ax.set_title("Refusal Rates Across Conditions")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.5, 4.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.savefig(os.path.join(OUT_DIR, "refusal_rates.pdf"))
    plt.close(fig)
    print("  Saved refusal_rates.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Layer-wise Dot Products
# ═══════════════════════════════════════════════════════════════════════════
def fig_layer_dot_products():
    la = data["layer_analysis"]
    layers = sorted(la.keys(), key=int)
    layer_ints = [int(l) for l in layers]
    means = [la[l]["dot_product_mean"] for l in layers]
    stds = [la[l]["dot_product_std"] for l in layers]
    peak = data["peak_layer"]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    colors = [BLACK if int(l) == peak["layer"] else MID for l in layers]
    ax.bar(layer_ints, means, color=colors, edgecolor=BLACK, linewidth=0.3, width=0.7)
    ax.errorbar(
        layer_ints, means, yerr=stds,
        fmt="none", ecolor=DARK, capsize=2.5, capthick=0.6, linewidth=0.6,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Dot Product")
    ax.set_title("Layer-wise Refusal Vector Alignment with Text Activations")
    ax.set_xticks(layer_ints)
    ax.axhline(y=0, color=LIGHT, linewidth=0.4, linestyle="--")

    fig.savefig(os.path.join(OUT_DIR, "layer_dot_products.pdf"))
    plt.close(fig)
    print("  Saved layer_dot_products.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def fig_layer_heatmap():
    la = data["layer_analysis"]
    layers = sorted(la.keys(), key=int)
    means = np.array([la[l]["dot_product_mean"] for l in layers]).reshape(1, -1)

    fig, ax = plt.subplots(figsize=(6.5, 1.2))
    im = ax.imshow(means, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([int(l) for l in layers])
    ax.set_yticks([])
    ax.set_xlabel("Layer")
    ax.set_title("Refusal Alignment Intensity")

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.02, pad=0.06)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(os.path.join(OUT_DIR, "layer_dot_heatmap.pdf"))
    plt.close(fig)
    print("  Saved layer_dot_heatmap.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — KL Divergence Summary
# ═══════════════════════════════════════════════════════════════════════════
def fig_kl_summary():
    kl = data["kl_divergence"]

    fig, ax = plt.subplots(figsize=(4.5, 2.2))

    stats_list = [
        ("Mean", kl["mean"]),
        ("Std", kl["std"]),
        ("Min", kl["min"]),
        ("Max", kl["max"]),
    ]
    stat_labels = [s[0] for s in stats_list]
    stat_vals = [s[1] for s in stats_list]
    y_pos = range(len(stats_list))

    ax.barh(y_pos, stat_vals, height=0.5, color=MID, edgecolor=BLACK, linewidth=0.4)
    for i, val in enumerate(stat_vals):
        ax.text(val + kl["max"] * 0.04, i, f"{val:.4f}", va="center", fontsize=8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(stat_labels)
    ax.set_xlabel("KL Divergence")
    ax.set_title("KL Divergence: Original vs. Ablated SSS")
    ax.invert_yaxis()

    fig.savefig(os.path.join(OUT_DIR, "kl_summary.pdf"))
    plt.close(fig)
    print("  Saved kl_summary.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Pipeline Schematic
# ═══════════════════════════════════════════════════════════════════════════
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    ax.set_title("Experimental Pipeline", fontsize=11, fontweight="bold", pad=12)

    # Two columns: left (data collection), right (analysis)
    steps = [
        # (x, y, label, width, height)
        (0.5, 7.2, "Step 1: USU Inference (950 samples)", 4.0, 0.7),
        (0.5, 6.0, "Step 2a: SSS Inference (150 samples)", 4.0, 0.7),
        (5.3, 6.6, "Step 2b: Compute Refusal Vector", 4.2, 0.7),
        (0.5, 4.8, "Step 2c: Ablated SSS (safety check)", 4.0, 0.7),
        (0.5, 3.6, "Step 3: SUU Baseline (150 samples)", 4.0, 0.7),
        (5.3, 4.8, "Step 4: Similarity Analysis", 4.2, 0.7),
        (5.3, 3.6, "Step 5: Ablated SUU Inference", 4.2, 0.7),
        (2.9, 2.1, "Steps 6--7: KL Divergence & Summary", 4.2, 0.7),
    ]

    shades = [FILL, FILL, VLIGHT, FILL, FILL, VLIGHT, VLIGHT, VLIGHT]

    for (x, y, label, w, h), shade in zip(steps, shades):
        box = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12",
            facecolor=shade, edgecolor=DARK, linewidth=0.8,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=7.5, color=BLACK)

    # Arrows (simple straight lines)
    arrow_kw = dict(arrowstyle="-|>", color=DARK, lw=0.9)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=arrow_kw)

    # Step 1 -> 2b
    arrow(4.5, 7.55, 5.3, 7.1)
    # Step 2a -> 2b
    arrow(4.5, 6.35, 5.3, 6.8)
    # Step 2b -> 2c
    arrow(5.3, 6.6, 4.5, 5.35)
    # Step 2b -> Step 5
    arrow(7.4, 6.6, 7.4, 4.3)
    # Step 3 -> Step 4
    arrow(4.5, 3.95, 5.3, 5.15)
    # Step 4 -> Step 5
    arrow(7.4, 4.8, 7.4, 4.3)
    # Step 5 -> 6-7
    arrow(7.1, 3.6, 5.8, 2.8)
    # Step 2c -> 6-7
    arrow(2.5, 4.8, 4.0, 2.8)

    fig.savefig(os.path.join(OUT_DIR, "experiment_pipeline.pdf"))
    plt.close(fig)
    print("  Saved experiment_pipeline.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 — Layer-wise dot product range
# ═══════════════════════════════════════════════════════════════════════════
def fig_layer_range():
    la = data["layer_analysis"]
    layers = sorted(la.keys(), key=int)
    layer_ints = [int(l) for l in layers]
    means = [la[l]["dot_product_mean"] for l in layers]
    mins = [la[l]["dot_product_min"] for l in layers]
    maxs = [la[l]["dot_product_max"] for l in layers]

    lower = [m - mi for m, mi in zip(means, mins)]
    upper = [ma - m for m, ma in zip(means, maxs)]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.errorbar(
        layer_ints, means, yerr=[lower, upper],
        fmt="o", markersize=3.5, color=BLACK, ecolor=LIGHT,
        capsize=3, capthick=0.6, linewidth=0.8, markeredgewidth=0,
    )
    ax.plot(layer_ints, means, color=DARK, linewidth=0.8, alpha=0.6)
    ax.fill_between(layer_ints, mins, maxs, alpha=0.08, color=BLACK)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Dot Product")
    ax.set_title("Per-Layer Dot Product Range (Min / Mean / Max)")
    ax.set_xticks(layer_ints)
    ax.axhline(y=0, color=LIGHT, linewidth=0.4, linestyle="--")

    fig.savefig(os.path.join(OUT_DIR, "layer_dot_range.pdf"))
    plt.close(fig)
    print("  Saved layer_dot_range.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures from", JSON_PATH)
    fig_refusal_rates()
    fig_layer_dot_products()
    fig_layer_heatmap()
    fig_kl_summary()
    fig_pipeline()
    fig_layer_range()
    print(f"\nDone. All figures saved to {OUT_DIR}/")
