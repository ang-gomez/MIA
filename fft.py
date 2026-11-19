"""
fft.py - Look at the Fourier spectrum and CHOOSE the frequency band
===============================================================================

Run this BEFORE main.py, and only when you want to (re)choose the band.
It uses ONLY the DEVELOPMENT images (choosing on test would be cheating).

What it does:
    1. For each development image: shows the channel, its Fourier spectrum and
       how the energy is spread over the frequencies.
    2. Shows the channel filtered with several cut-off frequencies, so you
       can SEE what each band keeps and what it removes.
    3. Criterion to choose: for every band (low cut-off x high cut-off) it
       segments the development images and measures the mean Dice.
       The band with the best mean Dice is suggested.

Then: write the chosen values in config.py
    FFT_LOW_CUTOFF  = ...
    FFT_HIGH_CUTOFF = ...
and run main.py, which compares the Fourier filter with the other methods.

Figures are saved in figures/fourier_explore/
"""

import numpy as np
import matplotlib.pyplot as plt

import config
import utils


# ============================================================================
# Settings of this exploration (only used here)
# ============================================================================
CHANNEL_TO_SHOW = "hematoxylin"    # channel used for the spectrum figures

# Cut-off frequencies tested, in cycles/pixel (0.5 = finest possible detail)
LOW_CUTOFFS = [0, 0.005, 0.01, 0.02]                        # remove BELOW (0 = nothing)
HIGH_CUTOFFS = [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4]  # remove ABOVE

# Opening radius used for the clean-up while testing the bands
TEST_OPENING_RADIUS = 2

FOLDER = config.FIGURES_FOLDER / "fourier_explore"


# ============================================================================
# Functions
# ============================================================================
def radial_profile(values, frequencies, n_bins=100):
    """
    Average of `values` on rings of equal frequency (centre -> edge).
    Turns the 2D spectrum into a 1D curve: energy vs frequency.
    """
    bins = np.linspace(0, frequencies.max(), n_bins + 1)
    ring = np.digitize(frequencies.ravel(), bins) - 1          # ring index of every pixel
    sums = np.bincount(ring, weights=values.ravel(), minlength=n_bins + 1)[:n_bins]
    counts = np.bincount(ring, minlength=n_bins + 1)[:n_bins]
    centres = (bins[:-1] + bins[1:]) / 2
    return centres, sums / np.maximum(counts, 1), sums


def figure_spectrum(channel, title):
    """Channel | log spectrum (centred) | energy vs frequency."""
    spectrum = np.fft.fft2(channel)
    power = np.abs(spectrum) ** 2
    frequencies = utils.frequency_grid(channel.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].imshow(channel, cmap="gray")
    axes[0].set_title(f"Channel: {CHANNEL_TO_SHOW}")
    axes[0].axis("off")

    # fftshift puts the frequency 0 in the CENTRE of the picture
    # log: the centre is millions of times stronger than the rest
    axes[1].imshow(np.log1p(np.fft.fftshift(np.abs(spectrum))), cmap="magma",
                   extent=[-0.5, 0.5, 0.5, -0.5])
    for cutoff in HIGH_CUTOFFS:
        axes[1].add_patch(plt.Circle((0, 0), cutoff, fill=False, color="cyan", lw=0.6))
    axes[1].set_title("log |FFT| (centre = low freq)\ncircles = tested high cut-offs")
    axes[1].set_xlabel("cycles/pixel")

    freqs, mean_power, ring_power = radial_profile(power, frequencies)
    # cumulative energy WITHOUT the frequency 0 (= the mean brightness)
    ring_power[0] = 0
    cumulative = np.cumsum(ring_power) / ring_power.sum() * 100
    axes[2].semilogy(freqs, mean_power, color="black")
    axes[2].set_xlabel("frequency (cycles/pixel)")
    axes[2].set_ylabel("mean power (log scale)")
    twin = axes[2].twinx()
    twin.plot(freqs, cumulative, color="tab:blue")
    twin.set_ylabel("cumulative energy (%)", color="tab:blue")
    twin.set_ylim(0, 101)
    for level in [90, 99]:
        f = freqs[np.argmax(cumulative >= level)]
        twin.axvline(f, color="tab:blue", ls="--", lw=0.8)
        twin.text(f, level - 8, f" {level}% at {f:.2f}", color="tab:blue", fontsize=8)
    axes[2].set_title("Energy vs frequency")

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def figure_filtered(channel, mask, title):
    """The channel low-pass filtered at every tested high cut-off."""
    fig, axes = plt.subplots(1, 1 + len(HIGH_CUTOFFS), figsize=(2.3 * (1 + len(HIGH_CUTOFFS)), 3))
    axes[0].imshow(channel, cmap="gray")
    axes[0].set_title("original", fontsize=9)
    for ax, cutoff in zip(axes[1:], HIGH_CUTOFFS):
        ax.imshow(utils.fft_bandpass(channel, 0, cutoff), cmap="gray")
        ax.set_title(f"keep < {cutoff}", fontsize=9)
    for ax in axes:
        ax.contour(mask, levels=[0.5], colors=config.REFERENCE_COLOR, linewidths=0.6)
        ax.axis("off")
    fig.suptitle(title)
    return fig


def band_dice(dev_data, low, high):
    """
    Mean Dice on the development images for one band, averaged over every
    channel and threshold method (= how much this band helps in general).
    """
    scores = []
    for d in dev_data:
        for channel in config.CHANNELS:
            filtered = utils.fft_bandpass(d["channels"][channel], low, high)
            for method in config.THRESHOLD_METHODS:
                mask = utils.THRESHOLD_FUNCTIONS[method](filtered)
                mask = utils.clean_up(mask, TEST_OPENING_RADIUS)
                scores.append(utils.dice(mask, d["mask"]))
    return float(np.mean(scores))


def figure_band_scores(scores, best):
    """Heatmap: one cell per (low cut-off, high cut-off), colour = mean Dice."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    image = ax.imshow(scores, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(HIGH_CUTOFFS)), [str(h) for h in HIGH_CUTOFFS])
    ax.set_yticks(range(len(LOW_CUTOFFS)), [str(l) for l in LOW_CUTOFFS])
    ax.set_xlabel("high cut-off (remove ABOVE)")
    ax.set_ylabel("low cut-off (remove BELOW)")
    for i in range(len(LOW_CUTOFFS)):
        for j in range(len(HIGH_CUTOFFS)):
            ax.text(j, i, f"{scores[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="red" if (i, j) == best else "white")
    fig.colorbar(image, label="mean Dice (development)")
    ax.set_title("Which band segments best? (red = suggested)")
    return fig


# ============================================================================
# Main program
# ============================================================================
def main():
    # Load the DEVELOPMENT images only
    dev_data = []
    for image_path, mask_path, group in utils.find_image_mask_pairs(config.DATA_FOLDER):
        if group == "development":
            image = utils.load_image(image_path)
            dev_data.append({"name": image_path.stem, "image": image,
                               "mask": utils.load_mask(mask_path),
                               "channels": utils.extract_channels(image)})
    if len(dev_data) == 0:
        print("[PROBLEM] No development image: check GROUP_OF_IMAGE in config.py.")
        return
    utils.print_banner(f"FOURIER - {len(dev_data)} development images loaded")

    # 1 + 2. Look at the spectrum and at the filtered images
    for d in dev_data:
        channel = d["channels"][CHANNEL_TO_SHOW]
        utils.save_figure(figure_spectrum(channel, d["name"]), FOLDER, f"spectrum_{d['name']}")
        utils.save_figure(figure_filtered(channel, d["mask"], d["name"]),
                          FOLDER, f"filtered_{d['name']}")
    utils.print_banner("FOURIER - spectrum and filtered images saved")

    # 3. Score every band
    print("Testing every band (this takes about a minute)...")
    scores = np.zeros((len(LOW_CUTOFFS), len(HIGH_CUTOFFS)))
    for i, low in enumerate(LOW_CUTOFFS):
        for j, high in enumerate(HIGH_CUTOFFS):
            scores[i, j] = band_dice(dev_data, low, high)

    i, j = np.unravel_index(np.argmax(scores), scores.shape)   # position of the best score
    utils.save_figure(figure_band_scores(scores, (i, j)), FOLDER, "band_scores")

    rows = [{"low_cutoff": low, "high_cutoff": high, "mean_dice_development": scores[a, b]}
            for a, low in enumerate(LOW_CUTOFFS) for b, high in enumerate(HIGH_CUTOFFS)]
    utils.save_table(rows, config.RESULTS_FOLDER / "fourier_band_scores.csv")

    utils.print_banner("FOURIER - band scores computed")
    print(f"Suggested band (best mean Dice on development = {scores[i, j]:.3f}):")
    print(f"    FFT_LOW_CUTOFF  = {LOW_CUTOFFS[i]}")
    print(f"    FFT_HIGH_CUTOFF = {HIGH_CUTOFFS[j]}")
    print("-> Look at the figures in figures/fourier_explore/, then write your")
    print("   choice in config.py and run main.py.")


if __name__ == "__main__":
    main()