"""
utils.py - All the FUNCTIONS of the project
===========================================

Each function does ONE small job. main.py calls them in order.
Parameters (folders, methods, colours...) come from config.py.

Sections:
     1. Finding files
     2. Loading images and masks
     3. Checking the data
     4. Messages and saving files
     5. STEP 2 - Colour channels
     6. STEP 3 - Smoothing (Gaussian / Fourier) + threshold
     7. STEP 4 - Clean-up
     8. STEP 5 - Metrics (Dice, IoU, HD95)
     9. STEP 6 - Recipes and ranking
    10. STEP 7 / 8 - Noise and ground-truth variability
    11. Figures
"""

import csv                                # to write tables (.csv)
import itertools                          # to build all the combinations

import numpy as np                        # arrays of numbers

import matplotlib
matplotlib.use("Agg")                     # save figures to files, never open windows
import matplotlib.pyplot as plt           # to draw figures

from scipy import ndimage                 # fill holes, distances
from skimage import io                    # read image files
from skimage import color, exposure       # colour conversions, rescaling
from skimage import filters, measure, morphology

import config                             # our parameters file


# ============================================================================
# 1. Finding files
# ============================================================================
def list_image_files(folder):
    """
    Return all the image files in `folder` (any extension listed in
    config.IMAGE_EXTENSIONS), sorted by name.
    Subfolders are searched only if config.SEARCH_SUBFOLDERS is True.
    The "figures" and "results" folders made by this program are ignored.
    """
    # glob("*")  = only this folder ; rglob("*") = this folder + all subfolders
    if config.SEARCH_SUBFOLDERS:
        candidates = folder.rglob("*")
    else:
        candidates = folder.glob("*")

    files = []
    for file in sorted(candidates):
        if config.FIGURES_FOLDER in file.parents or config.RESULTS_FOLDER in file.parents:
            continue
        # file.suffix = extension, e.g. ".jpg" ; .lower() so ".JPG" also works
        if file.is_file() and file.suffix.lower() in config.IMAGE_EXTENSIONS:
            files.append(file)
    return files


def get_group(file, data_folder):
    """
    Return the group of an image: "development" or "test".
      1. If the image is in a subfolder called development/test -> that name.
      2. Otherwise, look it up in config.GROUP_OF_IMAGE.
      3. Otherwise -> "no_group".
    """
    relative_path = file.relative_to(data_folder)   # e.g. test/lymphocyte.jpg
    if len(relative_path.parts) > 1 and relative_path.parts[0] in config.GROUPS:
        return relative_path.parts[0]
    return config.GROUP_OF_IMAGE.get(file.stem, "no_group")


def explain_empty_folder(folder):
    """Print what is inside `folder`, to understand why nothing was found."""
    print("\n[HELP] No pair found. Let's look at what is in the folder:")

    if not folder.exists():
        print(f"  The folder does not exist: {folder}")
        print("  -> Change DATA_FOLDER in config.py.")
        return

    content = sorted(folder.rglob("*"))
    if len(content) == 0:
        print("  The folder is EMPTY -> put the images and masks inside.")
        return

    for item in content[:50]:   # show at most 50 lines
        kind = "folder" if item.is_dir() else "file  "
        print(f"  {kind}  {item.relative_to(folder)}")

    print(f"  -> Check that extensions are in {config.IMAGE_EXTENSIONS}")
    print(f"     and mask names contain '{config.MASK_KEYWORD}'.")


def find_image_mask_pairs(folder=config.DATA_FOLDER):
    """
    Look in `folder` (and its subfolders) and return a list of
    (image_path, mask_path, group) triplets.

    Rule: the mask is in the SAME folder as the image, its name STARTS with
    the image name and CONTAINS config.MASK_KEYWORD.
    """
    if not folder.exists():
        explain_empty_folder(folder)
        return []

    all_files = list_image_files(folder)
    pairs = []

    for file in all_files:
        # Skip the masks themselves: we only start from the raw images
        if config.MASK_KEYWORD in file.name.lower():
            continue

        possible_masks = [
            f for f in all_files
            if f.parent == file.parent
            and f.name.startswith(file.stem)
            and config.MASK_KEYWORD in f.name.lower()
        ]

        if len(possible_masks) == 0:
            print(f"[WARNING] No mask found for {file.relative_to(folder)} -> skipped")
        else:
            pairs.append((file, possible_masks[0], get_group(file, folder)))

    if len(pairs) == 0:
        explain_empty_folder(folder)

    return pairs


# ============================================================================
# 2. Loading images and masks
# ============================================================================
def load_image(path):
    """
    Read a colour image and return an array of shape (height, width, 3).
    The 3 channels are Red, Green, Blue, with values from 0 to 255.
    """
    image = io.imread(path)
    return image[:, :, :3]   # drop a 4th "alpha" channel if there is one


def load_mask(path):
    """
    Read a hand-drawn mask and return a True/False array
    (True = nucleus = white in the file).
    Pixels brighter than config.MASK_THRESHOLD become True: this removes the
    grey pixels created by JPG compression.
    """
    mask = io.imread(path)
    if mask.ndim == 3:            # saved in colour -> keep one channel
        mask = mask[:, :, 0]
    return mask > config.MASK_THRESHOLD


# ============================================================================
# 3. Checking the data
# ============================================================================
def check_pair(image, mask, name):
    """
    Check one image/mask pair. Prints only if there is a problem.
    Returns True if everything looks fine, False otherwise.
    """
    if image.shape[:2] != mask.shape:
        print(f"[PROBLEM] {name}: image and mask do not have the same size!")
        return False
    if mask.sum() == 0:
        print(f"[PROBLEM] {name}: the mask is empty (all black)!")
        return False
    return True


# ============================================================================
# 4. Messages and saving files
# ============================================================================
def print_banner(message):
    """Print a message inside a box, to make it easy to see."""
    line = "=" * (len(message) + 4)
    print(f"\n{line}\n  {message}\n{line}")


def make_folder(folder):
    """Create a folder (and its parents) if it does not exist yet."""
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_figure(fig, folder, file_name):
    """Save a matplotlib figure as a .png in `folder`, then close it."""
    make_folder(folder)
    fig.savefig(folder / f"{file_name}.png", dpi=config.FIGURE_DPI,
                bbox_inches="tight")
    plt.close(fig)   # free the memory


def save_table(rows, path):
    """
    Save a list of dictionaries as a .csv table (opens with Excel).
    Each dictionary = one row; its keys = the column names.
    """
    make_folder(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# 5. STEP 2 - Colour channels
# ============================================================================
# Every function below returns a 2D array (height x width) with values
# between 0 and 1, where the NUCLEUS is BRIGHT. Same range and same
# "direction" for all channels -> the next steps treat them all the same way.

def channel_green_inverted(image):
    """1 - green. The nucleus absorbs green light, so it becomes bright."""
    return 1 - image[:, :, 1] / 255     # channel 1 = green


def channel_red_inverted(image):
    """1 - red. Same idea with the red channel."""
    return 1 - image[:, :, 0] / 255     # channel 0 = red


def channel_hematoxylin(image):
    """
    Amount of hematoxylin-like stain, by "colour deconvolution"
    (rgb2hed splits the image into Hematoxylin, Eosin and DAB stains).
    Our smears are May-Grunwald-Giemsa, not H&E, but the blue/purple part
    of the stain behaves similarly. The raw values are rescaled to 0-1.
    """
    hematoxylin = color.rgb2hed(image)[:, :, 0]
    return exposure.rescale_intensity(hematoxylin, out_range=(0, 1))


def channel_saturation(image):
    """Saturation (HSV colour space), already between 0 and 1."""
    return color.rgb2hsv(image)[:, :, 1]


# Channel name (as written in config.CHANNELS) -> function that computes it
CHANNEL_FUNCTIONS = {
    "green_inverted": channel_green_inverted,
    "red_inverted": channel_red_inverted,
    "hematoxylin": channel_hematoxylin,
    "saturation": channel_saturation,
}


def extract_channels(image):
    """Return a dictionary {channel name: 2D array} for config.CHANNELS."""
    return {name: CHANNEL_FUNCTIONS[name](image) for name in config.CHANNELS}


# ============================================================================
# 6. STEP 3 - Smoothing (Gaussian / Fourier) + threshold
# ============================================================================
def gaussian_blur(channel, sigma):
    """Gaussian blur. sigma = 0 means no blur (the channel is returned as is)."""
    if sigma == 0:
        return channel
    return filters.gaussian(channel, sigma=sigma)


def frequency_grid(shape):
    """
    For every pixel of the Fourier spectrum, its frequency (distance to the
    centre of the spectrum), in cycles/pixel: 0 = centre, 0.5 = corners of
    the axes. Same layout as np.fft.fft2 (NOT shifted).
    """
    fy = np.fft.fftfreq(shape[0])          # vertical frequencies
    fx = np.fft.fftfreq(shape[1])          # horizontal frequencies
    return np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def butterworth_lowpass(frequencies, cutoff, order=config.FFT_ORDER):
    """
    Smooth low-pass filter: ~1 below `cutoff`, ~0 above it.
    (A "hard" cut at exactly the cutoff creates ripples in the image,
    Butterworth avoids that; `order` sets how steep the edge is.)
    """
    return 1 / (1 + (frequencies / cutoff) ** (2 * order))


def fft_bandpass(channel, low_cutoff=None, high_cutoff=None):
    """
    Fourier band-pass filter:
      1. go to the frequency domain (fft2),
      2. multiply by a filter that keeps frequencies between low and high,
      3. come back to the image (ifft2),
      4. rescale to 0-1 so it can be compared with the other channels.
    Default cutoffs come from config.py.
    """
    if low_cutoff is None:
        low_cutoff = config.FFT_LOW_CUTOFF
    if high_cutoff is None:
        high_cutoff = config.FFT_HIGH_CUTOFF

    frequencies = frequency_grid(channel.shape)
    band = butterworth_lowpass(frequencies, high_cutoff)              # remove high freqs
    if low_cutoff > 0:
        band = band * (1 - butterworth_lowpass(frequencies, low_cutoff))  # remove low freqs

    spectrum = np.fft.fft2(channel)
    filtered = np.real(np.fft.ifft2(spectrum * band))
    return exposure.rescale_intensity(filtered, out_range=(0, 1))


def smooth(channel, smoothing):
    """
    Apply one smoothing option:
      a number (0, 1, 2...) -> Gaussian blur with that sigma
      "fft"                 -> Fourier band-pass with the cutoffs of config.py
    """
    if smoothing == "fft":
        return fft_bandpass(channel)
    return gaussian_blur(channel, smoothing)


def smoothing_name(smoothing):
    """Readable name: 'sigma=1' or 'fft'."""
    return "fft" if smoothing == "fft" else f"sigma={smoothing}"


def threshold_otsu(channel):
    """Otsu: one threshold, nucleus = pixels above it."""
    return channel > filters.threshold_otsu(channel)


def threshold_multi_otsu(channel):
    """
    Multi-Otsu: N_CLASSES classes (e.g. background / cytoplasm / nucleus).
    The nucleus is the brightest class = above the LAST threshold.
    """
    thresholds = filters.threshold_multiotsu(channel, classes=config.N_CLASSES)
    return channel > thresholds[-1]


def threshold_kmeans(channel, n_iterations=20):
    """
    K-means on the pixel intensities, written by hand (no extra library).
    1. Start with N_CLASSES "centres" spread over the intensities.
    2. Give each pixel to the closest centre.
    3. Move each centre to the mean of its pixels. Repeat.
    The nucleus is the cluster with the brightest centre.
    """
    values = channel.ravel()   # all pixels in one long list
    # starting centres: e.g. 3 classes -> the 1/6, 3/6, 5/6 quantiles
    centres = np.quantile(values, (np.arange(config.N_CLASSES) + 0.5) / config.N_CLASSES)

    for _ in range(n_iterations):
        # distance of every pixel to every centre -> index of the closest one
        labels = np.argmin(np.abs(values[:, None] - centres[None, :]), axis=1)
        for k in range(config.N_CLASSES):
            if np.any(labels == k):
                centres[k] = values[labels == k].mean()

    brightest = np.argmax(centres)
    return (labels == brightest).reshape(channel.shape)


# Threshold name (as written in config.THRESHOLD_METHODS) -> function
THRESHOLD_FUNCTIONS = {
    "otsu": threshold_otsu,
    "multi_otsu": threshold_multi_otsu,
    "kmeans": threshold_kmeans,
}


def smooth_and_threshold(channel, smoothing, method):
    """Smooth the channel, then threshold it. Returns a True/False mask."""
    return THRESHOLD_FUNCTIONS[method](smooth(channel, smoothing))


# ============================================================================
# 7. STEP 4 - Clean-up
# ============================================================================
def opening(mask, radius):
    """
    Opening = erosion then dilation with a disk of `radius` pixels.
    Removes white specks smaller than the disk, keeps the big shapes.
    """
    return ndimage.binary_opening(mask, structure=morphology.disk(radius))


def fill_holes(mask):
    """Fill the black holes completely surrounded by white."""
    return ndimage.binary_fill_holes(mask)


def keep_main_object(mask):
    """
    Keep the biggest white object, plus any other object at least
    config.KEEP_OBJECT_MIN_FRACTION of its size (separated neutrophil lobes).
    """
    labels = measure.label(mask)              # 0 = background, 1, 2, 3 = objects
    if labels.max() == 0:
        return mask                           # nothing to keep
    sizes = np.bincount(labels.ravel())       # number of pixels of each label
    sizes[0] = 0                              # ignore the background
    keep = sizes >= config.KEEP_OBJECT_MIN_FRACTION * sizes.max()
    keep[0] = False
    return keep[labels]


def clean_up(mask, opening_radius):
    """The 3 clean-up stages in order. Returns the final mask."""
    mask = opening(mask, opening_radius)
    mask = fill_holes(mask)
    return keep_main_object(mask)


def clean_up_stages(mask, opening_radius):
    """Same as clean_up, but returns every intermediate mask (for figures)."""
    stages = {"threshold": mask}
    stages["opening"] = opening(mask, opening_radius)
    stages["fill holes"] = fill_holes(stages["opening"])
    stages["main object"] = keep_main_object(stages["fill holes"])
    return stages


# ============================================================================
# 8. STEP 5 - Metrics (Dice, IoU, HD95)
# ============================================================================
def dice(prediction, reference):
    """
    Dice = 2 x |overlap| / (|prediction| + |reference|).
    1 = perfect, 0 = no overlap.
    """
    overlap = np.logical_and(prediction, reference).sum()
    total = prediction.sum() + reference.sum()
    return 2 * overlap / total if total > 0 else 1.0


def iou(prediction, reference):
    """
    IoU (Jaccard) = |overlap| / |union|.  1 = perfect, 0 = no overlap.
    Always lower than Dice for the same masks.
    """
    overlap = np.logical_and(prediction, reference).sum()
    union = np.logical_or(prediction, reference).sum()
    return overlap / union if union > 0 else 1.0


def border(mask):
    """Pixels of the mask that touch the background (the outline)."""
    return mask & ~ndimage.binary_erosion(mask)


def hd95(prediction, reference):
    """
    95th percentile Hausdorff distance, in PIXELS (pixel size is unknown).
    For every outline pixel of each mask, measure the distance to the
    closest outline pixel of the other mask; HD95 = the 95th percentile of
    all these distances. 0 = perfect; lower = better.
    Returns NaN (not a number) if one of the masks is empty.
    """
    if prediction.sum() == 0 or reference.sum() == 0:
        return float("nan")

    border_pred = border(prediction)
    border_ref = border(reference)

    # distance_transform_edt gives, for every pixel, the distance to the
    # closest 0-pixel -> we use "not border" so the 0-pixels are the border
    distance_to_ref = ndimage.distance_transform_edt(~border_ref)
    distance_to_pred = ndimage.distance_transform_edt(~border_pred)

    distances = np.concatenate([distance_to_ref[border_pred],
                                distance_to_pred[border_ref]])
    return float(np.percentile(distances, 95))


def compute_metrics(prediction, reference):
    """Return the 3 metrics in a dictionary."""
    return {"dice": dice(prediction, reference),
            "iou": iou(prediction, reference),
            "hd95": hd95(prediction, reference)}


# ============================================================================
# 9. STEP 6 - Recipes and ranking
# ============================================================================
def all_recipes():
    """
    Build every combination of channel x smoothing x threshold x opening radius.
    Each recipe is a dictionary, e.g.
        {"channel": "hematoxylin", "smoothing": 1, "threshold": "otsu", "opening_radius": 2}
    """
    recipes = []
    for ch, sm, th, r in itertools.product(config.CHANNELS, config.SMOOTHINGS,
                                           config.THRESHOLD_METHODS,
                                           config.OPENING_RADII):
        recipes.append({"channel": ch, "smoothing": sm,
                        "threshold": th, "opening_radius": r})
    return recipes


def recipe_name(recipe):
    """Short readable name, e.g. 'hematoxylin | sigma=1 | otsu | r=2'."""
    return (f"{recipe['channel']} | {smoothing_name(recipe['smoothing'])} | "
            f"{recipe['threshold']} | r={recipe['opening_radius']}")


def segment(channels, recipe):
    """Apply one recipe to the channels of one image -> final mask."""
    mask = smooth_and_threshold(channels[recipe["channel"]],
                                recipe["smoothing"], recipe["threshold"])
    return clean_up(mask, recipe["opening_radius"])


def mean_score_per_recipe(results, group, metric):
    """
    From the big results table, compute for each recipe the mean of `metric`
    over the images of `group`. Returns a list of (recipe name, mean),
    sorted from best to worst.
    """
    scores = {}
    for row in results:
        if row["group"] == group:
            scores.setdefault(row["recipe"], []).append(row[metric])
    means = [(name, float(np.mean(values))) for name, values in scores.items()]
    # higher is better for dice / iou
    return sorted(means, key=lambda item: item[1], reverse=True)


def recipe_from_name(name):
    """Find the recipe dictionary that has this name."""
    return next(r for r in all_recipes() if recipe_name(r) == name)


# ============================================================================
# 10. STEP 7 / 8 - Noise and ground-truth variability
# ============================================================================
def add_noise(image, noise_std, rng):
    """
    Add Gaussian noise to a colour image.
    noise_std is relative to the 0-1 range (0.1 = 10% of the full range).
    rng = random number generator (np.random.default_rng), for reproducibility.
    Returns an image back in 0-255 (uint8), like a normal image file.
    """
    noisy = image / 255 + rng.normal(0, noise_std, image.shape)
    noisy = np.clip(noisy, 0, 1)                # stay inside 0-1
    return (noisy * 255).astype(np.uint8)


def change_mask(mask, pixels):
    """
    Simulate another person tracing the nucleus:
      pixels > 0 -> mask grown by that many pixels  (dilation)
      pixels < 0 -> mask shrunk by that many pixels (erosion)
      pixels = 0 -> unchanged
    """
    if pixels == 0:
        return mask
    disk = morphology.disk(abs(pixels))
    if pixels > 0:
        return ndimage.binary_dilation(mask, structure=disk)
    return ndimage.binary_erosion(mask, structure=disk)


# ============================================================================
# 11. Figures
# ============================================================================
def draw_outlines(ax, image, reference=None, prediction=None):
    """Show the image with the reference (green) and/or algorithm (red) outlines."""
    ax.imshow(image)
    if reference is not None:
        ax.contour(reference, levels=[0.5], colors=config.REFERENCE_COLOR,
                   linewidths=config.CONTOUR_WIDTH)
    if prediction is not None and prediction.any():
        ax.contour(prediction, levels=[0.5], colors=config.ALGORITHM_COLOR,
                   linewidths=config.CONTOUR_WIDTH)
    ax.axis("off")


def figure_step1(image, mask, title):
    """Image | hand-drawn mask | overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Hand-drawn mask")
    draw_outlines(axes[2], image, reference=mask)
    axes[2].set_title("Overlay (green = reference)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    return fig


def figure_step2(image, mask, channels, title):
    """Image followed by every channel, with the reference outline."""
    fig, axes = plt.subplots(1, 1 + len(channels), figsize=(3 * (1 + len(channels)), 3.5))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    for ax, (name, channel) in zip(axes[1:], channels.items()):
        ax.imshow(channel, cmap=config.CHANNEL_COLORMAP, vmin=0, vmax=1)
        ax.contour(mask, levels=[0.5], colors=config.REFERENCE_COLOR,
                   linewidths=config.CONTOUR_WIDTH)
        ax.set_title(name)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    return fig


def figure_step3(masks, mask_ref, title):
    """
    Grid of the thresholded masks (before clean-up):
    one row per channel, one column per (smoothing, threshold method).
    `masks` is a dictionary {(channel, smoothing, method): mask}.
    """
    columns = list(itertools.product(config.SMOOTHINGS, config.THRESHOLD_METHODS))
    fig, axes = plt.subplots(len(config.CHANNELS), len(columns),
                             figsize=(1.8 * len(columns), 1.9 * len(config.CHANNELS)),
                             squeeze=False)
    for i, ch in enumerate(config.CHANNELS):
        for j, (s, th) in enumerate(columns):
            ax = axes[i, j]
            ax.imshow(masks[(ch, s, th)], cmap="gray")
            ax.contour(mask_ref, levels=[0.5], colors=config.REFERENCE_COLOR, linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(f"{smoothing_name(s)}\n{th}", fontsize=8)
            if j == 0:
                ax.set_ylabel(ch, fontsize=8)
    fig.suptitle(title)
    return fig


def figure_step4(image, stages, mask_ref, title):
    """The clean-up stages, one panel each, for the example recipe."""
    fig, axes = plt.subplots(1, 1 + len(stages), figsize=(3 * (1 + len(stages)), 3.5))
    axes[0].imshow(image)
    axes[0].set_title("Image")
    for ax, (name, mask) in zip(axes[1:], stages.items()):
        ax.imshow(mask, cmap="gray")
        ax.contour(mask_ref, levels=[0.5], colors=config.REFERENCE_COLOR, linewidths=1)
        ax.set_title(name)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    return fig


def figure_ranking(ranking, title, n_shown=20):
    """Horizontal bar chart of the mean score of the best recipes."""
    shown = ranking[:n_shown][::-1]   # best at the top
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(shown) + 1))
    ax.barh([name for name, _ in shown], [score for _, score in shown], color="steelblue")
    ax.set_xlabel(f"mean {config.RANKING_METRIC}")
    ax.set_xlim(0, 1)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title)
    return fig


def figure_all_recipes(ranking, best_name, title):
    """
    Mean score of ALL the recipes, sorted from best to worst.
    Shows that most recipes are NOT good: only the top ones are close to 1.
    """
    scores = [score for _, score in ranking]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(scores) + 1), scores, marker="o", markersize=3, color="steelblue")
    ax.plot(1, scores[0], marker="o", markersize=8, color=config.ALGORITHM_COLOR,
            label=f"best: {best_name}")
    ax.set_xlabel("recipe rank (1 = best)")
    ax.set_ylabel(f"mean {config.RANKING_METRIC} (development)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(title)
    return fig


def figure_parameter_effects(results, group, metric):
    """
    One box plot per parameter (channel, smoothing, threshold, opening radius).
    Each box = all the scores obtained WITH that option, on the images of `group`.
    Shows which choices make the segmentation good or bad.
    """
    # recipe names look like "green_inverted | sigma=0 | multi_otsu | r=2"
    parameter_names = ["channel", "smoothing", "threshold", "opening radius"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for i, (ax, parameter) in enumerate(zip(axes, parameter_names)):
        values_per_option = {}
        for row in results:
            if row["group"] == group:
                option = row["recipe"].split(" | ")[i]
                values_per_option.setdefault(option, []).append(row[metric])
        options = list(values_per_option.keys())
        ax.boxplot([values_per_option[o] for o in options])
        ax.set_xticks(range(1, len(options) + 1), options, rotation=30, fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, axis="y")
        ax.set_title(parameter)
    axes[0].set_ylabel(f"{metric} ({group} images)")
    fig.suptitle("Effect of each choice on the score (each box = every recipe using that option)")
    plt.tight_layout()
    return fig


def figure_result(image, reference, prediction, scores, title):
    """Image with reference (green) and algorithm (red) outlines + scores."""
    fig, ax = plt.subplots(figsize=(5, 5.4))
    draw_outlines(ax, image, reference, prediction)
    ax.set_title(f"{title}\nDice={scores['dice']:.3f}  IoU={scores['iou']:.3f}  "
                 f"HD95={scores['hd95']:.1f} px", fontsize=9)
    return fig


def figure_score_table(rows, title):
    """A table of scores drawn as a figure (to paste in the slides)."""
    columns = ["image", "group", "dice", "iou", "hd95"]
    cells = [[row["image"], row["group"], f"{row['dice']:.3f}",
              f"{row['iou']:.3f}", f"{row['hd95']:.1f}"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(rows) + 0.8))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=columns, loc="center",
                     colWidths=[0.44, 0.14, 0.14, 0.14, 0.14])   # wide name column
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title(title)
    return fig


def figure_curves(x_values, curves, x_label, y_label, title):
    """
    Line plot with one line per recipe.
    `curves` is a dictionary {line name: list of y values (one per x value)}.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, y_values in curves.items():
        ax.plot(x_values, y_values, marker="o", label=name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title(title)
    return fig


def figure_examples(images, references, predictions, titles, suptitle):
    """A row of images with reference (green) and algorithm (red) outlines."""
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3.4))
    for ax, image, ref, pred, title in zip(axes, images, references, predictions, titles):
        draw_outlines(ax, image, ref, pred)
        ax.set_title(title, fontsize=8)
    fig.suptitle(suptitle)
    return fig