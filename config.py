"""
config.py - All the PARAMETERS of the project, in one place
===========================================================

This file contains no code that "does" something: only settings.
If you want to change a folder, a parameter or the list of methods to test,
you change it HERE, and main.py / utils.py stay the same.
"""
 
from pathlib import Path

# ============================================================================
# Folders
# ============================================================================
# Path(__file__).parent = the folder where this config.py file is,
# so the code works no matter from where you run it.
PROJECT_FOLDER = Path(__file__).parent

# Folder with the raw images AND the hand-drawn masks
DATA_FOLDER = PROJECT_FOLDER            # images are next to the .py files
# DATA_FOLDER = PROJECT_FOLDER / "data"  # use this line instead if you move
#                                        # the images into a "data" folder

# False = look only at the files directly in DATA_FOLDER (ignore subfolders)
# True  = also look inside subfolders (e.g. development/ test/)
SEARCH_SUBFOLDERS = False

# Where the figures are saved (one subfolder per step, created automatically)
FIGURES_FOLDER = PROJECT_FOLDER / "figures"

# Where the tables of results (.csv, open them with Excel) are saved
RESULTS_FOLDER = PROJECT_FOLDER / "results"

# ============================================================================
# Files
# ============================================================================
# Word that appears in the mask file names
# (e.g. "lymphocyte__1604253483_036annotated1.jpg")
MASK_KEYWORD = "annotated"

# Accepted image extensions (upper/lower case does not matter)
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]

# JPG compression creates grey pixels at the edges of the mask.
# Every pixel brighter than this value (0-255) is considered "nucleus".
MASK_THRESHOLD = 127

# ============================================================================
# Groups: development / test
# ============================================================================
# Nothing is "trained": the groups only say which images are used to CHOOSE
# the best recipe (development) and which ones show the final result (test).
# All 9 images go through every step; the test scores are just never looked
# at when choosing.
GROUPS = ["development", "test"]

# Which group each image belongs to (from the "Data" slide of the presentation:
# the former train + validation images = development).
# Used when the images are NOT sorted into development/test subfolders.
GROUP_OF_IMAGE = {
    "lymphocyte__1604332836_052": "development",
    "monocyte__1604332699_048": "development",
    "segmented_neutrophil__1604315954_071": "development",
    "lymphocyte__1604368723_118": "development",
    "monocyte__1604276339_056": "development",
    "segmented_neutrophil__1604313532_014": "development",

    "lymphocyte__1604253483_036": "test",
    "monocyte__1604323433_090": "test",
    "segmented_neutrophil__1604369288_032": "test",
}

# ============================================================================
# STEP 2 - Colour channels
# ============================================================================
# The 4 ways of turning the colour image into a single grey image.
# In all of them the NUCLEUS appears BRIGHT (high values, between 0 and 1).
#   "green_inverted" : 1 - green  (the purple nucleus absorbs a lot of green)
#   "red_inverted"   : 1 - red    (the nucleus also absorbs red)
#   "hematoxylin"    : amount of hematoxylin-like (blue/purple) stain
#   "saturation"     : how "intense" the colour is (nucleus = very saturated)
CHANNELS = ["green_inverted", "red_inverted", "hematoxylin", "saturation"]

# ============================================================================
# STEP 3 - Smoothing (Gaussian or Fourier) + threshold
# ============================================================================
# Gaussian blur strengths to test. 0 = no blur.
SIGMAS = [0, 1, 2]

# Fourier (FFT) band-pass filter, tested as one more smoothing option.
# Frequencies are in cycles/pixel: 0 = flat background, 0.5 = finest detail.
#   FFT_LOW_CUTOFF  : frequencies BELOW are removed (0 = keep the background)
#   FFT_HIGH_CUTOFF : frequencies ABOVE are removed (small = strong blur)
# -> Choose them by running fft.py, then write them here.
USE_FFT_FILTER = True
FFT_LOW_CUTOFF = 0.0
FFT_HIGH_CUTOFF = 0.08   # chosen with fft.py (best mean Dice on development)
FFT_ORDER = 2   # steepness of the filter edge (Butterworth order)

# All the smoothing options tested (built automatically from the lines above):
# e.g. [0, 1, 2, "fft"]
SMOOTHINGS = SIGMAS + (["fft"] if USE_FFT_FILTER else [])

# How to separate nucleus from the rest:
#   "otsu"       : 1 threshold -> 2 classes (nucleus / rest)
#   "multi_otsu" : 2 thresholds -> 3 classes (background / cytoplasm+RBC / nucleus)
#   "kmeans"     : groups the pixels into 3 clusters by intensity
THRESHOLD_METHODS = ["otsu", "multi_otsu", "kmeans"]

# Number of classes used by multi_otsu and kmeans
N_CLASSES = 3

# ============================================================================
# STEP 4 - Clean-up
# ============================================================================
# Radius (in pixels) of the "opening" that removes small white specks.
# 3 values tested -> 4 channels x 3 sigmas x 3 thresholds x 3 radii = 108 recipes
# (+ the FFT option: 4 channels x 4 smoothings x 3 thresholds x 3 radii = 144)
OPENING_RADII = [1, 2, 3]

# "Keep main object": keep the biggest white object, plus any other object
# that is at least this fraction of its size (so the separated lobes of a
# neutrophil are not thrown away). 1.0 = keep ONLY the biggest object.
KEEP_OBJECT_MIN_FRACTION = 0.2

# Recipe used to illustrate the clean-up stages in the step 4 figures
EXAMPLE_RECIPE = {"channel": "hematoxylin", "smoothing": 1,
                  "threshold": "otsu", "opening_radius": 2}

# ============================================================================
# STEP 6 - Choosing the best recipe
# ============================================================================
# Metric used to rank the recipes ("dice" or "iou": higher = better)
RANKING_METRIC = "dice"

# ============================================================================
# STEP 7 - Sensitivity to noise (on the TEST images)
# ============================================================================
# Gaussian noise added to the colour image (image scaled 0-1).
# Standard deviations tested: 0 = original image.
NOISE_LEVELS = [0, 0.02, 0.05, 0.1, 0.15, 0.2]
NOISE_REPEATS = 3     # noise is random -> repeat and average
RANDOM_SEED = 0       # same seed = same "random" noise every run

# ============================================================================
# STEP 8 - Sensitivity to ground-truth variability (on the TEST images)
# ============================================================================
# Simulates another person tracing the nucleus a bit bigger or smaller:
# the hand-drawn mask is shrunk (negative) or grown (positive) by N pixels.
GT_CHANGES_PX = [-3, -2, -1, 0, 1, 2, 3]

# ============================================================================
# Display
# ============================================================================
REFERENCE_COLOR = "lime"   # hand-drawn outline (green, as in the presentation)
ALGORITHM_COLOR = "red"    # automatic outline (red, as in the presentation)
CONTOUR_WIDTH = 1.5        # thickness of the outlines
CHANNEL_COLORMAP = "gray"  # colour map used to display the channels
FIGURE_DPI = 150           # resolution of the saved figures