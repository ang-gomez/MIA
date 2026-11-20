# Leukocyte nucleus segmentation on blood smears

Mini-project for the **Medical Image Analysis (MIA)** course, BME-Paris Master, Bioimaging track.
Team: Angeles Gomez Impellizzeri (engineer) and Elise (MD).

The goal is to **design, implement and evaluate** a method that finds the **nucleus of white blood cells**
(lymphocytes, monocytes and neutrophils) in colour images of blood smears.

The method is **classical image processing**. There is no deep learning and no trained model:
every step is a fixed rule (colour channel, filter, threshold, clean-up).

---

## Method

```
colour image
   │
   ├── PRE-PROCESSING   colour → 4 grey channels (green inverted, red inverted,
   │                    hematoxylin, saturation) + smoothing
   │                    (none, Gaussian sigma = 1 or 2, or Fourier low-pass filter)
   │
   ├── SEGMENTATION     brightness threshold: Otsu, Multi-Otsu or K-means
   │
   └── POST-PROCESSING  opening (radius 1, 2 or 3 px) → fill holes → keep main object
   │
   ▼
automatic segmentation of the nucleus
```

Every combination is tested: 4 channels × 4 smoothing options × 3 thresholds × 3 opening radii
= **144 "recipes"**. Each one is compared with the manual mask, and the best one is kept.

### Data split

| Group | Images | Used for |
|---|---|---|
| **Development** | 6 (2 per cell type) | Choosing the Fourier band and the best recipe |
| **Test** | 3 (1 per cell type) | Final results only: never used to choose anything |

### Validation

- **Dice** (= F1 score for binary masks) and **IoU**: overlap with the manual mask (1 = perfect)
- **HD95**: 95th percentile Hausdorff distance between the outlines, in pixels (0 = perfect)
- **Sensitivity to noise**: Gaussian noise of increasing strength added to the test images
- **Sensitivity to ground-truth variability**: manual masks grown or shrunk by 1–3 px, to simulate
  another person drawing them

---

## Data

9 images (368 × 368 px, May-Grünwald-Giemsa stain) from the dataset of
*Yarıkan AE et al., Sci Data 2026;13:417*. 3 lymphocytes, 3 monocytes, 3 neutrophils, one patient per image.

The manual masks (ground truth) were drawn in GIMP by Elise: white = nucleus, black = background.

Put the images and their masks **in the same folder as the code**. The mask has the same name
as its image, plus `annotated` at the end:

```
lymphocyte__1604253483_036.jpg
lymphocyte__1604253483_036annotated1.jpg
```

The group of each image (development or test) is set in `config.py` (`GROUP_OF_IMAGE`).

---

## Files

| File | What it contains |
|---|---|
| `config.py` | All the settings: folders, channels, filters, thresholds, noise levels… Change things **here** |
| `utils.py` | All the functions (loading, channels, filters, thresholds, clean-up, metrics, figures) |
| `fft.py` | Separate script to look at the Fourier spectrum and **choose the frequency band** |
| `main.py` | The full pipeline, steps 1 to 8. This is the one you run |

---

## How to run

**1. Install the libraries (once)**

```
python -m pip install -r requirements.txt
```

**2. (Optional) Choose the Fourier band**

```
python fft.py
```

It uses only the development images and prints a suggested band. Look at the figures in
`figures/fourier_explore/`, then write your choice in `config.py`:

```python
FFT_LOW_CUTOFF = 0.0
FFT_HIGH_CUTOFF = 0.08
```

**3. Run the whole pipeline**

```
python main.py
```

It runs without any question (about 1–2 minutes) and prints `STEP 1 COMPLETED` … `STEP 8 COMPLETED`.

---

## What main.py does

| Step | What it does | Output folder |
|---|---|---|
| 1 | Load the images and manual masks, check them | `figures/step1_load/` |
| 2 | Pre-processing: colour → 4 grey channels | `figures/step2_channels/` |
| 3 | Pre-processing (smoothing) + segmentation (threshold) | `figures/step3_threshold/` |
| 4 | Post-processing: clean-up → 144 recipes per image | `figures/step4_cleanup/` |
| 5 | Score every recipe on every image (Dice, IoU, HD95) | `figures/step5_metrics/` |
| 6 | Choose the best recipe on development, show it on test (best case, worst case, score table) | `figures/step6_final/` |
| 7 | Sensitivity to noise (same recipe, only the smoothing changes) | `figures/step7_noise/` |
| 8 | Sensitivity to ground-truth variability | `figures/step8_ground_truth/` |

All the numbers are also saved as tables (`.csv`, open with Excel) in `results/`.
The `figures/` and `results/` folders are created automatically and are not stored on GitHub.

---

## Results (test images)

Chosen recipe: `green_inverted | sigma=0 | multi_otsu | r=2`

| Image | Dice | IoU | HD95 (px) |
|---|---|---|---|
| lymphocyte_036 | 0.975 | 0.951 | 2.2 |
| monocyte_090 | 0.985 | 0.971 | 2.0 |
| neutrophil_032 | 0.973 | 0.947 | 2.0 |
| **Mean** | **0.978** | **0.956** | **2.1** |

Main observations:

- The **colour channel** and the **threshold method** matter most. Smoothing and clean-up size
  change little on clean images.
- **Smoothing matters with noise**: without blur, the Dice drops strongly as noise increases;
  with a light Gaussian blur (sigma = 1) it stays high.
- **Fourier filtering**: the nucleus is a large, smooth shape, so it lives in the **low frequencies**.
  A low-pass filter (cut-off 0.08 cycles/pixel) works well; removing low frequencies destroys the segmentation.
- The difference between the automatic segmentation and the manual mask is about the same as
  the difference between two manual tracings 1 pixel apart.

## Limitations

- Only 9 images: the results show trends, not definitive proof.
- The best recipes are very close to each other (differences in the 3rd decimal).
- Noise and ground-truth variability are **simulated**.
- The manual mask of `monocyte__1604276339_056` is currently wrong (it is a copy of a neutrophil mask)
  and has to be redrawn.

---

## Libraries

Python 3, NumPy, SciPy, scikit-image, Matplotlib (see `requirements.txt`).
