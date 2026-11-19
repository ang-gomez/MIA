"""
main.py - The program you RUN
=============================

Leukocyte nucleus segmentation - MIA project (Master 2 Bioimaging)

The whole pipeline runs from start to end, without questions:
    STEP 1 - Load and check the images and hand-drawn masks
    STEP 2 - Extract the 4 colour channels
    STEP 3 - Smoothing (Gaussian or Fourier) + threshold
    STEP 4 - Clean-up (opening, fill holes, keep main object)
    STEP 5 - Compare every result with the hand-drawn mask (Dice, IoU, HD95)
    STEP 6 - Choose the best recipe on the development images,
             show its result on the test images
    STEP 7 - Sensitivity to noise (test images)
    STEP 8 - Sensitivity to ground-truth variability (test images)

The Fourier band is chosen BEFORE, with the separate script
fft.py, and written in config.py (FFT_LOW_CUTOFF, FFT_HIGH_CUTOFF).

Outputs (created next to this file):
    figures/step1_load/ ... figures/step6_final/   -> the images (.png)
    results/                                        -> the tables (.csv, Excel)

This file only tells the story. The work is done by the functions in
utils.py, and every setting lives in config.py.

How to run: open this file in VS Code and click "Run", or type in the
terminal:   python main.py

Libraries needed (install once):
    python -m pip install numpy scipy matplotlib scikit-image
"""

import numpy as np

import config
import utils


# ============================================================================
# STEP 1 - Load and check the images and their hand-drawn masks
# ============================================================================
def run_step1():
    """
    Returns a list of dictionaries, one per image:
        {"name": ..., "group": ..., "image": array, "mask": array}
    """
    pairs = utils.find_image_mask_pairs(config.DATA_FOLDER)
    folder = config.FIGURES_FOLDER / "step1_load"

    data = []
    for image_path, mask_path, group in pairs:
        image = utils.load_image(image_path)
        mask = utils.load_mask(mask_path)
        name = image_path.stem

        if utils.check_pair(image, mask, name):
            data.append({"name": name, "group": group, "image": image, "mask": mask})
            fig = utils.figure_step1(image, mask, f"[{group}] {name}")
            utils.save_figure(fig, folder, f"{group}_{name}")

    # How many images in each group
    for group in config.GROUPS + ["no_group"]:
        n = len([d for d in data if d["group"] == group])
        if n > 0:
            print(f"  {group}: {n} image(s)")

    utils.print_banner(f"STEP 1 COMPLETED - {len(data)} of {len(pairs)} pairs OK")
    return data


# ============================================================================
# STEP 2 - Extract the colour channels
# ============================================================================
def run_step2(data):
    """Adds the 4 channels to every image: data[i]["channels"]."""
    folder = config.FIGURES_FOLDER / "step2_channels"

    for d in data:
        d["channels"] = utils.extract_channels(d["image"])
        fig = utils.figure_step2(d["image"], d["mask"], d["channels"],
                                 f"[{d['group']}] {d['name']}")
        utils.save_figure(fig, folder, f"{d['group']}_{d['name']}")

    utils.print_banner(f"STEP 2 COMPLETED - {len(config.CHANNELS)} channels "
                       f"for {len(data)} images")


# ============================================================================
# STEP 3 - Smoothing (Gaussian or Fourier) + threshold
# ============================================================================
def run_step3(data):
    """
    For every image, every channel x smoothing x threshold method -> one mask.
    Adds data[i]["thresholded"] = {(channel, smoothing, method): mask}.
    """
    folder = config.FIGURES_FOLDER / "step3_threshold"

    for d in data:
        d["thresholded"] = {}
        for channel in config.CHANNELS:
            for smoothing in config.SMOOTHINGS:
                for method in config.THRESHOLD_METHODS:
                    d["thresholded"][(channel, smoothing, method)] = utils.smooth_and_threshold(
                        d["channels"][channel], smoothing, method)

        fig = utils.figure_step3(d["thresholded"], d["mask"],
                                 f"[{d['group']}] {d['name']} - before clean-up")
        utils.save_figure(fig, folder, f"{d['group']}_{d['name']}")

    n = len(config.CHANNELS) * len(config.SMOOTHINGS) * len(config.THRESHOLD_METHODS)
    utils.print_banner(f"STEP 3 COMPLETED - {n} thresholded masks per image")


# ============================================================================
# STEP 4 - Clean-up
# ============================================================================
def run_step4(data, recipes):
    """
    Applies the clean-up of every recipe.
    Adds data[i]["segmentations"] = {recipe name: final mask}.
    """
    folder = config.FIGURES_FOLDER / "step4_cleanup"
    example = config.EXAMPLE_RECIPE

    for d in data:
        d["segmentations"] = {}
        for recipe in recipes:
            mask = d["thresholded"][(recipe["channel"], recipe["smoothing"], recipe["threshold"])]
            d["segmentations"][utils.recipe_name(recipe)] = utils.clean_up(
                mask, recipe["opening_radius"])

        # Figure: the clean-up stages, for the example recipe
        mask = d["thresholded"][(example["channel"], example["smoothing"], example["threshold"])]
        stages = utils.clean_up_stages(mask, example["opening_radius"])
        fig = utils.figure_step4(d["image"], stages, d["mask"],
                                 f"[{d['group']}] {d['name']}  -  {utils.recipe_name(example)}")
        utils.save_figure(fig, folder, f"{d['group']}_{d['name']}")

    utils.print_banner(f"STEP 4 COMPLETED - {len(recipes)} recipes applied "
                       f"to {len(data)} images")


# ============================================================================
# STEP 5 - Metrics
# ============================================================================
def run_step5(data):
    """
    Compares every segmentation with the hand-drawn mask.
    Returns the big results table (one row per image x recipe) and saves it.
    """
    results = []
    for d in data:
        for name, prediction in d["segmentations"].items():
            row = {"image": d["name"], "group": d["group"], "recipe": name}
            row.update(utils.compute_metrics(prediction, d["mask"]))
            results.append(row)

    utils.save_table(results, config.RESULTS_FOLDER / "step5_all_results.csv")

    # Figures (development images only)
    folder = config.FIGURES_FOLDER / "step5_metrics"
    ranking = utils.mean_score_per_recipe(results, "development", config.RANKING_METRIC)
    if len(ranking) > 0:
        # a) all the recipes, from best to worst
        fig = utils.figure_all_recipes(ranking, ranking[0][0],
                                       "All recipes, sorted (development images)")
        utils.save_figure(fig, folder, "1_all_recipes_sorted")
        # b) the 20 best ones, zoomed
        fig = utils.figure_ranking(ranking, "20 best recipes (development images)")
        utils.save_figure(fig, folder, "2_top20_recipes")
        # c) which choices matter
        fig = utils.figure_parameter_effects(results, "development", config.RANKING_METRIC)
        utils.save_figure(fig, folder, "3_effect_of_each_choice")

    utils.print_banner(f"STEP 5 COMPLETED - {len(results)} scores computed")
    return results


# ============================================================================
# STEP 6 - Choose the best recipe and evaluate it on test
# ============================================================================
def run_step6(data, results):
    """
    Choose the recipe with the best mean score on the DEVELOPMENT images,
    then look at its result on every image (the TEST ones = final result).
    Returns the name of the best recipe (or None if no development image).
    """
    folder = config.FIGURES_FOLDER / "step6_final"
    metric = config.RANKING_METRIC

    # 6a. Ranking of all the recipes on the development images
    #     (the test scores are NOT used here)
    ranking = utils.mean_score_per_recipe(results, "development", metric)
    if len(ranking) == 0:
        print("[PROBLEM] No development image: check GROUP_OF_IMAGE in config.py.")
        return None
    best_recipe = ranking[0][0]

    # 6b. Save the ranking (all recipes, best first)
    selection = [{"rank": i + 1, "recipe": name, f"development_mean_{metric}": score}
                 for i, (name, score) in enumerate(ranking)]
    utils.save_table(selection, config.RESULTS_FOLDER / "step6_ranking.csv")
    print(f"  Best recipe: {best_recipe}")

    # 6c. Evaluate the chosen recipe on every image (test = the real result)
    final_rows = []
    for d in data:
        prediction = d["segmentations"][best_recipe]
        scores = utils.compute_metrics(prediction, d["mask"])
        final_rows.append({"image": d["name"], "group": d["group"], **scores})

        fig = utils.figure_result(d["image"], d["mask"], prediction, scores,
                                  f"[{d['group']}] {d['name']}")
        utils.save_figure(fig, folder / d["group"], d["name"])

    utils.save_table(final_rows, config.RESULTS_FOLDER / "step6_best_recipe_scores.csv")

    # 6d. Test set: score table + best and worst case (as in the presentation)
    test_rows = [row for row in final_rows if row["group"] == "test"]
    if len(test_rows) > 0:
        fig = utils.figure_score_table(test_rows, f"Test images - {best_recipe}")
        utils.save_figure(fig, folder, "test_score_table")

        test_rows_sorted = sorted(test_rows, key=lambda row: row[metric])
        for label, row in [("worst_case", test_rows_sorted[0]),
                           ("best_case", test_rows_sorted[-1])]:
            d = next(d for d in data if d["name"] == row["image"])
            fig = utils.figure_result(d["image"], d["mask"], d["segmentations"][best_recipe],
                                      row, f"{label.replace('_', ' ')}: {d['name']}")
            utils.save_figure(fig, folder, f"test_{label}")

        mean_test = sum(row[metric] for row in test_rows) / len(test_rows)
        print(f"  Mean {metric} on TEST: {mean_test:.3f}")

    utils.print_banner("STEP 6 COMPLETED - best recipe chosen and evaluated")
    return best_recipe


# ============================================================================
# STEP 7 - Sensitivity to noise
# ============================================================================
def recipes_to_compare(best_recipe):
    """
    The chosen recipe, and the SAME recipe with every other smoothing option
    (only the smoothing changes: same channel, threshold and clean-up).
    -> fair comparison: does blurring (Gaussian or Fourier) resist noise better?
    """
    chosen = utils.recipe_from_name(best_recipe)
    names = [best_recipe]
    for smoothing in config.SMOOTHINGS:
        variant = dict(chosen)            # copy of the chosen recipe...
        variant["smoothing"] = smoothing  # ...with another smoothing
        name = utils.recipe_name(variant)
        if name not in names:
            names.append(name)
    return names


def run_step7(data, results, best_recipe):
    """
    Add more and more noise to the TEST images, segment them again, and see
    how the Dice drops. Saves a curve, examples and a table.
    """
    folder = config.FIGURES_FOLDER / "step7_noise"
    test_data = [d for d in data if d["group"] == "test"]
    if len(test_data) == 0:
        print("[WARNING] No test image: step 7 skipped.")
        return

    rng = np.random.default_rng(config.RANDOM_SEED)   # reproducible noise
    names = recipes_to_compare(best_recipe)
    recipes = [utils.recipe_from_name(n) for n in names]

    # scores[recipe name][noise level] = list of Dice values
    scores = {n: {level: [] for level in config.NOISE_LEVELS} for n in names}
    examples = []   # for the example figure (first test image, chosen recipe)

    for level in config.NOISE_LEVELS:
        repeats = 1 if level == 0 else config.NOISE_REPEATS
        for repeat in range(repeats):
            for d in test_data:
                noisy = utils.add_noise(d["image"], level, rng)
                channels = utils.extract_channels(noisy)
                for name, recipe in zip(names, recipes):
                    prediction = utils.segment(channels, recipe)
                    scores[name][level].append(utils.dice(prediction, d["mask"]))
                    if d is test_data[0] and repeat == 0 and name == best_recipe:
                        examples.append((noisy, d["mask"], prediction, level))

    # Table + curve
    rows = []
    curves = {}
    for name, recipe in zip(names, recipes):
        means = [float(np.mean(scores[name][level])) for level in config.NOISE_LEVELS]
        if name == best_recipe:
            label = f"CHOSEN recipe: {name}"
        else:
            label = f"same recipe with {utils.smoothing_name(recipe['smoothing'])}"
        curves[label] = means
        for level, mean in zip(config.NOISE_LEVELS, means):
            rows.append({"recipe": name, "noise_std": level, "mean_dice_test": mean})
    utils.save_table(rows, config.RESULTS_FOLDER / "step7_noise.csv")

    fig = utils.figure_curves(config.NOISE_LEVELS, curves, "noise level (standard deviation)",
                              "mean Dice: automatic vs manual (test images)",
                              "Sensitivity to noise")
    utils.save_figure(fig, folder, "dice_vs_noise")

    fig = utils.figure_examples([e[0] for e in examples], [e[1] for e in examples],
                                [e[2] for e in examples],
                                [f"noise = {e[3]}" for e in examples],
                                f"{test_data[0]['name']} - {best_recipe}")
    utils.save_figure(fig, folder, "examples")

    utils.print_banner(f"STEP 7 COMPLETED - {len(config.NOISE_LEVELS)} noise levels, "
                       f"{len(names)} recipes compared")


# ============================================================================
# STEP 8 - Sensitivity to ground-truth variability
# ============================================================================
def run_step8(data, best_recipe):
    """
    Shrink / grow the hand-drawn masks of the TEST images (as if another person
    had traced them) and see how much the Dice of our segmentation changes.
    Also measures the Dice between the original and the changed tracing,
    i.e. how much the tracing alone can move the score.
    """
    folder = config.FIGURES_FOLDER / "step8_ground_truth"
    test_data = [d for d in data if d["group"] == "test"]
    if len(test_data) == 0:
        print("[WARNING] No test image: step 8 skipped.")
        return

    rows = []
    algo_curve = []
    tracing_curve = []
    for pixels in config.GT_CHANGES_PX:
        algo_scores = []
        tracing_scores = []
        for d in test_data:
            changed = utils.change_mask(d["mask"], pixels)
            prediction = d["segmentations"][best_recipe]
            algo_scores.append(utils.dice(prediction, changed))
            tracing_scores.append(utils.dice(d["mask"], changed))
        algo_curve.append(float(np.mean(algo_scores)))
        tracing_curve.append(float(np.mean(tracing_scores)))
        rows.append({"mask_change_px": pixels,
                     "mean_dice_algorithm_vs_changed_mask": algo_curve[-1],
                     "mean_dice_original_vs_changed_mask": tracing_curve[-1]})
    utils.save_table(rows, config.RESULTS_FOLDER / "step8_ground_truth.csv")

    curves = {"automatic segmentation vs modified manual mask": algo_curve,
              "original manual mask vs modified manual mask": tracing_curve}
    fig = utils.figure_curves(config.GT_CHANGES_PX, curves,
                              "manual mask change (pixels, - = smaller, + = bigger)",
                              "mean Dice (test images)",
                              "Sensitivity to ground-truth variability")
    utils.save_figure(fig, folder, "dice_vs_mask_change")

    # Example: the changed tracings on the first test image
    d = test_data[0]
    shown = [p for p in config.GT_CHANGES_PX if p in (min(config.GT_CHANGES_PX), 0,
                                                      max(config.GT_CHANGES_PX))]
    fig = utils.figure_examples([d["image"]] * len(shown),
                                [utils.change_mask(d["mask"], p) for p in shown],
                                [d["segmentations"][best_recipe]] * len(shown),
                                [f"mask change = {p:+d} px" for p in shown],
                                f"{d['name']} - green = modified manual mask, "
                                f"red = automatic segmentation")
    utils.save_figure(fig, folder, "examples")

    utils.print_banner(f"STEP 8 COMPLETED - {len(config.GT_CHANGES_PX)} mask changes tested")


# ============================================================================
# Main program
# ============================================================================
def main():
    data = run_step1()
    if len(data) == 0:
        return   # nothing to do without images

    recipes = utils.all_recipes()

    run_step2(data)
    run_step3(data)
    run_step4(data, recipes)
    results = run_step5(data)
    best_recipe = run_step6(data, results)
    if best_recipe is None:
        return

    run_step7(data, results, best_recipe)
    run_step8(data, best_recipe)

    print(f"\nAll done! Figures in: {config.FIGURES_FOLDER}")
    print(f"          Tables in : {config.RESULTS_FOLDER}")


# Run main() only when this file is executed directly
if __name__ == "__main__":
    main()