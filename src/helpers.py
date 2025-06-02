import random

import numpy as np
import shutil
from pathlib import Path

random.seed(42)
np.random.seed(42)

import seaborn as sns
import matplotlib.pyplot as plt


def list2text(filename, lst):
    """
    Write a list to a text file, each element on a new line.

    Args:
        filename (str): Path to the text file.
        lst (list): List of strings to write to the file.
    """
    with open(filename, "w") as f:
        for item in lst:
            f.write(f"{item}\n")

def text2list(filename):
    """
    Read a text file and return its contents as a list of lines.

    Args:
        filename (str): Path to the text file.

    Returns:
        list: List of lines from the file, stripped of leading/trailing whitespace.
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def make_aesthetic(
        hex_color_list=None,
        with_gridlines=False,
        bold_title=False,
        save_transparent=False,
        font_scale=2,
        custom_font="Arial",
):
    """
    Make Seaborn look clean and add space between title and plot

    Args:
        hex_color_list: List of hex colors for the palette
        with_gridlines: Whether to add gridlines
        bold_title: Whether to make the title bold
        save_transparent: Whether to save the plot with a transparent background
        font_scale: Scaling factor for font sizes
        custom_font: Font to use for the plot. Note:
            There is a very annoying thing where if you try this in Google Colab with, say,
            'Arial' but colab does not have Arial it will keep printing a warning about this,
            so I would recommend using 'Arial' only if you are sure it is available on the system.

    """

    # Note: To make some parts of title bold and others not bold, we have to use
    # latex rendering. This should work:
    # plt.title(r'$\mathbf{bolded\ title}$' + '\n' + 'And a non-bold subtitle')

    sns.set(style="white", context="paper", font_scale=font_scale)
    if not hex_color_list:
        # 2024-11-28: Reordered color list
        hex_color_list = [
            "#2C3531",  # Dark charcoal gray with green undertone
            "#D41876",  # Telemagenta
            "#00A896",  # Persian green
            "#826AED",  # Medium slate blue
            "#F45B69",  # Vibrant pinkish-red
            "#E3B505",  # Saffron
            "#89DAFF",  # Pale azure
            "#342E37",  # Dark grayish-purple
            "#7DCD85",  # Emerald
            "#F7B2AD",  # Melon
            "#D4B2D8",  # Pink lavender
            "#020887",  # Phthalo blue
            "#E87461",  # Medium-bright orange
            "#7E6551",  # Coyote
            "#F18805",  # Tangerine
        ]

    sns.set_palette(sns.color_palette(hex_color_list))

    plt.rcParams.update(
        {
            # font settings
            "font.family": custom_font,
            "font.weight": "regular",
            "axes.labelsize": 11 * font_scale,
            "axes.titlesize": 14 * font_scale,
            "xtick.labelsize": 10 * font_scale,
            "ytick.labelsize": 10 * font_scale,
            "legend.fontsize": 10 * font_scale,
            # spines/grids
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.8,  # Thinner spines
            "axes.grid": with_gridlines,
            "grid.alpha": 0.2,
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            # title
            "axes.titlelocation": "left",
            "axes.titleweight": "bold" if bold_title else "regular",
            "axes.titlepad": 15 * (font_scale / 1),
            # fig
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": 0.2,
            "figure.constrained_layout.w_pad": 0.2,
            # legend
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.facecolor": "white",
            "legend.borderpad": 0.4,
            "legend.borderaxespad": 1.0,
            "legend.handlelength": 1.5,
            "legend.handleheight": 0.7,
            "legend.handletextpad": 0.5,
            # export
            "savefig.dpi": 300,
            "savefig.transparent": save_transparent,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.2,
            "figure.autolayout": False,
            # math font
            "mathtext.fontset": "custom",
            "mathtext.rm": custom_font,
            "mathtext.it": f"{custom_font}:italic",
            "mathtext.bf": f"{custom_font}:bold",
        }
    )

    return hex_color_list





def path2correct_loc(source_path, destination_path, copy_instead_of_move=False):
    """
    Move or copy all files from source path to destination path. I am using this because
    Kaggle downloader installs things to a very weird location, the user's cache. The problem is that
    these files are very big and I want to keep track of them.

    Args:
        source_path (str): Path to the source files, get this from Kaggle dataset download
        destination_path (str): Relative path from current working directory where to move the files
        copy_instead_of_move (bool): If True, copy files instead of moving them

    Returns:
        str: Absolute path to the destination directory

    ###

    NOTE: Edge-casey behavior I noticed 2025-05-26 11:44:35:

    Let's say we are downloading `Cornell-University/arxiv` and newest version is 234. I'd use the function like this:

        >>> source_path = kagglehub.dataset_download("Cornell-University/arxiv" ) # dls to cache
        >>> new_loc = path2correct_loc(source_path, "") # moves main file to wd

    Here's what source_path actually is by default in kagglehub:

        users/<user>/.cache/kaggle/datasets/cornell/arxiv/versions/234/arxiv-metadata-oai-snapshot.json

    So then when you run `new_loc = path2correct_loc(source_path, "")`, this will move `arxiv-metadata-oai-snapshot.json`
    file to wd. However, `path2correct_loc` keeps the rest of the cache metadata.

    This can matter because now let's say you run this after the initial download and move:

        >>> source_path = kagglehub.dataset_download("Cornell-University/arxiv/versions/234" )

    Kaggle is going to skip because there are some metadata files that exist in cache. This is mostly desired behavior, but consider
    the edge case where you downloaded the file (to cache), moved it to destination folder (like wd), deleted from that destination folder
     --> it won't download again unless you delete all the stuff Kaggle created in cache.

    But one way to get around this edge case is to make it so kaggle does not use cache to check if downloaded
    by writing `force_download=True`, so then this will always re-download the dataset, even if it exists in cache.

        >>> source_path = kagglehub.dataset_download("Cornell-University/arxiv/versions/234" , force_download=True)

    """

    source = Path(source_path)
    dest = Path.cwd() / destination_path

    if not source.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    dest.mkdir(parents=True, exist_ok=True)

    files_moved = 0
    total_size = 0

    print(f"{'Copying' if copy_instead_of_move else 'Moving'} files from {source} to {dest}")

    # Walk through all files and subdirectories
    for item in source.rglob('*'):
        if item.is_file():
            # Calculate relative path to preserve directory structure
            relative_path = item.relative_to(source)
            dest_file = dest / relative_path

            # Create subdirectories if needed
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Move or copy the file
            try:
                if copy_instead_of_move:
                    shutil.copy2(item, dest_file)
                    action = "Copied"
                else:
                    shutil.move(str(item), str(dest_file))
                    action = "Moved"

                file_size = dest_file.stat().st_size
                total_size += file_size
                files_moved += 1

                print(f"{action}: {relative_path} ({file_size / (1024 * 1024):.2f} MB)")

            except Exception as e:
                print(f"Error processing {item}: {e}")

    print(f"\nCompleted! {files_moved} files {'copied' if copy_instead_of_move else 'moved'}")
    print(f"Total size: {total_size / (1024 * 1024 * 1024):.2f} GB")
    print(f"Files are now in: {dest.absolute()}")

    return str(dest.absolute())


def statsmodels2latex(model, beta_digits=2, se_digits=2, p_digits=3, ci_digits=2, print_sci_not=False):
    """
    This function summarizes the results from a fitted statistical model,
    printing a LaTeX formatted string for each parameter in the model that includes the beta coefficient,
    standard error, p-value, and 95% CI.

    Parameters:
    - model: A fitted statistical model with methods to extract parameters, standard errors,
             p-values, and confidence intervals.
    - beta_digits (default = 2): Number of decimal places for beta coefficients.
    - se_digits (default = 2): Number of decimal places for standard errors.
    - p_digits (default = 3): Number of decimal places for p-values.
    - ci_digits (default = 2): Number of decimal places for confidence intervals.
    - print_sci_not: Boolean to print very small p-values (p<0.001) in scientific notation or just write 'p<0.001'

    """

    summary_strs = []
    # Check if the necessary methods are available in the model
    if not all(hasattr(model, attr) for attr in ['params', 'bse', 'pvalues', 'conf_int']):
        raise ValueError("Model does not have the required methods (params, bse, pvalues, conf_int).")

    # Retrieve parameter estimates, standard errors, p-values, and confidence intervals
    params = model.params
    errors = model.bse
    pvalues = model.pvalues
    conf_int = model.conf_int()

    # Iterate through each parameter
    for param_name, beta in params.items():
        # Escape LaTeX special characters in parameter names
        safe_param_name = param_name.replace('_', '\\_')

        se = errors[param_name]
        p = pvalues[param_name]
        ci_lower, ci_upper = conf_int.loc[param_name]

        # Determine p-value format
        if p < 0.001:
            if print_sci_not:
                p_formatted = f"= {p:.2e}"
            else:
                p_formatted = f"<0.001"
        else:
            p_formatted = f"= {p:.{p_digits}f}"

        # Format the summary string for the current parameter with LaTeX formatting
        summary = (f"{safe_param_name}: $\\beta = {beta:.{beta_digits}f}$, "
                   f"$SE = {se:.{se_digits}f}$, $p {p_formatted}$, "
                   f"$95\\% CI = [{ci_lower:.{ci_digits}f}, {ci_upper:.{ci_digits}f}]$")
        print(summary)


def array_stats(data, digits=2, include_ci=False):
    import scipy.stats as stats
    from scipy.stats import bootstrap
    import statistics as st

    """Calculate and print summary statistics for an array.

    This function computes basic descriptive statistics (mean, median, standard
    deviation, and mode) for a given array of data. It also provides an option
    to calculate and include a 95% confidence interval for the mean using
    bootstrap resampling.

    Args:
        data (array-like): The input data array for which statistics will be
            calculated. Can be a list, numpy array, or any array-like structure.
        digits (int, optional): Number of decimal places for rounding the
            calculated statistics. Defaults to 2.
        include_ci (bool, optional): Whether to include a 95% confidence
            interval for the mean using bootstrap resampling. Defaults to False.

    Returns:
        dict: A dictionary containing the calculated statistics with the
            following keys:
            - 'mean': The arithmetic mean of the data
            - 'median': The median (middle value) of the data
            - 'sd': The sample standard deviation (using ddof=1)
            - 'mode': The most frequently occurring value
            - 'ci': (optional) A tuple containing the lower and upper bounds
              of the 95% confidence interval, only present if include_ci=True


    Note:
        The function prints formatted statistics to the console like this--
        "M = 2.83, SD = 1.47, Mdn = 2.50" \n "Mode = 2.00" in addition to
        returning them as a dictionary. If include_ci=True, it also prints the
        confidence interval as "95% CI [1.83, 3.83]". When multiple modes exist,
        the function will print a warning message and use the first occurrence.

        The bootstrap uses Scipy 10K iterations and is bootstrapping the mean.
    """
    data = np.array(data)
    mean_val = np.mean(data)
    median_val = np.median(data)
    sd_val = np.std(data, ddof=1)

    # Calculate mode - handling cases with multiple modes
    try:
        mode_val = st.mode(data)
    except st.StatisticsError:
        # If multiple modes, use scipy's mode which returns the first occurrence
        print("Multiple modes found, using the first one.")
        mode_val = stats.mode(data, keepdims=True)[0][0]
        print(mode_val)

    result = {
        'mean': round(mean_val, digits),
        'median': round(median_val, digits),
        'sd': round(sd_val, digits),
        'mode': round(mode_val, digits)
    }

    # Add confidence interval if requested
    if include_ci:
        def mean_func(x, axis):
            return np.mean(x, axis=axis)

        data_reshaped = np.array(data).reshape(-1, 1)
        bootstrap_result = bootstrap((data_reshaped,), mean_func,
                                     confidence_level=0.95,
                                     random_state=42,
                                     n_resamples=10 * 1000)
        ci_lower, ci_upper = bootstrap_result.confidence_interval
        result['ci'] = (round(float(ci_lower), digits), round(float(ci_upper), digits))

    print(f"M = {result['mean']:.{digits}f}, SD = {result['sd']:.{digits}f}, Mdn = {result['median']:.{digits}f}")
    print(f"Mode = {result['mode']:.{digits}f}")
    if include_ci and 'ci' in result:
        print(f"95% CI [{result['ci'][0]:.{digits}f}, {result['ci'][1]:.{digits}f}]")

    return result

