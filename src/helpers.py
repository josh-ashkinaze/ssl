import random

import numpy as np
import shutil
from pathlib import Path

random.seed(42)
np.random.seed(42)

import seaborn as sns
import matplotlib.pyplot as plt


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

    NOTE: Edge-casey behavior I noticed 2025-05-26 11:44:35:

    Let's say we are downloading `Cornell-University/arxiv` and newest version is 234.

        source_path = kagglehub.dataset_download("Cornell-University/arxiv" )

    Here's what source_path actually is by default in kagglehub:

        users/<user>/.cache/kaggle/datasets/cornell/arxiv/versions/234/arxiv-metadata-oai-snapshot.json

    So then when you run `new_loc = path2correct_loc(source_path, "")`, this will move the file to wd.

    But what this does is ONLY moves `arxiv-metadata-oai-snapshot.json` to wd. However, it keeps the
    rest of the folders. This can matter because now let's say you run

        source_path = kagglehub.dataset_download("Cornell-University/arxiv/versions/234" )

    Kaggle is going to skip because there are some metadata files that exist in cache. This is mostly desired behavior, but consider
    the edge case where you downloaded it, moved it to another folder, deleted from that folder --> it wont download again
    unless you delete all the stuff Kaggle created in cache.



    Args:
        source_path (str): Path to the source files
        destination_path (str): Relative path from current working directory where to move the files
        copy_instead_of_move (bool): If True, copy files instead of moving them

    Returns:
        str: Absolute path to the destination directory
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







