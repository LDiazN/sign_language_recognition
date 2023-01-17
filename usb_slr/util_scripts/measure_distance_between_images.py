"""
This script will measure the euclidian distance between images of the same class, and store results in specified folder.
    - First command line argument is name of folder storing trajectory map images
    - Second command line argument is name of folder where to store results
"""
from collections import defaultdict
import sys
import cv2
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

assert len(sys.argv) > 2, "Not enough arguments"

TRAJECTORY_MAPS_DIR = Path(sys.argv[1])
OUTPUT_DIR = Path(sys.argv[2])

# Sanity check
if not TRAJECTORY_MAPS_DIR.exists():
    print(f"Folder '{TRAJECTORY_MAPS_DIR}' does not exists", file=sys.stderr)
    exit(1)

# Create output folder if not exists
if not OUTPUT_DIR.exists():
    print(f"Folder '{OUTPUT_DIR}' does not exists, creating it...", file=sys.stderr)
    OUTPUT_DIR.mkdir(parents=True)

# Read images in trajectory map folder
img_list = defaultdict(list)
for img in glob.glob(f"{TRAJECTORY_MAPS_DIR}/*.png"):
    img_filename = Path(img).name
    class_name, _ = img_filename.split("_")
    img_list[class_name].append((1.0/255.0) * cv2.imread(img))

# Compute average difference between images
class_diffs = defaultdict(list)
weird_images_list = []
for (class_name, images) in img_list.items():
    n_images = len(images)
    for i in range(0, n_images):
        for j in range(i+1, n_images):
            new_dist = np.linalg.norm(images[i] - images[j])
            if (new_dist == 0):
                weird_images_list.append(f"Weird zero distance in images: {i}, {j} of class {class_name}")
            class_diffs[class_name].append(new_dist)

data_frame_dict = {
    "Clase" : [],
    "Promedio" : [],
    "Desviación estándar" : [],
    "Mediana" : [],
    "Mínimo" : [],
    "Máximo" : [],
    }

# compute median, mean and variance for each class
for (class_name, data) in class_diffs.items():
    mean = round(np.mean(data), 4)
    median = round(np.median(data), 4)
    std_dev = round(np.std(data), 4)
    mini = round(np.min(data), 4)
    maxi = round(np.max(data), 4)

    data_frame_dict["Clase"].append(class_name)
    data_frame_dict["Promedio"].append(mean)
    data_frame_dict["Desviación estándar"].append(std_dev)
    data_frame_dict["Mediana"].append(median)
    data_frame_dict["Mínimo"].append(mini)
    data_frame_dict["Máximo"].append(maxi)


aggregated_data_df = pd.DataFrame(data_frame_dict)
aggregated_data_df.set_index("Clase", inplace=True)
aggregated_data_df.sort_values("Promedio", inplace=True, ascending=False)


# Save results
def latex_with_lines(df, *args, **kwargs):
    kwargs['column_format'] = '|'.join([''] + ['l'] * df.index.nlevels
                                            + ['r'] * df.shape[1] + [''])
    res = df.to_latex(*args, **kwargs)
    return res.replace('\\\\\n', '\\\\ \\hline\n')

path_to_results_file = Path(OUTPUT_DIR, "results.txt")
with path_to_results_file.open("w") as f:
    # Save table as latex for the book
    print(latex_with_lines(aggregated_data_df), file=f)

    print("Identical images list: ", file=f)

    # Save repeated images, might be useful
    for s in weird_images_list:
        print(s, file=f)
