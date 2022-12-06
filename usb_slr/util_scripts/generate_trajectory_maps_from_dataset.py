"""
This script will generate trajectory maps for each sign repetition in a given dataset.
For example, if you have 3 videos for "hello" and 2 videos for "good morning", 
you'll get 3 trajectory maps for "hello" and 2 for "good morning"
"""

from slr.model.data_ingestor import DataIngestor
from slr.data_processing.image_parser import PoseGraph, PoseValues
import cv2
import numpy as np
from pathlib import Path

ing = DataIngestor()
x_train, x_val, y_train, y_val = ing.generate_train_data(video_limit=5000)

# Generate graphs sequences for every frame


signs = (PoseGraph.as_trajectory_cv_img([PoseValues.from_array(arr).as_graph(False) for arr in r if not np.all(arr == 0)]) for r in x_train)

# Group signs per label
labels = {np.argmax(i) : 0 for i in y_train}
for i in y_train:
    labels[np.argmax(i)] += 1

# Save images
path_to_img_folder = Path("img")
# Create path if not exists
if not path_to_img_folder.exists(): path_to_img_folder.mkdir()

for (img, y) in zip(signs, y_train):

    i = np.argmax(y)

    # Skip samples with too few repetitions
    if labels[i] <= 1:
        continue
    
    # Create folder for this sign if not exists
    path_to_sign_dir = Path(path_to_img_folder, str(i))
    if not path_to_sign_dir.exists(): path_to_sign_dir.mkdir()

    cv2.imshow("Next image", img)
    cv2.waitKey(500)

    # Next name is the amount of files in this dir
    next_name = str(len(list(path_to_sign_dir.glob("*"))))
    img = np.float32(img)
    img = 255 * img
    cv2.imwrite(str(Path(path_to_sign_dir, next_name + ".jpg")), img)
    print("saved image ", i)
