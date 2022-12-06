"""
Use this script to compute correlation between mean positions of every joint and 
value of label
"""

from slr.model.data_ingestor import DataIngestor
from slr.data_processing.image_parser import PoseGraph, PoseValues
import cv2
import numpy as np
from pathlib import Path

ing = DataIngestor()
x_train, x_val, y_train, y_val = ing.generate_train_data(video_limit=10)

# Generate graphs sequences for every frame
signs = np.array([np.mean(PoseGraph.mean_positions([PoseValues.from_array(arr).as_graph(False) for arr in r if not np.all(arr == 0)]), axis=1)  for r in x_train])
y_train = np.argmax(y_train, axis=1)
y_train = np.repeat(y_train, 75)
print(y_train.shape)
np.cov(signs, y_train)