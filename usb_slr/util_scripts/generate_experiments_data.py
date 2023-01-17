"""
    This file generates the data for experiments. This data includes plots, confusion matrices, performance metrics and 
    sample trajectory maps.
    Arguments:
        - Dataset : str = dataset to generate data for. MS for MS-ASL, SLA64 for, well, SLA64 and PERU for the PERUvian dataset
"""

# Local imports
from tkinter.filedialog import Open
import cv2
from slr.dataset_manager.dataset_managers import ArgentinaDatasetManager, MicrosoftDatasetManager, PeruDatasetManager, SignDescription
from slr.model.data_ingestor import DataIngestor, FeatureTransformer

# Python imports
from pathlib import Path
import pickle
import numpy as np
import sys

# Third party imports
import torch.cuda as cuda
import torch
import wandb
import termcolor as c
import pandas as pd

# -- < Which dataset to use > ---------------------------------
VALID_DATASETS = ["SLA64", "PERU", "MS"]

# Parse dataset from command line arguments
if len(sys.argv) < 2:
    print(f"Error: missing dataset argument. Possible options: {VALID_DATASETS}", file=sys.stderr)
    exit(1)

DATASET = sys.argv[1]

if DATASET not in VALID_DATASETS:
    print(f"Error: Invalid dataset '{DATASET}'. Valid options: {VALID_DATASETS}")


OUTPUT_DIR = Path(f"{DATASET}_generated_resources")

# -- < Initial set up > ---------------------------------------
# Set up wandb project
wandb.init(project="usb-slr", entity="usb-slr")

torch.cuda.set_device(torch.device("cuda:0"))
torch.set_default_tensor_type('torch.cuda.FloatTensor')


if cuda.is_available():
    print(f"GPU found. Trying to use GPU...")
else:
    print("No GPU found. Defaulting to CPU")

# -- < Load compiled data for this training > ---------------------
CACHE_TRAIN_X = Path(f"/run/media/luis/Storage/.usb_slr/cache/.train_x_{DATASET}.npy")
CACHE_TRAIN_Y = Path(f"/run/media/luis/Storage/.usb_slr/cache/.train_y_{DATASET}.npy")
CACHE_VAL_X = Path(f"/run/media/luis/Storage/.usb_slr/cache/.val_x_{DATASET}.npy")
CACHE_VAL_Y = Path(f"/run/media/luis/Storage/.usb_slr/cache/.val_y_{DATASET}.npy")
CACHE_LABELMAP = Path(f"/run/media/luis/Storage/.usb_slr/cache/.labelmap_{DATASET}.npy")
print("Reading data...")

N_CLASSES_ARGENTINA = 32
N_CLASSES_MS = 20
N_CLASSES_PERU = 5

n_frames_map = {
    "MS" : 60,
    "PERU" : 80,
    "SLA64" : 200
}
n_labels_map = {
    "MS" : N_CLASSES_MS,
    "PERU" : N_CLASSES_PERU,
    "SLA64" : N_CLASSES_ARGENTINA
}

def filter_ms(features : np.ndarray, description : SignDescription) -> bool:
    allowed =   [3,
                19,
                15,
                8,
                33,
                11,
                1,
                14,
                12,
                51,
                50,
                2,
                29,
                79,
                23,
                78,
                9,
                31,
                61,
                25,
                10,
                7,
                44,
                17,
                75,
                39,
                48,
                59,
                21,
                45,
                65,
                95,
                76,
                34,
                64,
                32,
                13,
                83,
                66,
                92,
                47,
                43,
                52,
                72,
                24,
                28,
                6,
                71,
                26,
                99,
                16][:N_CLASSES_MS]
    
    return description.label in allowed and len(features) > 0

def filter_argentina(features : np.ndarray, description : SignDescription) -> bool:
    allowed = [i for i in range(1, 65)][:N_CLASSES_ARGENTINA]

    return len(features) > 0 and description.label in allowed

# Load MS data
if not CACHE_TRAIN_X.exists():
    print("No cache found. Creating new dataset...")
    if DATASET == "MS":
        dataset_manager = MicrosoftDatasetManager()
        filter_fn = filter_ms
    elif DATASET == "PERU":
        dataset_manager = PeruDatasetManager()
        filter_fn = lambda a, b: True
    elif DATASET == "SLA64":
        dataset_manager = ArgentinaDatasetManager()
        filter_fn = filter_argentina
    else:
        assert False, f"Unrecognized dataset: '{DATASET}'"
    
    data_ingestor = DataIngestor(dataset_manager=dataset_manager)
    ft =  FeatureTransformer(80)
    X_train, X_val, Y_train, Y_val, label_map = data_ingestor.generate_train_data(predicate=filter_fn, padding_func=FeatureTransformer.pad_with_zeroes, normalize_location=True, reduce_labels=True, test_size=0.2)
    print("Data generation ready")
    # Try to cache data
    with CACHE_TRAIN_X.open('wb') as f:
        pickle.dump(X_train, f)
    with CACHE_TRAIN_Y.open('wb') as f:
        pickle.dump(Y_train, f)
    with CACHE_VAL_X.open('wb') as f:
        pickle.dump(X_val, f)
    with CACHE_VAL_Y.open('wb') as f:
        pickle.dump(Y_val, f)

    with CACHE_LABELMAP.open('wb') as f:
        pickle.dump(label_map, f)
else:
    print("Cache found. Using cached data...")
    with CACHE_TRAIN_X.open('rb') as f:
        X_train = pickle.load(f)
    with CACHE_TRAIN_Y.open('rb') as f:
        Y_train = pickle.load(f)
    with CACHE_VAL_X.open('rb') as f:
        X_val = pickle.load(f)
    with CACHE_VAL_Y.open('rb') as f:
        Y_val = pickle.load(f)
    with CACHE_LABELMAP.open('rb') as f:
        label_map = pickle.load(f)

assert not np.any(np.isnan(X_train))
assert not np.any(np.isnan(Y_train))
assert not np.any(np.isnan(X_val))
assert not np.any(np.isnan(X_val))


# Create dataloaders from data
tensor_train_x = torch.Tensor(X_train).cpu() # (amount, n_frames, n_features)
tensor_train_y = torch.Tensor(Y_train).cpu()
tensor_val_x = torch.Tensor(X_val).cpu()
tensor_val_y = torch.Tensor(Y_val).cpu()


# Pre process data for MS
from slr.model.trajectory_map_lstm_model import DataPreprocessor 
from slr.data_processing.image_parser import PoseValues

CACHE_TMCNNMODEL = Path(f"/run/media/luis/Storage/.usb_slr/cache/.tmcnn_train_data_fine_tuning_{DATASET}.pkl")

if CACHE_TMCNNMODEL.exists():
    print("Found cache for images, using cache")

    with CACHE_TMCNNMODEL.open("rb") as f:
        tensor_train_x, tensor_train_y, tensor_val_x, tensor_val_y, lstm_tensor_train_x, lstm_tensor_val_x = pickle.load(f)
else:
    print("Cache not found, creating data from scratch")
    # Preprocess data: create pose values from arrays to create matrices
    poses_train_x = [[PoseValues.from_array(r) for r in s if r.count_nonzero() != 0] for s in tensor_train_x]
    poses_val_x = [[PoseValues.from_array(r) for r in s if r.count_nonzero() != 0] for s in tensor_val_x]

    # Filter out empty signs
    new_poses_train_x = []  
    new_poses_val_x = []
    new_train_y = []
    new_val_y = []

    for (x,y) in zip(poses_train_x, tensor_train_y):
        if not x:
            continue

        new_poses_train_x.append(x)
        new_train_y.append(y.reshape(1, *y.shape))

    poses_train_x = new_poses_train_x
    del new_poses_train_x
    tensor_train_y = torch.cat(new_train_y)
    del new_train_y


    for (x,y) in zip(poses_val_x, tensor_val_y):
        if not x:
            continue

        new_poses_val_x.append(x)
        new_val_y.append(y.reshape(1, *y.shape))

    poses_val_x = new_poses_val_x
    del new_poses_val_x
    tensor_val_y = torch.cat(new_val_y)
    del new_val_y


    processor = DataPreprocessor()

    tensor_train_x, lstm_tensor_train_x = processor.process_many(poses_train_x, joint_pos_radius=3, image_size_x=128, image_size_y=128, process_type="limb_colored", n_frames=n_frames_map[DATASET])
    tensor_val_x, lstm_tensor_val_x = processor.process_many(poses_val_x, joint_pos_radius=3,  image_size_x=128, image_size_y=128, process_type="limb_colored", n_frames=n_frames_map[DATASET])

    tensor_train_y = tensor_train_y.cuda()
    tensor_val_y = tensor_val_y.cuda()

    with CACHE_TMCNNMODEL.open("wb") as f:
        pickle.dump((tensor_train_x, tensor_train_y, tensor_val_x, tensor_val_y, lstm_tensor_train_x, lstm_tensor_val_x), f)

# -- < Save trajectory map images to disk > -----------------------------------------------------
trajectory_maps_dir = Path(OUTPUT_DIR, "trajectory_maps")
trajectory_maps_dir.mkdir(parents=True, exist_ok=True)
count_per_class = {}
for (x,y) in zip(tensor_train_x, tensor_train_y):
    x = torch.moveaxis(x, 0, -1)
    x_class_id = torch.argmax(y, dim=0).item()
    x_class = label_map[x_class_id]

    if x_class not in count_per_class:
        count_per_class[x_class] = 0

    count_per_class[x_class] += 1

    x_tensor : torch.Tensor = x
    x_img = 255 * x_tensor.cpu().numpy()
    cv2.imwrite(str(Path(trajectory_maps_dir,f"{x_class}_{count_per_class[x_class]}.png")), x_img)

from slr.model.trajectory_map_lstm_model import TMLSTMDataset
train_dataset = TMLSTMDataset(tensor_train_x, tensor_train_y, lstm_data=lstm_tensor_train_x)
val_dataset = TMLSTMDataset(tensor_val_x, tensor_val_y, lstm_data=lstm_tensor_val_x)
# -------------------------------------------------------------------------------------------------

# -- < Starting training > ----------------------------------------
from slr.model.trajectory_map_lstm_model import TMLSTMCLassifier
from slr.model.trainer import Trainer


N_LABELS_MS = tensor_val_y.shape[1]
N_EPOCHS = 50
K_FOLDS = 6

# Model to train
model = TMLSTMCLassifier(n_labels_map[DATASET], n_frames=n_frames_map[DATASET])

# Select optimizer
def optim():
    return torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9,0.999), amsgrad=True, weight_decay=0.0001)
# optim = torch.optim.SGD(model.parameters(S), momentum=0.07, weight_decay=0.0003, lr=0.0005)
# optim =  torch.optim.RMSprop(model.parameters(), weight_decay=0.0003)
print("Dataset: ")
c.cprint(f"\t {DATASET}", color='blue')
print("Model: ")
print(model)
print("Optimizer: ")
print(optim())
print("Seed is: ", torch.seed()) 

# RECUERDA QUE ESTAMOS USANDO 30 FRAMES

trainer = Trainer(model, train_dataset, val_dataset, n_classes=n_labels_map[DATASET], loss_fn=torch.nn.CrossEntropyLoss(), optimizer=optim(), experiment_name="training_slr_with_lstm_and_colored_limbs_20_classes", batch_size=32, acceptance_th=0.8, use_wandb=True)
# result = trainer.run_train(N_EPOCHS, train_title="Training from scratch on gesture rec")  
results = trainer.train_k_folds(
    optim, 
    n_epochs=N_EPOCHS, 
    k_folds=K_FOLDS,
    save_stats_history=True,
    profile_model=True
    )

# Write results into a file
path_to_results_file = Path(OUTPUT_DIR, "results.txt")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Utility function to put lines in pandas latex dataframe
def latex_with_lines(df, *args, **kwargs):
    kwargs['column_format'] = '|'.join([''] + ['l'] * df.index.nlevels
                                            + ['r'] * df.shape[1] + [''])
    res = df.to_latex(*args, **kwargs)
    return res.replace('\\\\\n', '\\\\ \\hline\n')

with path_to_results_file.open("w") as f:

    table_data = {
        "Pérdida de validación" : [r.best_loss_on_validation for r in results],
        "Pérdida de entrenamiento" : [r.best_loss_on_train for r in results],
        "Precisión de validación" : [r.best_acc_on_validation for r in results],
        "Precisión de entrenamiento" : [r.best_acc_on_train for r in results],
        "Tiempo (segundos)" : [r.train_duration for r in results]
    }

    table_data = pd.DataFrame(table_data, index=[i for i in range(1, K_FOLDS + 1)])
    table_data.index.name = "Fold"

    table_data.loc['Prom.'] = table_data.mean()
    print(latex_with_lines(table_data, float_format="%.4f"), file=f)
    
    # Annotate used classes
    print("Classes: ", file=f)
    for class_name in label_map.values():
        print(class_name, file=f)

import matplotlib.pyplot as plt
# Save accuracy and loss plots to file
for (i, result) in enumerate(results):

    # Save loss plot
    result.loss_history.rename({"train" : "Entrenamiento", "validation" : "Validación"})
    result.loss_history.plot()
    plt.xlabel("Época")
    plt.title(f"Pérdida: Entrenamiento vs Validación ({i+1})")
    plt.savefig(str(Path(OUTPUT_DIR, f"loss_fold_{i+1}.png")))
    plt.clf()

    # Save acc plot
    result.acc_history.rename({"train" : "Entrenamiento", "validation" : "Validación"})
    result.acc_history.plot()
    plt.xlabel("Época")
    plt.title(f"Precisión: Entrenamiento vs Validación ({i+1})")
    plt.savefig(str(Path(OUTPUT_DIR, f"accuracy_fold_{i+1}.png")))
    plt.clf()    

    # Save confusion matrix
    trainer.plot_confusion_matrix(result.confusion_matrix, np.array([i for i in range(n_labels_map[DATASET])]), labelmap=label_map, file_to_save=str(Path(OUTPUT_DIR, "conf_matrix.png")))
    plt.savefig(str(Path(OUTPUT_DIR, f"conf_matrix_fold_{i+1}.png")))
    plt.clf()
