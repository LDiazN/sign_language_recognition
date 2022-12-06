"""
Count how many instances we have available for the MS-ASL dataset. 
- Count how many vids per class there are in the dataset
- Count how many vids per class there are currently downloaded in this computer 
"""
import pandas as pd
from typing import Iterable
from slr.dataset_manager.dataset_managers import MicrosoftDatasetManager, SignDescription
from slr.model.data_ingestor import DataIngestor


ms_manager = MicrosoftDatasetManager()
ingestor = DataIngestor(file_manager=ms_manager.file_manager, dataset_manager=ms_manager)
def get_dataset_description(manager : MicrosoftDatasetManager) -> Iterable[SignDescription]:
    """
        Utility function to read all descriptions from this dataset manager
    """    
    desc_funcs = [manager.read_test_dataset_description, manager.read_train_dataset_description, manager.read_val_dataset_description]
    for read_func in desc_funcs:
        for elem in read_func():
            yield elem

# Total amount of instances per vid. class -> int
absolute_count = {}
absolute_vids = set({})  # to count how many vids there's available vs how many we actually have
for desc in get_dataset_description(ms_manager):
    if desc.label not in absolute_count:
        absolute_count[desc.label] = 0
    absolute_count[desc.label] += 1
    absolute_vids.add(desc.url)


# To count the amount of repetitions available, create a map from vid url to labels, and 
# check which videos we do have
actual_count = {}
actual_vids = set({})
for (_, desc) in ingestor.retrieve_all():

    if desc.label not in actual_count:
        actual_count[desc.label] = 0

    actual_vids.add(desc.url)
    actual_count[desc.label] +=1


label_map = ms_manager.label_map

data = {
    "Clase" : [],
    "Total"  : [],
    "Disponible" : [],
    "% disponible" : [] 
    }

for key in label_map.keys():
    
    data["Clase"].append(label_map[key])
    data["Total"].append(absolute_count[key])
    
    if key not in actual_count:
        data["Disponible"].append(0)
        data["% disponible"].append(0)
        continue

    data["Disponible"].append(actual_count[key])
    data["% disponible"].append(100 * (actual_count[key] / absolute_count[key]))

# compute total instance available vs total instances gathered
total_instances = sum(data["Total"])
total_available = sum(data["Disponible"])
total_available_percent = (total_available / total_instances) * 100

def latex_with_lines(df : pd.DataFrame, *args, **kwargs):
    kwargs['column_format'] = '|'.join([''] + ['l'] * df.index.nlevels
                                            + ['r'] * df.shape[1] + [''])
    res = df.to_latex(*args, **kwargs)
    return res.replace('\\\\\n', '\\\\ \\hline\n')
df = pd.DataFrame(data)
df.sort_values(by="% disponible", ascending=False, inplace=True)

print("Availability per class")
print(latex_with_lines(df,float_format="%.4f", index = False))

print("Overall results: ")
print("Dataset vids: ", len(absolute_vids))
print("Available vids: ", len(actual_vids))
print("% of vids in disk: ", (len(actual_vids) / len(absolute_vids)) * 100)
print("Total instances: ", total_instances)
print("Total available: ", total_available)
print("% Total available: ", total_available_percent)