"""
Use this script to generate a plot from a tensorboard csv 
"""

from pathlib import Path
import pandas as pd
import sys
import matplotlib.pyplot as plt

# Sanity check
if len(sys.argv) < 2:
    print("Missing file argument", file=sys.stderr)
    exit(1)

file_path = Path(sys.argv[1])
if not file_path.exists():
    print(f"Provided file '{file_path}' does not exists")


data = pd.read_csv(str(file_path))
data[['Value', 'Step']].plot()
plt.xlabel("epochs")
plt.title("acc")
plt.savefig("my_plot.png")
data[['Step']].plot()
plt.savefig("my_plot2.png")