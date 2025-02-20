# Simulation Data (sim_data)

This folder contains the simulation data used in our work.

## Options for Obtaining Data

You have two options to populate this folder:

1. **Download Data from the Paper:**  
   If you have access to the dataset published with the paper, download the data and place the files in this folder.

2. **Generate Your Own Data:**  
   Alternatively, you can use the provided data collection scripts in the repository to generate your own simulation data.  
   - Run the data collection script ( `python3 data_collection.py`).
   - The collected data will be automatically saved as `.npz` files in this folder.

## Data Format

The data is saved as `.npz` files containing the following keys:
- **images:** Array of processed images.
- **poses:** Array of pose matrices (if applicable).
- **angles:** Array of joint angles.
- **focal:** Focal length value of the camera.

## Usage

Once the data is available in this folder, you can load it in your analysis scripts using NumPy, e.g.:

```python
import numpy as np
robot_ID = 1 # 1, 2 or 3
data = np.load(f'data/sim_data/sim_data_robo{robot_ID}(arm).npz')
images = data['images']
poses = data['poses']
angles = data['angles']
focal = data['focal']
