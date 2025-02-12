import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from MS_unsteady_Analysis_lib import *


# Initialize the visualizer
flow_field = FlowFieldData(
    "all_slices_x_2.00e-05.npz",
    sphere_center_y=1.03e-5,   # Match visualization parameters
    sphere_center_z=8.5e-6,
    sphere_diameter=1e-6
)
visualizer = FlowFieldVisualizer(flow_field)


# Create animations for multiple fields
fields = ['temperature', 'pressure', 'u', 'v', 'w', 'mach', 'vol', 'NumPart1']
for field in fields:
    print(f"\nProcessing {field} animation...")
    visualizer.create_field_animation(
        field=field,
        fps=15,
        save_path=f'{field}_evolution.mp4'
    )
