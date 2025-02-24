import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from MS_unsteady_Analysis_lib import *


# Initialize the visualizer
flow_field = FlowFieldData(
    npz_file="all_slices_x_2.00e-05.npz",
    sphere_center_y=2.03e-5,  # Adjust to your obstacle’s y-center
    sphere_center_z=2e-5,     # Adjust to your obstacle’s z-center
    sphere_diameter=16e-6,    # Matches a sphere with radius 8e-6 meters
    scale_z=0.3               # Matches z-scaling of 0.3 in your STL
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
