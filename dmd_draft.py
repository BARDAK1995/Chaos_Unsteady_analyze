import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD
from matplotlib.patches import Circle
from scipy import signal
from MS_unsteady_Analysis_lib import *
from matplotlib.patches import Ellipse  # Add this import at the top of your file
# Example usage
# flow_field = FlowFieldData("all_slices_x_1.00e-05.npz")
# print_data_summary(flow_field)


flow_field = FlowFieldData(
    npz_file="all_slices_x_2.00e-05.npz",
    sphere_center_y=2.03e-5,  # Adjust to your obstacle’s y-center
    sphere_center_z=2e-5,     # Adjust to your obstacle’s z-center
    sphere_diameter=16e-6,    # Matches a sphere with radius 8e-6 meters
    scale_z=0.3               # Matches z-scaling of 0.3 in your STL
)
print_data_summary(flow_field)

# Get first available timestep
timesteps = flow_field.get_timesteps()
timestep = timesteps[0]

snapshot = flow_field.get_snapshot(timestep)
# Analyze initial snapshot
analyze_snapshot(snapshot)

# Get temperature data
temperature_data = get_field_snapshots(flow_field, 'temperature')
print(f"\nTemperature data shape: {temperature_data.shape}")
print(f"First snapshot values:\n{temperature_data[0]}")

# Visualize fields
visualizer = FlowFieldVisualizer(flow_field)
target_fields = ['temperature', 'mach', 'pressure','NumPart1']
fig, axes = visualizer.plot_multiple_fields(timestep, target_fields)
plt.suptitle(f"Flow Field Comparison - Timestep {timestep}")
plt.show()

# visualize_temperature_snapshot(flow_field, timestep, show_scatter=True)
visualize_snapshot(flow_field, timestep, field='temperature')
# For pressure field
visualize_snapshot(flow_field, timestep, field='pressure', cmap='viridis')
# For volume fraction
visualize_snapshot(flow_field, timestep, field='vol', cmap='YlOrRd')

y, z = snapshot.get_coordinates()
temperature = snapshot.get_field_data('temperature')
# Use helper function directly
yi, zi, interpolated_temp = interpolate_2d_data(
    y, z, temperature,
    num_grid_points=100,
    sphere_center_y=flow_field.sphere_center_y,
    sphere_center_z=flow_field.sphere_center_z,
    sphere_radius=flow_field.sphere_diameter / 2,
    scale_z=flow_field.scale_z
    )
# Create custom visualization
plt.figure(figsize=(10, 8))
plt.pcolormesh(yi, zi, interpolated_temp, shading='auto', cmap='coolwarm')
plt.colorbar(label='Temperature')
plt.xlabel('Y Position')
plt.ylabel('Z Position')
plt.title('Custom Temperature Visualization')
plt.gca().set_aspect('equal')
plt.show()

Volume_data = get_field_snapshots(flow_field, 'vol')
dx_scale = Volume_data**(1/3) * 10**6#microns
A_x_scale = dx_scale**2
A_x_scale_correction = A_x_scale / A_x_scale.max()


yi, zi, interpolated_temp = interpolate_2d_data(
    y, z, A_x_scale_correction[10, :],
    num_grid_points=100,
    sphere_center_y=flow_field.sphere_center_y,
    sphere_center_z=flow_field.sphere_center_z,
    sphere_radius=flow_field.sphere_diameter / 2,
    scale_z=flow_field.scale_z
)

# Create custom visualization
plt.figure(figsize=(10, 8))
plt.pcolormesh(yi, zi, interpolated_temp, shading='auto', cmap='coolwarm')
plt.colorbar(label='Temperature')
plt.xlabel('Y Position')
plt.ylabel('Z Position')
plt.title('Custom Temperature Visualization')
plt.gca().set_aspect('equal')
plt.show()


temperature_data = get_field_snapshots(flow_field, 'pressure') * A_x_scale_correction
temperature_data_corrected=temperature_data[:,:]
# Reshape data for DMD (snapshots as columns)
X = temperature_data_corrected.reshape((temperature_data_corrected.shape[0], -1)).T
# Mean subtraction
X_mean = np.mean(X, axis=1, keepdims=True)
X_centered = X - X_mean

# Perform DMD with 10 modes
dmd = DMD(
    svd_rank=10,          # Number of modes to extract
    exact=True,           # Use exact DMD formulation
    opt=True,             # Optimized DMD implementation
    tikhonov_regularization=None  # No regularization for vanilla analysis
)
dmd.fit(X_centered)
# Get coordinates from a snapshot
dt = 5e-9  # timestep in seconds

# Get DMD eigenvalues
eigenvalues = dmd.eigs

# Calculate frequencies in Hz
# freq = ln(λ)/(2π*dt) where λ are the DMD eigenvalues
frequencies_hz = np.log(eigenvalues).imag / (2 * np.pi * dt)

# Convert to kHz
frequencies_khz = frequencies_hz / 1000

# Print frequencies and sort by magnitude of mode
mode_magnitudes = np.linalg.norm(dmd.modes, axis=0)
sorted_indices = np.argsort(mode_magnitudes)[::-1]  # Sort in descending order

print("\nMode frequencies and magnitudes:")
print("Mode | Frequency (kHz) | Mode Magnitude")
print("-" * 40)
for i in sorted_indices:
    print(f"{i+1:4d} | {abs(frequencies_khz[i]):13.2f} | {mode_magnitudes[i]:13.2e}")

# Get coordinates from a snapshot
y, z = flow_field.get_snapshot(flow_field.get_timesteps()[0]).get_coordinates()
cylinder_y = 1.03e-5  # y position
cylinder_z = 7e-6     # z position
cylinder_diameter = 6e-6  # diameter

# Plot modes with frequencies in titles
for i in range(10):
    plt.figure(figsize=(10, 8))
    yi, zi, interpolated_temp = interpolate_2d_data(
        y, z, 
        np.abs(dmd.modes[:, i]) / A_x_scale_correction[12, :],
        num_grid_points=100,
        sphere_center_y=flow_field.sphere_center_y,
        sphere_center_z=flow_field.sphere_center_z,
        sphere_radius=flow_field.sphere_diameter / 2,
        scale_z=flow_field.scale_z
    )
    im = plt.pcolormesh(yi, zi, interpolated_temp, shading='auto', cmap='coolwarm')
    plt.colorbar(im, label=f'Mode {i+1} Magnitude')

    # Add the ellipse with orange crosshatch
    ellipse = Ellipse(
        xy=(flow_field.sphere_center_y, flow_field.sphere_center_z),  # Center
        width=flow_field.sphere_diameter,                             # Full width
        height=flow_field.sphere_diameter * flow_field.scale_z,       # Scaled height
        angle=0,                                                      # No rotation
        fill=False,                                                   # No solid fill
        edgecolor='orange',                                           # Edge color
        hatch='+',                                                    # Crosshatch pattern
        linewidth=2,                                                  # Thicker lines for visibility
        alpha=0.7                                                     # Transparency
    )
    plt.gca().add_patch(ellipse)
    
    # Add labels and title with frequency
    plt.xlabel('Y Position')
    plt.ylabel('Z Position')
    plt.title(f'DMD Mode {i+1} (f = {abs(frequencies_khz[i]):.2f} kHz)')
    
    # Set aspect ratio to equal
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

probe_points = [
    (1.03e-5, 3.5e-6),
    (1.5e-5, 4.0e-6),
    (2.0e-5, 5.0e-6),
    (2.5e-5, 6.0e-6)
]
field_name = 'pressure'  # Field to analyze and visualize

# Get time series data for all probe points
times, probe_coords, pressure_series_list = get_time_series_at_locations(flow_field, probe_points, field_name)

# Get timesteps for selecting a snapshot
timesteps = flow_field.get_timesteps()

# Plot the flow field with probe points at the first timestep
plot_flow_field_with_probes(flow_field, timesteps[0], field_name, probe_coords)

# Create figure for time series
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['b', 'g', 'r', 'c']  # Colors for four probe points
for i, (probe_coord, series) in enumerate(zip(probe_coords, pressure_series_list)):
    ax.plot(times * 1e6, series, color=colors[i], 
            label=f'Probe {i+1}: y={probe_coord[0]:.2e}, z={probe_coord[1]:.2e}')
ax.set_xlabel('Time (μs)')
ax.set_ylabel(f'{field_name.capitalize()}')
ax.set_title(f'{field_name.capitalize()} Time Series at Probe Points')
ax.grid(True)
ax.legend()
plt.show()

# Create figure for PSD
fig, ax = plt.subplots(figsize=(12, 6))
fs = 1 / (1.25e-8)  # Sampling frequency in Hz
for i, series in enumerate(pressure_series_list):
    f, Pxx = signal.welch(series, fs, nperseg=256)
    f_khz = f / 1000  # Convert to kHz
    ax.semilogy(f_khz, Pxx, color=colors[i], label=f'Probe {i+1}')
ax.set_xlabel('Frequency (kHz)')
ax.set_ylabel('PSD')
ax.set_title('Power Spectral Density at Probe Points')
ax.grid(True)
ax.legend()
plt.show()

# # Print dominant frequencies
# peak_indices = signal.find_peaks(Pxx)[0]
# dominant_freqs = f_khz[peak_indices]
# dominant_powers = Pxx[peak_indices]

# # Sort by power
# sorted_indices = np.argsort(dominant_powers)[::-1]
# dominant_freqs = dominant_freqs[sorted_indices]
# dominant_powers = dominant_powers[sorted_indices]

# print("\nDominant frequencies (top 5):")
# print("Frequency (kHz) | Power")
# print("-" * 30)
# for freq, power in zip(dominant_freqs[:5], dominant_powers[:5]):
#     print(f"{freq:13.2f} | {power:.2e}")