import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD
from matplotlib.patches import Circle
from scipy import signal
from MS_unsteady_Analysis_lib import *
# Example usage
# flow_field = FlowFieldData("all_slices_x_1.00e-05.npz")
# print_data_summary(flow_field)


flow_field = FlowFieldData(
    "all_slices_x_1.00e-05.npz",
    sphere_center_y=1.03e-5,   # Match visualization parameters
    sphere_center_z=7e-6,
    sphere_diameter=6e-6
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
yi, zi, interpolated_temp = interpolate_2d_data(y, z, temperature)

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
yi, zi, interpolated_temp = interpolate_2d_data(y, z, A_x_scale_correction[10,:] )
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
    yi, zi, interpolated_temp = interpolate_2d_data(y, z, 
                                                np.abs(dmd.modes[:, i])/A_x_scale_correction[12,:])  # Add this parameter
    # yi, zi, interpolated_temp = interpolate_2d_data(y, z, 
    #                                             (dmd.modes[:, i]).real/A_x_scale_correction[12,:])
    im = plt.pcolormesh(yi, zi, interpolated_temp, shading='auto', cmap='coolwarm')
    plt.colorbar(im, label=f'Mode {i+1} Magnitude')

    # Add cylinder as a circle
    circle = Circle((cylinder_y, cylinder_z), cylinder_diameter/2,
                   fill=True, color='black', alpha=0.1)
    plt.gca().add_patch(circle)
    
    # Add labels and title with frequency
    plt.xlabel('Y Position')
    plt.ylabel('Z Position')
    plt.title(f'DMD Mode {i+1} (f = {abs(frequencies_khz[i]):.2f} kHz)')
    
    # Set aspect ratio to equal
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()



# Example usage:
target_y = 1.03e-5  # Example y coordinate
target_z = 3.5e-6     # Example z coordinate
field_name = 'pressure'  # or any other available field

# Get time series data
times, pressure_values = get_time_series_at_location(flow_field, target_y, target_z, field_name)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot time series
ax1.plot(times*1e6, pressure_values, 'b-', linewidth=2)
ax1.set_xlabel('Time (μs)')
ax1.set_ylabel(f'{field_name.capitalize()}')
ax1.set_title(f'{field_name.capitalize()} Time Series at (y={target_y:.2e}, z={target_z:.2e})')
ax1.grid(True)

# Calculate PSD using Welch's method
fs = 1/(1.25e-8)  # Sampling frequency in Hz
f, Pxx = signal.welch(pressure_values, fs, nperseg=256)

# Convert frequency to kHz for better readability
f_khz = f/1000

# Plot PSD
ax2.semilogy(f_khz, Pxx)
ax2.set_xlabel('Frequency (kHz)')
ax2.set_ylabel('PSD')
ax2.set_title('Power Spectral Density')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Print dominant frequencies
peak_indices = signal.find_peaks(Pxx)[0]
dominant_freqs = f_khz[peak_indices]
dominant_powers = Pxx[peak_indices]

# Sort by power
sorted_indices = np.argsort(dominant_powers)[::-1]
dominant_freqs = dominant_freqs[sorted_indices]
dominant_powers = dominant_powers[sorted_indices]

print("\nDominant frequencies (top 5):")
print("Frequency (kHz) | Power")
print("-" * 30)
for freq, power in zip(dominant_freqs[:5], dominant_powers[:5]):
    print(f"{freq:13.2f} | {power:.2e}")