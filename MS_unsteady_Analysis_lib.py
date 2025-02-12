import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from pydmd import MrDMD, DMD
from scipy.interpolate import griddata
from matplotlib.patches import Circle
from scipy import signal
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from matplotlib.patches import Rectangle
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Snapshot:
    """Represents flow field data at a single timestep"""
    timestep: int
    centers: np.ndarray  # Structured array with 'y' and 'z' coordinates
    values: np.ndarray   # Structured array containing field values
    x_position: float
    
    def get_field_data(self, field: str) -> np.ndarray:
        """Retrieve specified field data"""
        if field not in self.values.dtype.names:
            raise ValueError(f"Field {field} not available. Options: {self.values.dtype.names}")
        return self.values[field]
    
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Extract y and z coordinates"""
        return self.centers['y'], self.centers['z']
    
    def get_field_range(self, field: str) -> Tuple[float, float]:
        """Calculate min/max values for a given field"""
        data = self.get_field_data(field)
        return float(np.min(data)), float(np.max(data))

class FlowFieldData:
    """Loads and manages flow field data"""
    def __init__(self, npz_file: str, 
                 sphere_center_y: float, 
                 sphere_center_z: float, 
                 sphere_diameter: float):
        self.npz_file = npz_file
        self.sphere_center_y = sphere_center_y
        self.sphere_center_z = sphere_center_z
        self.sphere_radius = sphere_diameter / 2
        self.snapshots: Dict[int, Snapshot] = {}
        self.x_position: float = None
        self.available_fields: Tuple[str] = None
        self._load_data()

    def _load_data(self):
        """Load and process data from NPY cache if available, otherwise from NPZ file"""
        logger.info(f"Loading and processing data...")
        # Construct NPY cache filename from NPZ filename
        npy_cache = Path(self.npz_file).with_suffix('.npy')
        try:
            # First try to load from NPY cache
            if npy_cache.exists():
                logger.info(f"Loading from NPY cache: {npy_cache}")
                cache_data = np.load(npy_cache, allow_pickle=True).item()
                self.x_position = cache_data['x_position']
                self.available_fields = cache_data['available_fields']
                self.snapshots = cache_data['snapshots']
                logger.info(f"Successfully loaded {len(self.snapshots)} snapshots from cache")
                return
                
            # If no cache exists, load from NPZ and create cache
            logger.info(f"No cache found, loading from NPZ: {self.npz_file}")
            with np.load(self.npz_file, allow_pickle=True) as data:
                self.x_position = float(data['x_position'])
                self.available_fields = data['values'][0].dtype.names
                
                for i, timestep in enumerate(data['timesteps']):
                    # Copy raw data
                    centers = data['centers'][i].copy()
                    values = data['values'][i].copy()
                    
                    # ==== MODIFIED SPHERE MASKING ====
                    # Calculate distance from sphere center (y-z plane)
                    y_coords = centers['y']
                    z_coords = centers['z']
                    dist_sq = (y_coords - self.sphere_center_y)**2 + (z_coords - self.sphere_center_z)**2
                    mask = dist_sq <= (self.sphere_radius)**2
                    
                    # Zero out all fields EXCEPT 'vol' within sphere
                    for field in values.dtype.names:
                        if field != 'vol':  # Skip volume field
                            values[field][mask] = 0.0
                    # ==== END MODIFIED MASKING ====
                    
                    self.snapshots[int(timestep)] = Snapshot(
                        timestep=int(timestep),
                        centers=centers,
                        values=values,
                        x_position=self.x_position
                    )
                
                # Create cache dictionary
                cache_data = {
                    'x_position': self.x_position,
                    'available_fields': self.available_fields,
                    'snapshots': self.snapshots
                }
                
                # Save cache
                logger.info(f"Creating NPY cache: {npy_cache}")
                np.save(npy_cache, cache_data)
                logger.info(f"Successfully processed {len(self.snapshots)} snapshots and created cache")
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    def get_snapshot(self, timestep: int) -> Optional[Snapshot]:
        """Retrieve snapshot by timestep"""
        return self.snapshots.get(timestep)
    
    def get_timesteps(self) -> List[int]:
        """Return sorted list of available timesteps"""
        return sorted(self.snapshots.keys())
    
    def get_field_range(self, field: str) -> Tuple[float, float]:
        """Calculate global range for a field across all timesteps"""
        if field not in self.available_fields:
            raise ValueError(f"Invalid field: {field}. Available: {self.available_fields}")
        
        global_min = min(snapshot.get_field_range(field)[0] for snapshot in self.snapshots.values())
        global_max = max(snapshot.get_field_range(field)[1] for snapshot in self.snapshots.values())
        return global_min, global_max

class FlowFieldVisualizer:
    """Handles visualization of flow field data"""
    
    def __init__(self, flow_field: FlowFieldData):
        self.flow_field = flow_field
        self.cmap = self._create_diverging_colormap()
    
    @staticmethod
    def _create_diverging_colormap():
        """Create blue-white-red colormap"""
        return LinearSegmentedColormap.from_list("custom_diverging", ['blue', 'white', 'red'], N=256)
    @staticmethod
    def _get_field_colormap(field):
        """Get appropriate colormap for each field type."""
        colormaps = {
            'pressure': 'viridis',
            'temperature': 'plasma',
            'u': 'RdBu_r',  # Diverging colormap for velocities
            'v': 'RdBu_r',
            'w': 'RdBu_r',
            'mach': 'rainbow',
            'vol': 'YlOrRd',
            'NumPart1': 'rainbow',
        }
        return colormaps.get(field, 'viridis')
    
    @staticmethod
    def _get_field_label(field):
        """Get the appropriate label and units for each field."""
        labels = {
            'pressure': 'Pressure (Pa)',
            'temperature': 'Temperature (K)',
            'u': 'X-Velocity (m/s)',
            'v': 'Y-Velocity (m/s)',
            'w': 'Z-Velocity (m/s)',
            'mach': 'Mach Number',
            'vol': 'Volume Fraction',
            'NumPart1': 'N_sim_particles'
        }
        return labels.get(field, field)
    
    def _create_field_patches(self, y_coords: np.ndarray, 
                         z_coords: np.ndarray, 
                         values: np.ndarray,
                         vmin: float = None,
                         vmax: float = None,
                         cmap: str = 'viridis') -> List[Rectangle]:
        """Create patches for each cell."""
        from matplotlib.patches import Rectangle
        
        # Normalize values for colormap
        if vmin is None: vmin = np.min(values)
        if vmax is None: vmax = np.max(values)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.get_cmap(cmap)
        
        # Get cell sizes (assume square cells with side = min distance to neighbors)
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack((y_coords, z_coords)))
        distances, _ = tree.query(np.column_stack((y_coords, z_coords)), k=2)
        cell_sizes = distances[:, 1]  # Use distance to nearest neighbor
        
        # Create patches
        patches = []
        for y, z, size, val in zip(y_coords, z_coords, cell_sizes, values):
            half_size = size/2
            rect = Rectangle((y-half_size, z-half_size), 
                            size, size,
                            facecolor=cmap(norm(val)),
                            edgecolor='none')
            patches.append(rect)
        
        return patches
    

    def plot_field(self, timestep: int, field: str, ax: plt.Axes = None,
                   vmin: float = None, vmax: float = None) -> plt.Axes:
        """Plot a single field at specified timestep"""
        snapshot = self.flow_field.get_snapshot(timestep)
        if not snapshot:
            raise ValueError(f"Timestep {timestep} not found")
        
        ax = ax or plt.gca()
        y, z = snapshot.get_coordinates()
        values = snapshot.get_field_data(field)
        
        scatter = ax.scatter(y, z, c=values, cmap=self.cmap, s=20, vmin=vmin, vmax=vmax)
        plt.colorbar(scatter, ax=ax, label=field)
        
        ax.set(xlabel='Y Position', ylabel='Z Position',
               title=f'{field} at Timestep {timestep}\nX={snapshot.x_position:.2e}')
        ax.set_aspect('equal')
        return ax
    
    def create_field_animation(self, field: str, 
                           debug: bool = False, 
                           fps: int = 10,
                           start_idx: Optional[int] = None,
                           end_idx: Optional[int] = None,
                           step: int = 1,
                           save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create an animation of field evolution over time using imshow for speed.
        This version parallelizes the pre-calculation of interpolated grids.
        """
        # Prepare grid and figure
        Y, Z, (y_min, y_max, z_min, z_max) = self._prepare_interpolation_grid()
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Add cylinder (if needed)
        cylinder = plt.Circle((10e-6, 5e-6), 1e-6, 
                                color='black', alpha=0.9, zorder=5)
        ax.add_patch(cylinder)
        
        # Get timesteps and data ranges
        timesteps = self.flow_field.get_timesteps()
        if start_idx is None: start_idx = 0
        if end_idx is None: end_idx = len(timesteps)
        frame_indices = list(range(start_idx, end_idx, step))
        
        vmin, vmax = self.flow_field.get_field_range(field)
        
        # Pre-calculate interpolated grid values for each frame in parallel.
        logger.info(f"Pre-calculating interpolation grids for {field} using parallel processing...")
        
        def compute_grid(idx):
            snapshot = self.flow_field.get_snapshot(timesteps[idx])
            return self._interpolate_timestep(snapshot.centers, snapshot.values, field, Y, Z)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            grid_values = list(executor.map(compute_grid, frame_indices))
        
        # Instead of using contourf, use imshow to display the grid.
        extent = (y_min*1e6, y_max*1e6, z_min*1e6, z_max*1e6)
        im = ax.imshow(grid_values[0], origin='lower', extent=extent,
                    cmap=self._get_field_colormap(field), vmin=vmin, vmax=vmax,
                    aspect='equal')
        plt.colorbar(im, ax=ax, label=self._get_field_label(field))
        
        # Set up axes
        ax.set_xlabel('Y coordinate (μm)')
        ax.set_ylabel('Z coordinate (μm)')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.grid(True)
        
        def update(frame):
            # Update the image data
            im.set_data(grid_values[frame])
            timestep = timesteps[frame_indices[frame]]
            ax.set_title(f'{self._get_field_label(field)} at X = {self.flow_field.x_position*1e6:.2f} μm\n'
                        f'Timestep {timestep}')
            return [im]
        
        # Create animation with blitting enabled for further speedup
        anim = FuncAnimation(fig, update, 
                            frames=len(frame_indices),
                            interval=1000/fps, 
                            blit=True)
        
        if save_path:
            logger.info(f"Saving animation to {save_path}")
            writer = mpl.animation.FFMpegWriter(fps=fps, bitrate=5000)
            anim.save(save_path, writer=writer)
            plt.close()
        else:
            plt.show()
            
        return anim

    def plot_multiple_fields(self, timestep: int, fields: List[str],
                            use_global_range: bool = False) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Plot multiple fields from the same timestep"""
        n_fields = len(fields)
        fig, axes = plt.subplots(1, n_fields, figsize=(5*n_fields, 6))
        axes = axes if isinstance(axes, np.ndarray) else [axes]
        
        for ax, field in zip(axes, fields):
            vmin, vmax = self.flow_field.get_field_range(field) if use_global_range else (None, None)
            self.plot_field(timestep, field, ax, vmin, vmax)
        
        plt.tight_layout()
        return fig, axes
    
    def _prepare_interpolation_grid(self, y_min=0, y_max=20e-6, z_min=0, z_max=20e-6, grid_size=200):
        """Prepare common interpolation grid for animations"""
        yi = np.linspace(y_min, y_max, grid_size)
        zi = np.linspace(z_min, z_max, grid_size)
        Y, Z = np.meshgrid(yi, zi)
        return Y, Z, (y_min, y_max, z_min, z_max)

    def _interpolate_timestep(self, centers, values, field, Y, Z):
        """Interpolate field values for a single timestep onto regular grid"""
        points = np.column_stack((centers['y'], centers['z']))
        field_values = values[field]
        grid_values = self._interpolate_field_values(points, field_values, Y, Z)
        return grid_values

    def _interpolate_field_values(self, points, values, Y, Z):
        """Interpolate values using the hybrid approach"""
        return interpolate_2d_data(points[:, 0], points[:, 1], values, Y.shape[0])[2]

def print_data_summary(flow_field: FlowFieldData):
    """Print structured data overview"""
    print("\n=== Dataset Summary ===")
    print(f"X-position: {flow_field.x_position:.2e}")
    print(f"Available fields: {flow_field.available_fields}")
    print(f"Timesteps available ({len(flow_field.get_timesteps())}): {flow_field.get_timesteps()[:3]}...")
    print(f"Total snapshots: {len(flow_field.snapshots)}")

def analyze_snapshot(snapshot: Snapshot):
    """Perform detailed analysis of a snapshot"""
    print("\n=== Snapshot Analysis ===")
    print(f"Timestep: {snapshot.timestep}")
    print(f"Coordinates shape: {snapshot.centers.shape}")
    print(f"Field value ranges:")
    for field in snapshot.values.dtype.names:
        fmin, fmax = snapshot.get_field_range(field)
        print(f"- {field}: {fmin:.2f} to {fmax:.2f}")

def get_field_snapshots(flow_field: FlowFieldData, field_name: str) -> np.ndarray:
    """
    Returns time series of field snapshots as a numpy array.
    
    Args:
        field_name: Name of the field to collect (must be in available_fields)
        
    Returns:
        np.ndarray: Array of shape (num_snapshots, y_dim, z_dim) containing
                    the requested field values across all timesteps
    """
    if field_name not in flow_field.available_fields:
        raise ValueError(f"Invalid field '{field_name}'. Available fields: {flow_field.available_fields}")
    
    timesteps = flow_field.get_timesteps()
    first_snapshot = flow_field.get_snapshot(timesteps[0])
    sample_data = first_snapshot.get_field_data(field_name)
    data_array = np.zeros((len(timesteps), *sample_data.shape))
    
    for i, ts in enumerate(timesteps):
        snapshot = flow_field.get_snapshot(ts)
        data_array[i] = snapshot.get_field_data(field_name)
        
    return data_array

def plot_interpolated_temperature(y_coords: np.ndarray, 
                                z_coords: np.ndarray, 
                                temperature_values: np.ndarray,
                                num_grid_points: int = 200,
                                title: str = "Temperature Field",
                                cmap: str = 'coolwarm',
                                figsize: Tuple[int, int] = (10, 8)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a 2D interpolated plot of temperature field data.
    
    Args:
        y_coords: Array of y coordinates
        z_coords: Array of z coordinates
        temperature_values: Array of temperature values at each (y,z) point
        num_grid_points: Number of points to use in each dimension for interpolation grid
        title: Plot title
        cmap: Colormap to use for visualization
        figsize: Figure size in inches
    
    Returns:
        fig, ax: The figure and axes objects
    """
    # Create regular grid to interpolate the data
    y_min, y_max = y_coords.min(), y_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()
    
    # Create meshgrid
    yi = np.linspace(y_min, y_max, num_grid_points)
    zi = np.linspace(z_min, z_max, num_grid_points)
    yi, zi = np.meshgrid(yi, zi)
    
    # Interpolate temperature values on regular grid
    points = np.column_stack((y_coords.flatten(), z_coords.flatten()))
    temperature_grid = griddata(points, 
                              temperature_values.flatten(), 
                              (yi, zi), 
                              method='cubic',
                              fill_value=np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot interpolated data
    im = ax.pcolormesh(yi, zi, temperature_grid, 
                      shading='auto', 
                      cmap=cmap)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Temperature')
    
    # Set labels and title
    ax.set_xlabel('Y Position')
    ax.set_ylabel('Z Position')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    return fig, ax

# def interpolate_2d_data(y_coords: np.ndarray,
#                        z_coords: np.ndarray,
#                        values: np.ndarray,
#                        num_grid_points: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Create a 2D interpolated grid from flattened data arrays.
    
#     Args:
#         y_coords: Flattened array of y coordinates
#         z_coords: Flattened array of z coordinates
#         values: Flattened array of values at each (y,z) point
#         num_grid_points: Number of points to use in each dimension for interpolation grid
    
#     Returns:
#         yi: 2D array of interpolated y coordinates
#         zi: 2D array of interpolated z coordinates
#         interpolated_values: 2D array of interpolated values
#     """
#     # Ensure inputs are flattened
#     y_coords = np.asarray(y_coords).flatten()
#     z_coords = np.asarray(z_coords).flatten()
#     values = np.asarray(values).flatten()
    
#     # Create regular grid
#     y_min, y_max = y_coords.min(), y_coords.max()
#     z_min, z_max = z_coords.min(), z_coords.max()
    
#     yi = np.linspace(y_min, y_max, num_grid_points)
#     zi = np.linspace(z_min, z_max, num_grid_points)
#     yi, zi = np.meshgrid(yi, zi)
    
#     # Interpolate values
#     points = np.column_stack((y_coords, z_coords))
#     interpolated_values = griddata(points, values, (yi, zi), 
#                                  method='cubic', 
#                                  fill_value=np.nan)
    
#     return yi, zi, interpolated_values
    
def interpolate_2d_data(y_coords: np.ndarray,
                       z_coords: np.ndarray,
                       values: np.ndarray,
                       num_grid_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D interpolated grid using a hybrid approach for better accuracy.
    
    Args:
        y_coords: Flattened array of y coordinates
        z_coords: Flattened array of z coordinates
        values: Flattened array of values at each (y,z) point
        num_grid_points: Number of points for interpolation grid
    """
    # Ensure inputs are flattened
    y_coords = np.asarray(y_coords).flatten()
    z_coords = np.asarray(z_coords).flatten()
    values = np.asarray(values).flatten()
    
    # Create regular grid
    y_min, y_max = y_coords.min(), y_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()
    
    yi = np.linspace(y_min, y_max, num_grid_points)
    zi = np.linspace(z_min, z_max, num_grid_points)
    yi, zi = np.meshgrid(yi, zi)
    
    # Points for interpolation
    points = np.column_stack((y_coords, z_coords))
    
    # First pass: Use cubic interpolation
    interpolated_cubic = griddata(points, values, (yi, zi), 
                                method='cubic', 
                                fill_value=np.nan)
    
    # Second pass: Use linear interpolation
    interpolated_linear = griddata(points, values, (yi, zi), 
                                 method='linear', 
                                 fill_value=np.nan)
    
    # Third pass: Use nearest neighbor for remaining NaN values
    interpolated_nearest = griddata(points, values, (yi, zi), 
                                  method='nearest', 
                                  fill_value=np.nan)
    
    # Combine results:
    # 1. Use cubic where it's well-behaved (not NaN and within data range)
    # 2. Use linear where cubic fails
    # 3. Use nearest for remaining points
    
    # Get valid value range from input data
    valid_min = np.nanmin(values)
    valid_max = np.nanmax(values)
    
    # Create mask for invalid cubic interpolation results
    cubic_mask = (np.isnan(interpolated_cubic) | 
                 (interpolated_cubic < valid_min) | 
                 (interpolated_cubic > valid_max))
    
    # Create final interpolated grid
    interpolated_values = interpolated_cubic.copy()
    
    # Replace invalid cubic values with linear interpolation
    linear_mask = cubic_mask & ~np.isnan(interpolated_linear)
    interpolated_values[linear_mask] = interpolated_linear[linear_mask]
    
    # Fill remaining NaN values with nearest neighbor
    remaining_mask = np.isnan(interpolated_values)
    interpolated_values[remaining_mask] = interpolated_nearest[remaining_mask]
    
    return yi, zi, interpolated_values

def get_time_series_at_location(flow_field: FlowFieldData, 
                              target_y: float, 
                              target_z: float, 
                              field: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get time series data from the cell closest to the specified (y,z) coordinates.
    
    Args:
        flow_field: FlowFieldData object
        target_y: Y coordinate of interest
        target_z: Z coordinate of interest
        field: Name of the field to extract (e.g., 'pressure', 'temperature', etc.)
    
    Returns:
        times: Array of timesteps
        values: Array of field values at the specified location
    """
    # Get first snapshot to find the closest point
    timesteps = flow_field.get_timesteps()
    first_snapshot = flow_field.get_snapshot(timesteps[0])
    y, z = first_snapshot.get_coordinates()
    
    # Calculate distances to all points
    distances = np.sqrt((y - target_y)**2 + (z - target_z)**2)
    
    # Find index of closest point
    closest_idx = np.argmin(distances)
    
    # Get minimum distance
    min_distance = distances[closest_idx]
    
    # Extract time series data
    times = np.array(timesteps) * 1.25e-8  # Convert to seconds using given timestep
    values = np.zeros(len(timesteps))
    
    for i, ts in enumerate(timesteps):
        snapshot = flow_field.get_snapshot(ts)
        field_data = snapshot.get_field_data(field)
        values[i] = field_data[closest_idx]
    
    print(f"Closest point found at y={y[closest_idx]:.2e}, z={z[closest_idx]:.2e}")
    print(f"Distance from requested point: {min_distance:.2e} meters")
    
    return times, values

def plot_interpolated_field(y_coords: np.ndarray, 
                          z_coords: np.ndarray, 
                          field_values: np.ndarray,
                          field_name: str,
                          num_grid_points: int = 200,
                          title: str = None,
                          cmap: str = 'coolwarm',
                          figsize: Tuple[int, int] = (10, 8)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a 2D interpolated plot of any field data.
    
    Args:
        y_coords: Array of y coordinates
        z_coords: Array of z coordinates
        field_values: Array of field values at each (y,z) point
        field_name: Name of the field being plotted
        num_grid_points: Number of points to use in each dimension for interpolation grid
        title: Plot title (if None, will use field name)
        cmap: Colormap to use for visualization
        figsize: Figure size in inches
    
    Returns:
        fig, ax: The figure and axes objects
    """
    # Create regular grid to interpolate the data
    y_min, y_max = y_coords.min(), y_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()
    
    # Create meshgrid
    yi = np.linspace(y_min, y_max, num_grid_points)
    zi = np.linspace(z_min, z_max, num_grid_points)
    yi, zi = np.meshgrid(yi, zi)
    
    # Interpolate values on regular grid
    points = np.column_stack((y_coords.flatten(), z_coords.flatten()))
    field_grid = griddata(points, 
                         field_values.flatten(), 
                         (yi, zi), 
                         method='cubic',
                         fill_value=np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot interpolated data
    im = ax.pcolormesh(yi, zi, field_grid, 
                      shading='auto', 
                      cmap=cmap)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label=field_name)
    
    # Set labels and title
    ax.set_xlabel('Y Position')
    ax.set_ylabel('Z Position')
    ax.set_title(title if title else f"{field_name} Field")
    ax.set_aspect('equal')
    
    return fig, ax

def visualize_snapshot(flow_field: FlowFieldData, 
                      timestep: int,
                      field: str,
                      show_scatter: bool = True,
                      cmap: str = 'coolwarm') -> None:
    """
    Visualize a specific field snapshot with both interpolation and original points.
    
    Args:
        flow_field: FlowFieldData object containing the snapshots
        timestep: Timestep to visualize
        field: Name of the field to visualize (e.g., 'temperature', 'pressure', 'vol', etc.)
        show_scatter: If True, also shows the original data points
        cmap: Colormap to use for visualization
    """
    # Get snapshot data
    snapshot = flow_field.get_snapshot(timestep)
    if snapshot is None:
        raise ValueError(f"Timestep {timestep} not found")
    
    # Verify field exists
    if field not in flow_field.available_fields:
        raise ValueError(f"Field '{field}' not found. Available fields: {flow_field.available_fields}")
    
    # Get coordinates and field data
    y, z = snapshot.get_coordinates()
    field_data = snapshot.get_field_data(field)
    
    if show_scatter:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot interpolated field
        plot_interpolated_field(y, z, field_data, 
                              field_name=field,
                              title=f"Interpolated {field} Field (t={timestep})",
                              cmap=cmap,
                              figsize=(15, 6))
        
        # Plot original points
        scatter = ax2.scatter(y, z, c=field_data, 
                            cmap=cmap, s=20)
        plt.colorbar(scatter, ax=ax2, label=field)
        ax2.set_aspect('equal')
        ax2.set_title(f"Original Data Points (t={timestep})")
        ax2.set_xlabel('Y Position')
        ax2.set_ylabel('Z Position')
        
    else:
        # Plot only interpolated field
        plot_interpolated_field(y, z, field_data,
                              field_name=field,
                              title=f"{field} Field (t={timestep})",
                              cmap=cmap)
    
    plt.tight_layout()
    plt.show()
