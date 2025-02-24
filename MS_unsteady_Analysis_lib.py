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
from typing import Tuple

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
             sphere_diameter: float,
             scale_z: float = 1.0):
        self.npz_file = npz_file
        self.sphere_center_y = sphere_center_y
        self.sphere_center_z = sphere_center_z
        self.sphere_diameter = sphere_diameter  # Store diameter
        self.sphere_radius = sphere_diameter / 2  # Compute and store radius
        self.scale_z = scale_z
        self.snapshots: Dict[int, Snapshot] = {}
        self.x_position: float = None
        self.available_fields: Tuple[str] = None
        self._load_data()

    def _apply_elliptical_mask(self, centers: np.ndarray, values: np.ndarray) -> None:
        """Apply elliptical masking to the data in-place"""
        y_coords = centers['y']
        z_coords = centers['z']
        dy = (y_coords - self.sphere_center_y) / self.sphere_radius
        dz = (z_coords - self.sphere_center_z) / (self.sphere_radius * self.scale_z)
        dist_sq = dy**2 + dz**2
        mask = dist_sq <= 1.0  # Inside the ellipse if ≤ 1
        print(f"Points inside mask: {np.sum(mask)} out of {len(y_coords)}")
        for field in values.dtype.names:
            if field != 'vol':  # Skip volume field
                values[field][mask] = 0.0

    def _load_data(self):
        """Load and process data from NPY cache if available, otherwise from NPZ file"""
        logger.info(f"Loading and processing data...")
        npy_cache = Path(self.npz_file).with_suffix('.npy')
        try:
            # Try to load from NPY cache
            if npy_cache.exists():
                logger.info(f"Loading from NPY cache: {npy_cache}")
                cache_data = np.load(npy_cache, allow_pickle=True).item()
                
                # Check if mask parameters match current instance
                params_match = (
                    cache_data.get('sphere_center_y') == self.sphere_center_y and
                    cache_data.get('sphere_center_z') == self.sphere_center_z and
                    cache_data.get('sphere_diameter') == self.sphere_diameter and
                    cache_data.get('scale_z') == self.scale_z
                )
                
                self.x_position = cache_data['x_position']
                self.available_fields = cache_data['available_fields']
                self.snapshots = cache_data['snapshots']
                
                if not params_match:
                    logger.info("Mask parameters have changed, reapplying mask to cached data")
                    # Reapply mask to all snapshots in cache
                    for snapshot in self.snapshots.values():
                        self._apply_elliptical_mask(snapshot.centers, snapshot.values)
                    # Update cache with new parameters after masking
                    cache_data = {
                        'x_position': self.x_position,
                        'available_fields': self.available_fields,
                        'snapshots': self.snapshots,
                        'sphere_center_y': self.sphere_center_y,
                        'sphere_center_z': self.sphere_center_z,
                        'sphere_diameter': self.sphere_diameter,
                        'scale_z': self.scale_z
                    }
                    logger.info(f"Updating NPY cache: {npy_cache}")
                    np.save(npy_cache, cache_data)
                else:
                    logger.info(f"Mask parameters match, using cached data as-is")
                
                logger.info(f"Successfully loaded {len(self.snapshots)} snapshots from cache")
                return
                
            # If no cache exists, load from NPZ and create cache
            logger.info(f"No cache found, loading from NPZ: {self.npz_file}")
            with np.load(self.npz_file, allow_pickle=True) as data:
                self.x_position = float(data['x_position'])
                self.available_fields = data['values'][0].dtype.names
                
                for i, timestep in enumerate(data['timesteps']):
                    centers = data['centers'][i].copy()
                    values = data['values'][i].copy()
                    
                    # Apply elliptical masking
                    self._apply_elliptical_mask(centers, values)
                    
                    self.snapshots[int(timestep)] = Snapshot(
                        timestep=int(timestep),
                        centers=centers,
                        values=values,
                        x_position=self.x_position
                    )
                
                # Create cache dictionary with mask parameters
                cache_data = {
                    'x_position': self.x_position,
                    'available_fields': self.available_fields,
                    'snapshots': self.snapshots,
                    'sphere_center_y': self.sphere_center_y,
                    'sphere_center_z': self.sphere_center_z,
                    'sphere_diameter': self.sphere_diameter,
                    'scale_z': self.scale_z
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
        
        from matplotlib.patches import Ellipse

        # Add elliptical cylinder representation
        ellipse = Ellipse(
            xy=(self.flow_field.sphere_center_y, self.flow_field.sphere_center_z),
            width=2 * self.flow_field.sphere_radius,  # Full width = 2 * semi-axis along y
            height=2 * self.flow_field.sphere_radius * self.flow_field.scale_z,  # Full height = 2 * semi-axis along z
            angle=0,
            color='black',
            alpha=0.1,
            zorder=5
        )
        ax.add_patch(ellipse)
        
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
    
def interpolate_2d_data(y_coords: np.ndarray,
                       z_coords: np.ndarray,
                       values: np.ndarray,
                       num_grid_points: int = 100,
                       sphere_center_y: float = None,
                       sphere_center_z: float = None,
                       sphere_radius: float = None,
                       scale_z: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D interpolated grid from non-uniform data using a hybrid approach.
    Optionally applies an elliptical mask to the interpolated grid.

    Args:
        y_coords: Flattened array of y coordinates
        z_coords: Flattened array of z coordinates
        values: Flattened array of values at each (y,z) point
        num_grid_points: Number of points for the interpolation grid
        sphere_center_y: Y-coordinate of the ellipse center (optional)
        sphere_center_z: Z-coordinate of the ellipse center (optional)
        sphere_radius: Semi-major axis of the ellipse (optional)
        scale_z: Scaling factor for the z-direction (optional)

    Returns:
        yi: 2D array of y coordinates on the uniform grid
        zi: 2D array of z coordinates on the uniform grid
        interpolated_values: 2D array of interpolated values
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

    # Perform interpolation (using existing hybrid approach)
    interpolated_values = griddata(points, values, (yi, zi), method='cubic', fill_value=np.nan)
    interpolated_linear = griddata(points, values, (yi, zi), method='linear', fill_value=np.nan)
    interpolated_nearest = griddata(points, values, (yi, zi), method='nearest', fill_value=np.nan)

    # Combine results: cubic -> linear -> nearest
    valid_min, valid_max = np.nanmin(values), np.nanmax(values)
    cubic_mask = (np.isnan(interpolated_values) | 
                  (interpolated_values < valid_min) | 
                  (interpolated_values > valid_max))
    interpolated_values[cubic_mask] = interpolated_linear[cubic_mask]
    remaining_mask = np.isnan(interpolated_values)
    interpolated_values[remaining_mask] = interpolated_nearest[remaining_mask]

    # Apply elliptical mask if parameters are provided
    if all(p is not None for p in [sphere_center_y, sphere_center_z, sphere_radius]):
        dy = (yi - sphere_center_y) / sphere_radius
        dz = (zi - sphere_center_z) / (sphere_radius * scale_z)
        dist_sq = dy**2 + dz**2
        mask = dist_sq <= 1.0
        interpolated_values[mask] = 0.0  # Set to zero inside the ellipse

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
    Applies elliptical masking to the interpolated data.
    
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
    
    # Interpolate data with elliptical masking
    yi, zi, interpolated_data = interpolate_2d_data(
        y, z, field_data,
        num_grid_points=100,  # Adjust as needed
        sphere_center_y=flow_field.sphere_center_y,
        sphere_center_z=flow_field.sphere_center_z,
        sphere_radius=flow_field.sphere_diameter / 2,
        scale_z=flow_field.scale_z
    )
    
    if show_scatter:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot interpolated field
        im = ax1.pcolormesh(yi, zi, interpolated_data, shading='auto', cmap=cmap)
        plt.colorbar(im, ax=ax1, label=field)
        ax1.set_aspect('equal')
        ax1.set_title(f"Interpolated {field} Field (t={timestep})")
        ax1.set_xlabel('Y Position')
        ax1.set_ylabel('Z Position')
        
        # Plot original points
        scatter = ax2.scatter(y, z, c=field_data, cmap=cmap, s=20)
        plt.colorbar(scatter, ax=ax2, label=field)
        ax2.set_aspect('equal')
        ax2.set_title(f"Original Data Points (t={timestep})")
        ax2.set_xlabel('Y Position')
        ax2.set_ylabel('Z Position')
        
    else:
        # Plot only interpolated field
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.pcolormesh(yi, zi, interpolated_data, shading='auto', cmap=cmap)
        plt.colorbar(im, ax=ax, label=field)
        ax.set_aspect('equal')
        ax.set_title(f"{field} Field (t={timestep})")
        ax.set_xlabel('Y Position')
        ax.set_ylabel('Z Position')
    
    plt.tight_layout()
    plt.show()