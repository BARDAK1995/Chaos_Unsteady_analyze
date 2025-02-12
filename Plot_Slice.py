import tecplot
import numpy as np
import os
from tqdm import tqdm
import re
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

def extract_timestep(filename):
    """Extract timestep number from PLT filename."""
    match = re.search(r'_(\d+)\.plt$', filename)
    return int(match.group(1)) if match else 0

def process_single_plt(plt_file, directory_path, x_slice):
    """
    Process a single PLT file and return its slice data.
    This function is designed to be used with multiprocessing.
    """
    try:
        full_path = os.path.join(directory_path, plt_file)
        timestep = extract_timestep(plt_file)
        
        # Load and process the PLT file
        data, metadata = analyze_plt_structure(full_path)

        slice_data = find_slice_data(data, metadata, x_slice)
        
        # Skip if no cells found in slice
        if len(slice_data['cells']) == 0:
            return None
            
        # Create structured arrays for efficient storage
        centers = np.array(list(zip(
            slice_data['centers']['y'],
            slice_data['centers']['z']
        )), dtype=[('y', 'f8'), ('z', 'f8')])
        
        values = np.array(list(zip(
            slice_data['values']['pressure'],
            slice_data['values']['temperature'],
            slice_data['values']['u'],
            slice_data['values']['v'],
            slice_data['values']['w'],
            slice_data['values']['mach'],
            slice_data['values']['vol'],
            slice_data['values']['NumPart1'] 
 
        )), dtype=[('pressure', 'f8'), 
                  ('temperature', 'f8'),
                  ('u', 'f8'),
                  ('v', 'f8'),
                  ('w', 'f8'),
                  ('mach', 'f8'),
                  ('vol', 'f8'),
                  ('NumPart1', 'f8')])
        
        return {
            'timestep': timestep,
            'centers': centers,
            'values': values,
            'edges': np.array(slice_data['edges']),
            'sizes': np.array(slice_data['sizes'])
        }
        
    except Exception as e:
        print(f"\nError processing {plt_file}: {str(e)}")
        return None

def analyze_plt_structure(plt_file):
    """Analyze the structure of the PLT file and return relevant data and metadata"""
    dataset = tecplot.data.load_tecplot(plt_file)
    zone = dataset.zone(0)
    
    # Extract basic data with scaling correction
    scaling_factor = 1  # Correction factor for flow variables
    data = {
        'x': np.array(zone.values('x')[:]),
        'y': np.array(zone.values('y')[:]),
        'z': np.array(zone.values('z')[:]),
        'pressure': np.array(zone.values('Pressure')[:]) * scaling_factor,
        'temperature': np.array(zone.values('T1')[:]) * scaling_factor,
        'u': np.array(zone.values('u1')[:]) * scaling_factor,
        'v': np.array(zone.values('v1')[:]) * scaling_factor,
        'w': np.array(zone.values('w1')[:]) * scaling_factor,
        'mach': np.array(zone.values('Mach')[:]),
        'vol': np.array(zone.values('vol')[:]) * scaling_factor,
        'NumPart1': np.array(zone.values('NumPart1')[:]) * scaling_factor 
    }
    
    # Get connectivity if available
    if hasattr(zone, 'nodemap'):
        data['connectivity'] = zone.nodemap
    
    metadata = {
        'x_range': (float(np.min(data['x'])), float(np.max(data['x']))),
        'y_range': (float(np.min(data['y'])), float(np.max(data['y']))),
        'z_range': (float(np.min(data['z'])), float(np.max(data['z']))),
        'pressure_range': (float(np.min(data['pressure'])), float(np.max(data['pressure']))),
        'temperature_range': (float(np.min(data['temperature'])), float(np.max(data['temperature']))),
        'velocity_ranges': {
            'u': (float(np.min(data['u'])), float(np.max(data['u']))),
            'v': (float(np.min(data['v'])), float(np.max(data['v']))),
            'w': (float(np.min(data['w'])), float(np.max(data['w'])))
        },
        'mach_range': (float(np.min(data['mach'])), float(np.max(data['mach']))),
        'vol_range': (float(np.min(data['vol'])), float(np.max(data['vol']))),
        'NumPart1_range': (float(np.min(data['NumPart1'])), float(np.max(data['NumPart1']))),  # Add this line
        'scaling_factor': scaling_factor,
        'num_nodes': len(data['x']),
        'num_cells': len(data['pressure'])
    }
    
    return data, metadata

def find_slice_data(data, metadata, x_slice):
    """Find cells that intersect with the slice plane and return detailed slice information"""
    nodes_per_cell = 8  # Assuming hexahedral cells
    slice_data = {
        'cells': [],
        'edges': [],
        'centers': {'x': [], 'y': [], 'z': []},
        'values': {
            'pressure': [],
            'temperature': [],
            'u': [],
            'v': [],
            'w': [],
            'mach': [],
            'vol': [],
            'NumPart1': []
        },
        'sizes': []
    }
    
    # Define cell edge connectivity for a hexahedron
    edge_connections = [
        (0,1), (1,2), (2,3), (3,0),  # Bottom face
        (4,5), (5,6), (6,7), (7,4),  # Top face
        (0,4), (1,5), (2,6), (3,7)   # Vertical edges
    ]
    
    for cell in range(metadata['num_cells']):
        # Get indices for this cell's nodes
        start_idx = cell * nodes_per_cell
        node_indices = range(start_idx, start_idx + nodes_per_cell)
        
        # Get node coordinates
        cell_x = data['x'][node_indices]
        cell_y = data['y'][node_indices]
        cell_z = data['z'][node_indices]
        
        # Check if cell intersects slice plane
        if np.min(cell_x) <= x_slice <= np.max(cell_x):
            # Calculate intersection points for edges that cross the slice plane
            cell_edges = []
            for edge in edge_connections:
                x1, x2 = cell_x[edge[0]], cell_x[edge[1]]
                y1, y2 = cell_y[edge[0]], cell_y[edge[1]]
                z1, z2 = cell_z[edge[0]], cell_z[edge[1]]
                
                # If edge crosses the slice plane
                if (x1 <= x_slice <= x2) or (x2 <= x_slice <= x1):
                    # Calculate intersection point using linear interpolation
                    if x2 != x1:
                        t = (x_slice - x1) / (x2 - x1)
                        y_int = y1 + t * (y2 - y1)
                        z_int = z1 + t * (z2 - z1)
                        cell_edges.append([[y_int, z_int]])
            
            if cell_edges:
                slice_data['edges'].extend(cell_edges)
                
                # Store cell center
                slice_data['centers']['x'].append(np.mean(cell_x))
                slice_data['centers']['y'].append(np.mean(cell_y))
                slice_data['centers']['z'].append(np.mean(cell_z))
                
                # Store all field values
                slice_data['values']['pressure'].append(data['pressure'][cell])
                slice_data['values']['temperature'].append(data['temperature'][cell])
                slice_data['values']['u'].append(data['u'][cell])
                slice_data['values']['v'].append(data['v'][cell])
                slice_data['values']['w'].append(data['w'][cell])
                slice_data['values']['mach'].append(data['mach'][cell])
                slice_data['values']['vol'].append(data['vol'][cell])
                slice_data['values']['NumPart1'].append(data['NumPart1'][cell])

                # Calculate and store cell size
                cell_size = np.max([np.max(cell_y) - np.min(cell_y),
                                  np.max(cell_z) - np.min(cell_z)])
                slice_data['sizes'].append(cell_size)
                
                slice_data['cells'].append(cell)
    
    # Convert lists to numpy arrays
    slice_data['cells'] = np.array(slice_data['cells'])
    slice_data['edges'] = np.array(slice_data['edges'])
    for key in slice_data['centers']:
        slice_data['centers'][key] = np.array(slice_data['centers'][key])
    for key in slice_data['values']:
        slice_data['values'][key] = np.array(slice_data['values'][key])
    slice_data['sizes'] = np.array(slice_data['sizes'])
    
    return slice_data

def process_plt_directory(directory_path, x_slice, output_filename, num_processes=None):
    """
    Process all PLT files in a directory in parallel and save slice data to a single NPZ file.
    
    Args:
        directory_path: Path to directory containing PLT files
        x_slice: X-coordinate for slicing
        output_filename: Name of output NPZ file
        num_processes: Number of processes to use (default: CPU count - 1)
    """
    # Get list of all PLT files and sort by timestep
    plt_files = [f for f in os.listdir(directory_path) if f.endswith('.plt')]
    plt_files.sort(key=extract_timestep)
    # data0, metadata0 = analyze_plt_structure(plt_files[0])

    print(f"Found {len(plt_files)} PLT files")
    
    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    print(f"Using {num_processes} processes")
    
    # Create partial function with fixed arguments
    process_func = partial(process_single_plt, 
                         directory_path=directory_path,
                         x_slice=x_slice)
    
    # Process files in parallel
    results = []
    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(process_func, plt_files), 
                         total=len(plt_files),
                         desc="Processing PLT files"):
            if result is not None:
                results.append(result)
    
    # Sort results by timestep
    results.sort(key=lambda x: x['timestep'])
    
    # Separate data into arrays
    timesteps = np.array([r['timestep'] for r in results])
    centers = np.array([r['centers'] for r in results])
    values = np.array([r['values'] for r in results])
    edges = np.array([r['edges'] for r in results], dtype=object)
    sizes = np.array([r['sizes'] for r in results])
    
    metadata_zip={
            'num_timesteps': len(timesteps),
            'x_slice': x_slice,
            'first_timestep': timesteps[0] if len(timesteps) > 0 else None,
            'last_timestep': timesteps[-1] if len(timesteps) > 0 else None,
            # 'x_range': metadata["x_range"],
            # 'y_range':  metadata['y_range'],
            # 'z_range':  metadata['z_range']
        }
    # Save all data to a single NPZ file
    np.savez_compressed(
        output_filename,
        timesteps=timesteps,
        x_position=x_slice,
        centers=centers,
        values=values,
        edges=edges,
        sizes=sizes,
        metadata=metadata_zip
    )
    
    print(f"\nProcessing complete!")
    print(f"Processed {len(timesteps)} timesteps")
    print(f"Data saved to: {output_filename}.npz")
    
    # Print some statistics
    if len(timesteps) > 0:
        print("\nData statistics:")
        print(f"Timestep range: {timesteps[0]} to {timesteps[-1]}")
        print(f"Average number of cells per slice: {np.mean([len(v) for v in values]):.1f}")
        print(f"Total storage size: {os.path.getsize(output_filename + '.npz') / 1e6:.1f} MB")

if __name__ == "__main__":
    # Configuration
    directory_path = "outputsmall"
    x_slice = 2.0e-5  # Adjust this value as needed
    output_filename = f"all_slices_x_{x_slice:.2e}"
    
    # Optional: specify number of processes (default is CPU count - 1)
    num_processes = None  # Set to a specific number if desired
    
    # Process all PLT files
    process_plt_directory(directory_path, x_slice, output_filename, num_processes)
