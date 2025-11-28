import pyvista as pv
import numpy as np
import sys
import os

def view_full_volume(filename):
    print(f"--- Initializing Volumetric Viewer ---")
    
    # Print error to terminal if file is not found
    if not os.path.exists(filename):
        print(f"Error: File not found at {filename}")
        return

    print(f"Loading data from {filename}...")
    # Handle file loading errors gracefully to prevent crashes
    try:
        data = np.load(filename)
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        return
        
    print(f"Data Shape: {data.shape}")
    
    # Step 1: Create the Alpha Channel (Transparency) 
    print("Calculating transparency...")
    
    # Calculate intensity (Max of RGB) to find where tissue is
    intensity = np.max(data, axis=-1)
    
    # Normalize 0 - 255 to 0.0 - 1.0
    opacity = intensity.astype(np.float32) / 255.0
    
    # Threshold for transparent: make background (darker than 10/255) completely transparent
    opacity[opacity < 0.05] = 0.0 
    
    # Step 2: Combine RGB + Alpha 
    # Concatenate RGB channels with alpha channel to create RGBA format
    data_rgba = np.concatenate(
        [data, opacity[..., np.newaxis] * 255], 
        axis=-1
    ).astype(np.uint8)

    # Step 3: Setup Grid 
    print("Building 3D Grid...")
    grid = pv.ImageData() # Create structured grid for regular 3D volume data
    grid.dimensions = np.array(data.shape[:3][::-1]) # Reverse dimensions because PyVista uses (X, Y, Z) while NumPy uses (Z, Y, X)
    grid.spacing = (1, 1, 1) # to stretch volume change 3th value
    
    grid.point_data["colors"] = data_rgba.reshape(-1, 4) # Add RGBA data

    # Step 4: Plotting 
    print("Opening Window...")
    p = pv.Plotter(window_size=[1000, 800]) # Define the plotter
    p.set_background("white")

    # Add the Volume
    vol = p.add_volume(
        grid, 
        scalars="colors", 
        shade=True,      # Adds shadows for depth
        diffuse=0.8      # Matte finish
    )
    
    # Create the interactive widget to slice the volume
    try:
        p.add_volume_clip_plane(vol, normal='y')
    except AttributeError:
        # Fallback for older PyVista versions
        print("Warning: Interactive clipping not supported in this version. Showing full volume.")

    p.add_text("3D Volumetric Rendering", position='upper_left', color='black') # Add title to the window (top-left)
    
    print("\n=== Controls ===")
    print("- Rotate: Left Click + Drag")
    print("- Zoom: Scroll")
    print("- Cut: Drag the arrow/plane widget to slice inside")
    print("================\n")
    
    p.show() # Start window

if __name__ == "__main__":
    target_file = "./../final_histology_volume.npy" # Path to volume file
    # Allow command line argument to override default file path
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    view_full_volume(target_file)