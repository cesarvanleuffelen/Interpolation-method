import os
import cv2
import numpy as np
import imageio.v2 as imageio
import tensorflow as tf
from natsort import natsorted
from tqdm import tqdm
from pystackreg import StackReg
import re
import gc

# ==========================================
# TENSORFLOW SETUP
# ==========================================
# Instruct tensorflow to use CPU only (prevent mac freezing)
try:
    tf.config.set_visible_devices([], 'GPU')
    print("System Stability Mode: GPU Disabled (Running on CPU to prevent freeze)")
except:
    pass

# ==========================================
# HELPER FUNCTIONS
# ==========================================

# Loads and normalizes image to float32 format 0 - 1 for model input
def load_image(img_path: str):
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max) # Maximum uint8 value (255.0) for normalization
    image = imageio.imread(img_path)
    if len(image.shape) == 2: # Convert grayscale to RGB by stacking channels
        image = np.stack((image,)*3, axis=-1)
    if image.shape[2] == 4: # Remove alpha channel if present, keep only RGB
        image = image[:, :, :3]
    image = image.astype(np.float32) / _UINT8_MAX_F # Normalize pixel values to 0 - 1 range
    return image

# function to get the file extension to check for correct format
def get_file_extension(filename):
    return os.path.splitext(filename)[1]

# function to get the order number from file name
def extract_number_from_filename(filename):
    match = re.search(r'(\d+)(?=\D*$)', filename)
    return int(match.group(1)) if match else None

# Detects gaps between consecutive image files matching the expected frame spacing
def identify_gaps(pthims, frames_between):
    """Strict gap detection to prevent processing wrong files."""
    files = [f for f in os.listdir(pthims) if os.path.isfile(os.path.join(pthims, f))] # Get all files in directory
    image_files = natsorted([f for f in files if f.endswith(('tif', 'png', 'jpg'))]) # Filter and sort image files naturally
    
    skip_inputs = {}
    print(f"Scanning {len(image_files)} aligned files for gaps...")
    
    # Iterate through consecutive file pairs to detect gaps
    for i in range(len(image_files) - 1):
        file1 = image_files[i]
        file2 = image_files[i+1]
        
        num1 = extract_number_from_filename(file1)
        num2 = extract_number_from_filename(file2)
        
        # Calculate gap
        gap = num2 - num1 - 1
        
        # Only process gaps matching expected size to avoid wrong interpolations
        if gap == frames_between:
            if gap not in skip_inputs:
                skip_inputs[gap] = []
            skip_inputs[gap].append([num1, num2])
            
    return skip_inputs

# ==========================================
# INTERPOLATION LOGIC (Maximum Stability)
# ==========================================
def interpolate_from_image_list(pthims, skip_images, TILE_SIZE, model, image_files):
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max) # Maximum uint8 value (255.0) for denormalization
    image_files = natsorted(image_files)
    file_extension = get_file_extension(image_files[0])
    image_dict = {extract_number_from_filename(f): f for f in image_files} # Map frame numbers to filenames for quick lookup

    # Process each gap size separately to handle different interpolation counts
    for skip_num, image_pairs in skip_images.items():
        print(f'--- Processing {len(image_pairs)} gaps of size {skip_num} ---')
        
        # Create output folder for interpolated frames of this gap size
        output_folder = os.path.join(pthims, f'int_{skip_num}')
        os.makedirs(output_folder, exist_ok=True)
        
        times = np.linspace(0, 1, skip_num + 2)[1:-1] # Generate interpolation time points, excluding endpoints (0 and 1)

        # Loop through pairs
        for img1_num, img2_num in tqdm(image_pairs, desc="Interpolating Pairs"):
            
            # 1. Check if ALL output files exist for this pair
            # check first to avoid loading heavy images into RAM if not needed
            missing_files = False
            for idx in range(len(times)):
                base_name = os.path.splitext(image_dict[img1_num])[0] # Extract filename without extension
                match = re.search(r'(\d+)$', base_name)
                # If the filename ends with numbers, increment them for interpolated frames
                if match:
                    number_part = match.group(1)
                    new_number = str(int(number_part) + (idx + 1)).zfill(len(number_part))
                    filename = f"{base_name[:-len(number_part)]}{new_number}{file_extension}"
                else:
                    filename = f"{base_name}_int{idx + 1}{file_extension}"
                
                # Check if this specific interpolated frame already exists
                if not os.path.exists(os.path.join(output_folder, filename)):
                    missing_files = True
                    break
            
            if not missing_files:
                continue

            # 2. Load Images (Only done once per pair)
            try:
                image1_path = os.path.join(pthims, image_dict[img1_num])
                image2_path = os.path.join(pthims, image_dict[img2_num])
                
                image1 = load_image(image1_path)
                image2 = load_image(image2_path)
                
                # Expand image dimension for batching
                x0 = np.expand_dims(image1, axis=0)
                x1 = np.expand_dims(image2, axis=0)

                # 3. Processing Loop (Frame by Frame)
                for idx, time_value in enumerate(times):
                    # Naming logic
                    base_name = os.path.splitext(image_dict[img1_num])[0]
                    match = re.search(r'(\d+)$', base_name)
                    if match:
                        number_part = match.group(1)
                        new_number = str(int(number_part) + (idx + 1)).zfill(len(number_part)) # Increment number preserving zero-padding
                        filename = f"{base_name[:-len(number_part)]}{new_number}{file_extension}" # Construct filename with incremented number
                    else:
                        filename = f"{base_name}_int{idx + 1}{file_extension}" # Fallback naming if no number is found
                    
                    output_path = os.path.join(output_folder, filename)
                    
                    # Skip if specific frame exists
                    if os.path.exists(output_path): 
                        continue

                    # Inference of interpolation model
                    time_in = np.array([time_value], dtype=np.float32)
                    input_data = {
                        'time': np.expand_dims(time_in, axis=0),
                        'x0': x0,
                        'x1': x1
                    }
                    
                    mid_frame = model(input_data)
                    generated_image = mid_frame['image'][0].numpy()
                    
                    image_in_uint8 = np.clip(generated_image * _UINT8_MAX_F, 0, _UINT8_MAX_F).astype(np.uint8) # Denormalize from 0 - 1, to uint8 0 - 255 and clip values
                    imageio.imwrite(output_path, image_in_uint8, format=file_extension.lstrip('.')) # Save the interpolated image to the output directory

                    # Free memory immediately after each frame to prevent RAM buildup
                    del mid_frame
                    del generated_image
                    del input_data
                    gc.collect()

            except Exception as e:
                print(f"Error processing pair {img1_num}-{img2_num}: {e}")
            
            # Cleanup pair data from RAM to prevent buildup
            if 'image1' in locals(): del image1
            if 'image2' in locals(): del image2
            if 'x0' in locals(): del x0
            if 'x1' in locals(): del x1
            gc.collect()

# ==========================================
# STEPS
# ==========================================
def step_1_downsize(input_dir, output_dir, target_size=(1500, 1500)):
    if not os.path.exists(output_dir): os.makedirs(output_dir) # Create output directory if it doesn't exist
    files = natsorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.png', '.jpg'))]) # Get all images that have one of the specified extensions
    print(f"--- Step 1: Downsizing {len(files)} images ---")
    
    target_w, target_h = target_size
    # Process each image to fit within target dimensions while maintaining aspect ratio
    for f in tqdm(files):
        img_path = os.path.join(input_dir, f) # Construct full path to image file
        img = cv2.imread(img_path) 
        if img is None: continue

        h, w = img.shape[:2] # Get width & height of the image
        scale = min(target_w / w, target_h / h) # Calculate scale factor to fit image within target size
        new_w, new_h = int(w * scale), int(h * scale) # Compute new dimensions preserving aspect ratio
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA) # Resize using area interpolation for downscaling
        
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8) # Create black canvas of target size
        y_off = (target_h - new_h) // 2 # Calculate vertical offset to center image
        x_off = (target_w - new_w) // 2 # Calculate horizontal offset to center image
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized # Place resized image centered on canvas
        cv2.imwrite(os.path.join(output_dir, f), canvas) # Save the final downscaled and centered image

def step_2_align(input_dir, output_dir):
    if not os.path.exists(output_dir): os.makedirs(output_dir) # Create output directory if it doesn't exist
    files = natsorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.png', '.jpg'))]) # Get all images that have one of the specified extensions
    print(f"--- Step 2: Aligning {len(files)} images (RGB) ---")
    
    # Print warning message to terminal if no files were found
    if len(files) == 0:
        print("Warning: No images found in step1 folder. Check Step 1.")
        return

    ref_idx = len(files) // 2 # Use middle image as reference for alignment
    ref_img = imageio.imread(os.path.join(input_dir, files[ref_idx])) # Load reference image
    if len(ref_img.shape) == 2: ref_img = np.stack((ref_img,)*3, axis=-1) # Convert grayscale to RGB
    
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY) # Convert reference to grayscale for registration
    sr = StackReg(StackReg.RIGID_BODY) # Initialize rigid body transformation (rotation + translation)
    
    # Align each image to the reference using rigid body transformation
    for f in tqdm(files):
        img = imageio.imread(os.path.join(input_dir, f)) # Get the image
        if len(img.shape) == 2: img = np.stack((img,)*3, axis=-1) # Convert grayscale to RGB
        if img.shape[2] == 4: img = img[:,:,:3] # Remove alpha channel if present
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Convert to grayscale for registration
        t_matrix = sr.register(ref_gray, img_gray) # Compute transformation matrix to align with reference
        
        out_img = np.zeros_like(img) # Initialize output image with same shape

        # Apply transformation to each RGB channel separately
        for c in range(3):
            transformed_channel = sr.transform(img[:,:,c], tmat=t_matrix)
            out_img[:,:,c] = np.clip(transformed_channel, 0, 255)
            
        imageio.imwrite(os.path.join(output_dir, f), out_img.astype(np.uint8)) # Save aligned image

def step_3_prepare_gaps(input_dir, frames_between):
    print("--- Step 3: Renaming files ---")
    files = natsorted([f for f in os.listdir(input_dir)]) # Get all files sorted naturally
    step_size = frames_between + 1
    # Rename files with spacing to leave room for interpolated frames
    for i, f in enumerate(files):
        old_path = os.path.join(input_dir, f)
        ext = os.path.splitext(f)[1]
        new_name = f"{str(i * step_size).zfill(5)}{ext}" 
        new_path = os.path.join(input_dir, new_name)
        if old_path != new_path:
            os.rename(old_path, new_path)

def step_4_stack_and_save_memmap(align_dir, final_output_name):
    """Stacking compatible with Napari (.npy with header)"""
    from numpy.lib.format import open_memmap 
    
    print("--- Step 4: Stacking Final Volume (Low RAM Mode) ---")
    all_files = []
    # Recursively find all image files in directory and subdirectories
    for root, files in os.walk(align_dir):
        for file in files:
            if file.endswith(('.tif', '.png', '.jpg')):
                all_files.append(os.path.join(root, file))
    
    all_files = natsorted(all_files)
    count = len(all_files)
    
    # Print error message to terminal if their are no files and return
    if count == 0:
        print("Error: No images found to stack!")
        return

    print(f"Found {count} slices. Preparing disk storage...")
    first_img = imageio.imread(all_files[0]) # Get the first image
    h, w = first_img.shape[:2] # Get the image's height & width
    channels = 3 # RGB channels for color images
    
    output_shape = (count, h, w, channels) # Difine the shape of the output
    fp = open_memmap(final_output_name, mode='w+', dtype='uint8', shape=output_shape) # Create memory-mapped array for efficient disk storage
    
    # Load and stack each image into the volume array
    for i, f in enumerate(tqdm(all_files, desc="Stacking")):
        img = imageio.imread(f)
        if len(img.shape) == 2: img = np.stack((img,)*3, axis=-1) # Convert grayscale to RGB
        if img.shape[2] == 4: img = img[:,:,:3] # Remove alpha channel if present
        fp[i] = img # Write image to volume at index i
        if i % 50 == 0: fp.flush() # Periodically flush to disk to prevent data loss
            
    fp.flush() # Final flush to ensure all data is written to disk
    print(f"Volume saved successfully to {final_output_name}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # === CONFIG ===
    RAW_DIR = "./split1" # Path to starting images directory
    TEMP_RESIZED = "./images/step1" # Path to save directory for aligned images
    TEMP_ALIGNED = "./images/step2" # Path to save directory for interpolated images
    MODEL_PATH = "./../model" # Path to interpolation model
    
    FRAMES_BETWEEN = 25 # Amount of frames that should be interpolated for each gap
    TILE_SIZE = (512, 512) # Size of interpolation tiles
    
    # Create output folders
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(TEMP_RESIZED, exist_ok=True)
    os.makedirs(TEMP_ALIGNED, exist_ok=True)
    
    # 1. Downsize
    step_1_downsize(RAW_DIR, TEMP_RESIZED, target_size=(1500, 1500))
    
    # 2. Align
    step_2_align(TEMP_RESIZED, TEMP_ALIGNED)
    
    # 3. Rename
    step_3_prepare_gaps(TEMP_ALIGNED, FRAMES_BETWEEN)
    
    # 4. Interpolate
    print("Loading Model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    
    model = tf.saved_model.load(MODEL_PATH) # Load model
    
    image_files = [f for f in os.listdir(TEMP_ALIGNED) if f.endswith(('tif', 'png', 'jpg'))] # Get all image files from aligned directory
    skip_images = identify_gaps(TEMP_ALIGNED, FRAMES_BETWEEN)
    
    interpolate_from_image_list(TEMP_ALIGNED, skip_images, TILE_SIZE, model, image_files)
    
    # 5. Stack
    step_4_stack_and_save_memmap(TEMP_ALIGNED, "final_histology_volume.npy")