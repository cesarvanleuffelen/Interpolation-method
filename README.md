# Histology Volume Reconstruction with Frame Interpolation

## Table of Contents

1. [Introduction](#1-introduction)
2. [Code Explanation](#2-code-explanation)
3. [How to Run](#3-how-to-run)
4. [Tested Hardware](#4-tested-hardware)


## 1. Introduction

This project implements a complete pipeline for reconstructing 3D volumes from sparse histology image slices using deep learning-based frame interpolation. The system takes a series of histology images that are spaced apart and generates intermediate frames between them, creating a dense 3D volume suitable for visualization and analysis.

The workflow consists of five main steps: image downsizing, alignment, gap preparation, interpolation, and volume stacking. The final output is a 3D numpy array that can be visualized using the included PyVista viewer, which provides interactive 3D rendering with transparency and slicing capabilities.

The interpolation process uses a TensorFlow-based neural network model to generate realistic intermediate frames between existing slices. This approach is particularly useful for histology imaging where physical sectioning creates gaps of 100 micrometers between slices, and margin assessment needs a continuous volume representation.


## 2. Code Explanation

### 2.1 build_volume_v3.py

This is the main processing script that orchestrates the entire volume reconstruction pipeline. It processes raw histology images through multiple stages to create a final 3D volume.

#### TensorFlow Configuration

The script begins by configuring TensorFlow to run on CPU only. This is a critical step for Mac systems, particularly M1 MacBooks, where GPU usage can cause system freezes due to unified memory.

```python
try:
    tf.config.set_visible_devices([], 'GPU')
    print("System Stability Mode: GPU Disabled (Running on CPU to prevent freeze)")
except:
    pass
```

This configuration prevents TensorFlow from attempting to use the GPU, which can cause system instability on certain Mac configurations. The CPU-only mode ensures stable operation, though it may be slower than GPU processing.

#### Helper Functions

The script includes several utility functions that support the main processing pipeline.

The `load_image` function handles image loading and normalization. It reads images using imageio, converts grayscale images to RGB format, removes alpha channels if present, and normalizes pixel values to the 0-1 range required by the interpolation model.

```python
def load_image(img_path: str):
    _UINT8_MAX_F = float(np.iinfo(np.uint8).max)
    image = imageio.imread(img_path)
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    image = image.astype(np.float32) / _UINT8_MAX_F
    return image
```

The normalization step converts uint8 pixel values (0-255) to float32 values (0.0-1.0), which is the expected input format for the TensorFlow model. This ensures consistent data representation regardless of the source image format.

The `identify_gaps` function scans through aligned images and detects gaps between consecutive frames that match the expected spacing. This is crucial for determining which image pairs need interpolation.

```python
def identify_gaps(pthims, frames_between):
    files = [f for f in os.listdir(pthims) if os.path.isfile(os.path.join(pthims, f))]
    image_files = natsorted([f for f in files if f.endswith(('tif', 'png', 'jpg'))])
    
    skip_inputs = {}
    for i in range(len(image_files) - 1):
        file1 = image_files[i]
        file2 = image_files[i+1]
        num1 = extract_number_from_filename(file1)
        num2 = extract_number_from_filename(file2)
        gap = num2 - num1 - 1
        
        if gap == frames_between:
            if gap not in skip_inputs:
                skip_inputs[gap] = []
            skip_inputs[gap].append([num1, num2])
    
    return skip_inputs
```

The function uses natural sorting to ensure files are processed in the correct numerical order, even when filenames contain zero-padded numbers. It only processes gaps that exactly match the expected frame spacing, preventing incorrect interpolations.

#### Processing Steps

The main pipeline consists of four sequential steps, each handling a specific aspect of volume preparation.

**Step 1: Downsizing**

The first step resizes all input images to a uniform target size while maintaining aspect ratio. This standardization is necessary for consistent processing and reduces computational requirements.

```python
def step_1_downsize(input_dir, output_dir, target_size=(1500, 1500)):
    target_w, target_h = target_size
    for f in tqdm(files):
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
```

The function calculates a scale factor that ensures the image fits within the target dimensions without distortion. It uses area interpolation for downscaling, which provides better quality than linear interpolation. The resized image is then centered on a black canvas of the target size, ensuring all output images have identical dimensions.

**Step 2: Alignment**

Alignment is critical for creating a coherent 3D volume. The script uses rigid body registration to align all images to a reference frame, compensating for any misalignment that occurred during the imaging process.

```python
def step_2_align(input_dir, output_dir):
    ref_idx = len(files) // 2
    ref_img = imageio.imread(os.path.join(input_dir, files[ref_idx]))
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    sr = StackReg(StackReg.RIGID_BODY)
    
    for f in tqdm(files):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        t_matrix = sr.register(ref_gray, img_gray)
        
        for c in range(3):
            transformed_channel = sr.transform(img[:,:,c], tmat=t_matrix)
            out_img[:,:,c] = np.clip(transformed_channel, 0, 255)
```

The middle image is selected as the reference frame, as it typically represents the most stable position in the sequence. The registration process uses grayscale images for computational efficiency, then applies the computed transformation matrix to each RGB channel separately. Rigid body transformation allows rotation and translation but preserves scale, which is appropriate for histology sections that may have shifted but not deformed.

**Step 3: Gap Preparation**

Before interpolation can begin, the script renames files to create numerical spacing that leaves room for interpolated frames.

```python
def step_3_prepare_gaps(input_dir, frames_between):
    step_size = frames_between + 1
    for i, f in enumerate(files):
        new_name = f"{str(i * step_size).zfill(5)}{ext}"
        os.rename(old_path, new_path)
```

If `FRAMES_BETWEEN` is 25, files are renamed with spacing of 26 (0, 26, 52, etc.), leaving exactly 25 slots for interpolated frames between each pair. This naming scheme ensures that interpolated frames can be inserted with sequential numbering.

**Step 4: Interpolation**

The interpolation step is the most computationally intensive part of the pipeline. It uses a pre-trained TensorFlow model to generate intermediate frames between existing image pairs.

```python
def interpolate_from_image_list(pthims, skip_images, TILE_SIZE, model, image_files):
    for skip_num, image_pairs in skip_images.items():
        times = np.linspace(0, 1, skip_num + 2)[1:-1]
        
        for img1_num, img2_num in tqdm(image_pairs):
            image1 = load_image(image1_path)
            image2 = load_image(image2_path)
            x0 = np.expand_dims(image1, axis=0)
            x1 = np.expand_dims(image2, axis=0)
            
            for idx, time_value in enumerate(times):
                input_data = {
                    'time': np.expand_dims(time_in, axis=0),
                    'x0': x0,
                    'x1': x1
                }
                mid_frame = model(input_data)
                generated_image = mid_frame['image'][0].numpy()
                image_in_uint8 = np.clip(generated_image * _UINT8_MAX_F, 0, _UINT8_MAX_F).astype(np.uint8)
```

The function generates interpolation time points between 0 and 1, excluding the endpoints. For each time point, it calls the model with the two source images and the time value, which determines the interpolation position. The model output is denormalized back to uint8 format and saved. The script includes extensive memory management, deleting intermediate results and calling garbage collection after each frame to prevent RAM buildup during long processing sessions.

> You can download the model here [model download](https://drive.google.com/drive/folders/16a4zhopq8AfKCADXxBwuYccGr_PnBRlt) (source: Joshi, S., Forjaz, A., Han, K.S. et al. InterpolAI: deep learning-based optical flow interpolation and restoration of biomedical images for improved 3D tissue mapping. Nat Methods 22, 1556–1567 (2025). https://doi.org/10.1038/s41592-025-02712-4)

**Step 5: Stacking**

The final step combines all images into a single 3D numpy array using memory-mapped storage for efficiency.

```python
def step_4_stack_and_save_memmap(align_dir, final_output_name):
    fp = open_memmap(final_output_name, mode='w+', dtype='uint8', shape=output_shape)
    
    for i, f in enumerate(tqdm(all_files, desc="Stacking")):
        img = imageio.imread(f)
        fp[i] = img
        if i % 50 == 0: fp.flush()
    
    fp.flush()
```

Memory-mapped arrays allow the script to work with volumes larger than available RAM by storing data directly on disk. The array is accessed like a regular numpy array, but data is read from and written to disk as needed. Periodic flushing ensures data is written incrementally, preventing data loss if the process is interrupted.

### 2.2 viewer/viewer_pyvista.py

The viewer script provides interactive 3D visualization of the generated volume using PyVista, a powerful visualization library built on VTK.

#### Volume Loading and Transparency

The viewer begins by loading the numpy array and calculating transparency values based on image intensity.

```python
def view_full_volume(filename):
    data = np.load(filename)
    intensity = np.max(data, axis=-1)
    opacity = intensity.astype(np.float32) / 255.0
    opacity[opacity < 0.05] = 0.0
```

The transparency calculation uses the maximum value across RGB channels to determine tissue presence. Darker regions (background) are made completely transparent, while brighter regions (tissue) remain visible. This creates a natural visualization where only the actual tissue is shown, not the background.

#### RGBA Conversion

The RGB data is combined with the calculated alpha channel to create RGBA format required for volume rendering.

```python
data_rgba = np.concatenate(
    [data, opacity[..., np.newaxis] * 255], 
    axis=-1
).astype(np.uint8)
```

The alpha channel is expanded to match the spatial dimensions of the RGB data, then multiplied by 255 to convert from 0-1 range to 0-255 range for uint8 storage.

#### 3D Grid Setup

PyVista requires a structured grid to represent the volume data. The grid dimensions must be reversed because PyVista uses (X, Y, Z) coordinate ordering while NumPy arrays use (Z, Y, X) ordering.

```python
grid = pv.ImageData()
grid.dimensions = np.array(data.shape[:3][::-1])
grid.spacing = (1, 1, 1)
grid.point_data["colors"] = data_rgba.reshape(-1, 4)
```

The `ImageData` class represents a regular grid where each point has associated data. The spacing parameter controls the physical size of each voxel; changing the third value would stretch the volume along the Z-axis, which can be useful for adjusting aspect ratios if slice spacing differs from pixel spacing.

#### Interactive Rendering

The viewer creates an interactive window with volume rendering and clipping capabilities.

```python
p = pv.Plotter(window_size=[1000, 800])
vol = p.add_volume(
    grid, 
    scalars="colors", 
    shade=True,
    diffuse=0.8
)
p.add_volume_clip_plane(vol, normal='y')
```

The volume is rendered with shading enabled for depth perception and a matte finish for realistic appearance. The clip plane widget allows users to interactively slice through the volume by dragging a plane, revealing internal structures. This is particularly useful for exploring the 3D structure of histology volumes.

## 3. How to Run

### Prerequisites

Before running the scripts, ensure you have all required dependencies installed. The project requires Python 3.8 or higher and several scientific computing libraries.

```bash
pip install numpy opencv-python imageio tensorflow natsort tqdm pystackreg pyvista
```

For M1 MacBooks, you may need to install TensorFlow using a specific method compatible with Apple Silicon. Consult the TensorFlow documentation for the latest installation instructions for Apple Silicon.

### Directory Structure

Organize your files according to the expected directory structure. Place your raw histology images in the `split1` directory, and ensure the interpolation model is located at the path specified in the configuration section.

```
interpolation_stacking_method/
├── split1/                    # Raw input images
├── images/
│   ├── step1/                 # Downsized images (created automatically)
│   └── step2/                 # Aligned images and interpolated frames (created automatically)
├── viewer/
│   └── viewer_pyvista.py
├── build_volume_v3.py
└── final_histology_volume.npy  # Final output (created automatically)
```

### Configuration

Before running, adjust the configuration variables in `build_volume_v3.py` to match your setup.

```python
RAW_DIR = "./split1"
TEMP_RESIZED = "./images/step1"
TEMP_ALIGNED = "./images/step2"
MODEL_PATH = "./../model"
FRAMES_BETWEEN = 25
TILE_SIZE = (512, 512)
```

Set `RAW_DIR` to the path containing your input images. Adjust `MODEL_PATH` to point to your TensorFlow SavedModel directory. The `FRAMES_BETWEEN` parameter determines how many frames to interpolate between each pair of existing images.

### Running the Pipeline

Execute the main processing script from the project root directory.

```bash
python build_volume_v3.py
```

The script will process images through all five steps automatically. Progress bars will indicate the status of each step. Processing time depends on the number of images, image size, and available computational resources. On an M1 MacBook, expect several minutes to hours depending on the dataset size.

The script includes resume capability: if interrupted, it will skip already-processed interpolated frames when restarted, allowing you to continue from where it left off.

### Viewing the Result

Once processing is complete, use the viewer to visualize the generated volume.

```bash
python viewer/viewer_pyvista.py
```

You can also specify a custom volume file path as a command-line argument.

```bash
python viewer/viewer_pyvista.py path/to/your/volume.npy
```

The viewer will open an interactive window. Use the mouse to rotate the volume, scroll to zoom, and drag the clipping plane widget to slice through the volume. The controls are displayed in the terminal when the viewer starts.


## 4. Tested Hardware

This project has been tested and verified to work on an M1 MacBook. The system configuration includes specific considerations for Apple Silicon architecture.

### System Specifications

The code has been tested on an M1 MacBook with the following characteristics:
- Apple Silicon M1 chip
- 32GB RAM
- macOS Sequoia 15.0.1
- Python 3.11.9
- TensorFlow configured for CPU-only operation

### Performance Considerations

The CPU-only TensorFlow configuration, while necessary for system stability, means processing will be slower than GPU-accelerated systems. For large datasets, expect processing times in the range of hours rather than minutes. The memory-mapped stacking approach helps manage RAM usage, but very large volumes may still require significant memory.

The interpolation step is the most time-consuming, as it requires running the neural network model for each interpolated frame. The script's memory management ensures stable operation even during extended processing sessions, but monitor system resources if processing very large datasets.

### Known Limitations

On M1 MacBooks, TensorFlow GPU support is not available, so all processing runs on CPU. This is handled automatically by the script's configuration. The viewer should work normally, but very large volumes may experience slower rendering performance compared to systems with dedicated graphics hardware.

