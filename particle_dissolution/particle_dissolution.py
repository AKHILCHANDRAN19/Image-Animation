import cv2
import numpy as np
import glob
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def particle_dissolution(image, dx_field, dy_field, dissolve_threshold, factor, max_disp):
    """
    Applies a particle dissolution effect to the image.
    
    1. Displaces pixels using a pre-generated random displacement field, scaled by 'factor'.
    2. Applies a dissolve mask based on a fixed random threshold so that pixels gradually vanish.
    
    Parameters:
      image: Input image in BGR format.
      dx_field, dy_field: Displacement fields (same shape as image height/width) with values in [-1,1].
      dissolve_threshold: A fixed random field (values between 0 and 1) used to decide when a pixel dissolves.
      factor: Current displacement scale (0 to max_disp).
      max_disp: The maximum displacement scale.
      
    Returns:
      Processed image with the particle dissolution effect applied.
    """
    rows, cols, _ = image.shape
    # Create a meshgrid of original pixel coordinates.
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Compute new coordinates by adding scaled displacement.
    new_x = (x + dx_field * factor).astype(np.float32)
    new_y = (y + dy_field * factor).astype(np.float32)
    
    # Remap the image using the computed coordinate maps.
    displaced = cv2.remap(image, new_x, new_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Normalize current factor to get a "dissolve level" (0 to 1).
    dissolve_level = factor / max_disp
    # Create an alpha mask: pixels with a random threshold less than the dissolve level become transparent.
    alpha_mask = (dissolve_threshold > dissolve_level).astype(np.float32)
    alpha_mask = cv2.merge([alpha_mask, alpha_mask, alpha_mask])
    
    # Blend the displaced image with a black background using the alpha mask.
    result = (displaced.astype(np.float32) * alpha_mask).astype(np.uint8)
    return result

# Define the folder containing your images.
downloads_folder = '/storage/emulated/0/Download'
extensions = ['*.jpg', '*.jpeg', '*.png']

# Gather all image files in the folder.
image_files = []
for ext in extensions:
    image_files.extend(glob.glob(os.path.join(downloads_folder, ext)))
    
if not image_files:
    print("No image files found in the specified folder.")
    exit()

# Video settings: 25 frames per second for a 5-second video.
fps = 25
duration = 5  # seconds
num_frames = fps * duration

# Define maximum displacement in pixels.
max_disp = 50  # Adjust as needed for stronger or subtler dispersion.

# Process each image file.
for img_path in image_files:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Unable to read image: {img_path}. Skipping...")
        continue

    rows, cols, _ = image.shape
    # Generate a fixed random displacement field (values in [-1, 1]).
    dx_field = np.random.uniform(-1, 1, (rows, cols))
    dy_field = np.random.uniform(-1, 1, (rows, cols))
    # Generate a fixed dissolve threshold for each pixel.
    dissolve_threshold = np.random.uniform(0, 1, (rows, cols))
    
    frames = []
    # For each frame, gradually increase the displacement factor.
    for i in range(num_frames):
        # 'factor' increases linearly from 0 to max_disp.
        factor = (i / num_frames) * max_disp
        processed = particle_dissolution(image, dx_field, dy_field, dissolve_threshold, factor, max_disp)
        # Convert from BGR to RGB (MoviePy expects RGB format).
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        frames.append(processed_rgb)
    
    # Create a video clip from the generated frames.
    clip = ImageSequenceClip(frames, fps=fps)
    base_name, _ = os.path.splitext(os.path.basename(img_path))
    output_video = os.path.join(downloads_folder, base_name + '_particle_dissolution.mp4')
    clip.write_videofile(output_video, codec='libx264')
