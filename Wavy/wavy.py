import cv2
import numpy as np
import glob
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def sinusoidal_warp(image, amplitude=10, wavelength=50, phase=0):
    """
    Applies a sinusoidal warp to the image by shifting pixels horizontally.
    
    Parameters:
      image: Input image in BGR format.
      amplitude: Maximum horizontal displacement in pixels.
      wavelength: Wavelength of the sine wave (controls the frequency).
      phase: Phase shift of the sine wave.
      
    Returns:
      The warped image.
    """
    rows, cols, _ = image.shape
    # Create a meshgrid of original x and y coordinates.
    x_map, y_map = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Apply a sinusoidal shift to the x coordinates based on the y coordinate.
    x_map_new = x_map + amplitude * np.sin(2 * np.pi * y_map / wavelength + phase)
    y_map_new = y_map  # Keep vertical coordinates unchanged
    
    # Convert maps to float32 as required by cv2.remap.
    x_map_new = x_map_new.astype(np.float32)
    y_map_new = y_map_new.astype(np.float32)
    
    # Remap the image using the computed coordinate maps.
    warped_image = cv2.remap(image, x_map_new, y_map_new, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped_image

# Define the folder containing your images.
downloads_folder = '/storage/emulated/0/Download'
extensions = ['*.jpg', '*.jpeg', '*.png']

# Gather all image files from the specified folder.
image_files = []
for ext in extensions:
    image_files.extend(glob.glob(os.path.join(downloads_folder, ext)))

if not image_files:
    print("No image files found in the specified folder.")
    exit()

# Video settings: 25 fps for a 5-second video.
fps = 25
duration = 5  # seconds
num_frames = fps * duration

# Process each image file.
for img_path in image_files:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Unable to read image: {img_path}. Skipping...")
        continue

    frames = []
    # Create frames with a dynamic sinusoidal warp by varying the phase.
    for i in range(num_frames):
        # Vary phase over time to animate the distortion (one full cycle over the video).
        phase = (2 * np.pi * i) / num_frames  
        warped = sinusoidal_warp(image, amplitude=10, wavelength=50, phase=phase)
        # Convert from BGR to RGB for MoviePy.
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        frames.append(warped_rgb)
    
    # Create a video clip from the frames.
    clip = ImageSequenceClip(frames, fps=fps)
    base_name, _ = os.path.splitext(os.path.basename(img_path))
    output_video = os.path.join(downloads_folder, base_name + '_wavy.mp4')
    clip.write_videofile(output_video, codec='libx264')
