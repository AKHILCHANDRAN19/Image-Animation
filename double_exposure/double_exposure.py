import cv2
import numpy as np
import glob
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def double_exposure(image, dx=10, dy=10, alpha=0.5):
    """
    Creates a double exposure effect by blending the original image with a shifted copy.
    
    Parameters:
      image: The input image in BGR format.
      dx: Horizontal offset in pixels for the shifted copy.
      dy: Vertical offset in pixels for the shifted copy.
      alpha: Weight for the original image when blending (the shifted copy uses 1-alpha).
      
    Returns:
      The blended image showing a double exposure effect.
    """
    rows, cols, _ = image.shape
    # Create a translation matrix for the offset
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(image, M, (cols, rows))
    # Blend the original and shifted image
    blended = cv2.addWeighted(image, alpha, shifted, 1 - alpha, 0)
    return blended

# Define the folder containing your images.
downloads_folder = '/storage/emulated/0/Download'
extensions = ['*.jpg', '*.jpeg', '*.png']

# Gather all image files with the specified extensions.
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

# Process each image file.
for img_path in image_files:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Unable to read image: {img_path}. Skipping...")
        continue

    frames = []
    # For a slight animation effect, we vary the offset slightly over time.
    for i in range(num_frames):
        # Optionally, vary the offset with a small oscillation over time.
        dx = 10 + 2 * np.sin(2 * np.pi * i / num_frames)
        dy = 10 + 2 * np.cos(2 * np.pi * i / num_frames)
        
        # Apply double exposure effect
        blended = double_exposure(image, dx=int(dx), dy=int(dy), alpha=0.5)
        
        # Convert from BGR to RGB (MoviePy expects RGB images)
        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        frames.append(blended_rgb)
    
    # Create a video clip from the generated frames.
    clip = ImageSequenceClip(frames, fps=fps)
    base_name, _ = os.path.splitext(os.path.basename(img_path))
    output_video = os.path.join(downloads_folder, base_name + '_double_exposure.mp4')
    clip.write_videofile(output_video, codec='libx264')
