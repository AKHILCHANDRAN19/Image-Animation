from PIL import Image, ImageOps, ImageFilter
import numpy as np
import glob
import os
import random
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def pull_out_shake_effect(image, zoom_factor, shake_intensity=5):
    """
    Applies the pull-out and shake effect using Pillow.
    
    Parameters:
      image: PIL.Image object (original image, full resolution).
      zoom_factor: Current zoom factor (1.0 = full image; >1 = zoomed-in crop).
      shake_intensity: Maximum shake offset in pixels.
      
    Returns:
      A PIL.Image frame of the same size as the original image.
    """
    orig_width, orig_height = image.size
    
    # Determine the crop size based on the zoom factor.
    crop_width = int(orig_width / zoom_factor)
    crop_height = int(orig_height / zoom_factor)
    
    # Compute the ideal top-left coordinates for a centered crop.
    center_x, center_y = orig_width // 2, orig_height // 2
    left = center_x - crop_width // 2
    upper = center_y - crop_height // 2
    
    # Apply random shake offsets.
    offset_x = random.randint(-shake_intensity, shake_intensity)
    offset_y = random.randint(-shake_intensity, shake_intensity)
    left += offset_x
    upper += offset_y
    
    # Ensure crop region remains within image boundaries.
    left = max(0, min(left, orig_width - crop_width))
    upper = max(0, min(upper, orig_height - crop_height))
    
    right = left + crop_width
    lower = upper + crop_height
    
    # Crop and then resize back to original size.
    cropped = image.crop((left, upper, right, lower))
    # Use LANCZOS resampling for high-quality scaling.
    frame = cropped.resize((orig_width, orig_height), Image.Resampling.LANCZOS)
    return frame

# Video settings.
fps = 25
duration = 5  # seconds
num_frames = fps * duration

# Define the folder containing your images.
downloads_folder = '/storage/emulated/0/Download'
extensions = ['*.jpg', '*.jpeg', '*.png']

# Gather image files.
image_files = []
for ext in extensions:
    image_files.extend(glob.glob(os.path.join(downloads_folder, ext)))

if not image_files:
    print("No image files found in the specified folder.")
    exit()

frames = []
# Process the first image (for demonstration).
with Image.open(image_files[0]) as img:
    # Use the image in its original size as the target.
    orig_width, orig_height = img.size
    
    # Define zoom factors: start with a zoomed-in view (e.g., 1.5 times the full image)
    # and gradually zoom out to 1.0 (i.e., full image).
    start_zoom = 1.5
    end_zoom = 1.0
    
    # Generate frames for the video.
    for i in range(num_frames):
        # Linear interpolation of the zoom factor.
        current_zoom = start_zoom + (end_zoom - start_zoom) * (i / (num_frames - 1))
        frame = pull_out_shake_effect(img, current_zoom, shake_intensity=5)
        # Convert PIL.Image to a NumPy array.
        frames.append(np.array(frame))

# Create a video clip using MoviePy.
clip = ImageSequenceClip(frames, fps=fps)
base_name = os.path.splitext(os.path.basename(image_files[0]))[0]
output_video = os.path.join(downloads_folder, base_name + '_pull_out_shake_pil.mp4')
clip.write_videofile(output_video, codec='libx264')
