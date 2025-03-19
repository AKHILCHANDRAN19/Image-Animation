from PIL import Image
import numpy as np
import glob
import os
import random
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def pull_out_shake_rotate_effect(image, target_size, dynamic_zoom, rotation_angle, shake_intensity=5):
    """
    Applies a combined pull-out (zoom out), shake, and rotation effect.
    
    Parameters:
      image: PIL.Image object (original full-resolution image).
      target_size: Tuple (width, height) for the video frame (we use original image size).
      dynamic_zoom: Zoom factor applied to the crop. For example, 1.2 means the crop is 1/1.2
                    of the full image (zoomed in); 1.0 means no zoom (full image).
      rotation_angle: Angle in degrees to rotate the image. Negative means anticlockwise.
      shake_intensity: Maximum pixel offset for random shake.
    
    Returns:
      A PIL.Image frame of size target_size with the effect applied.
    """
    target_width, target_height = target_size
    orig_width, orig_height = image.size

    # Step 1: Rotate the original image by the specified angle.
    # Use BICUBIC resampling and expand so the entire rotated image is available.
    rotated = image.rotate(rotation_angle, resample=Image.Resampling.BICUBIC, expand=True)
    rot_width, rot_height = rotated.size
    center_x, center_y = rot_width // 2, rot_height // 2

    # Step 2: Determine the crop size based on dynamic_zoom.
    # A dynamic_zoom > 1.0 gives a smaller crop (zoomed in).
    crop_width = int(orig_width / dynamic_zoom)
    crop_height = int(orig_height / dynamic_zoom)
    
    # Step 3: Calculate the crop region centered at the rotated image's center,
    # with random shake offsets.
    # Compute ideal top-left corner for a centered crop.
    left = center_x - crop_width // 2
    upper = center_y - crop_height // 2
    
    # Apply random shake offsets.
    offset_x = random.randint(-shake_intensity, shake_intensity)
    offset_y = random.randint(-shake_intensity, shake_intensity)
    left += offset_x
    upper += offset_y
    
    # Ensure the crop region is within the bounds of the rotated image.
    left = max(0, min(left, rot_width - crop_width))
    upper = max(0, min(upper, rot_height - crop_height))
    right = left + crop_width
    lower = upper + crop_height

    # Step 4: Crop the rotated image.
    cropped = rotated.crop((left, upper, right, lower))
    
    # Step 5: Resize the cropped region back to the target size.
    frame = cropped.resize(target_size, Image.Resampling.LANCZOS)
    return frame

# Video settings.
fps = 25
duration = 5  # seconds
num_frames = fps * duration

# Use the original image's size as the video frame size.
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
    orig_size = img.size  # (width, height)
    
    # Define parameters for the effect.
    # We'll start with a zoom factor of 1.2 (zoomed in) and end at 1.0 (full image).
    start_zoom = 1.2
    end_zoom = 1.0
    # Start rotated anticlockwise at -15° and end at 0°.
    start_angle = -15
    end_angle = 0
    shake_intensity = 5

    # Generate frames.
    for i in range(num_frames):
        # Interpolate zoom factor linearly.
        dynamic_zoom = start_zoom + (end_zoom - start_zoom) * (i / (num_frames - 1))
        # Interpolate rotation angle linearly.
        rotation_angle = start_angle + (end_angle - start_angle) * (i / (num_frames - 1))
        
        frame = pull_out_shake_rotate_effect(img, orig_size, dynamic_zoom, rotation_angle, shake_intensity)
        # Convert PIL image to numpy array.
        frames.append(np.array(frame))

# Create video clip using MoviePy.
clip = ImageSequenceClip(frames, fps=fps)
base_name = os.path.splitext(os.path.basename(image_files[0]))[0]
output_video = os.path.join(downloads_folder, base_name + '_pull_out_shake_rotate.mp4')
clip.write_videofile(output_video, codec='libx264')
