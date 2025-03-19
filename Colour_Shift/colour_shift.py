import cv2
import numpy as np
import glob
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def color_shift_desaturate(image, hue_shift=0, sat_factor=1.0, brightness_factor=1.0):
    """
    Alters the hue, saturation, and brightness of the image.
    
    Parameters:
      image: Input image in BGR format.
      hue_shift: Amount to add to the hue channel (OpenCV hue range is 0-180).
      sat_factor: Factor to multiply the saturation channel.
      brightness_factor: Factor to multiply the value (brightness) channel.
      
    Returns:
      The image with modified color properties.
    """
    # Convert image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Adjust hue (wrap around using modulo 180)
    hsv[:,:,0] = (hsv[:,:,0] + hue_shift) % 180
    # Adjust saturation and brightness
    hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_factor, 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * brightness_factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    # Convert back to BGR
    shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return shifted

# Define the folder containing your images.
downloads_folder = '/storage/emulated/0/Download'
extensions = ['*.jpg', '*.jpeg', '*.png']

# Gather all image files.
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

# Parameters for gradual effect
initial_sat_factor = 1.0      # Start fully saturated
final_sat_factor = 0.5        # End at 50% saturation (desaturated)
initial_hue = 0               # Start with no hue shift
final_hue = 30              # Gradually shift hue by 30 units (roughly toward cooler tones)

# Process each image file.
for img_path in image_files:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Unable to read image: {img_path}. Skipping...")
        continue

    frames = []
    # Create frames with a gradual color shift/desaturation.
    for i in range(num_frames):
        # Linear interpolation for hue shift and saturation factor.
        current_hue = initial_hue + (final_hue - initial_hue) * (i / num_frames)
        current_sat_factor = initial_sat_factor + (final_sat_factor - initial_sat_factor) * (i / num_frames)
        # Optionally, you can also adjust brightness by modifying brightness_factor.
        shifted = color_shift_desaturate(image, hue_shift=current_hue, sat_factor=current_sat_factor)
        
        # Convert from BGR to RGB for MoviePy.
        shifted_rgb = cv2.cvtColor(shifted, cv2.COLOR_BGR2RGB)
        frames.append(shifted_rgb)
    
    # Create a video clip from the frames.
    clip = ImageSequenceClip(frames, fps=fps)
    base_name, _ = os.path.splitext(os.path.basename(img_path))
    output_video = os.path.join(downloads_folder, base_name + '_color_shift.mp4')
    clip.write_videofile(output_video, codec='libx264')
