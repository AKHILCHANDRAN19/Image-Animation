import cv2
import numpy as np
import glob
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def flicker_effect(image, alpha):
    """
    Applies an intermittent flicker effect by blending the input image with a black background.
    
    Parameters:
      image: Input image in BGR format.
      alpha: Opacity level for the image (0.0 = fully transparent, 1.0 = fully visible).
      
    Returns:
      The image blended with black according to the alpha value.
    """
    # Create a black background image of the same size.
    black = np.zeros_like(image)
    # Blend the original image with black.
    flickered = cv2.addWeighted(image, alpha, black, 1 - alpha, 0)
    return flickered

# Define the folder where your images are stored.
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

# Process each image file.
for img_path in image_files:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Unable to read image: {img_path}. Skipping...")
        continue

    frames = []
    # For each frame, generate a random opacity value to simulate flickering.
    for i in range(num_frames):
        # Generate a random alpha between 0 (ghost disappears) and 1 (fully visible).
        alpha = np.random.uniform(0.0, 1.0)
        # Optionally, you could mix in a periodic component:
        # alpha = 0.5 + 0.5 * np.sin(2 * np.pi * i / num_frames) * np.random.uniform(0.5, 1.0)
        
        flickered_frame = flicker_effect(image, alpha)
        # Convert the frame from BGR to RGB for MoviePy.
        flickered_frame_rgb = cv2.cvtColor(flickered_frame, cv2.COLOR_BGR2RGB)
        frames.append(flickered_frame_rgb)
    
    # Create a video clip from the generated frames.
    clip = ImageSequenceClip(frames, fps=fps)
    base_name, _ = os.path.splitext(os.path.basename(img_path))
    output_video = os.path.join(downloads_folder, base_name + '_flicker.mp4')
    clip.write_videofile(output_video, codec='libx264')
