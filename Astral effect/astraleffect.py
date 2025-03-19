import cv2
import numpy as np
import glob
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def astral_effect(image, max_offset=10, blur_kernel=(15, 15), ghost_alpha=0.4):
    """
    Applies an astral effect by shifting, blurring, and blending the image.
    
    Parameters:
      image: The input image in BGR format.
      max_offset: Maximum pixel shift for translation.
      blur_kernel: Kernel size for Gaussian blur.
      ghost_alpha: Blending factor for the ghost layer.
      
    Returns:
      The image with an astral (ghostly) effect applied.
    """
    rows, cols, _ = image.shape
    # Generate small random offsets
    dx = np.random.randint(-max_offset, max_offset)
    dy = np.random.randint(-max_offset, max_offset)
    
    # Create a translation matrix and shift the image
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(image, M, (cols, rows))
    
    # Apply Gaussian blur to the shifted (ghost) copy
    blurred = cv2.GaussianBlur(shifted, blur_kernel, 0)
    
    # Blend the original image with the blurred, shifted version
    astral_img = cv2.addWeighted(image, 1 - ghost_alpha, blurred, ghost_alpha, 0)
    return astral_img

# Define the full path to your images folder.
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

# Process each image file found.
for img_path in image_files:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Unable to read image: {img_path}. Skipping...")
        continue

    frames = []
    # Generate a series of frames with the astral effect.
    for i in range(num_frames):
        astral_frame = astral_effect(image)
        # Convert from BGR to RGB (MoviePy expects RGB).
        astral_frame_rgb = cv2.cvtColor(astral_frame, cv2.COLOR_BGR2RGB)
        frames.append(astral_frame_rgb)
    
    # Create a video clip from the generated frames.
    clip = ImageSequenceClip(frames, fps=fps)
    # Save the video with a modified filename.
    base_name, _ = os.path.splitext(os.path.basename(img_path))
    output_video = os.path.join(downloads_folder, base_name + '_astral.mp4')
    clip.write_videofile(output_video, codec='libx264')
