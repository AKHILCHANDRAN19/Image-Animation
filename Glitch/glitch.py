import cv2
import numpy as np
import glob
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def glitch_image(image, max_offset=20):
    """
    Applies a simple glitch effect by randomly shifting each color channel.
    OpenCV uses BGR format, so this effect shifts each channel horizontally.
    """
    # Split image into Blue, Green, and Red channels
    b, g, r = cv2.split(image)
    
    # Generate random horizontal offsets for each channel
    offset_b = np.random.randint(-max_offset, max_offset)
    offset_g = np.random.randint(-max_offset, max_offset)
    offset_r = np.random.randint(-max_offset, max_offset)
    
    # Apply horizontal shift (roll) on each channel
    b_glitched = np.roll(b, offset_b, axis=1)
    g_glitched = np.roll(g, offset_g, axis=1)
    r_glitched = np.roll(r, offset_r, axis=1)
    
    # Merge the channels back together
    glitched = cv2.merge([b_glitched, g_glitched, r_glitched])
    return glitched

# Define the full path to the Download folder.
downloads_folder = '/storage/emulated/0/Download'
# Define file extensions to search for.
extensions = ['*.jpg', '*.jpeg', '*.png']

# Gather all image files with the specified extensions.
image_files = []
for ext in extensions:
    image_files.extend(glob.glob(os.path.join(downloads_folder, ext)))

if not image_files:
    print("No image files found in the Download folder.")
    exit()

# Video settings: 25 frames per second for a 5-second video.
fps = 25
duration = 5  # seconds
num_frames = fps * duration

# Process each image file found in the Download folder.
for img_path in image_files:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Unable to read image: {img_path}. Skipping...")
        continue

    frames = []
    # Generate a series of frames with the glitch effect.
    for i in range(num_frames):
        glitched_frame = glitch_image(image)
        # Convert from BGR to RGB (MoviePy expects RGB).
        glitched_frame_rgb = cv2.cvtColor(glitched_frame, cv2.COLOR_BGR2RGB)
        frames.append(glitched_frame_rgb)
    
    # Create a video clip from the generated frames.
    clip = ImageSequenceClip(frames, fps=fps)
    # Save the video in the Download folder with a modified filename.
    base_name, _ = os.path.splitext(os.path.basename(img_path))
    output_video = os.path.join(downloads_folder, base_name + '_glitch.mp4')
    clip.write_videofile(output_video, codec='libx264')
