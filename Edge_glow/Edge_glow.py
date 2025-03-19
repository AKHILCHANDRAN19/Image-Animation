import cv2
import numpy as np
import glob
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def edge_glow_effect(image, lower_thresh=100, upper_thresh=200, glow_color=(255, 0, 0), glow_intensity=0.7, blur_kernel=(15, 15)):
    """
    Applies an edge glowing outline effect.
    
    Steps:
      1. Convert the image to grayscale.
      2. Detect edges using Canny edge detection.
      3. Dilate the edges to make them thicker.
      4. Apply a Gaussian blur to soften the edges.
      5. Colorize the blurred edges using the specified glow_color.
      6. Overlay the glowing edges onto the original image using weighted blending.
    
    Parameters:
      image: Input image in BGR format.
      lower_thresh, upper_thresh: Thresholds for Canny edge detection.
      glow_color: The color (in BGR) used for the glow (e.g., (255, 0, 0) for blue).
      glow_intensity: Weight for blending the glow layer with the original image.
      blur_kernel: Kernel size for the Gaussian blur.
      
    Returns:
      The image with the edge glowing outline effect applied.
    """
    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, lower_thresh, upper_thresh)
    
    # Dilate edges to thicken them
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Apply Gaussian blur to soften edges (creating the glow)
    edges_blurred = cv2.GaussianBlur(edges_dilated, blur_kernel, 0)
    
    # Normalize the blurred edges to range [0, 1]
    edges_norm = edges_blurred.astype(np.float32) / 255.0
    
    # Create a glow layer with the desired color (BGR) by multiplying the color by the normalized edge mask.
    glow_layer = np.zeros_like(image, dtype=np.float32)
    glow_layer[:] = glow_color  # Set the entire layer to the glow color.
    glow_layer = glow_layer * edges_norm[:, :, np.newaxis]
    glow_layer = glow_layer.astype(np.uint8)
    
    # Blend the glow layer with the original image.
    output = cv2.addWeighted(image, 1.0, glow_layer, glow_intensity, 0)
    return output

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
    # Animate the glow intensity to create a pulsating effect.
    for i in range(num_frames):
        # Vary the glow intensity using a sine function (oscillates between 0.7 and 1.0, for example)
        dynamic_intensity = 0.7 + 0.3 * np.sin(2 * np.pi * i / num_frames)
        effect_frame = edge_glow_effect(image, lower_thresh=100, upper_thresh=200, 
                                          glow_color=(255, 0, 0),  # Blue-ish glow in BGR (255, 0, 0)
                                          glow_intensity=dynamic_intensity,
                                          blur_kernel=(15, 15))
        # Convert from BGR to RGB (MoviePy expects RGB)
        effect_frame_rgb = cv2.cvtColor(effect_frame, cv2.COLOR_BGR2RGB)
        frames.append(effect_frame_rgb)
    
    # Create a video clip from the frames.
    clip = ImageSequenceClip(frames, fps=fps)
    base_name, _ = os.path.splitext(os.path.basename(img_path))
    output_video = os.path.join(downloads_folder, base_name + '_edge_glow.mp4')
    clip.write_videofile(output_video, codec='libx264')
