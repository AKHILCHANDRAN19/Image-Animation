import cv2
import numpy as np
import glob
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def generate_light_leak_mask(shape, center, radius, intensity=1.0):
    """
    Generates a radial gradient mask that decays with distance from the center.
    
    Parameters:
      shape: Tuple representing the shape of the image (height, width, channels).
      center: Tuple (x, y) for the leak center.
      radius: Radius over which the leak fades.
      intensity: Maximum intensity multiplier for the mask.
      
    Returns:
      A 2D mask with values in [0,1], scaled by intensity.
    """
    rows, cols = shape[:2]
    # Create a coordinate grid.
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    # Compute Euclidean distance from the center.
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    # Create a mask that decays linearly with distance.
    mask = np.clip(1 - (dist / radius), 0, 1)
    mask = mask * intensity
    return mask

def apply_light_leaks(image, num_leaks=2, intensity_range=(0.2, 0.5)):
    """
    Overlays a number of random light leaks onto the image.
    
    Parameters:
      image: Input image in BGR format.
      num_leaks: Number of separate leaks to overlay.
      intensity_range: Tuple specifying the random intensity range for each leak.
      
    Returns:
      The image with light leaks/lens flares overlaid.
    """
    # Create an empty overlay.
    overlay = np.zeros_like(image, dtype=np.float32)
    rows, cols, _ = image.shape
    
    for _ in range(num_leaks):
        # Random center for the leak.
        center = (np.random.randint(0, cols), np.random.randint(0, rows))
        # Random radius between 10% and 50% of the minimum dimension.
        radius = np.random.uniform(min(rows, cols) * 0.1, min(rows, cols) * 0.5)
        # Random intensity within the specified range.
        intensity = np.random.uniform(*intensity_range)
        # Choose a warm, flare-like color in BGR (e.g., low blue, moderate green, high red).
        B = np.random.randint(0, 80)
        G = np.random.randint(100, 200)
        R = np.random.randint(150, 255)
        color = (B, G, R)
        
        # Generate the leak mask.
        mask = generate_light_leak_mask(image.shape, center, radius, intensity)
        # Expand mask to three channels.
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        
        # Create a colored leak by multiplying the mask with the chosen color.
        leak = np.zeros_like(image, dtype=np.float32)
        leak[:, :, 0] = mask * color[0]
        leak[:, :, 1] = mask * color[1]
        leak[:, :, 2] = mask * color[2]
        
        # Accumulate leaks.
        overlay += leak

    # Clip overlay values.
    overlay = np.clip(overlay, 0, 255)
    # Blend the overlay with the original image.
    result = cv2.addWeighted(image.astype(np.float32), 1.0, overlay, 1.0, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

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

# Video settings.
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
    # For each frame, apply the light leak effect.
    # To simulate unpredictability, new random leaks are generated for each frame.
    for i in range(num_frames):
        leak_frame = apply_light_leaks(image, num_leaks=2, intensity_range=(0.2, 0.5))
        # Optionally add a flicker by modulating overall brightness using a sine function.
        brightness_factor = 0.9 + 0.2 * np.sin(2 * np.pi * i / num_frames)
        leak_frame = cv2.convertScaleAbs(leak_frame, alpha=brightness_factor, beta=0)
        # Convert from BGR to RGB (MoviePy expects RGB images).
        leak_frame_rgb = cv2.cvtColor(leak_frame, cv2.COLOR_BGR2RGB)
        frames.append(leak_frame_rgb)
    
    # Compile frames into a video.
    clip = ImageSequenceClip(frames, fps=fps)
    base_name, _ = os.path.splitext(os.path.basename(img_path))
    output_video = os.path.join(downloads_folder, base_name + '_light_leaks.mp4')
    clip.write_videofile(output_video, codec='libx264')
