import cv2
import numpy as np
import glob
import os
import random
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def dolly_zoom_map(shape, center, r0, s_dynamic, shake_intensity):
    """
    Computes mapping arrays for a dolly zoom effect with shake.
    
    For each pixel (x,y), we compute its distance r from the center.
    If r < r0, scale factor = 1 (subject remains unchanged).
    If r >= r0, the scale factor linearly interpolates from 1 at r=r0 to s_dynamic at r=max_r.
    Then new coordinates are computed as:
       new_x = cx + scale * (x - cx) + random_shake_x
       new_y = cy + scale * (y - cy) + random_shake_y
       
    Parameters:
      shape: Tuple (height, width)
      center: Tuple (cx, cy)
      r0: Radius below which no scaling is applied.
      s_dynamic: The dynamic scale factor for the background (e.g. 1.2 to 1.0)
      shake_intensity: Maximum pixel shake.
      
    Returns:
      map_x, map_y: Floating-point mapping arrays for cv2.remap.
    """
    rows, cols = shape
    cx, cy = center
    # Maximum possible radius from center (corner distance)
    max_r = np.sqrt(cx**2 + cy**2)
    
    # Create meshgrid of pixel coordinates.
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    # Compute distance from center for each pixel.
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Compute scaling factor for each pixel.
    # For r < r0: scale = 1.
    # For r >= r0: linearly interpolate: scale = 1 + (s_dynamic - 1) * ((r - r0) / (max_r - r0))
    scale = np.ones_like(r, dtype=np.float32)
    mask = r >= r0
    scale[mask] = 1 + (s_dynamic - 1) * ((r[mask] - r0) / (max_r - r0))
    
    # Compute new coordinates without shake.
    map_x = cx + scale * (x - cx)
    map_y = cy + scale * (y - cy)
    
    # Add random shake offsets uniformly for each frame.
    shake_x = np.random.uniform(-shake_intensity, shake_intensity)
    shake_y = np.random.uniform(-shake_intensity, shake_intensity)
    map_x += shake_x
    map_y += shake_y
    
    return map_x.astype(np.float32), map_y.astype(np.float32)

# Video settings.
fps = 25
duration = 5  # seconds
num_frames = fps * duration

# Define folder and gather image files.
downloads_folder = '/storage/emulated/0/Download'
extensions = ['*.jpg', '*.jpeg', '*.png']
image_files = []
for ext in extensions:
    image_files.extend(glob.glob(os.path.join(downloads_folder, ext)))
    
if not image_files:
    print("No image files found in the specified folder.")
    exit()

# Process the first image (for demonstration).
# We assume the video frame size is the original image size.
image = cv2.imread(image_files[0])
if image is None:
    print("Unable to read image.")
    exit()

rows, cols, _ = image.shape
center = (cols // 2, rows // 2)
# Define subject radius as 25% of the minimum dimension.
r0 = 0.25 * min(cols, rows)

frames = []
# We'll vary s_dynamic from 1.2 (zoomed background) to 1.0 (normal) over time.
s_start = 1.2
s_end = 1.0

for i in range(num_frames):
    t = i / (num_frames - 1)
    s_dynamic = s_start + (s_end - s_start) * t
    # Compute mapping arrays for this frame.
    map_x, map_y = dolly_zoom_map((rows, cols), center, r0, s_dynamic, shake_intensity=5)
    # Apply the mapping.
    frame = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # Append the frame (convert BGR to RGB for MoviePy).
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)

# Create a video clip from the frames.
clip = ImageSequenceClip(frames, fps=fps)
base_name = os.path.splitext(os.path.basename(image_files[0]))[0]
output_video = os.path.join(downloads_folder, base_name + '_dolly_zoom.mp4')
clip.write_videofile(output_video, codec='libx264')
