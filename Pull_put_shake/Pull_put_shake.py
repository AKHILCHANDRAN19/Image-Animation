import cv2
import numpy as np
import glob
import os
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

def pull_out_and_shake(image, scale_factor, shake_intensity=5):
    """
    Applies a pull-out (zoom out) and shake effect.
    
    The image is scaled down according to the scale_factor (where a value less than 1 means zooming out)
    and then placed on a black canvas of the original image size with a random offset (shake effect).
    
    Parameters:
      image: Input image in BGR format.
      scale_factor: Scaling factor for the image (e.g. 1.0 = original size, 0.7 = zoomed out).
      shake_intensity: Maximum pixel displacement in any direction for the shake effect.
      
    Returns:
      The resulting frame with the pull-out and shake effect.
    """
    h, w = image.shape[:2]
    # Compute new size after scaling.
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    # Resize the image.
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create a black canvas of the original size.
    canvas = np.zeros_like(image)
    
    # Generate random offsets for shake effect.
    offset_x = np.random.randint(-shake_intensity, shake_intensity + 1)
    offset_y = np.random.randint(-shake_intensity, shake_intensity + 1)
    
    # Compute position to place the scaled image such that it is centered plus the shake offsets.
    start_x = (w - new_w) // 2 + offset_x
    start_y = (h - new_h) // 2 + offset_y
    end_x = start_x + new_w
    end_y = start_y + new_h
    
    # Handle boundaries: determine the overlapping region between the canvas and the scaled image.
    canvas_x1 = max(start_x, 0)
    canvas_y1 = max(start_y, 0)
    canvas_x2 = min(end_x, w)
    canvas_y2 = min(end_y, h)
    
    # Corresponding region from the scaled image.
    img_x1 = canvas_x1 - start_x
    img_y1 = canvas_y1 - start_y
    img_x2 = img_x1 + (canvas_x2 - canvas_x1)
    img_y2 = img_y1 + (canvas_y2 - canvas_y1)
    
    # Paste the scaled (and shaken) image onto the canvas.
    canvas[canvas_y1:canvas_y2, canvas_x1:canvas_x2] = scaled[img_y1:img_y2, img_x1:img_x2]
    return canvas

# Define the folder containing your images.
downloads_folder = '/storage/emulated/0/Download'
extensions = ['*.jpg', '*.jpeg', '*.png']

# Gather all image files.
image_files = []
for ext in extensions:
    image_files.extend(glob.glob(os.path.join(downloads_folder, ext)))

if not image_files:
    print("No images found in the folder.")
    exit()

# Video settings: 25 fps for a 5-second video.
fps = 25
duration = 5  # seconds
num_frames = fps * duration

# Process each image file.
for img_path in image_files:
    image = cv2.imread(img_path)
    if image is None:
        print(f"Unable to read {img_path}. Skipping...")
        continue

    frames = []
    # For a pull-out effect, we simulate a zoom out:
    # Scale factor will go from 1.0 (full size) to 0.7 (zoomed out).
    start_scale = 1.0
    end_scale = 0.7
    for i in range(num_frames):
        # Linear interpolation between start_scale and end_scale.
        scale_factor = start_scale + (end_scale - start_scale) * (i / (num_frames - 1))
        frame = pull_out_and_shake(image, scale_factor, shake_intensity=5)
        # Convert frame from BGR to RGB (MoviePy expects RGB).
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    # Create a video clip from the frames.
    clip = ImageSequenceClip(frames, fps=fps)
    base_name, _ = os.path.splitext(os.path.basename(img_path))
    output_video = os.path.join(downloads_folder, base_name + '_pull_out_shake.mp4')
    clip.write_videofile(output_video, codec='libx264')
