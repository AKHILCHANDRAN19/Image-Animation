import cv2
import numpy as np
import os
import time

# --- Configuration ---
image_path = "/storage/emulated/0/Download/input.png"
output_video_path = "/storage/emulated/0/Download/output_intense_flicker_alpha.mp4"
duration_seconds = 10
fps = 30  # Frames per second - Higher FPS can make flicker feel more intense

# --- Flicker Parameters (using Alpha Blending) ---
# Range for random alpha (opacity) per frame.
# 0.0 = fully black, 1.0 = fully visible image
min_alpha = 0.0
max_alpha = 1.0

# --- End Configuration ---

# 1. Check if input image exists
if not os.path.exists(image_path):
    print(f"Error: Input image not found at {image_path}")
    exit()

# 2. Load the input image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image file: {image_path}")
    exit()

# Ensure image is in standard 8-bit format (most common)
if img.dtype != np.uint8:
    print(f"Warning: Image dtype is {img.dtype}, expected uint8. Converting.")
    # Basic conversion, might need adjustment depending on original range
    if img.max() > 255:
        img = (img / img.max() * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

# 3. Get image dimensions (height, width)
h, w = img.shape[:2]
print(f"Input image dimensions: Width={w}, Height={h}")

# --- Pre-calculate for efficiency ---
# Create a black background image *once*
black_background = np.zeros_like(img)
# Calculate total frames
total_frames = int(duration_seconds * fps)

# 4. Setup Video Writer
frame_size = (w, h)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

if not out.isOpened():
    print(f"Error: Could not open video writer for path: {output_video_path}")
    exit()

print(f"Generating INTENSE ALPHA FLICKER video: {output_video_path} ({duration_seconds}s @ {fps}fps)")
print(f"Alpha range per frame: [{min_alpha:.2f}, {max_alpha:.2f}]")
print(f"Total frames to generate: {total_frames}")

try:
    # 5. Generate frames with intense alpha flicker animation
    for i in range(total_frames):
        # --- CORE FLICKER LOGIC ---
        # Generate a random alpha (opacity) value for this specific frame
        current_alpha = np.random.uniform(min_alpha, max_alpha)

        # Blend the original image with the black background using the random alpha
        # alpha for img, (1-alpha) for black_background, gamma=0
        flickered_frame = cv2.addWeighted(img, current_alpha, black_background, 1.0 - current_alpha, 0)
        # --- End Flicker Logic ---

        # Write the frame
        out.write(flickered_frame)

        # Optional: Print progress
        if (i + 1) % fps == 0: # Print every second
             print(f"Processed frame {i + 1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
             # time.sleep(0.001) # Usually not needed

finally:
    # 6. Release the VideoWriter
    print("Releasing video writer...")
    out.release()

print(f"\nVideo saved successfully to: {output_video_path}")
