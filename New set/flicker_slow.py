import cv2
import numpy as np
import os
import time

# --- Configuration ---
image_path = "/storage/emulated/0/Download/input.png"
output_video_path = "/storage/emulated/0/Download/output_slow_flicker_animation.mp4"
duration_seconds = 10
fps = 30  # Frames per second

# --- Flicker Parameters ---
# Controls the maximum brightness change due to the slow wave (0-255 scale)
# Smaller value = more subtle overall drift
slow_wave_amplitude = 20

# How many full cycles of the slow brightness wave occur over the video duration
# Smaller value = slower overall drift
slow_wave_cycles = 1.5 # e.g., 1.5 cycles over 10 seconds

# Controls the maximum random brightness fluctuation per frame
# Smaller value = less noticeable instantaneous flicker
random_noise_level = 5 # +/- this value randomly added/subtracted each frame

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

# Ensure image is in standard 8-bit format
if img.dtype != np.uint8:
    print("Warning: Image depth is not uint8. Converting.")
    # Handle potential scaling if needed, assuming input range is standard
    if img.max() > 255: # e.g., for 16-bit images
        img = (img / 256).astype(np.uint8)
    else:
        img = img.astype(np.uint8)


# 3. Get image dimensions (height, width)
h, w = img.shape[:2]
print(f"Input image dimensions: Width={w}, Height={h}")

# --- Pre-calculate for efficiency ---
# Convert the base image to float32 once for calculations
img_float = img.astype(np.float32)
# Calculate total frames
total_frames = int(duration_seconds * fps)
# Calculate frequency for the slow sine wave
# omega = 2 * pi * f, where f = cycles / total_time = cycles / (total_frames / fps)
# frequency_rad_per_frame = 2 * np.pi * slow_wave_cycles / total_frames
frequency_rad_per_frame = (2.0 * np.pi * slow_wave_cycles) / total_frames

# 4. Setup Video Writer
frame_size = (w, h)  # Output video dimensions will match input image
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

if not out.isOpened():
    print(f"Error: Could not open video writer for path: {output_video_path}")
    exit()

print(f"Generating SLOW FLICKER video: {output_video_path} ({duration_seconds}s @ {fps}fps)")
print(f"Slow wave: Amplitude={slow_wave_amplitude}, Cycles={slow_wave_cycles}")
print(f"Random noise level: +/- {random_noise_level}")
print(f"Total frames to generate: {total_frames}")

try:
    # 5. Generate frames with slow flicker animation
    for i in range(total_frames):
        # Calculate the slow brightness offset using sine wave
        sine_offset = slow_wave_amplitude * np.sin(frequency_rad_per_frame * i)

        # Calculate a small random offset for the flicker effect
        random_offset = np.random.randint(-random_noise_level, random_noise_level + 1)

        # Combine the offsets
        total_brightness_offset = sine_offset + random_offset

        # Apply the brightness offset to the float image
        # Adding the offset to all channels equally
        flicker_frame_float = img_float + total_brightness_offset

        # --- IMPORTANT: Clip the values ---
        # Ensure pixel values stay within the valid 0-255 range
        flicker_frame_float = np.clip(flicker_frame_float, 0, 255)

        # Convert back to uint8 for writing to video
        flicker_frame_uint8 = flicker_frame_float.astype(np.uint8)

        # Write the frame
        out.write(flicker_frame_uint8)

        # Optional: Print progress
        if (i + 1) % (fps * 2) == 0: # Print every 2 seconds
             print(f"Processed frame {i + 1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
             # time.sleep(0.001) # Seldom needed, but available

finally:
    # 6. Release the VideoWriter
    print("Releasing video writer...")
    out.release()

print(f"\nVideo saved successfully to: {output_video_path}")
