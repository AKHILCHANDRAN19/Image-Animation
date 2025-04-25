import cv2
import numpy as np
import os
import time

# --- Configuration ---
image_path = "/storage/emulated/0/Download/input.png"
output_video_path = "/storage/emulated/0/Download/output_fast_flicker_animation.mp4"
duration_seconds = 10
fps = 30  # Frames per second

# --- !! KEY CHANGES FOR FASTNESS !! ---
# Controls the maximum brightness change due to the rapid wave (0-255 scale)
slow_wave_amplitude = 25 # Slightly higher amplitude for the faster wave

# How many full cycles of the brightness wave occur over the video duration
# Higher value = faster overall pulse/wave
slow_wave_cycles = 15 # e.g., 15 cycles over 10 seconds - much faster wave!

# Controls the maximum random brightness fluctuation per frame
# Higher value = more intense instantaneous flicker
random_noise_level = 15 # +/- this value randomly added/subtracted each frame

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
    if img.max() > 255:
        img = (img / 256).astype(np.uint8)
    else:
        img = img.astype(np.uint8)


# 3. Get image dimensions (height, width)
h, w = img.shape[:2]
print(f"Input image dimensions: Width={w}, Height={h}")

# --- Pre-calculate for efficiency ---
img_float = img.astype(np.float32) # Convert base image to float32
total_frames = int(duration_seconds * fps)
# Calculate frequency for the *fast* sine wave
frequency_rad_per_frame = (2.0 * np.pi * slow_wave_cycles) / total_frames

# 4. Setup Video Writer
frame_size = (w, h)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

if not out.isOpened():
    print(f"Error: Could not open video writer for path: {output_video_path}")
    exit()

print(f"Generating FAST FLICKER video: {output_video_path} ({duration_seconds}s @ {fps}fps)")
print(f"Fast wave: Amplitude={slow_wave_amplitude}, Cycles={slow_wave_cycles}")
print(f"Random noise level: +/- {random_noise_level}")
print(f"Total frames to generate: {total_frames}")

try:
    # 5. Generate frames with fast flicker animation
    for i in range(total_frames):
        # Calculate the *fast* brightness offset using sine wave
        sine_offset = slow_wave_amplitude * np.sin(frequency_rad_per_frame * i)

        # Calculate a *larger* random offset for the flicker effect
        random_offset = np.random.randint(-random_noise_level, random_noise_level + 1)

        # Combine the offsets
        total_brightness_offset = sine_offset + random_offset

        # Apply the brightness offset to the float image
        flicker_frame_float = img_float + total_brightness_offset

        # --- IMPORTANT: Clip the values ---
        flicker_frame_float = np.clip(flicker_frame_float, 0, 255)

        # Convert back to uint8 for writing to video
        flicker_frame_uint8 = flicker_frame_float.astype(np.uint8)

        # Write the frame
        out.write(flicker_frame_uint8)

        # Optional: Print progress (might print more often now)
        if (i + 1) % fps == 0: # Print every second
             print(f"Processed frame {i + 1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
             # time.sleep(0.001) # Unlikely needed

finally:
    # 6. Release the VideoWriter
    print("Releasing video writer...")
    out.release()

print(f"\nVideo saved successfully to: {output_video_path}")
