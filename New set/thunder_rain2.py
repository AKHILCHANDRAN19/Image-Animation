import cv2
import numpy as np
import random
import os
import time

# --- Configuration ---
input_image_path = "/storage/emulated/0/Download/input.png"
# New output name for continuous flicker
output_video_path = "/storage/emulated/0/Download/output_video_constant_flicker.mp4"

# Video settings
duration_seconds = 10 # You can adjust duration as needed
fps = 30
num_frames = duration_seconds * fps

# Base Image Settings
darkness_factor = 0.3 # How dark the 'low' part of the flicker should be

# Rain effect settings
num_raindrops = 700
rain_length_min = 15
rain_length_max = 40
rain_speed_min = 10
rain_speed_max = 20
rain_thickness = 1
rain_color_min = 150
rain_color_max = 220

# Flicker effect settings
# Probability of showing the BRIGHT (original) frame each time
flicker_brightness_probability = 0.50 # 0.5 gives ~50/50 bright/dark frames

# --- End Configuration ---

# 1. Load the Input Image
print(f"Loading image: {input_image_path}")
original_image = cv2.imread(input_image_path)

if original_image is None:
    print(f"Error: Could not load image from {input_image_path}")
    print("Please ensure the file exists and the path is correct.")
    exit()

height, width, _ = original_image.shape
print(f"Image loaded successfully (Width: {width}, Height: {height})")

# 2. Create the Darkened Base Image
print(f"Creating darkened base image (factor: {darkness_factor})...")
dark_image_float = original_image.astype(np.float32) * darkness_factor
dark_image = np.clip(dark_image_float, 0, 255).astype(np.uint8)
print("Darkened image created.")

# 3. Initialize Video Writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
print(f"Initializing video writer for: {output_video_path}")
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

if not video_writer.isOpened():
    print(f"Error: Could not open video writer for {output_video_path}")
    print("Check permissions and if the codec 'mp4v' is supported.")
    exit()

# 4. Initialize Raindrops with Variable Brightness
raindrops = []
for _ in range(num_raindrops):
    x = random.randint(0, width)
    y = random.randint(-int(height*0.5), height)
    length = random.randint(rain_length_min, rain_length_max)
    speed = random.randint(rain_speed_min, rain_speed_max)
    gray_value = random.randint(rain_color_min, rain_color_max)
    raindrops.append([x, y, length, speed, gray_value])

# 5. Main Loop - Generate Frames with Constant Flicker
print(f"Generating {num_frames} frames with constant flicker...")
start_time = time.time()
for frame_count in range(num_frames):

    # --- Apply Constant Flicker ---
    # Decide randomly for EACH frame whether to use the bright or dark base
    if random.random() < flicker_brightness_probability:
        # Use the bright original image for this frame
        frame = original_image.copy()
    else:
        # Use the dark image for this frame
        frame = dark_image.copy()

    # --- Apply Rain (Draw on top of the selected base frame) ---
    for i in range(num_raindrops):
        # Move drop down
        raindrops[i][1] += raindrops[i][3] # y = y + speed

        # Get current drop details
        x, y, length, speed, gray_value = raindrops[i]
        rain_color = (gray_value, gray_value, gray_value)

        # Draw the raindrop line
        start_point = (int(x), int(y))
        end_point = (int(x), int(y + length))
        cv2.line(frame, start_point, end_point, rain_color, rain_thickness)

        # Reset drop if it goes off screen
        if y > height:
            raindrops[i][0] = random.randint(0, width)
            raindrops[i][1] = random.randint(-length*3, -length)
            raindrops[i][2] = random.randint(rain_length_min, rain_length_max)
            raindrops[i][3] = random.randint(rain_speed_min, rain_speed_max)
            raindrops[i][4] = random.randint(rain_color_min, rain_color_max)

    # --- Write Frame to Video ---
    video_writer.write(frame)

    # Optional: Display progress
    if (frame_count + 1) % fps == 0:
        elapsed = time.time() - start_time
        processed_percent = int((frame_count + 1) * 100 / num_frames)
        print(f"Processed frame {frame_count + 1}/{num_frames} ({processed_percent}%) - Elapsed: {elapsed:.2f}s")

# 6. Release Resources
video_writer.release()
end_time = time.time()
print("-" * 30)
print(f"Video generation complete! Took {end_time - start_time:.2f} seconds.")
print(f"Constant flicker video saved successfully to: {output_video_path}")
print("-" * 30)
