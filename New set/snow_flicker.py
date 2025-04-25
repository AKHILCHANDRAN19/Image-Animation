import cv2
import numpy as np
import random
import os
import time

# --- Configuration ---
input_image_path = "/storage/emulated/0/Download/input.png"
# New output name for snow effect
output_video_path = "/storage/emulated/0/Download/output_video_snow_flicker.mp4"

# Video settings
duration_seconds = 15 # Increased duration slightly to enjoy the snow
fps = 30
num_frames = duration_seconds * fps

# Base Image Settings
darkness_factor = 0.35 # Slightly adjusted darkness maybe

# Snow effect settings
num_snowflakes = 1500    # More particles for snow often looks better
snow_size_min = 1        # Smallest snowflake radius
snow_size_max = 4        # Largest snowflake radius
snow_speed_y_min = 1     # Snow falls slower vertically
snow_speed_y_max = 4
snow_drift_factor = 0.5  # How much horizontal movement (0 = none)
snow_color = (240, 240, 240) # Off-white color for snow

# Flicker effect settings
flicker_brightness_probability = 0.50 # Probability of showing the bright frame

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

# 4. Initialize Snowflakes
# Store as [x, y, radius, speed_y, speed_x]
snowflakes = []
for _ in range(num_snowflakes):
    x = random.randint(0, width)
    y = random.randint(-int(height*0.2), height) # Start some above screen
    radius = random.randint(snow_size_min, snow_size_max)
    speed_y = random.randint(snow_speed_y_min, snow_speed_y_max)
    # Horizontal speed (drift) - can be slightly left or right
    speed_x = random.uniform(-snow_drift_factor, snow_drift_factor)
    snowflakes.append([x, y, radius, speed_y, speed_x])

# 5. Main Loop - Generate Frames with Constant Flicker and Snow
print(f"Generating {num_frames} frames with snow and constant flicker...")
start_time = time.time()
for frame_count in range(num_frames):

    # --- Apply Constant Flicker ---
    if random.random() < flicker_brightness_probability:
        frame = original_image.copy()
    else:
        frame = dark_image.copy()

    # --- Apply Snow (Draw on top of the selected base frame) ---
    for i in range(num_snowflakes):
        # Update position based on speeds
        snowflakes[i][0] += snowflakes[i][4] # x = x + speed_x
        snowflakes[i][1] += snowflakes[i][3] # y = y + speed_y

        # Get current snowflake details
        x, y, radius, speed_y, speed_x = snowflakes[i]

        # Draw the snowflake (filled circle)
        # Ensure coordinates and radius are integers for drawing
        center_point = (int(x), int(y))
        cv2.circle(frame, center_point, radius, snow_color, thickness=-1) # -1 thickness fills the circle

        # Reset snowflake if it goes off screen (bottom or sides)
        # If it goes off the bottom OR drifts too far horizontally
        if y > height + radius or x < -radius or x > width + radius:
            # Reset position to somewhere above the screen
            snowflakes[i][0] = random.randint(0, width) # New random x
            snowflakes[i][1] = random.randint(-int(height*0.1), -radius*2) # Reset Y above screen
            # Assign new random properties
            snowflakes[i][2] = random.randint(snow_size_min, snow_size_max) # New radius
            snowflakes[i][3] = random.randint(snow_speed_y_min, snow_speed_y_max) # New speed_y
            snowflakes[i][4] = random.uniform(-snow_drift_factor, snow_drift_factor) # New speed_x

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
print(f"Snow flicker video saved successfully to: {output_video_path}")
print("-" * 30)
