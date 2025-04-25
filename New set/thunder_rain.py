import cv2
import numpy as np
import random
import os
import time # To add slight delay for effect (optional)

# --- Configuration ---
input_image_path = "/storage/emulated/0/Download/input.png"
output_video_path = "/storage/emulated/0/Download/output_video.mp4"

# Video settings
duration_seconds = 10  # How long the video should be
fps = 30              # Frames per second
num_frames = duration_seconds * fps

# Rain effect settings
num_raindrops = 600    # More drops = heavier rain
rain_length_min = 10
rain_length_max = 30
rain_speed_min = 8
rain_speed_max = 15
rain_color = (200, 200, 200) # Light grey/blueish
rain_thickness = 1

# Thunder effect settings
thunder_probability_per_frame = 0.015 # Chance of thunder starting in any given frame (e.g., 0.01 = 1%)
thunder_flash_duration_min = 2   # Min frames the flash lasts
thunder_flash_duration_max = 5   # Max frames the flash lasts
thunder_brightness_increase = 80 # How much brighter the screen gets during flash (0-255)

# --- End Configuration ---

# 1. Load the Input Image
print(f"Loading image: {input_image_path}")
image = cv2.imread(input_image_path)

if image is None:
    print(f"Error: Could not load image from {input_image_path}")
    print("Please ensure the file exists and the path is correct.")
    exit()

height, width, _ = image.shape
print(f"Image loaded successfully (Width: {width}, Height: {height})")

# 2. Initialize Video Writer
# Use 'mp4v' codec for MP4 files. You might need to install codecs if it fails.
# Other options include 'XVID' for AVI
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
print(f"Initializing video writer for: {output_video_path}")
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

if not video_writer.isOpened():
    print(f"Error: Could not open video writer for {output_video_path}")
    print("Check permissions and if the codec 'mp4v' is supported.")
    exit()

# 3. Initialize Raindrops
# Store as [x, y, length, speed]
raindrops = []
for _ in range(num_raindrops):
    x = random.randint(0, width)
    y = random.randint(-height, height) # Start some drops off-screen above
    length = random.randint(rain_length_min, rain_length_max)
    speed = random.randint(rain_speed_min, rain_speed_max)
    raindrops.append([x, y, length, speed])

# 4. Initialize Thunder State
thunder_active = False
thunder_frames_left = 0

print(f"Generating {num_frames} frames for a {duration_seconds}s video at {fps}fps...")

# 5. Main Loop - Generate Frames
start_time = time.time()
for frame_count in range(num_frames):
    # Start with a fresh copy of the original image for each frame
    frame = image.copy()

    # --- Apply Rain ---
    for i in range(num_raindrops):
        # Move drop down based on its speed
        raindrops[i][1] += raindrops[i][3] # y = y + speed

        # Get current drop details
        x, y, length, speed = raindrops[i]

        # Draw the raindrop line
        # Ensure coordinates are integers for drawing
        start_point = (int(x), int(y))
        end_point = (int(x), int(y + length))
        cv2.line(frame, start_point, end_point, rain_color, rain_thickness)

        # If drop goes off the bottom screen, reset it to the top
        if y > height:
            raindrops[i][0] = random.randint(0, width) # New random x
            raindrops[i][1] = random.randint(-length*3, -length) # Reset well above screen
            raindrops[i][2] = random.randint(rain_length_min, rain_length_max) # New length
            raindrops[i][3] = random.randint(rain_speed_min, rain_speed_max)   # New speed

    # --- Apply Thunder ---
    if thunder_active:
        # Apply brightness flash using NumPy for efficiency
        # Convert to float32 to prevent overflow when adding, clip, then convert back
        frame_float = frame.astype(np.float32)
        frame_float += thunder_brightness_increase
        np.clip(frame_float, 0, 255, out=frame_float) # Clip values to 0-255 range
        frame = frame_float.astype(np.uint8)

        thunder_frames_left -= 1
        if thunder_frames_left <= 0:
            thunder_active = False
            # print(f"Thunder ended at frame {frame_count}") # Optional debug
    else:
        # Check if new thunder should start based on probability
        if random.random() < thunder_probability_per_frame:
            thunder_active = True
            thunder_frames_left = random.randint(thunder_flash_duration_min, thunder_flash_duration_max)
            print(f"--- Thunder! --- (Starting at frame {frame_count}, lasting {thunder_frames_left} frames)")


    # --- Write Frame to Video ---
    video_writer.write(frame)

    # Optional: Display progress
    if (frame_count + 1) % fps == 0: # Print update every second
        elapsed = time.time() - start_time
        print(f"Processed frame {frame_count + 1}/{num_frames} ({int((frame_count + 1) * 100 / num_frames)}%) - Elapsed: {elapsed:.2f}s")

# 6. Release Resources
video_writer.release()
end_time = time.time()
print("-" * 30)
print(f"Video generation complete! Took {end_time - start_time:.2f} seconds.")
print(f"Video saved successfully to: {output_video_path}")
print("-" * 30)

# Optional: Close any OpenCV windows if you were displaying frames during debug
# cv2.destroyAllWindows()
