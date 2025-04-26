import cv2
import numpy as np
import os
import math
import time

# --- Configuration ---

# Input Image Path (as specified)
image_folder = '/storage/emulated/0/Download/'
img_name = '1.png'
img_path = os.path.join(image_folder, img_name)

# Video Output Configuration
output_filename = 'whip_pan_effect.mp4' # Output video filename
output_path = os.path.join(image_folder, output_filename) # Save video in the Download folder
fps = 30.0           # Frames per second for the output video
duration_seconds = 1.0 # How long the whip pan effect should last
max_blur_kernel_size = 71 # Max horizontal blur kernel size (ODD number, adjust for more/less blur)

# --- End Configuration ---

print(f"Attempting to load image from: {img_path}")

# Load the input image
img = cv2.imread(img_path)

# Check if image loaded successfully
if img is None:
    print(f"Error: Could not load image from: {img_path}")
    print("Please ensure:")
    print(f"  1. The file '{img_name}' exists exactly in the Download folder.")
    print("  2. Termux (or the script runner) has storage permissions.")
    exit()

print("Image loaded successfully.")

# Get image dimensions (height, width)
img_h, img_w, _ = img.shape
frame_size = (img_w, img_h) # IMPORTANT: VideoWriter expects (width, height)

# --- Initialize Video Writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use MP4V codec for .mp4 file

try:
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not video_writer.isOpened():
         raise IOError(f"Could not open video writer. Check codec and path: {output_path}")
    print(f"Video writer initialized. Saving to: {output_path}")
except Exception as e:
    print(f"Error initializing VideoWriter: {e}")
    print("Ensure OpenCV installation supports video writing and the path is valid.")
    exit()

# --- Animation Generation ---
num_frames = int(fps * duration_seconds)
print(f"Generating {num_frames} frames for a {duration_seconds}s transition at {fps} FPS...")
start_time = time.time()

# Ensure max_blur_kernel_size is odd
if max_blur_kernel_size % 2 == 0:
    max_blur_kernel_size += 1
    print(f"Adjusted max_blur_kernel_size to odd number: {max_blur_kernel_size}")

for frame_num in range(num_frames):
    # Calculate progress (0.0 to 1.0)
    progress = frame_num / (num_frames - 1) if num_frames > 1 else 1.0

    # --- Calculate Blur Intensity ---
    # We want the blur to increase rapidly. Using an ease-in curve (like quadratic)
    # makes the acceleration more pronounced, like a real whip pan.
    ease_in_progress = progress ** 2 # Makes blur accelerate
    #ease_in_progress = progress # Linear increase (less dramatic)

    # Calculate current kernel size based on eased progress
    # Start with kernel size 1 (no blur) and increase up to max_blur_kernel_size
    current_kernel_size = 1 + int(ease_in_progress * (max_blur_kernel_size - 1))

    # Ensure kernel size is always odd and at least 1
    if current_kernel_size < 1:
        current_kernel_size = 1
    elif current_kernel_size % 2 == 0:
        current_kernel_size += 1

    # --- Apply Horizontal Motion Blur ---
    if current_kernel_size > 1:
        # Create the horizontal motion blur kernel
        kernel_motion_blur = np.zeros((current_kernel_size, current_kernel_size))
        # Assign 1s to the middle row
        kernel_motion_blur[int((current_kernel_size - 1) / 2), :] = np.ones(current_kernel_size)
        # Normalize the kernel
        kernel_motion_blur = kernel_motion_blur / current_kernel_size

        # Apply the kernel using filter2D
        try:
            output_frame = cv2.filter2D(img, -1, kernel_motion_blur)
        except Exception as e:
            print(f"Error applying filter at frame {frame_num} with kernel size {current_kernel_size}: {e}")
            output_frame = img.copy() # Use original image on error
    else:
        # No blur needed for kernel size 1
        output_frame = img.copy() # Make a copy to avoid modifying original


    # --- Write the frame to the video file ---
    try:
        video_writer.write(output_frame)
    except Exception as e:
         print(f"Error writing frame {frame_num} to video: {e}")
         break # Stop if writing fails

    # --- Progress Indicator ---
    if (frame_num + 1) % 10 == 0 or frame_num == num_frames - 1:
        elapsed = time.time() - start_time
        print(f"Processed frame {frame_num + 1}/{num_frames} (Blur Kernel: {current_kernel_size}px) [{elapsed:.2f}s]")


# --- Cleanup ---
video_writer.release() # IMPORTANT: Finalize and save the video file
end_time = time.time()
print("-" * 30)
print("Whip Pan effect generation finished.")
print(f"Video saved successfully to: {output_path}")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print("-" * 30)
print("You can now view the video using a player on your Android device.")
