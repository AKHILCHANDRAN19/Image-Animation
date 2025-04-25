import cv2
import numpy as np
import os
import time

# --- Configuration ---
# Input image path (as specified by user)
image_path = "/storage/emulated/0/Download/input.png"

# Output video path (in the same folder as input)
input_dir = os.path.dirname(image_path)
output_filename = "output_ghostly_blur.mp4"
output_video_path = os.path.join(input_dir, output_filename)

# Video parameters
duration_seconds = 10 # How long the video should be
fps = 30              # Frames per second

# --- Ghostly Blur Parameters ---

# Blur Kernel Size: Controls the intensity of the blur.
# MUST be an ODD positive integer (e.g., 5, 11, 21, 35).
# Higher values result in more blur.
blur_kernel_size = 25 # Adjust this value

# Ghost Transparency (Alpha): Controls how visible the blurred image is.
# 0.0 = fully transparent (invisible black screen)
# 1.0 = fully opaque (just the blurred image, no transparency effect)
# Values between 0.3 and 0.8 often work well for ghosts.
ghost_alpha = 0.65    # Adjust this value

# --- End Configuration ---

# --- Script Start ---
print("--- Ghostly Blur Video Generator ---")

# --- Parameter Validation ---
if not isinstance(blur_kernel_size, int) or blur_kernel_size <= 0:
    print(f"Error: blur_kernel_size ({blur_kernel_size}) must be a positive integer.")
    exit(1)
if blur_kernel_size % 2 == 0:
    blur_kernel_size += 1 # Make it odd
    print(f"Warning: blur_kernel_size was even, adjusted to {blur_kernel_size}.")

if not (0.0 <= ghost_alpha <= 1.0):
     print(f"Warning: ghost_alpha ({ghost_alpha}) should be between 0.0 and 1.0. Clamping.")
     ghost_alpha = np.clip(ghost_alpha, 0.0, 1.0)

# 1. Check if input image exists
print(f"Checking for input image at: {image_path}")
if not os.path.exists(image_path):
    print(f"Error: Input image not found at '{image_path}'")
    print("Please ensure the 'image_path' variable is set correctly.")
    exit(1)
else:
    print("Input image found.")

# 2. Load the input image
print("Loading image...")
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image file: '{image_path}'.")
    print("The file might be corrupted or in an unsupported format.")
    exit(1)
else:
    print("Image loaded successfully.")

# Ensure image is in standard 8-bit format (most common)
if img.dtype != np.uint8:
    print(f"Warning: Image dtype is {img.dtype}, expected uint8. Attempting conversion.")
    try:
        if img.max() <= 1.0 and img.min() >= 0.0: # Likely float 0-1 range
             img = (img * 255).astype(np.uint8)
             print("Converted float image (0-1 range) to uint8.")
        elif img.max() > 255: # Likely higher bit depth, normalize
             img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
             print(f"Normalized image from range [{img.min()}-{img.max()}] to uint8.")
        else: # Assume it's already convertible
             img = img.astype(np.uint8)
             print("Converted image to uint8.")
    except Exception as e:
        print(f"Error during dtype conversion: {e}")
        print(f"Cannot proceed with image type {img.dtype}.")
        exit(1)

# 3. Get image dimensions (height, width)
h, w = img.shape[:2]
print(f"Input image dimensions: Width={w}, Height={h}")

# --- Pre-calculate and Initialize ---
# Create a black background image *once*
black_background = np.zeros_like(img, dtype=np.uint8)
# Calculate total frames
total_frames = int(duration_seconds * fps)
if total_frames <= 0:
    print("Error: Duration and FPS result in zero or negative frames.")
    exit(1)

# 4. Setup Video Writer
frame_size = (w, h) # Width, Height order for VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 file
print(f"Attempting to create video writer for: {output_video_path}")
print(f"Codec: mp4v, FPS: {fps}, Frame Size: {frame_size}")

# Ensure the output directory exists
if input_dir and not os.path.exists(input_dir):
    print(f"Warning: Output directory '{input_dir}' does not exist. Creating it.")
    try:
        os.makedirs(input_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory '{input_dir}': {e}")
        exit(1)

out = cv2.VideoWriter(output_video_path, fourcc, float(fps), frame_size)

if not out.isOpened():
    print(f"Error: Could not open video writer for path: '{output_video_path}'")
    print("Check permissions, available disk space, and codec support.")
    exit(1)
else:
    print("Video writer opened successfully.")


print(f"\nGenerating GHOSTLY BLUR video...")
print(f"Parameters: {duration_seconds}s @ {fps}fps")
print(f"Blur Kernel Size: {blur_kernel_size}, Ghost Alpha: {ghost_alpha:.2f}")
print(f"Total frames to generate: {total_frames}")

start_time = time.time()

try:
    # --- Pre-apply blur *once* if it's constant ---
    # This is more efficient than blurring in every loop iteration
    print("Pre-calculating blurred image...")
    blurred_img = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
    print("Blurring complete.")

    # 5. Generate frames with the ghostly blur effect
    for i in range(total_frames):
        # --- CORE GHOSTLY BLUR LOGIC ---

        # Blend the pre-blurred image with the black background
        # using the desired ghost_alpha for transparency.
        processed_frame = cv2.addWeighted(
            src1=blurred_img,       # The pre-blurred version of the image
            alpha=ghost_alpha,      # Transparency of the blurred image
            src2=black_background,  # Blend with black
            beta=(1.0 - ghost_alpha),# Transparency of the background
            gamma=0.0
        )
        # Ensure frame remains uint8
        # processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)

        # --- End Ghostly Blur Logic ---

        # Write the frame
        out.write(processed_frame)

        # Optional: Print progress
        if (i + 1) % fps == 0 or (i + 1) == total_frames: # Print every second or on the last frame
             elapsed_time = time.time() - start_time
             estimated_total_time = (elapsed_time / (i + 1)) * total_frames if i > 0 else 0
             remaining_time = estimated_total_time - elapsed_time
             print(f"Processed frame {i + 1}/{total_frames} ({(i+1)/total_frames*100:.1f}%) | "
                   f"Elapsed: {elapsed_time:.1f}s | "
                   f"ETA: {remaining_time:.1f}s", end='\r')

except Exception as e:
    print(f"\nError during frame generation or writing: {e}")
    import traceback
    traceback.print_exc()
finally:
    # 6. Release the VideoWriter
    print("\nReleasing video writer...")
    out.release()
    # cv2.destroyAllWindows()

end_time = time.time()
total_time = end_time - start_time

print(f"\nVideo generation finished in {total_time:.2f} seconds.")
# Check if the output file was actually created and has size
if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
     print(f"Video saved successfully to: {output_video_path}")
else:
     print(f"Error: Output video file '{output_video_path}' was not created or is empty.")
     print("Please check for errors above, permissions, and disk space.")

print("--- Script End ---")
