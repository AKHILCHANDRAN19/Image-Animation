import cv2
import numpy as np
import os
import time

# --- Configuration ---
# Input image path (as specified by user)
image_path = "/storage/emulated/0/Download/input.png"

# Output video path (in the same folder as input)
input_dir = os.path.dirname(image_path)
output_filename = "output_ghost_trails.mp4"
output_video_path = os.path.join(input_dir, output_filename)

# Video parameters
duration_seconds = 10 # How long the video should be
fps = 30              # Frames per second

# --- Ghostly Trails Parameters ---
# How opaque the "leading edge" of the ghost is in each new frame.
# 1.0 = fully opaque original image added each frame
# 0.5 = semi-transparent original image added each frame
current_ghost_alpha = 0.4 # Good starting point, adjust as needed

# How much of the *previous* frame persists into the *current* frame.
# Closer to 1.0 means longer, stronger trails.
# Closer to 0.0 means shorter, weaker trails (less persistence).
trail_persistence = 0.88 # High persistence for noticeable trails

# --- End Configuration ---

# --- Script Start ---
print("--- Ghostly Trails Video Generator ---")

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

# Initialize the 'previous_frame' buffer. Start with black.
previous_frame = np.zeros_like(img, dtype=np.uint8)

# 4. Setup Video Writer
frame_size = (w, h) # Width, Height order for VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 file
print(f"Attempting to create video writer for: {output_video_path}")
print(f"Codec: mp4v, FPS: {fps}, Frame Size: {frame_size}")

# Ensure the output directory exists (though it should if input exists)
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


print(f"\nGenerating GHOSTLY TRAILS video...")
print(f"Parameters: {duration_seconds}s @ {fps}fps")
print(f"Current Ghost Alpha: {current_ghost_alpha:.2f}, Trail Persistence: {trail_persistence:.2f}")
print(f"Total frames to generate: {total_frames}")

start_time = time.time()

try:
    # 5. Generate frames with ghostly trails effect
    for i in range(total_frames):
        # --- CORE GHOSTLY TRAILS LOGIC ---

        # 1. Define the "current ghost" layer for this frame.
        #    This is the original image blended with black to make it semi-transparent.
        current_ghost_layer = cv2.addWeighted(
            src1=img,               # The original input image
            alpha=current_ghost_alpha, # How opaque this new layer is
            src2=black_background,  # Blend with black
            beta=(1.0 - current_ghost_alpha), # Weight for black
            gamma=0.0
        )

        # 2. Blend the new 'current_ghost_layer' with the 'previous_frame'.
        #    The 'previous_frame' holds the accumulated trails from past frames.
        processed_frame = cv2.addWeighted(
            src1=current_ghost_layer, # The semi-transparent ghost for *this* frame
            alpha=(1.0 - trail_persistence), # How much of the *new* ghost layer to add
            src2=previous_frame,      # The accumulated trails from the *last* frame
            beta=trail_persistence,   # How much of the old trail to keep
            gamma=0.0
        )
        # Ensure frame remains uint8 (usually handled by addWeighted, but safe)
        # processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)

        # 3. Update `previous_frame` for the *next* iteration.
        #    IMPORTANT: Use .copy() so you don't modify the buffer used by `processed_frame`.
        previous_frame = processed_frame.copy()

        # --- End Ghostly Trails Logic ---

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
    # cv2.destroyAllWindows() # Not needed as we don't display windows

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
