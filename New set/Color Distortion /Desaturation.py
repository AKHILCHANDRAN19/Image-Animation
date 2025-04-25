import cv2
import numpy as np
import os
import time

# --- Configuration ---
# Input image path (as specified by user)
image_path = "/storage/emulated/0/Download/input.png"

# Output video path (in the same folder as input)
input_dir = os.path.dirname(image_path)
output_filename = "output_color_distortion.mp4"
output_video_path = os.path.join(input_dir, output_filename)

# Video parameters
duration_seconds = 10 # How long the video should be
fps = 30              # Frames per second

# --- Color Distortion / Desaturation Parameters ---

# Choose ONE effect type:
# "grayscale": Completely black and white.
# "desaturate": Reduce color saturation (partially towards grayscale).
# "tint": Apply a color overlay.
EFFECT_TYPE = "desaturate" # Options: "grayscale", "desaturate", "tint"

# --- Parameters for "desaturate" ---
# 0.0 = fully grayscale, 1.0 = original saturation
saturation_factor = 0.2

# --- Parameters for "tint" ---
# BGR color format (Blue, Green, Red)
tint_color_bgr = (180, 120, 120) # Example: A slightly desaturated cyan/blue tint
# How strong the tint is (0.0 = no tint, 1.0 = fully tinted color)
tint_intensity = 0.4

# --- Common Parameter: Ghost Transparency (Alpha) ---
# Applied *after* color distortion. Controls overall visibility.
# 0.0 = fully transparent (invisible black screen)
# 1.0 = fully opaque (just the color-distorted image)
ghost_alpha = 0.75 # Adjust as needed

# --- End Configuration ---

# --- Script Start ---
print(f"--- Color Distortion / Desaturation Video Generator ---")
print(f"Selected Effect Type: {EFFECT_TYPE}")

# --- Parameter Validation ---
if not (0.0 <= ghost_alpha <= 1.0):
     print(f"Warning: ghost_alpha ({ghost_alpha}) should be between 0.0 and 1.0. Clamping.")
     ghost_alpha = np.clip(ghost_alpha, 0.0, 1.0)
if EFFECT_TYPE == "desaturate" and not (0.0 <= saturation_factor <= 1.0):
     print(f"Warning: saturation_factor ({saturation_factor}) should be between 0.0 and 1.0. Clamping.")
     saturation_factor = np.clip(saturation_factor, 0.0, 1.0)
if EFFECT_TYPE == "tint" and not (0.0 <= tint_intensity <= 1.0):
     print(f"Warning: tint_intensity ({tint_intensity}) should be between 0.0 and 1.0. Clamping.")
     tint_intensity = np.clip(tint_intensity, 0.0, 1.0)


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
img_original = cv2.imread(image_path) # Keep original safe
if img_original is None:
    print(f"Error: Could not read image file: '{image_path}'.")
    print("The file might be corrupted or in an unsupported format.")
    exit(1)
else:
    print("Image loaded successfully.")

# Ensure image is in standard 8-bit format (most common)
if img_original.dtype != np.uint8:
    print(f"Warning: Image dtype is {img_original.dtype}, expected uint8. Attempting conversion.")
    try:
        if img_original.max() <= 1.0 and img_original.min() >= 0.0: # Likely float 0-1 range
             img_original = (img_original * 255).astype(np.uint8)
             print("Converted float image (0-1 range) to uint8.")
        elif img_original.max() > 255: # Likely higher bit depth, normalize
             img_original = ((img_original - img_original.min()) / (img_original.max() - img_original.min()) * 255).astype(np.uint8)
             print(f"Normalized image from range [{img_original.min()}-{img_original.max()}] to uint8.")
        else: # Assume it's already convertible
             img_original = img_original.astype(np.uint8)
             print("Converted image to uint8.")
    except Exception as e:
        print(f"Error during dtype conversion: {e}")
        print(f"Cannot proceed with image type {img_original.dtype}.")
        exit(1)

# Make a working copy
img = img_original.copy()

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

# --- Pre-process color distortion if constant ---
# More efficient to do this once outside the loop
processed_color_img = None
print("Pre-processing color distortion...")

if EFFECT_TYPE == "grayscale":
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert back to 3 channels for VideoWriter compatibility
    processed_color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    print(f"Effect: Grayscale applied.")

elif EFFECT_TYPE == "desaturate":
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Multiply saturation channel (channel 1) by the factor
    # Ensure calculation uses float and result is clipped back to uint8 range
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1].astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
    processed_color_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    print(f"Effect: Desaturation applied (factor: {saturation_factor:.2f}).")

elif EFFECT_TYPE == "tint":
    # Create a solid color image of the tint color
    tint_color_img = np.zeros_like(img)
    tint_color_img[:] = tint_color_bgr
    # Blend the original image with the tint color image
    processed_color_img = cv2.addWeighted(
        src1=img,                 # Original image
        alpha=(1.0 - tint_intensity), # Weight for original
        src2=tint_color_img,      # Tint color image
        beta=tint_intensity,      # Weight for tint color
        gamma=0.0
    )
    print(f"Effect: Tint applied (Color: {tint_color_bgr}, Intensity: {tint_intensity:.2f}).")

else:
    print(f"Error: Unknown EFFECT_TYPE '{EFFECT_TYPE}'. Using original image.")
    processed_color_img = img.copy() # Fallback to original


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


print(f"\nGenerating COLOR DISTORTION video ({EFFECT_TYPE})...")
print(f"Parameters: {duration_seconds}s @ {fps}fps")
print(f"Final Ghost Alpha: {ghost_alpha:.2f}")
print(f"Total frames to generate: {total_frames}")

start_time = time.time()

try:
    # 5. Generate frames applying the final transparency
    for i in range(total_frames):
        # --- CORE TRANSPARENCY LOGIC ---
        # Blend the pre-processed color image with the black background
        processed_frame = cv2.addWeighted(
            src1=processed_color_img, # The color-distorted image
            alpha=ghost_alpha,        # Final transparency
            src2=black_background,    # Blend with black
            beta=(1.0 - ghost_alpha), # Transparency for background
            gamma=0.0
        )
        # Ensure frame remains uint8
        # processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)

        # --- End Transparency Logic ---

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
