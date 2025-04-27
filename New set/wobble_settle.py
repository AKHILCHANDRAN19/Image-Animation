import cv2
import numpy as np
import os
import math
import time
import random

# --- Configuration ---

# Paths (as specified for Termux/Android)
image_folder = '/storage/emulated/0/Download/'
img_name = '1.png' # MAKE SURE THIS IS YOUR IMAGE NAME
img_path = os.path.join(image_folder, img_name)

# Video Output Configuration
output_filename = 'wobble_settle_output.mp4' # Descriptive filename
output_path = os.path.join(image_folder, output_filename)
fps = 30.0           # Frames per second
total_duration_sec = 2.5 # Target total duration

# --- Animation Parameters ---
# Wobble & Deformation Control
wobble_frequency1 = 2.0 * math.pi * 1.5 # Base frequency (Hz * 2pi)
wobble_frequency2 = 2.0 * math.pi * 2.7 # Second frequency for complexity
wobble_amplitude_initial = 0.05 # Initial max corner offset as fraction of image width/height
deformation_intensity = 1.2 # How much corners move relative to center wobble ( > 1 exaggerates)
noise_factor = 0.3          # How much random noise affects the wobble (0 to 1)
damping_factor = 4.0        # How quickly the wobble dies down (higher = faster stop)

# Final Resting Position (Top-left corner - usually 0,0 for full image)
final_x = 0
final_y = 0

# --- End Configuration ---

# --- Load Image ---
print(f"Loading image from: {img_path}")
img_orig_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # Load potentially with alpha

if img_orig_raw is None:
    print(f"Error: Failed to load {img_path}. Check path and permissions."); exit()
print("Image loaded successfully.")

# --- Determine Canvas and Image Dimensions ---
img_h_orig, img_w_orig = img_orig_raw.shape[:2]
canvas_h, canvas_w = img_h_orig, img_w_orig # Canvas matches image size
print(f"Image/Canvas Dimensions (HxW): {canvas_h}x{canvas_w}")

# --- Handle Transparency (Copied from previous example) ---
def get_image_components(img_to_process):
    h, w = img_to_process.shape[:2]
    if len(img_to_process.shape) == 3 and img_to_process.shape[2] == 4: # BGRA
        print("Detected BGRA image.")
        bgr = img_to_process[:, :, 0:3].copy()
        alpha_norm = img_to_process[:, :, 3].astype(np.float32) / 255.0
        alpha_mask = cv2.merge([alpha_norm] * 3)
        return bgr, alpha_mask
    elif len(img_to_process.shape) == 3: # BGR
        print("Detected BGR image (no alpha). Assuming opaque.")
        bgr = img_to_process.copy()
        alpha_mask = np.ones((h, w, 3), dtype=np.float32)
        return bgr, alpha_mask
    elif len(img_to_process.shape) == 2: # Grayscale
         print("Detected Grayscale image. Converting to BGR, assuming opaque.")
         bgr = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)
         alpha_mask = np.ones((h, w, 3), dtype=np.float32)
         return bgr, alpha_mask
    else:
        raise ValueError(f"Error: Unexpected image shape: {img_to_process.shape}")

# Get original components ONCE
try:
    img_bgr_orig, img_alpha_mask_orig = get_image_components(img_orig_raw)
except Exception as e:
    print(f"Error processing image components: {e}")
    exit()

# --- Calculate Frame Counts ---
total_frames = max(1, int(fps * total_duration_sec))
print(f"Total Duration: {total_duration_sec:.2f}s => {total_frames} total frames.")

# --- Initialize Video Writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_size_writer = (canvas_w, canvas_h) # (width, height)
try:
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size_writer)
    if not video_writer.isOpened(): raise IOError("Video writer failed to open.")
    print(f"Video writer initialized. Saving to: {output_path}")
except Exception as e:
    print(f"Error initializing VideoWriter: {e}"); exit()

# --- Animation Loop ---
print(f"Generating {total_frames} frames...")
background = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) # Black background for canvas
start_time = time.time()

# Define original corner points (Source for perspective transform)
src_pts = np.float32([
    [0, 0],              # Top-left
    [img_w_orig, 0],     # Top-right
    [img_w_orig, img_h_orig], # Bottom-right
    [0, img_h_orig]      # Bottom-left
])

for frame_num in range(total_frames):
    canvas = background.copy()
    phase = "Wobbling"

    # Calculate progress and dampening factor for this frame
    # Use time instead of frame count for smoother frequency calculation
    current_time_sec = frame_num / fps
    progress = frame_num / (total_frames - 1) if total_frames > 1 else 1.0
    dampening = math.exp(-damping_factor * progress)

    # --- Calculate Wobble Offsets ---
    # Combine multiple sine waves for complex motion + add dampened noise
    max_offset_pixels_w = wobble_amplitude_initial * img_w_orig * dampening
    max_offset_pixels_h = wobble_amplitude_initial * img_h_orig * dampening

    # X offset calculation
    noise_x = (random.random() * 2 - 1) * noise_factor if noise_factor > 0 else 0
    offset_x = max_offset_pixels_w * (
        0.6 * math.sin(wobble_frequency1 * current_time_sec) +
        0.4 * math.sin(wobble_frequency2 * current_time_sec + 0.5) + # Add phase shift
        noise_x
    )

    # Y offset calculation (slightly different frequencies/phases/noise)
    noise_y = (random.random() * 2 - 1) * noise_factor if noise_factor > 0 else 0
    offset_y = max_offset_pixels_h * (
        0.5 * math.sin(wobble_frequency1 * 1.1 * current_time_sec + 0.2) + # Slightly different freq/phase
        0.5 * math.sin(wobble_frequency2 * 0.9 * current_time_sec + 0.8) +
        noise_y
    )

    # --- Calculate Destination Corner Points ---
    # Apply offsets differently to each corner for deformation, scaled by intensity
    intensity = deformation_intensity
    dst_pts = np.float32([
        [final_x + offset_x * intensity,             final_y + offset_y],                        # Top-left
        [final_x + img_w_orig + offset_x,            final_y + offset_y * intensity * 0.8],      # Top-right
        [final_x + img_w_orig + offset_x * intensity * 0.9, final_y + img_h_orig + offset_y * intensity], # Bottom-right
        [final_x + offset_x * 0.8,                   final_y + img_h_orig + offset_y]            # Bottom-left
    ])

    # --- Perform Perspective Warp ---
    try:
        # Get the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Warp the BGR image
        warped_bgr = cv2.warpPerspective(img_bgr_orig, matrix, (canvas_w, canvas_h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0)) # Black background fill

        # Warp the alpha mask (needs single channel, then merge back)
        # Crucially use borderValue=0 for alpha to make padding transparent
        alpha_ch = img_alpha_mask_orig[:,:,0] # Get one channel
        warped_alpha_single = cv2.warpPerspective(alpha_ch, matrix, (canvas_w, canvas_h),
                                                flags=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=0) # Transparent padding
        warped_alpha = cv2.merge([warped_alpha_single] * 3) # Make 3-channel again

    except cv2.error as warp_err:
        print(f"\nError during warpPerspective at frame {frame_num}: {warp_err}")
        # Fallback: use original image if warp fails
        warped_bgr = img_bgr_orig.copy() # Make sure to copy
        warped_alpha = img_alpha_mask_orig.copy()


    # --- Drawing the Warped Image ---
    # Since warpPerspective generates the image within the target canvas size,
    # we essentially overlay it directly onto the background using the warped alpha.
    # The wobble is *contained within* the warped_bgr/warped_alpha result.
    try:
        # Ensure data types are correct for blending
        canvas_float = canvas.astype(np.float32)
        warped_bgr_float = warped_bgr.astype(np.float32)
        # Warped alpha is already float32

        inv_alpha_mask = 1.0 - warped_alpha

        # Blend: (foreground * alpha) + (background * (1 - alpha))
        blended_float = (warped_bgr_float * warped_alpha) + (canvas_float * inv_alpha_mask)

        # Convert back to uint8 and assign to canvas
        canvas = np.clip(blended_float, 0, 255).astype(np.uint8)

    except Exception as e:
        print(f"\nError during blending at frame {frame_num}: {e}")
        # If blending fails, maybe just write the warped image directly (if opaque)
        # or skip drawing for this frame. Let's just use the canvas as is.
        pass


    # --- Write the frame ---
    try:
        video_writer.write(canvas)
    except Exception as e:
         print(f"\nError writing frame {frame_num} to video: {e}")
         break # Stop if writing fails

    # --- Progress Indicator ---
    if (frame_num + 1) % 5 == 0 or frame_num == total_frames - 1: # Update progress indicator
        elapsed = time.time() - start_time
        percent_done = ((frame_num + 1) / total_frames) * 100
        # Display current offset magnitude for debugging wobble
        current_magnitude = np.sqrt(offset_x**2 + offset_y**2)
        print(f"\rProcessed frame {frame_num + 1}/{total_frames} ({percent_done:.1f}%) ({phase}) Wobble Mag: {current_magnitude:.1f}px [{elapsed:.2f}s]", end="")


# --- Cleanup ---
video_writer.release()
# No cv2.destroyAllWindows() needed in Termux
end_time = time.time()
print("\n" + "-" * 30)
print("Animation finished.")
print(f"Video saved successfully to: {output_path}")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print("-" * 30)
