# -*- coding: utf-8 -*-
"""
Creates a 'Spin & Slam' video effect for an input image.

The image starts above the canvas, drops down while rotating,
and then 'slams' into its final centered position with a brief screen shake effect.
The output video dimensions can be set explicitly.
"""

import cv2
import numpy as np
import os
import math
import time
import random

# --- Configuration ---

# Paths (Adjust for your system, e.g., Termux/Android or PC)
# Use Environment Variable if available (good for Termux), otherwise default
# Try getting DOWNLOAD folder path common on Android
try:
    # Common Termux storage link location
    download_folder = '/storage/emulated/0/Download/'
    if not os.path.isdir(download_folder):
        # Fallback if the above doesn't exist (e.g., different Android setup or PC)
        download_folder = os.path.join(os.path.expanduser('~'), 'Downloads')
        if not os.path.isdir(download_folder):
             # Last resort: current directory
            download_folder = '.'
except Exception:
    download_folder = '.' # Default to current directory on error

image_folder = download_folder # Input and Output Folder
img_name = '1.png' # <<< --- PUT YOUR IMAGE FILENAME HERE
img_path = os.path.join(image_folder, img_name)

# Video Output Configuration
output_filename = 'spin_slam_output_9x16.mp4' # Changed default name
output_path = os.path.join(image_folder, output_filename)

# --- NEW: Explicit Output Video Dimensions ---
# Set your desired output video size here (e.g., 1080x1920 for 9:16 HD Portrait)
output_w = 1080
output_h = 1920
# --- End New Configuration ---

fps = 30.0           # Frames per second

# --- Animation Parameters ---
# Durations
drop_duration_sec = 0.6  # Time image takes to fall and rotate
slam_duration_sec = 0.15 # Duration of the screen shake effect after landing
settle_duration_sec = 0.2 # Time the image stays static at the end

# Motion
# final_x / final_y will now be calculated automatically to center the image.
# You only need to control the *starting* position relative to the center.
start_y_offset = 500     # How many pixels *above* the final centered position the image starts
total_rotation_deg = 360 * 2 # Total degrees the image rotates during the drop (e.g., 720 for two spins)

# Slam Effect
shake_intensity = 8     # Max pixels the image position shifts during the slam/shake

# --- Easing Functions ---
def ease_in_out_quad(t):
    """Quadratic easing in/out: acceleration until halfway, then deceleration."""
    t = max(0.0, min(1.0, t)) # Clamp t to [0, 1]
    if t < 0.5:
        return 2 * t * t
    else:
        return -1 + (4 - 2 * t) * t

def ease_out_quad(t):
    """Quadratic easing out: decelerating to zero velocity."""
    t = max(0.0, min(1.0, t)) # Clamp t to [0, 1]
    return t * (2 - t)

# --- End Configuration ---

# --- Helper: Calculate Rotated Bounding Box ---
def get_rotated_bounding_box(width, height, angle_degrees):
    """Calculates the width and height of the bounding box containing a rotated rectangle."""
    angle_rad = math.radians(angle_degrees)
    abs_cos = abs(math.cos(angle_rad))
    abs_sin = abs(math.sin(angle_rad))
    new_w = int(height * abs_sin + width * abs_cos)
    new_h = int(height * abs_cos + width * abs_sin)
    return new_w, new_h

# --- Helper: Transparency Handling ---
def get_image_components(img_to_process):
    """Separates image into BGR and 3-channel float32 alpha mask."""
    if img_to_process is None:
        raise ValueError("Input image to get_image_components is None")
    h, w = img_to_process.shape[:2]

    if len(img_to_process.shape) == 3 and img_to_process.shape[2] == 4: # BGRA
        bgr = img_to_process[:, :, 0:3].copy()
        alpha_channel = img_to_process[:, :, 3]
        alpha_norm = alpha_channel.astype(np.float32) / 255.0
        alpha_mask = cv2.merge([alpha_norm] * 3)
        return bgr, alpha_mask
    elif len(img_to_process.shape) == 3: # BGR (Assume opaque)
        bgr = img_to_process.copy()
        alpha_mask = np.ones((h, w, 3), dtype=np.float32)
        return bgr, alpha_mask
    elif len(img_to_process.shape) == 2: # Grayscale (Convert to BGR, assume opaque)
         bgr = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)
         alpha_mask = np.ones((h, w, 3), dtype=np.float32)
         return bgr, alpha_mask
    else:
        raise ValueError(f"Error: Unexpected image shape: {img_to_process.shape}")

# --- Check Input ---
print(f"Attempting to load image from: {img_path}")
if not os.path.isfile(img_path):
    print(f"Error: Input image not found at: {img_path}")
    print(f"Please ensure '{img_name}' exists in the '{image_folder}' folder.")
    exit()

img_orig_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if img_orig_raw is None:
    print(f"Error: Failed to load image. Check file integrity, permissions, or OpenCV installation.")
    exit()
print("Image loaded successfully.")

# --- Get Image Components ---
try:
    img_bgr_orig, img_alpha_mask_orig = get_image_components(img_orig_raw)
    img_h_orig, img_w_orig = img_bgr_orig.shape[:2]
    img_center_x = img_w_orig // 2
    img_center_y = img_h_orig // 2
except Exception as e:
    print(f"Error processing image components: {e}")
    exit()

# --- Set Canvas Size to Explicit Output Dimensions ---
canvas_w = output_w
canvas_h = output_h

# --- Calculate Final Position to Center the Image ---
final_x = (canvas_w - img_w_orig) // 2
final_y = (canvas_h - img_h_orig) // 2

# --- Sanity Check: Image vs Canvas Size ---
if img_w_orig > canvas_w or img_h_orig > canvas_h:
    print("Warning: Input image dimensions ({img_w_orig}x{img_h_orig}) are larger than")
    print(f"         the specified output canvas dimensions ({canvas_w}x{canvas_h}).")
    print("         The image will be clipped in its final position.")
    # Optional: Add logic here to resize the image if it's too big,
    # but the request was to *not* resize, so we'll just warn.
    # Example resize (uncomment if needed, but changes original goal):
    # scale_w = canvas_w / img_w_orig
    # scale_h = canvas_h / img_h_orig
    # scale = min(scale_w, scale_h) # Maintain aspect ratio
    # if scale < 1.0:
    #     print(f"         Resizing image down by scale factor {scale:.2f}")
    #     new_w = int(img_w_orig * scale)
    #     new_h = int(img_h_orig * scale)
    #     img_bgr_orig = cv2.resize(img_bgr_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
    #     img_alpha_mask_orig = cv2.resize(img_alpha_mask_orig, (new_w, new_h), interpolation=cv2.INTER_AREA)
    #     img_w_orig, img_h_orig = new_w, new_h
    #     img_center_x = img_w_orig // 2
    #     img_center_y = img_h_orig // 2
    #     # Recalculate final position after potential resize
    #     final_x = (canvas_w - img_w_orig) // 2
    #     final_y = (canvas_h - img_h_orig) // 2


# --- Recalculate Absolute Start Y based on new centered final_y ---
start_y_abs = final_y - start_y_offset # Starting Y position (top-left)

print(f"Original Image Dimensions (HxW): {img_h_orig}x{img_w_orig}")
print(f"Output Video Dimensions (WxH): {canvas_w}x{canvas_h}")
print(f"Final Centered Position (Top-Left): ({final_x}, {final_y})")
print(f"Absolute Start Y (Top-Left): {start_y_abs}")

# --- Calculate Frame Counts ---
drop_frames = max(1, int(fps * drop_duration_sec))
slam_frames = max(1, int(fps * slam_duration_sec))
settle_frames = max(1, int(fps * settle_duration_sec))
total_frames = drop_frames + slam_frames + settle_frames
total_duration_sec = total_frames / fps

print(f"Total Duration: {total_duration_sec:.2f}s => {total_frames} frames")
print(f" - Drop:   {drop_duration_sec:.2f}s ({drop_frames} frames)")
print(f" - Slam:   {slam_duration_sec:.2f}s ({slam_frames} frames)")
print(f" - Settle: {settle_duration_sec:.2f}s ({settle_frames} frames)")

# --- Initialize Video Writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
frame_size_writer = (canvas_w, canvas_h) # Use the fixed (width, height)
try:
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size_writer)
    if not video_writer.isOpened():
        raise IOError(f"Video writer failed to open. Check codec ('mp4v'), path ({output_path}), permissions, or available disk space.")
    print(f"Video writer initialized. Saving to: {output_path}")
except Exception as e:
    print(f"Error initializing VideoWriter: {e}")
    exit()

# --- Animation Loop ---
print(f"Generating {total_frames} frames...")
# Create background once if it's static (e.g., black)
background = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) # Black background
start_time = time.time()

for frame_num in range(total_frames):
    # Start with a fresh copy of the background for each frame
    canvas = background.copy()

    current_x = final_x # Default to final centered X
    current_y = final_y # Default to final centered Y
    current_angle = 0.0
    current_scale = 1.0 # Keep scale at 1.0 - no resizing
    image_to_draw_bgr = img_bgr_orig
    alpha_to_draw = img_alpha_mask_orig
    phase_desc = "Unknown"

    # Determine phase and calculate parameters
    if frame_num < drop_frames:
        # --- Drop Phase ---
        phase_desc = "Drop"
        # Prevent division by zero if drop_frames is 1
        drop_progress = frame_num / (drop_frames -1) if drop_frames > 1 else 1.0
        eased_drop_progress = ease_in_out_quad(drop_progress)

        # Interpolate Y position from start_y_abs to final_y
        current_y = int(round(start_y_abs + (final_y - start_y_abs) * eased_drop_progress))

        # Interpolate Rotation (linear)
        current_angle = total_rotation_deg * drop_progress

        # --- Rotation Handling (Same as before, but critical for placement) ---
        # Get rotation matrix centered on the image
        M = cv2.getRotationMatrix2D((img_center_x, img_center_y), current_angle, current_scale) # Scale is 1.0

        # Calculate the bounding box of the rotated image
        rotated_w, rotated_h = get_rotated_bounding_box(img_w_orig, img_h_orig, current_angle)

        # Adjust the translation part of the matrix M for warpAffine's coordinate system
        M[0, 2] += (rotated_w / 2) - img_center_x
        M[1, 2] += (rotated_h / 2) - img_center_y

        # Warp BGR image
        image_to_draw_bgr = cv2.warpAffine(img_bgr_orig, M, (rotated_w, rotated_h),
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)) # Use transparent black border for alpha

        # Warp Alpha mask (ensure it remains float32)
        alpha_to_draw_float = cv2.warpAffine(img_alpha_mask_orig, M, (rotated_w, rotated_h),
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)) # Fully transparent border

        # Clip alpha values just in case interpolation goes outside [0, 1]
        alpha_to_draw = np.clip(alpha_to_draw_float, 0.0, 1.0)

        # --- Calculate top-left corner for the *rotated* image on the canvas ---
        # We want the *center* of the rotated image to land at the interpolated (final_x + center_x, current_y + center_y) point
        target_center_x = final_x + img_center_x # Target X center remains fixed horizontally
        target_center_y = current_y + img_center_y # Target Y center moves vertically

        current_x = target_center_x - (rotated_w // 2)
        current_y = target_center_y - (rotated_h // 2)


    elif frame_num < drop_frames + slam_frames:
        # --- Slam Phase ---
        phase_desc = "Slam"
        # Prevent division by zero if slam_frames is 1
        slam_progress = (frame_num - drop_frames) / (slam_frames - 1) if slam_frames > 1 else 1.0
        # Intensity decreases rapidly using ease_out (starts high, fades)
        shake_magnitude = shake_intensity * (1.0 - ease_out_quad(slam_progress))

        # Apply random offset around the final centered position
        offset_x = random.uniform(-shake_magnitude, shake_magnitude)
        offset_y = random.uniform(-shake_magnitude, shake_magnitude)

        current_x = final_x + int(round(offset_x))
        current_y = final_y + int(round(offset_y))
        current_angle = 0 # No rotation during shake
        image_to_draw_bgr = img_bgr_orig # Use original unrotated image
        alpha_to_draw = img_alpha_mask_orig
        # Ensure dimensions match the image being drawn
        rotated_w, rotated_h = img_w_orig, img_h_orig # No rotation

    else:
        # --- Settle Phase ---
        phase_desc = "Settle"
        current_x = final_x # Final centered X
        current_y = final_y # Final centered Y
        current_angle = 0
        image_to_draw_bgr = img_bgr_orig
        alpha_to_draw = img_alpha_mask_orig
        # Ensure dimensions match the image being drawn
        rotated_w, rotated_h = img_w_orig, img_h_orig # No rotation


    # --- Compositing (Check if image needs drawing) ---
    # Rotated image dimensions are used here from calculation above
    h_draw, w_draw = rotated_h, rotated_w # Dimensions of the potentially rotated image

    # Check if the image is currently visible on the canvas at all
    if not (current_x + w_draw <= 0 or current_x >= canvas_w or current_y + h_draw <= 0 or current_y >= canvas_h):

        # Calculate canvas ROI coordinates, clamping to canvas boundaries
        y1_c = max(0, current_y)
        y2_c = min(canvas_h, current_y + h_draw)
        x1_c = max(0, current_x)
        x2_c = min(canvas_w, current_x + w_draw)

        # Calculate corresponding image slice coordinates (relative to potentially rotated image)
        y1_i = max(0, -current_y)
        y2_i = y1_i + (y2_c - y1_c) # Height of the ROI
        x1_i = max(0, -current_x)
        x2_i = x1_i + (x2_c - x1_c) # Width of the ROI

        # Ensure dimensions match and are valid before slicing and blending
        if (y2_c > y1_c) and (x2_c > x1_c) and \
           (y2_i > y1_i) and (x2_i > x1_i) and \
           (y2_i <= h_draw) and (x2_i <= w_draw): # Check slice bounds on the drawing image

            try:
                # Extract slices/ROIs
                canvas_roi = canvas[y1_c:y2_c, x1_c:x2_c]
                # Use the potentially rotated image_to_draw_bgr here
                img_slice = image_to_draw_bgr[y1_i:y2_i, x1_i:x2_i]
                mask_slice = alpha_to_draw[y1_i:y2_i, x1_i:x2_i] # Should be float32

                # Ensure data types for blending
                img_slice_float = img_slice.astype(np.float32)
                canvas_roi_float = canvas_roi.astype(np.float32)
                inv_alpha_mask = 1.0 - mask_slice

                # Perform alpha blending
                blended_roi_float = (img_slice_float * mask_slice) + (canvas_roi_float * inv_alpha_mask)

                # Place blended ROI back, converting back to uint8
                canvas[y1_c:y2_c, x1_c:x2_c] = np.clip(blended_roi_float, 0, 255).astype(np.uint8)

            except Exception as e:
                print(f"\nError during blending frame {frame_num} ({phase_desc}): {e}")
                print(f" - Canvas ROI shape: {canvas_roi.shape if 'canvas_roi' in locals() else 'N/A'} -> ({x1_c}:{x2_c}, {y1_c}:{y2_c})")
                print(f" - Image Slice shape: {img_slice.shape if 'img_slice' in locals() else 'N/A'} -> ({x1_i}:{x2_i}, {y1_i}:{y2_i})")
                print(f" - Mask Slice shape: {mask_slice.shape if 'mask_slice' in locals() else 'N/A'}")
                print(f" - Draw Img Dims (HxW): {h_draw}x{w_draw}")
                print(f" - Current Pos (X,Y): ({current_x},{current_y})")
                # Optionally break or continue
                break # Safer to stop if blending fails unexpectedly

    # --- Write Frame ---
    try:
        video_writer.write(canvas)
    except Exception as e:
         print(f"\nError writing frame {frame_num} to video: {e}")
         break # Stop processing if writing fails

    # --- Progress Indicator ---
    if (frame_num + 1) % 10 == 0 or frame_num == total_frames - 1:
        elapsed = time.time() - start_time
        percent_done = ((frame_num + 1) / total_frames) * 100
        # Prevent division by zero on first frame for ETA calc
        est_total_time = (elapsed / (frame_num + 1)) * total_frames if frame_num >= 0 and (frame_num + 1) > 0 else 0
        eta = est_total_time - elapsed if est_total_time > 0 else 0
        print(f"\rProcessed frame {frame_num + 1}/{total_frames} ({percent_done:.1f}%) Phase: {phase_desc} | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s ", end="")


# --- Cleanup ---
video_writer.release()
end_time = time.time()
print("\n" + "-" * 30)
print("Animation finished.")

# --- Verification ---
if os.path.exists(output_path):
    if os.path.getsize(output_path) > 0:
        print(f"Video saved successfully to: {output_path}")
    else:
        print(f"Error: Output video file exists but is empty: {output_path}")
        print("This might indicate an issue during writing, codec problems, or disk space limits.")
else:
    print(f"Error: Output video file was not created at {output_path}.")
    print("Check console for errors during video writing or permission issues.")

print(f"Total time taken: {end_time - start_time:.2f} seconds")
print("-" * 30)
