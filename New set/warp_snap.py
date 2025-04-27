# -*- coding: utf-8 -*-
"""
Creates a 'Warp In & Snap' video effect for an input image.

The image flies in from above, distorting with a perspective warp,
and then 'snaps' into its final centered, undistorted shape.
The output video dimensions can be set explicitly.
"""

import cv2
import numpy as np
import os
import math
import time
import random # Keep for potential future use, but not used in warp

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
img_name = '1.png' # <<< --- YOUR IMAGE FILENAME (as requested)
img_path = '/storage/emulated/0/Download/1.png' # <<< --- USE SPECIFIC INPUT PATH (as requested)

# --- Video Output Configuration ---
output_filename = 'warp_snap_output_9x16.mp4' # Changed effect name
output_path = os.path.join(image_folder, output_filename)

# --- NEW: Explicit Output Video Dimensions ---
# Set your desired output video size here (e.g., 1080x1920 for 9:16 HD Portrait)
output_w = 1080
output_h = 1920
# --- End New Configuration ---

fps = 30.0           # Frames per second

# --- Animation Parameters ---
# Durations
warp_duration_sec = 0.8  # Time image takes to fly in and warp
snap_duration_sec = 0.05 # Duration of the "snapped" state before settling (very short)
settle_duration_sec = 0.4 # Time the image stays static at the end

# Motion & Warp
# final_x / final_y will be calculated automatically to center the image.
# You only need to control the *starting* position relative to the center.
start_y_offset = 600     # How many pixels *above* the final centered position the image starts
initial_warp_factor = 0.4 # Controls how much the corners are distorted initially (0=none, 0.5=significant)
                         # This factor pushes the top corners inwards/upwards.

# --- Easing Functions ---
def ease_in_out_quad(t):
    """Quadratic easing in/out: acceleration until halfway, then deceleration."""
    t = max(0.0, min(1.0, t)) # Clamp t to [0, 1]
    if t < 0.5:
        return 2 * t * t
    else:
        return -1 + (4 - 2 * t) * t

# Removed ease_out_quad as it's not used for shake anymore
# Removed get_rotated_bounding_box function

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
    print(f"Please ensure the file exists and the path is correct.")
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
    # No center needed for perspective warp calculation itself
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
    print(f"Warning: Input image dimensions ({img_w_orig}x{img_h_orig}) are larger than")
    print(f"         the specified output canvas dimensions ({canvas_w}x{canvas_h}).")
    print("         The image will be clipped in its final position.")
    # No resizing logic here as per original code's intent

# --- Recalculate Absolute Start Y based on new centered final_y ---
start_y_abs = final_y - start_y_offset # Starting Y position (top-left)

print(f"Original Image Dimensions (HxW): {img_h_orig}x{img_w_orig}")
print(f"Output Video Dimensions (WxH): {canvas_w}x{canvas_h}")
print(f"Final Centered Position (Top-Left): ({final_x}, {final_y})")
print(f"Absolute Start Y (Top-Left): {start_y_abs}")
print(f"Initial Warp Factor: {initial_warp_factor}")

# --- Define Source and Target Corner Points for Perspective Warp ---
# Source points are the corners of the original image
src_pts = np.float32([
    [0, 0],             # Top-left
    [img_w_orig, 0],    # Top-right
    [img_w_orig, img_h_orig], # Bottom-right
    [0, img_h_orig]     # Bottom-left
])

# Destination points for the *start* of the animation (maximum warp)
# Example: Pinch top corners inwards and slightly up
wf = initial_warp_factor
dst_pts_start = np.float32([
    [img_w_orig * wf, img_h_orig * wf],             # Top-left (inward, upward)
    [img_w_orig * (1 - wf), img_h_orig * wf],       # Top-right (inward, upward)
    [img_w_orig, img_h_orig],                       # Bottom-right (fixed)
    [0, img_h_orig]                                 # Bottom-left (fixed)
])
# Alternative warp (e.g., trapezoid):
# dst_pts_start = np.float32([
#     [img_w_orig * wf, 0],             # Top-left (inward)
#     [img_w_orig * (1 - wf), 0],       # Top-right (inward)
#     [img_w_orig * (1 + wf/2), img_h_orig], # Bottom-right (outward)
#     [img_w_orig * (-wf/2), img_h_orig]     # Bottom-left (outward) - careful with negative coords
# ])


# Destination points for the *end* of the warp phase (no warp)
dst_pts_end = src_pts.copy() # The target is the original shape

# --- Calculate Frame Counts ---
warp_frames = max(1, int(fps * warp_duration_sec))
snap_frames = max(1, int(fps * snap_duration_sec))
settle_frames = max(1, int(fps * settle_duration_sec))
total_frames = warp_frames + snap_frames + settle_frames
total_duration_sec = total_frames / fps

print(f"Total Duration: {total_duration_sec:.2f}s => {total_frames} frames")
print(f" - Warp:   {warp_duration_sec:.2f}s ({warp_frames} frames)")
print(f" - Snap:   {snap_duration_sec:.2f}s ({snap_frames} frames)")
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

    current_x = final_x # Default position X
    current_y = final_y # Default position Y
    image_to_draw_bgr = img_bgr_orig # Default image
    alpha_to_draw = img_alpha_mask_orig # Default mask
    phase_desc = "Unknown"

    # Determine phase and calculate parameters
    if frame_num < warp_frames:
        # --- Warp Phase ---
        phase_desc = "Warp"
        # Prevent division by zero if warp_frames is 1
        warp_progress = frame_num / (warp_frames - 1) if warp_frames > 1 else 1.0
        eased_warp_progress = ease_in_out_quad(warp_progress)

        # Interpolate Y position from start_y_abs to final_y
        current_y = int(round(start_y_abs + (final_y - start_y_abs) * eased_warp_progress))
        # Keep X centered for this effect
        current_x = final_x

        # Interpolate destination points for perspective warp
        current_dst_pts = np.zeros_like(src_pts, dtype=np.float32)
        for i in range(4): # Iterate through 4 corner points
            for j in range(2): # Iterate through x and y coordinates
                current_dst_pts[i, j] = dst_pts_start[i, j] + (dst_pts_end[i, j] - dst_pts_start[i, j]) * eased_warp_progress

        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, current_dst_pts)

        # Warp the BGR image
        # Output size is set to original image dimensions for simplicity in placement
        warped_bgr = cv2.warpPerspective(img_bgr_orig, M, (img_w_orig, img_h_orig),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)) # Transparent black border

        # Warp the Alpha mask (ensure it remains float32)
        warped_alpha_float = cv2.warpPerspective(img_alpha_mask_orig, M, (img_w_orig, img_h_orig),
                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0)) # Fully transparent border

        # Use the warped images for drawing
        image_to_draw_bgr = warped_bgr
        # Clip alpha values just in case interpolation goes outside [0, 1]
        alpha_to_draw = np.clip(warped_alpha_float, 0.0, 1.0)

    elif frame_num < warp_frames + snap_frames:
        # --- Snap Phase ---
        # This phase simply shows the final, un-distorted image in its final position
        phase_desc = "Snap"
        current_x = final_x
        current_y = final_y
        image_to_draw_bgr = img_bgr_orig # Use original un-warped image
        alpha_to_draw = img_alpha_mask_orig

    else:
        # --- Settle Phase ---
        phase_desc = "Settle"
        current_x = final_x # Final centered X
        current_y = final_y # Final centered Y
        image_to_draw_bgr = img_bgr_orig # Use original un-warped image
        alpha_to_draw = img_alpha_mask_orig

    # --- Compositing (Check if image needs drawing) ---
    # Dimensions of the image being drawn (original size in all phases now)
    h_draw, w_draw = img_h_orig, img_w_orig

    # Check if the image is currently visible on the canvas at all
    if not (current_x + w_draw <= 0 or current_x >= canvas_w or current_y + h_draw <= 0 or current_y >= canvas_h):

        # Calculate canvas ROI coordinates, clamping to canvas boundaries
        y1_c = max(0, current_y)
        y2_c = min(canvas_h, current_y + h_draw)
        x1_c = max(0, current_x)
        x2_c = min(canvas_w, current_x + w_draw)

        # Calculate corresponding image slice coordinates
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
                # Use the potentially warped or original image_to_draw_bgr here
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
                # Provide more detailed info for debugging blending errors
                print(f"  Canvas ROI: y={y1_c}:{y2_c}, x={x1_c}:{x2_c} (shape: {canvas_roi.shape if 'canvas_roi' in locals() else 'N/A'})")
                print(f"  Image Slice: y={y1_i}:{y2_i}, x={x1_i}:{x2_i} (shape: {img_slice.shape if 'img_slice' in locals() else 'N/A'})")
                print(f"  Mask Slice: y={y1_i}:{y2_i}, x={x1_i}:{x2_i} (shape: {mask_slice.shape if 'mask_slice' in locals() else 'N/A'})")
                print(f"  Draw Img Dims (HxW): {h_draw}x{w_draw}")
                print(f"  Current Pos (X,Y): ({current_x},{current_y})")
                print(f"  image_to_draw_bgr shape: {image_to_draw_bgr.shape}")
                print(f"  alpha_to_draw shape: {alpha_to_draw.shape}")
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
