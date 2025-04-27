import cv2
import numpy as np
import os
import math
import time

# --- Configuration ---

# Paths (Adjust for your system, e.g., Termux/Android or PC)
image_folder = '/storage/emulated/0/Download/' # Input and Output Folder
img_name = '1.png' # <<< --- PUT YOUR IMAGE FILENAME HERE
img_path = os.path.join(image_folder, img_name)

# Video Output Configuration
output_filename = 'scale_pop_output.mp4' # Name for the output video file
output_path = os.path.join(image_folder, output_filename)

fps = 30.0           # Frames per second
total_duration_sec = 0.7 # Duration of the pop animation (Keep it short for a 'pop')

# --- Animation Parameters ---
# Final Resting Position (Top-left corner of the image relative to canvas top-left)
final_x = 50
final_y = 50

# Scaling Parameters
start_scale = 0.01 # Initial scale factor (e.g., 0.01 = 1% size, cannot be 0 for resize)

# --- Easing Function ---
def ease_out_cubic(t):
    """Cubic easing out: decelerating to zero velocity."""
    t -= 1
    return t * t * t + 1
    # Alternative: ease_out_quad(t): return t * (2 - t)

# --- End Configuration ---

# --- Check if input folder exists ---
if not os.path.isdir(image_folder):
    print(f"Error: The specified folder does not exist: {image_folder}")
    print("Please ensure the path is correct and accessible.")
    exit()

# --- Load Image ---
print(f"Loading image from: {img_path}")
if not os.path.exists(img_path):
    print(f"Error: Input image not found at: {img_path}")
    print(f"Please ensure '{img_name}' exists in the '{image_folder}' folder.")
    exit()

img_orig_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if img_orig_raw is None:
    print(f"Error: Failed to load {img_path}. Check file integrity and permissions."); exit()
print("Image loaded successfully.")

# --- Determine Canvas and Image Dimensions ---
img_h_orig, img_w_orig = img_orig_raw.shape[:2]

# Canvas needs to be large enough to hold the image at its final position
canvas_margin = 10
canvas_h = final_y + img_h_orig + canvas_margin
canvas_w = final_x + img_w_orig + canvas_margin
print(f"Image Dimensions (HxW): {img_h_orig}x{img_w_orig}")
print(f"Canvas Dimensions (WxH): {canvas_w}x{canvas_h}") # Note: WxH for print

# --- Handle Transparency ---
def get_image_components(img_to_process):
    """Separates image into BGR and 3-channel float32 alpha mask."""
    h, w = img_to_process.shape[:2]
    if img_to_process is None: raise ValueError("Input image to get_image_components is None")
    if len(img_to_process.shape) == 3 and img_to_process.shape[2] == 4: # BGRA
        bgr = img_to_process[:, :, 0:3].copy()
        alpha_channel = img_to_process[:, :, 3]
        alpha_norm = alpha_channel.astype(np.float32) / 255.0
        alpha_mask = cv2.merge([alpha_norm] * 3)
        return bgr, alpha_mask
    elif len(img_to_process.shape) == 3: # BGR
        bgr = img_to_process.copy()
        alpha_mask = np.ones((h, w, 3), dtype=np.float32)
        return bgr, alpha_mask
    elif len(img_to_process.shape) == 2: # Grayscale
         bgr = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)
         alpha_mask = np.ones((h, w, 3), dtype=np.float32)
         return bgr, alpha_mask
    else:
        raise ValueError(f"Error: Unexpected image shape: {img_to_process.shape}")

try:
    img_bgr_orig, img_alpha_mask_orig = get_image_components(img_orig_raw)
except Exception as e:
    print(f"Error processing image components: {e}"); exit()

# --- Calculate Frame Counts & Positioning Constants ---
total_frames = max(1, int(fps * total_duration_sec))
if total_duration_sec <= 0: total_duration_sec = 1/fps # Avoid division by zero

# Calculate the center point where the image should finally rest
final_center_x = final_x + img_w_orig / 2.0
final_center_y = final_y + img_h_orig / 2.0

print(f"Total Duration: {total_duration_sec:.2f}s => {total_frames} total frames.")
print(f"Scaling from {start_scale*100:.1f}% to 100% size.")
print(f"Final Top-Left: ({final_x}, {final_y})")
print(f"Final Center: ({final_center_x:.1f}, {final_center_y:.1f})")


# --- Initialize Video Writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4
frame_size_writer = (canvas_w, canvas_h) # (width, height)
try:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size_writer)
    if not video_writer.isOpened():
        raise IOError(f"Video writer failed to open. Check codec ('mp4v'), path ({output_path}), and permissions.")
    print(f"Video writer initialized. Saving to: {output_path}")
except Exception as e:
    print(f"Error initializing VideoWriter: {e}"); exit()

# --- Animation Loop ---
print(f"Generating {total_frames} frames...")
background = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) # Black background
start_time = time.time()

for frame_num in range(total_frames):
    canvas = background.copy()
    progress = frame_num / (total_frames - 1) if total_frames > 1 else 1.0

    # --- Calculate Scale and Position ---
    current_scale = 1.0 # Default to final scale
    scaled_bgr = img_bgr_orig
    scaled_alpha = img_alpha_mask_orig
    current_w, current_h = img_w_orig, img_h_orig
    pos_x = final_x
    pos_y = final_y
    phase_desc = "Final Frame" # Assume final frame initially

    if frame_num < total_frames - 1 or total_frames == 1: # Check if not the forced final frame
        phase_desc = "Scaling"
        eased_progress = ease_out_cubic(progress) # Apply easing

        # Scale goes from start_scale to 1.0 based on eased progress
        current_scale = start_scale + (1.0 - start_scale) * eased_progress
        current_scale = max(start_scale, min(1.0, current_scale)) # Clamp scale

        # Calculate new dimensions (ensure at least 1x1 pixel)
        current_w = max(1, int(round(img_w_orig * current_scale)))
        current_h = max(1, int(round(img_h_orig * current_scale)))

        # Resize the original BGR image and Alpha Mask
        interp_flag = cv2.INTER_AREA if current_scale < 0.5 else cv2.INTER_LINEAR # Use AREA for significant shrink
        try:
            if current_w != img_w_orig or current_h != img_h_orig: # Only resize if needed
                scaled_bgr = cv2.resize(img_bgr_orig, (current_w, current_h), interpolation=interp_flag)

                # Resize alpha mask (single channel first, then merge back)
                alpha_ch = img_alpha_mask_orig[:,:,0]
                scaled_alpha_single = cv2.resize(alpha_ch, (current_w, current_h), interpolation=interp_flag)
                scaled_alpha = cv2.merge([scaled_alpha_single] * 3)
            else: # If scale results in original size, use originals
                scaled_bgr = img_bgr_orig
                scaled_alpha = img_alpha_mask_orig

        except cv2.error as resize_err:
            print(f"\nError during resize frame {frame_num}: {resize_err}")
            print(f" Target size: ({current_w}, {current_h}), Scale: {current_scale:.3f}")
            # Fallback: use originals (might look jumpy)
            scaled_bgr = img_bgr_orig
            scaled_alpha = img_alpha_mask_orig
            current_w, current_h = img_w_orig, img_h_orig

        # Calculate top-left position (pos_x, pos_y) so the *center*
        # of the scaled image aligns with the final_center position
        scaled_center_x_rel = current_w / 2.0
        scaled_center_y_rel = current_h / 2.0
        pos_x = int(round(final_center_x - scaled_center_x_rel))
        pos_y = int(round(final_center_y - scaled_center_y_rel))

    # --- Drawing the Scaled Image (Robust slicing logic) ---
    # Calculate the slice of the canvas to draw onto
    y_start_canvas = max(0, pos_y)
    y_end_canvas = min(canvas_h, pos_y + current_h)
    x_start_canvas = max(0, pos_x)
    x_end_canvas = min(canvas_w, pos_x + current_w)

    # Calculate the corresponding slice of the scaled image to use
    y_start_img = max(0, -pos_y)
    y_end_img = y_start_img + (y_end_canvas - y_start_canvas)
    x_start_img = max(0, -pos_x)
    x_end_img = x_start_img + (x_end_canvas - x_start_canvas)

    # Proceed only if there's a valid, non-empty area to draw and image data is valid
    if y_start_canvas < y_end_canvas and x_start_canvas < x_end_canvas and \
       y_start_img < y_end_img and x_start_img < x_end_img and \
       scaled_bgr is not None and scaled_alpha is not None:

        h_roi = y_end_canvas - y_start_canvas
        w_roi = x_end_canvas - x_start_canvas
        h_img_slice = y_end_img - y_start_img
        w_img_slice = x_end_img - x_start_img

        # Sanity check dimensions before slicing
        if h_roi == h_img_slice and w_roi == w_img_slice and h_roi > 0 and w_roi > 0:
            # Ensure image slice indices are within the bounds of the scaled image
            if y_start_img >= 0 and x_start_img >= 0 and \
               y_end_img <= scaled_bgr.shape[0] and x_end_img <= scaled_bgr.shape[1] and \
               y_end_img <= scaled_alpha.shape[0] and x_end_img <= scaled_alpha.shape[1]:
                try:
                    # Extract the region of interest (ROI) from the canvas
                    roi = canvas[y_start_canvas : y_end_canvas, x_start_canvas : x_end_canvas]

                    # Extract the corresponding slice from the scaled image and alpha mask
                    img_slice = scaled_bgr[y_start_img : y_end_img, x_start_img : x_end_img]
                    mask_slice = scaled_alpha[y_start_img : y_end_img, x_start_img : x_end_img]

                    # Blend the image slice onto the ROI using the alpha mask
                    inv_alpha_mask = 1.0 - mask_slice
                    img_slice_float = img_slice.astype(np.float32)
                    roi_float = roi.astype(np.float32)

                    blended_roi_float = (img_slice_float * mask_slice) + (roi_float * inv_alpha_mask)

                    # Place the blended ROI back onto the canvas
                    canvas[y_start_canvas:y_end_canvas, x_start_canvas:x_end_canvas] = np.clip(blended_roi_float, 0, 255).astype(np.uint8)

                except Exception as e:
                    # Error during blending/slicing
                    print(f"\nError during blending/slicing frame {frame_num} ({phase_desc}): {e}")
                    print(f"  Canvas ROI: y[{y_start_canvas}:{y_end_canvas}], x[{x_start_canvas}:{x_end_canvas}] (Shape: {roi.shape if 'roi' in locals() else 'N/A'})")
                    print(f"  Image Slice Indices: y[{y_start_img}:{y_end_img}], x[{x_start_img}:{x_end_img}]")
                    print(f"  Scaled BGR Shape: {scaled_bgr.shape}, Scaled Alpha Shape: {scaled_alpha.shape}")
                    # No 'pass' needed here, except block finishes
            else:
                # Slice indices out of bounds case
                # print(f"\nWarning: Slice indices out of bounds frame {frame_num}") # Uncomment for debugging
                pass # This pass belongs to the 'if y_end_img <= ...' bounds check
        else:
             # Dimension mismatch case
             # print(f"\nWarning: Dimension mismatch frame {frame_num}...") # Uncomment for debugging
             pass # This pass belongs to the 'if h_roi == h_img_slice...' dimension check - THIS is likely where the error was

    # --- Write the frame ---
    try:
        video_writer.write(canvas)
    except Exception as e:
         print(f"\nError writing frame {frame_num} to video: {e}")
         break # Stop processing if writing fails

    # --- Progress Indicator ---
    if (frame_num + 1) % 5 == 0 or frame_num == total_frames - 1:
        elapsed = time.time() - start_time
        percent_done = ((frame_num + 1) / total_frames) * 100
        # Display scale as percentage
        print(f"\rProcessed frame {frame_num + 1}/{total_frames} ({percent_done:.1f}%) {phase_desc} Scale:{current_scale*100:.1f}% Pos:({pos_x},{pos_y}) [{elapsed:.2f}s]", end="")


# --- Cleanup ---
video_writer.release()
end_time = time.time()
print("\n" + "-" * 30)
print("Animation finished.")
# Verify output file existence
if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
    print(f"Video saved successfully to: {output_path}")
else:
    print(f"Error: Output video file was either not created or is empty at {output_path}.")
    print("Check for errors during video writing or permission issues.")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print("-" * 30)
