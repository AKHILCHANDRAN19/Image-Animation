import cv2
import numpy as np
import os
import math
import time

# --- Configuration ---

# Paths (as specified for Termux/Android)
image_folder = '/storage/emulated/0/Download/'
img_name = '1.png' # MAKE SURE THIS IS YOUR IMAGE NAME
img_path = os.path.join(image_folder, img_name)

# Video Output Configuration
output_filename = 'fling_in_output.mp4' # Descriptive filename
output_path = os.path.join(image_folder, output_filename)
fps = 30.0           # Frames per second
total_duration_sec = 0.8 # Duration for the fling (should be relatively short)

# --- Animation Parameters ---
# Where the image is flung FROM
fling_from_direction = 'top_left' # Options: 'top', 'bottom', 'left', 'right',
                                 # 'top_left', 'top_right', 'bottom_left', 'bottom_right'

# Final Resting Position (Top-left corner - usually 0,0 for full image)
final_x = 0
final_y = 0

# Fling Rotation (Optional)
fling_rotation_degrees = -15 # Total degrees to rotate during the fling (negative for clockwise)

# --- Easing Function ---
def ease_out_expo(t):
    """Exponential easing out: decelerating from zero velocity."""
    # t is progress from 0.0 to 1.0
    return 1.0 if t == 1.0 else 1.0 - pow(2, -10 * t)

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

# --- Calculate Frame Counts & Start/End Positions ---
total_frames = max(1, int(fps * total_duration_sec))
end_x = final_x
end_y = final_y

# Determine start position based on direction
start_x, start_y = 0, 0
margin = 10 # Small pixel margin to ensure fully off-screen

if fling_from_direction == 'top':
    start_x, start_y = final_x, -img_h_orig - margin
elif fling_from_direction == 'bottom':
    start_x, start_y = final_x, canvas_h + margin
elif fling_from_direction == 'left':
    start_x, start_y = -img_w_orig - margin, final_y
elif fling_from_direction == 'right':
    start_x, start_y = canvas_w + margin, final_y
elif fling_from_direction == 'top_left':
    start_x, start_y = -img_w_orig - margin, -img_h_orig - margin
elif fling_from_direction == 'top_right':
    start_x, start_y = canvas_w + margin, -img_h_orig - margin
elif fling_from_direction == 'bottom_left':
    start_x, start_y = -img_w_orig - margin, canvas_h + margin
elif fling_from_direction == 'bottom_right':
    start_x, start_y = canvas_w + margin, canvas_h + margin
else:
    print(f"Warning: Invalid fling_from_direction '{fling_from_direction}'. Defaulting to 'left'.")
    start_x, start_y = -img_w_orig - margin, final_y

print(f"Total Duration: {total_duration_sec:.2f}s => {total_frames} total frames.")
print(f"Flinging from ({start_x}, {start_y}) to ({end_x}, {end_y})")

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
background = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) # Black background
start_time = time.time()

# Calculate image center for rotation
center_orig_x = img_w_orig / 2.0
center_orig_y = img_h_orig / 2.0

for frame_num in range(total_frames):
    canvas = background.copy()
    phase = "Flinging"

    # Calculate progress (0.0 to 1.0)
    progress = frame_num / (total_frames - 1) if total_frames > 1 else 1.0

    # Apply easing function to progress
    eased_progress = ease_out_expo(progress)

    # --- Calculate Current Position using Eased Progress ---
    pos_x = int(start_x + (end_x - start_x) * eased_progress)
    pos_y = int(start_y + (end_y - start_y) * eased_progress)

    # --- Calculate Current Rotation using Eased Progress ---
    # Rotation angle goes from 0 to fling_rotation_degrees as progress goes 0 to 1
    current_angle = fling_rotation_degrees * eased_progress

    # --- Perform Rotation (if needed) ---
    if abs(current_angle) > 0.01: # Check if rotation is significant
        try:
            rot_mat = cv2.getRotationMatrix2D(center=(center_orig_x, center_orig_y),
                                              angle=current_angle,
                                              scale=1.0)

            # Apply rotation to BGR and Alpha mask
            rotated_bgr = cv2.warpAffine(img_bgr_orig, rot_mat, (img_w_orig, img_h_orig),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            alpha_ch = img_alpha_mask_orig[:,:,0]
            rotated_alpha_single = cv2.warpAffine(alpha_ch, rot_mat, (img_w_orig, img_h_orig),
                                                  flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            rotated_alpha = cv2.merge([rotated_alpha_single] * 3)

        except cv2.error as warp_err:
             print(f"\nError during warpAffine frame {frame_num}: {warp_err}")
             rotated_bgr = img_bgr_orig # Fallback
             rotated_alpha = img_alpha_mask_orig
    else:
        # No significant rotation, use original image components
        rotated_bgr = img_bgr_orig
        rotated_alpha = img_alpha_mask_orig

    # Dimensions remain original image dimensions after rotation warp
    current_h, current_w = img_h_orig, img_w_orig

    # --- Drawing the Image (Using Robust Logic) ---
    y_start_canvas = max(0, pos_y)
    y_end_canvas = min(canvas_h, pos_y + current_h)
    x_start_canvas = max(0, pos_x)
    x_end_canvas = min(canvas_w, pos_x + current_w)

    y_start_img = max(0, -pos_y)
    y_end_img = y_start_img + (y_end_canvas - y_start_canvas)
    x_start_img = max(0, -pos_x)
    x_end_img = x_start_img + (x_end_canvas - x_start_canvas)

    if y_start_canvas < y_end_canvas and x_start_canvas < x_end_canvas and \
       y_start_img < y_end_img and x_start_img < x_end_img:

        h_roi = y_end_canvas - y_start_canvas
        w_roi = x_end_canvas - x_start_canvas
        h_img_slice = y_end_img - y_start_img
        w_img_slice = x_end_img - x_start_img

        if h_roi == h_img_slice and w_roi == w_img_slice:
            try:
                roi = canvas[y_start_canvas : y_end_canvas, x_start_canvas : x_end_canvas]
                img_slice = rotated_bgr[y_start_img : y_end_img, x_start_img : x_end_img]
                mask_slice = rotated_alpha[y_start_img : y_end_img, x_start_img : x_end_img]

                inv_alpha_mask = 1.0 - mask_slice
                blended_roi = cv2.addWeighted(img_slice.astype(np.float32) * mask_slice, 1.0,
                                              roi.astype(np.float32) * inv_alpha_mask, 1.0, 0.0)

                canvas[y_start_canvas:y_end_canvas, x_start_canvas:x_end_canvas] = np.clip(blended_roi, 0, 255).astype(np.uint8)

            except Exception as e:
                print(f"\nError during blending/slicing at frame {frame_num}: {e}")
                pass # Continue
        else:
             print(f"\nWarning: Dimension mismatch frame {frame_num}. ROI:({h_roi}x{w_roi}), ImgSlice:({h_img_slice}x{w_img_slice}). Skipping draw.")

    # --- Write the frame ---
    try:
        video_writer.write(canvas)
    except Exception as e:
         print(f"\nError writing frame {frame_num} to video: {e}")
         break

    # --- Progress Indicator ---
    if (frame_num + 1) % 5 == 0 or frame_num == total_frames - 1:
        elapsed = time.time() - start_time
        percent_done = ((frame_num + 1) / total_frames) * 100
        print(f"\rProcessed frame {frame_num + 1}/{total_frames} ({percent_done:.1f}%) ({phase}) Pos:({pos_x},{pos_y}) Angle:{current_angle:.1f} [{elapsed:.2f}s]", end="")


# --- Cleanup ---
video_writer.release()
# No cv2.destroyAllWindows() needed in Termux
end_time = time.time()
print("\n" + "-" * 30)
print("Animation finished.")
print(f"Video saved successfully to: {output_path}")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print("-" * 30)
