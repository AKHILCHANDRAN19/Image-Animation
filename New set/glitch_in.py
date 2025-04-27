import cv2
import numpy as np
import os
import math
import time
import random

# --- Configuration ---

# Paths (Adjust for your system, e.g., Termux/Android or PC)
image_folder = '/storage/emulated/0/Download/' # Input and Output Folder
img_name = '1.png' # <<< --- PUT YOUR IMAGE FILENAME HERE
img_path = os.path.join(image_folder, img_name)

# Video Output Configuration
output_filename = 'glitch_in_0.4s_output.mp4' # Updated filename
output_path = os.path.join(image_folder, output_filename)

fps = 30.0           # Frames per second
glitch_duration_sec = 0.4 # <<< --- GLITCH DURATION INCREASED TO 0.4 SECONDS
total_duration_sec = 0.8  # Total animation duration (>= glitch_duration_sec)

# --- Animation Parameters ---
# Final Resting Position (Top-left corner of the image relative to canvas top-left)
final_x = 50
final_y = 50

# Glitch Parameters
max_channel_shift = 10   # Max horizontal/vertical pixel shift for R/B channels
max_block_shift = 15     # Max horizontal pixel shift for image blocks/strips
num_glitch_strips = 12   # Number of horizontal strips for block glitching
max_pos_offset = 3       # Max random pixel offset for the whole image position during glitch

# --- Easing Function ---
def ease_out_quad(t):
    """Quadratic easing out: decelerating to zero velocity. Used for glitch fade."""
    # Clamping input t between 0 and 1
    t = max(0.0, min(1.0, t))
    return t * (2 - t)

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

# Canvas needs to be large enough to hold the image at its final position + potential offset
canvas_margin = max(10, max_pos_offset * 2)
canvas_h = final_y + img_h_orig + canvas_margin
canvas_w = final_x + img_w_orig + canvas_margin

print(f"Image Dimensions (HxW): {img_h_orig}x{img_w_orig}")
print(f"Canvas Dimensions (WxH): {canvas_w}x{canvas_h}")

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
    img_alpha_mask_orig = img_alpha_mask_orig.astype(np.float32) # Ensure float32
except Exception as e:
    print(f"Error processing image components: {e}"); exit()

# --- Calculate Frame Counts ---
total_frames = max(1, int(fps * total_duration_sec))
glitch_frames = max(1, int(fps * glitch_duration_sec))
# Ensure total duration accommodates the glitch (it does here)
if total_frames < glitch_frames:
    total_frames = glitch_frames
    print(f"Warning: total_duration_sec increased to {total_frames/fps:.2f}s to accommodate glitch_duration_sec.")

print(f"Total Duration: {total_frames/fps:.2f}s => {total_frames} total frames.")
print(f"Glitch Duration: {glitch_duration_sec:.2f}s => {glitch_frames} frames.")
print(f"Final Position (Top-Left): ({final_x}, {final_y})")

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

# --- Helper function for Glitch ---
def apply_glitch(image_bgr, alpha_mask, intensity, frame_h, frame_w):
    """Applies random glitches based on intensity."""
    if intensity <= 0:
        return image_bgr.copy(), alpha_mask # Return copies if no glitch

    glitched_bgr = image_bgr.copy()
    current_h, current_w = glitched_bgr.shape[:2]

    # 1. Channel Shift
    current_channel_shift = int(round(max_channel_shift * intensity))
    if current_channel_shift > 0:
        b, g, r = cv2.split(glitched_bgr)
        # Shift Red channel
        shift_r_x = random.randint(-current_channel_shift, current_channel_shift)
        shift_r_y = random.randint(-current_channel_shift // 2, current_channel_shift // 2) # Less vertical shift usually
        r_shifted = np.roll(r, shift_r_x, axis=1)
        r_shifted = np.roll(r_shifted, shift_r_y, axis=0)
        # Shift Blue channel (different amount)
        shift_b_x = random.randint(-current_channel_shift, current_channel_shift)
        shift_b_y = random.randint(-current_channel_shift // 2, current_channel_shift // 2)
        b_shifted = np.roll(b, shift_b_x, axis=1)
        b_shifted = np.roll(b_shifted, shift_b_y, axis=0)
        # Merge back (keep original Green for stability or shift it less)
        glitched_bgr = cv2.merge([b_shifted, g, r_shifted])

    # 2. Block Displacement (Horizontal Strips)
    current_block_shift = int(round(max_block_shift * intensity))
    if current_block_shift > 0 and num_glitch_strips > 0:
        strip_h = max(1, current_h // num_glitch_strips)
        for i in range(num_glitch_strips):
            # Probability of glitching this strip decreases with intensity
            if random.random() < intensity * 0.7: # Adjust 0.7 probability factor as needed
                y_start = i * strip_h
                y_end = min((i + 1) * strip_h, current_h)
                if y_start >= y_end: continue

                shift_amount = random.randint(-current_block_shift, current_block_shift)
                if shift_amount == 0: continue

                strip = glitched_bgr[y_start:y_end, :]
                shifted_strip = np.roll(strip, shift_amount, axis=1)
                glitched_bgr[y_start:y_end, :] = shifted_strip

    # We typically don't glitch the alpha mask itself, it looks strange.
    # Return the glitched BGR and the original alpha mask.
    return glitched_bgr, alpha_mask


# --- Animation Loop ---
print(f"Generating {total_frames} frames...")
background = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) # Black background
start_time = time.time()

for frame_num in range(total_frames):
    canvas = background.copy()

    # Determine glitch intensity and image/position for this frame
    glitch_intensity = 0.0
    image_to_draw_bgr = img_bgr_orig
    alpha_to_draw = img_alpha_mask_orig # Keep original alpha
    pos_x = final_x
    pos_y = final_y
    phase_desc = "Settled"

    if frame_num < glitch_frames:
        # Calculate progress only within the glitch duration (0 to 1)
        glitch_progress = frame_num / (glitch_frames - 1) if glitch_frames > 1 else 1.0

        # Intensity fades out using easing (peaks at start)
        glitch_intensity = ease_out_quad(1.0 - glitch_progress)
        glitch_intensity = max(0.0, min(1.0, glitch_intensity)) # Clamp

        phase_desc = "Glitching"

        if glitch_intensity > 0.01: # Only apply if glitch is noticeable
            # Apply glitch effects
            image_to_draw_bgr, alpha_to_draw = apply_glitch(img_bgr_orig, img_alpha_mask_orig, glitch_intensity, img_h_orig, img_w_orig)

            # Apply slight random position offset
            current_pos_offset = int(round(max_pos_offset * glitch_intensity))
            if current_pos_offset > 0:
                pos_x += random.randint(-current_pos_offset, current_pos_offset)
                pos_y += random.randint(-current_pos_offset, current_pos_offset)

    # Ensure final frame is perfect
    if frame_num == total_frames - 1:
        image_to_draw_bgr = img_bgr_orig
        alpha_to_draw = img_alpha_mask_orig
        pos_x = final_x
        pos_y = final_y
        glitch_intensity = 0.0
        phase_desc = "Final Frame"

    # --- Drawing the Image (Glitched or Sharp) ---
    current_h, current_w = image_to_draw_bgr.shape[:2] # Use dimensions of potentially glitched image

    # Calculate the slice of the canvas to draw onto based on current pos_x, pos_y
    y_start_canvas = max(0, pos_y)
    y_end_canvas = min(canvas_h, pos_y + current_h)
    x_start_canvas = max(0, pos_x)
    x_end_canvas = min(canvas_w, pos_x + current_w)

    # Calculate the corresponding slice of the image to use
    y_start_img = max(0, -pos_y)
    y_end_img = y_start_img + (y_end_canvas - y_start_canvas)
    x_start_img = max(0, -pos_x)
    x_end_img = x_start_img + (x_end_canvas - x_start_canvas)

    # Proceed only if there's a valid, non-empty area to draw
    if y_start_canvas < y_end_canvas and x_start_canvas < x_end_canvas and \
       y_start_img < y_end_img and x_start_img < x_end_img:

        h_roi = y_end_canvas - y_start_canvas
        w_roi = x_end_canvas - x_start_canvas
        h_img_slice = y_end_img - y_start_img
        w_img_slice = x_end_img - x_start_img

        # Sanity check dimensions before slicing
        if h_roi == h_img_slice and w_roi == w_img_slice and h_roi > 0 and w_roi > 0:
            # Ensure image slice indices are within bounds
            if y_start_img >= 0 and x_start_img >= 0 and \
               y_end_img <= image_to_draw_bgr.shape[0] and x_end_img <= image_to_draw_bgr.shape[1] and \
               y_end_img <= alpha_to_draw.shape[0] and x_end_img <= alpha_to_draw.shape[1]:
                try:
                    # Extract the region of interest (ROI) from the canvas
                    roi = canvas[y_start_canvas : y_end_canvas, x_start_canvas : x_end_canvas]

                    # Extract the corresponding slice from the image and alpha mask
                    img_slice = image_to_draw_bgr[y_start_img : y_end_img, x_start_img : x_end_img]
                    mask_slice = alpha_to_draw[y_start_img : y_end_img, x_start_img : x_end_img] # Should be float32

                    # Blend the image slice onto the ROI using the alpha mask
                    inv_alpha_mask = 1.0 - mask_slice # Needs float mask
                    img_slice_float = img_slice.astype(np.float32)
                    roi_float = roi.astype(np.float32)

                    blended_roi_float = (img_slice_float * mask_slice) + (roi_float * inv_alpha_mask)

                    # Place the blended ROI back onto the canvas
                    canvas[y_start_canvas:y_end_canvas, x_start_canvas:x_end_canvas] = np.clip(blended_roi_float, 0, 255).astype(np.uint8)

                except Exception as e:
                    # Error during blending/slicing
                    print(f"\nError during blending/slicing frame {frame_num} ({phase_desc}): {e}")
                    # ... error details ...
            else:
                # Slice indices out of bounds case
                pass
        else:
             # Dimension mismatch case
             pass

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
        print(f"\rProcessed frame {frame_num + 1}/{total_frames} ({percent_done:.1f}%) {phase_desc} Intensity:{glitch_intensity:.2f} Pos:({pos_x},{pos_y}) [{elapsed:.2f}s]", end="")


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
