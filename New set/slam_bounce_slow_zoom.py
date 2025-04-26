import cv2
import numpy as np
import os
import math
import time

# --- Configuration ---

# Paths (as specified for Termux/Android)
image_folder = '/storage/emulated/0/Download/'
img_name = '1.png' # USE YOUR IMAGE NAME HERE
img_path = os.path.join(image_folder, img_name)

# Video Output Configuration
output_filename = 'slam_bounce_slow_zoom.mp4' # Updated filename
output_path = os.path.join(image_folder, output_filename) # Save video in the Download folder
fps = 30.0           # Frames per second

# --- Animation Parameters ---
# Canvas size will be determined by the image dimensions after loading

# --- Speed & Effect Parameters ---
total_video_duration_sec = 5.0 # << NEW: Total length of the output video

drop_duration_sec = 0.3   # VERY FAST Drop duration
bounce_duration_sec = 0.6 # Duration for the entire bounce sequence

num_bounces = 2.0         # How many full up-and-down bounces
bounce_height_initial = 0.2 # Initial bounce height factor
squash_factor_max = 0.85  # Max vertical squash at impact
damping_factor = 4.0      # How quickly bounce dampens

final_zoom_scale = 1.05   # << NEW: Final scale factor at the VERY END of the video.
                          # 1.05 = 5% zoom in. Use 1.02 or 1.03 for even more minimal zoom.

# --- Sanity Check ---
if total_video_duration_sec <= (drop_duration_sec + bounce_duration_sec):
    print(f"Error: total_video_duration_sec ({total_video_duration_sec}s) must be longer than drop ({drop_duration_sec}s) + bounce ({bounce_duration_sec}s) duration.")
    exit()

# --- End Configuration ---

# --- Load Image ---
print(f"Loading image from: {img_path}")
img_orig_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if img_orig_raw is None:
    print(f"Error: Failed to load {img_path}"); exit()
print("Image loaded successfully.")

# --- Determine Canvas and Image Dimensions ---
img_h_orig, img_w_orig = img_orig_raw.shape[:2]
canvas_h, canvas_w = img_h_orig, img_w_orig # Canvas matches image size
print(f"Image/Canvas Dimensions (HxW): {canvas_h}x{canvas_w}")

# --- Handle Transparency ---
def get_image_components(img_to_process):
    h, w = img_to_process.shape[:2]
    if len(img_to_process.shape) == 3 and img_to_process.shape[2] == 4: # RGBA
        bgr = img_to_process[:, :, 0:3].copy()
        alpha_norm = img_to_process[:, :, 3] / 255.0
        alpha_mask = cv2.merge([alpha_norm] * 3)
        return bgr, alpha_mask
    elif len(img_to_process.shape) == 3: # BGR
        print("Warning: Input image has no alpha channel. Assuming opaque.")
        bgr = img_to_process.copy()
        alpha_mask = np.ones((h, w, 3), dtype=np.float32)
        return bgr, alpha_mask
    elif len(img_to_process.shape) == 2: # Grayscale
         print("Warning: Input image is grayscale. Converting to BGR.")
         bgr = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)
         alpha_mask = np.ones((h, w, 3), dtype=np.float32)
         return bgr, alpha_mask
    else:
        print(f"Error: Unexpected image shape: {img_to_process.shape}")
        exit()

try:
    img_bgr_orig, img_alpha_mask_orig = get_image_components(img_orig_raw)
except Exception as e:
    print(f"Error processing image components: {e}")
    exit()

# --- Calculate Frame Counts & Positions ---
num_drop_frames = max(1, int(fps * drop_duration_sec))
num_bounce_frames = max(1, int(fps * bounce_duration_sec))
total_frames = max(1, int(fps * total_video_duration_sec)) # << Use total video duration

# Calculate the frame number where the bounce phase ends (and zoom begins)
bounce_end_frame = num_drop_frames + num_bounce_frames
num_zoom_frames = total_frames - bounce_end_frame # Frames dedicated to slow zoom

start_y = -img_h_orig
end_y = 0
center_x_orig = (canvas_w - img_w_orig) // 2 # Center based on original width

max_bounce_pixels = int(img_h_orig * bounce_height_initial)

# --- Initialize Video Writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_size_writer = (canvas_w, canvas_h)
try:
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size_writer)
    if not video_writer.isOpened(): raise IOError("Video writer failed to open.")
    print(f"Video writer initialized. Saving to: {output_path}")
except Exception as e:
    print(f"Error initializing VideoWriter: {e}"); exit()

# --- Animation Loop ---
print(f"Generating {total_frames} frames ({num_drop_frames} drop + {num_bounce_frames} bounce + {num_zoom_frames} slow zoom)...")
background = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
start_time = time.time()

img_center_x_orig_pos = img_w_orig // 2 + center_x_orig

for frame_num in range(total_frames):
    canvas = background.copy()

    current_h, current_w = img_h_orig, img_w_orig
    pos_x = center_x_orig
    pos_y = end_y
    img_to_draw_bgr = img_bgr_orig # Start assuming original image needed
    img_to_draw_alpha = img_alpha_mask_orig

    # --- Phase Logic ---
    if frame_num < num_drop_frames:
        # Phase 1: Dropping
        phase = "Dropping"
        drop_progress = frame_num / (num_drop_frames - 1) if num_drop_frames > 1 else 1.0
        current_y_drop = int(start_y + (end_y - start_y) * drop_progress)
        pos_y = current_y_drop
        # No resize needed, dimensions are original

    elif frame_num < bounce_end_frame:
        # Phase 2: Bouncing & Squashing/Stretching
        phase = "Bouncing"
        bounce_frame_num = frame_num - num_drop_frames
        bounce_progress = bounce_frame_num / (num_bounce_frames - 1) if num_bounce_frames > 1 else 1.0

        # Bounce height calculation
        angle = bounce_progress * num_bounces * math.pi
        dampening = math.exp(-damping_factor * bounce_progress)
        current_bounce_offset = int(max_bounce_pixels * abs(math.sin(angle)) * dampening)
        current_y_bottom = end_y + img_h_orig
        pos_y_bottom = current_y_bottom - current_bounce_offset

        # Squash/stretch calculation
        squash_lerp = (1.0 - abs(math.cos(angle))) * dampening
        current_h_factor = squash_factor_max + (1.0 - squash_factor_max) * (1.0 - squash_lerp)
        current_w_factor = 1.0 / current_h_factor if current_h_factor > 0.01 else 1.0

        # Apply deformation and resize if significant
        if abs(current_h_factor - 1.0) > 0.01 or abs(current_w_factor - 1.0) > 0.01:
            current_h = max(1, int(img_h_orig * current_h_factor))
            current_w = max(1, int(img_w_orig * current_w_factor))
            interpolation = cv2.INTER_LINEAR
            img_to_draw_bgr = cv2.resize(img_bgr_orig, (current_w, current_h), interpolation=interpolation)
            img_to_draw_alpha = cv2.resize(img_alpha_mask_orig, (current_w, current_h), interpolation=interpolation)
        # else: keep original size assigned earlier

        # Calculate top-left corner position
        pos_y = pos_y_bottom - current_h
        pos_x = center_x_orig + (img_w_orig - current_w) // 2

    else:
        # Phase 3: Slow Continuous Zoom In (after bounce settles)
        phase = "Slow Zoom"
        # Calculate progress within the zoom phase (0 to 1)
        zoom_frame_num = frame_num - bounce_end_frame
        zoom_progress = zoom_frame_num / (num_zoom_frames - 1) if num_zoom_frames > 1 else 1.0

        # Linearly interpolate scale factor from 1.0 (start of zoom) to final_zoom_scale
        current_scale_factor = 1.0 + (final_zoom_scale - 1.0) * zoom_progress

        # Calculate new dimensions based on scale
        current_h = max(1, int(img_h_orig * current_scale_factor))
        current_w = max(1, int(img_w_orig * current_scale_factor))

        # Resize the original image components
        interpolation = cv2.INTER_LINEAR # Can use INTER_AREA if shrinking significantly (not the case here)
        img_to_draw_bgr = cv2.resize(img_bgr_orig, (current_w, current_h), interpolation=interpolation)
        img_to_draw_alpha = cv2.resize(img_alpha_mask_orig, (current_w, current_h), interpolation=interpolation)

        # Calculate position to keep the image centered during zoom
        # Base position is the final resting place after bounce: end_y, center_x_orig
        pos_y = end_y + (img_h_orig - current_h) // 2 # Keep vertical center aligned with original center
        pos_x = center_x_orig + (img_w_orig - current_w) // 2 # Keep horizontal center aligned with original center


    # --- Drawing the Image ---
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
                img_slice = img_to_draw_bgr[y_start_img : y_end_img, x_start_img : x_end_img]
                mask_slice = img_to_draw_alpha[y_start_img : y_end_img, x_start_img : x_end_img]

                inv_alpha_mask = 1.0 - mask_slice
                blended_roi = cv2.addWeighted(img_slice.astype(np.float32) * mask_slice, 1.0,
                                              roi.astype(np.float32) * inv_alpha_mask, 1.0, 0.0)
                canvas[y_start_canvas:y_end_canvas, x_start_canvas:x_end_canvas] = np.clip(blended_roi, 0, 255).astype(np.uint8)

            except Exception as e:
                print(f"\nError during blending at frame {frame_num} ({phase}): {e}")
                # Attempt fallback copy
                try:
                    h_copy = min(h_roi, h_img_slice)
                    w_copy = min(w_roi, w_img_slice)
                    canvas[y_start_canvas:y_start_canvas+h_copy, x_start_canvas:x_start_canvas+w_copy] = img_slice[:h_copy, :w_copy]
                except Exception as fallback_e:
                     print(f"\nFallback copy failed frame {frame_num}: {fallback_e}")
        else:
             print(f"\nWarning: Dimension mismatch frame {frame_num}. ROI:({h_roi}x{w_roi}), ImgSlice:({h_img_slice}x{w_img_slice}).")
             try:
                 h_copy = min(h_roi, h_img_slice)
                 w_copy = min(w_roi, w_img_slice)
                 img_slice_safe = img_to_draw_bgr[y_start_img : y_start_img+h_copy, x_start_img : x_start_img+w_copy]
                 canvas[y_start_canvas:y_start_canvas+h_copy, x_start_canvas:x_start_canvas+w_copy] = img_slice_safe
             except Exception as fallback_e:
                 print(f"\nFallback limited copy failed frame {frame_num}: {fallback_e}")


    # --- Write the frame ---
    try:
        video_writer.write(canvas)
    except Exception as e:
         print(f"\nError writing frame {frame_num} to video: {e}")
         break

    # --- Progress Indicator ---
    if (frame_num + 1) % int(fps) == 0 or frame_num == total_frames - 1: # Update every second
        elapsed = time.time() - start_time
        percent_done = ((frame_num + 1) / total_frames) * 100
        print(f"\rProcessed frame {frame_num + 1}/{total_frames} ({percent_done:.1f}%) ({phase}) [{elapsed:.2f}s]", end="")


# --- Cleanup ---
video_writer.release()
end_time = time.time()
print("\n" + "-" * 30)
print("Animation finished.")
print(f"Video saved successfully to: {output_path}")
print(f"Total video duration: {total_video_duration_sec:.2f} seconds")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print("-" * 30)
