import cv2
import numpy as np
import os
import math
import time

# --- Configuration ---

# Paths (as specified for Termux/Android)
image_folder = '/storage/emulated/0/Download/'
img_name = '1.png'
img_path = os.path.join(image_folder, img_name)

# Video Output Configuration
output_filename = 'fast_drop_5shakes_fit.mp4' # Updated filename
output_path = os.path.join(image_folder, output_filename) # Save video in the Download folder
fps = 30.0           # Frames per second

# --- Animation Parameters ---
# Canvas size will be determined by the image dimensions after loading

# --- Speed & Effect Parameters ---
drop_duration_sec = 0.4   # VERY FAST Drop duration
shake_cycles = 5          # << CHANGED: Exactly five full back-and-forth shakes
shake_freq_hz = 4.0       # VERY FAST Shake frequency (cycles per second)
shake_amplitude = 25      # Horizontal shake distance (pixels)

# --- End Configuration ---

# --- Load Image ---
print(f"Loading image from: {img_path}")
img_orig = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if img_orig is None:
    print(f"Error: Failed to load {img_path}"); exit()
print("Image loaded successfully.")

# --- Determine Canvas and Image Dimensions ---
img_h, img_w = img_orig.shape[:2]
canvas_h, canvas_w = img_h, img_w # Canvas matches image size
print(f"Image/Canvas Dimensions (HxW): {canvas_h}x{canvas_w}")

# --- Handle Transparency ---
def get_image_components(img_to_process):
    h, w = img_to_process.shape[:2]
    if len(img_to_process.shape) == 3 and img_to_process.shape[2] == 4: # Check for 3 dims and 4 channels
        bgr = img_to_process[:, :, 0:3]
        alpha_norm = img_to_process[:, :, 3] / 255.0
        alpha_mask = cv2.merge([alpha_norm] * 3)
        return bgr, alpha_mask
    elif len(img_to_process.shape) == 3: # Standard 3-channel BGR
        bgr = img_to_process
        alpha_mask = np.ones((h, w, 3), dtype=np.float32)
        return bgr, alpha_mask
    elif len(img_to_process.shape) == 2: # Grayscale image
         print("Warning: Input image is grayscale. Converting to BGR.")
         bgr = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)
         alpha_mask = np.ones((h, w, 3), dtype=np.float32)
         return bgr, alpha_mask
    else:
        print(f"Error: Unexpected image shape: {img_to_process.shape}")
        exit()


try:
    img_bgr, img_alpha_mask = get_image_components(img_orig)
except Exception as e:
    print(f"Error processing image components: {e}")
    exit()

# --- Calculate Frame Counts & Positions ---
num_drop_frames = max(1, int(fps * drop_duration_sec))
# Shake duration now depends on 5 cycles
shake_duration_sec = shake_cycles / shake_freq_hz
num_shake_frames = max(1, int(fps * shake_duration_sec))
total_frames = num_drop_frames + num_shake_frames

start_y = -img_h # Start fully off-screen vertically
end_y = 0       # End position where image top aligns with canvas top
center_x = 0    # Image horizontally fills the canvas

# --- Initialize Video Writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_size_writer = (canvas_w, canvas_h) # (width, height) for VideoWriter
try:
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size_writer)
    if not video_writer.isOpened(): raise IOError("Video writer failed to open.")
    print(f"Video writer initialized. Saving to: {output_path}")
except Exception as e:
    print(f"Error initializing VideoWriter: {e}"); exit()

# --- Animation Loop ---
print(f"Generating {total_frames} frames ({num_drop_frames} drop + {num_shake_frames} shake)...")
background = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
start_time = time.time()

for frame_num in range(total_frames):
    canvas = background.copy()

    current_x = center_x
    current_y = 0

    # --- Phase Logic ---
    if frame_num < num_drop_frames:
        # Phase 1: Dropping
        drop_progress = frame_num / (num_drop_frames - 1) if num_drop_frames > 1 else 1.0
        current_y = int(start_y + (end_y - start_y) * drop_progress)
        current_x = center_x
        phase = "Dropping"

    else:
        # Phase 2: Shaking
        shake_frame_num = frame_num - num_drop_frames
        shake_progress = shake_frame_num / (num_shake_frames - 1) if num_shake_frames > 1 else 1.0
        total_angle = shake_cycles * 2 * math.pi # Uses the updated shake_cycles=5
        current_angle = shake_progress * total_angle
        shake_offset_x = int(shake_amplitude * math.sin(current_angle))
        current_y = end_y
        current_x = center_x + shake_offset_x
        phase = "Shaking"

    # --- Drawing the Image ---
    pos_x = current_x
    pos_y = current_y
    y_start_canvas = max(0, pos_y)
    y_end_canvas = min(canvas_h, pos_y + img_h)
    x_start_canvas = max(0, pos_x)
    x_end_canvas = min(canvas_w, pos_x + img_w)

    y_start_img = max(0, -pos_y)
    y_end_img = y_start_img + (y_end_canvas - y_start_canvas)
    x_start_img = max(0, -pos_x)
    x_end_img = x_start_img + (x_end_canvas - x_start_canvas)

    if y_start_canvas < y_end_canvas and x_start_canvas < x_end_canvas and \
       y_start_img < y_end_img and x_start_img < x_end_img:

        try:
            roi = canvas[y_start_canvas : y_end_canvas, x_start_canvas : x_end_canvas]
            img_slice = img_bgr[y_start_img : y_end_img, x_start_img : x_end_img]
            mask_slice = img_alpha_mask[y_start_img : y_end_img, x_start_img : x_end_img]

            if roi.shape[:2] == img_slice.shape[:2] == mask_slice.shape[:2]:
                inv_alpha_mask = 1.0 - mask_slice
                blended_roi = cv2.addWeighted(img_slice.astype(np.float32) * mask_slice, 1.0,
                                              roi.astype(np.float32) * inv_alpha_mask, 1.0, 0.0)
                canvas[y_start_canvas:y_end_canvas, x_start_canvas:x_end_canvas] = blended_roi.astype(np.uint8)
            else:
                 h_copy = min(roi.shape[0], img_slice.shape[0])
                 w_copy = min(roi.shape[1], img_slice.shape[1])
                 canvas[y_start_canvas:y_start_canvas+h_copy, x_start_canvas:x_start_canvas+w_copy] = img_slice[:h_copy, :w_copy]

        except Exception as e:
            print(f"\nError during drawing at frame {frame_num} ({phase}): {e}")

    # --- Write the frame ---
    try:
        video_writer.write(canvas)
    except Exception as e:
         print(f"\nError writing frame {frame_num} to video: {e}")
         break

    # --- Progress Indicator ---
    if (frame_num + 1) % int(fps/2) == 0 or frame_num == total_frames - 1:
        elapsed = time.time() - start_time
        print(f"\rProcessed frame {frame_num + 1}/{total_frames} ({phase}) [{elapsed:.2f}s]", end="")


# --- Cleanup ---
video_writer.release()
end_time = time.time()
print("\n" + "-" * 30)
print("Animation finished.")
print(f"Video saved successfully to: {output_path}")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print("-" * 30)
