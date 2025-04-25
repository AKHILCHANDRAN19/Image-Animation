import cv2
import numpy as np
import os
import time

# --- Configuration ---
# Input image path
image_path = "/storage/emulated/0/Download/input.png"

# Output video path
output_dir = os.path.dirname(image_path)
output_filename = "output_simulated_dolly_zoom.mp4"
output_video_path = os.path.join(output_dir, output_filename)

# Video parameters
duration_seconds = 7 # Dolly zooms are often relatively short
fps = 30

# --- Dolly Zoom Simulation Parameters ---

# Effect Direction:
# 'in': Simulates Camera Dollying OUT, Lens Zooming IN (Background Expands)
# 'out': Simulates Camera Dollying IN, Lens Zooming OUT (Background Compresses)
effect_direction = 'in' # Choose 'in' or 'out'

# Zoom Range (Scale Factor):
# How much the image effectively scales during the effect.
# A value of 1.0 is the original size.
# For 'in': Start near 1.0, end > 1.0 (e.g., 1.0 to 2.0)
# For 'out': Start near 1.0, end < 1.0 (e.g., 1.0 to 0.5)
min_scale = 1.0
max_scale = 2.0 # For 'in' effect
# min_scale = 0.5 # For 'out' effect
# max_scale = 1.0 # For 'out' effect


# Focus Point (relative coordinates 0.0 to 1.0) - Center is usually best
focus_x_rel = 0.5 # 0.5 = center horizontally
focus_y_rel = 0.5 # 0.5 = center vertically

# --- End Configuration ---

# --- Script Start ---
print("--- Simulated Dolly Zoom Video Generator ---")
print("Note: This simulates the scaling effect on a 2D image, not true perspective change.")

# 1. Check if input image exists
print(f"Checking for input image at: {image_path}")
if not os.path.exists(image_path):
    print(f"Error: Input image not found at '{image_path}'")
    exit(1)

# 2. Load the input image
print("Loading image...")
img_original = cv2.imread(image_path)
if img_original is None:
    print(f"Error: Could not read image file: '{image_path}'.")
    exit(1)

# Ensure uint8
if img_original.dtype != np.uint8:
    print(f"Warning: Converting image dtype {img_original.dtype} to uint8.")
    try:
        img_original = np.clip(img_original, 0, 255).astype(np.uint8) # Basic clip/convert
    except Exception as e:
        print(f"Error during dtype conversion: {e}")
        exit(1)


# 3. Get image dimensions
h, w = img_original.shape[:2]
print(f"Input image dimensions: Width={w}, Height={h}")

# Calculate absolute focus point
focus_x_abs = int(w * focus_x_rel)
focus_y_abs = int(h * focus_y_rel)

# 4. Setup Video Writer
total_frames = int(duration_seconds * fps)
frame_size = (w, h) # Output frame size is always the original size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
print(f"Attempting to create video writer for: {output_video_path}")

out = cv2.VideoWriter(output_video_path, fourcc, float(fps), frame_size)

if not out.isOpened():
    print(f"Error: Could not open video writer for path: '{output_video_path}'")
    exit(1)

print("Video writer opened successfully.")
print(f"\nGenerating Simulated Dolly Zoom video...")
print(f"Direction: '{effect_direction}', Scale Range: [{min_scale:.2f} - {max_scale:.2f}]")
print(f"Duration: {duration_seconds}s @ {fps}fps ({total_frames} frames)")

start_time = time.time()

try:
    # 5. Generate frames
    for i in range(total_frames):
        # Calculate progress (0.0 to 1.0)
        progress = i / max(1, total_frames - 1)

        # Calculate current scale factor based on direction
        if effect_direction == 'in': # Dolly Out, Zoom In -> Scale increases
            current_scale = min_scale + (max_scale - min_scale) * progress
        elif effect_direction == 'out': # Dolly In, Zoom Out -> Scale decreases
            current_scale = max_scale - (max_scale - min_scale) * progress
        else: # Default to 'in' if invalid direction specified
            print(f"Warning: Invalid effect_direction '{effect_direction}'. Defaulting to 'in'.")
            effect_direction = 'in_default' # Prevent repeating warning
            current_scale = min_scale + (max_scale - min_scale) * progress

        # --- Scaling (Simulates Zoom) ---
        scaled_w = int(w * current_scale)
        scaled_h = int(h * current_scale)

        # Ensure dimensions are at least 1x1
        if scaled_w < 1 or scaled_h < 1:
            # If scale is too small, just use a 1x1 pixel from original (or black)
            # To avoid error, let's just hold the last valid frame or use black
             if i > 0:
                 out.write(final_frame) # Write previous frame again
             else:
                 out.write(np.zeros((h,w,3), dtype=np.uint8)) # Write black frame
             print(f"\nWarning: Scale too small at frame {i}. Holding frame.", end='')
             continue # Skip rest of loop for this frame

        # Resize the original image
        scaled_image = cv2.resize(img_original, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        # --- Cropping / Padding (Simulates Dolly) ---
        final_frame = np.zeros((h, w, 3), dtype=np.uint8) # Start with black canvas (useful for 'out' effect)

        if current_scale >= 1.0: # Simulating 'in' effect (Dolly Out, Zoom In) - Crop the scaled image
            # Calculate the top-left corner of the crop window in the scaled image
            # We want the focus point in the scaled image to map to the focus point in the original frame size
            crop_x = int(focus_x_abs * current_scale - focus_x_abs)
            crop_y = int(focus_y_abs * current_scale - focus_y_abs)

            # Calculate crop boundaries
            crop_x1 = np.clip(crop_x, 0, scaled_w)
            crop_y1 = np.clip(crop_y, 0, scaled_h)
            crop_x2 = np.clip(crop_x + w, 0, scaled_w)
            crop_y2 = np.clip(crop_y + h, 0, scaled_h)

            # Calculate the size of the valid crop region
            valid_crop_w = crop_x2 - crop_x1
            valid_crop_h = crop_y2 - crop_y1

            if valid_crop_w > 0 and valid_crop_h > 0:
                 # Extract the valid region from scaled image
                 cropped_part = scaled_image[crop_y1:crop_y2, crop_x1:crop_x2]

                 # Calculate where to paste this onto the final (original size) frame
                 paste_x = (w - valid_crop_w) // 2
                 paste_y = (h - valid_crop_h) // 2

                 # Adjust paste coordinates based on how much was clipped from top/left
                 if crop_x < 0: paste_x = 0
                 if crop_y < 0: paste_y = 0

                 # Ensure pasting within bounds (should fit if calculations are right)
                 paste_x1 = np.clip(paste_x, 0, w)
                 paste_y1 = np.clip(paste_y, 0, h)
                 paste_x2 = np.clip(paste_x + valid_crop_w, 0, w)
                 paste_y2 = np.clip(paste_y + valid_crop_h, 0, h)

                 # Adjust cropped part size if pasting area is smaller (due to extreme clipping)
                 final_crop_w = paste_x2-paste_x1
                 final_crop_h = paste_y2-paste_y1

                 final_frame[paste_y1:paste_y2, paste_x1:paste_x2] = cropped_part[:final_crop_h, :final_crop_w]


        else: # Simulating 'out' effect (Dolly In, Zoom Out) - Place scaled image onto black canvas
            # Calculate top-left corner to paste the small image onto the black canvas
            # such that the focus points align
            paste_x = focus_x_abs - int(focus_x_abs * current_scale)
            paste_y = focus_y_abs - int(focus_y_abs * current_scale)

            # Clip paste coordinates to ensure they are within the final frame bounds
            paste_x1 = np.clip(paste_x, 0, w)
            paste_y1 = np.clip(paste_y, 0, h)
            paste_x2 = np.clip(paste_x + scaled_w, 0, w)
            paste_y2 = np.clip(paste_y + scaled_h, 0, h)

             # Calculate which part of the scaled_image corresponds to the valid paste area
            scaled_crop_x1 = 0 if paste_x >= 0 else -paste_x
            scaled_crop_y1 = 0 if paste_y >= 0 else -paste_y
            scaled_crop_x2 = scaled_w if (paste_x + scaled_w) <= w else w - paste_x
            scaled_crop_y2 = scaled_h if (paste_y + scaled_h) <= h else h - paste_y

            # Ensure crop indices are valid
            scaled_crop_x1 = np.clip(scaled_crop_x1, 0, scaled_w)
            scaled_crop_y1 = np.clip(scaled_crop_y1, 0, scaled_h)
            scaled_crop_x2 = np.clip(scaled_crop_x2, 0, scaled_w)
            scaled_crop_y2 = np.clip(scaled_crop_y2, 0, scaled_h)

            # Extract the valid part from the scaled image
            if scaled_crop_x2 > scaled_crop_x1 and scaled_crop_y2 > scaled_crop_y1:
                part_to_paste = scaled_image[scaled_crop_y1:scaled_crop_y2, scaled_crop_x1:scaled_crop_x2]

                # Paste onto the final frame
                final_frame[paste_y1:paste_y2, paste_x1:paste_x2] = part_to_paste


        # Write the final frame
        out.write(final_frame)

        # Optional: Print progress
        if (i + 1) % fps == 0 or (i + 1) == total_frames:
            elapsed_time = time.time() - start_time
            eta = (elapsed_time / (i + 1)) * (total_frames - (i + 1)) if i > 0 else 0
            print(f"Processed frame {i + 1}/{total_frames} ({(i+1)/total_frames*100:.1f}%) | Scale: {current_scale:.2f} | ETA: {eta:.1f}s", end='\r')


except Exception as e:
    print(f"\nError during frame generation or writing: {e}")
    import traceback
    traceback.print_exc()
finally:
    # 6. Release the VideoWriter
    print("\nReleasing video writer...")
    out.release()

end_time = time.time()
total_time = end_time - start_time

print(f"\nVideo generation finished in {total_time:.2f} seconds.")
if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
     print(f"Video saved successfully to: {output_video_path}")
else:
     print(f"Error: Output video file '{output_video_path}' was not created or is empty.")

print("--- Script End ---")
