import cv2
import numpy as np
import os
import time

# --- Configuration ---
# Input image path
image_path = "/storage/emulated/0/Download/input.png" # <<< CHANGE THIS TO YOUR IMAGE PATH

# Output video path
output_dir = os.path.dirname(image_path)
output_filename = "output_medium_dolly_zoom.mp4" # <<< Changed filename
output_video_path = os.path.join(output_dir, output_filename)

# --- Parameters for MEDIUM SPEED ---

# 1. SET A MODERATE DURATION
duration_seconds = 10 # e.g., 10 seconds (Adjust as needed)

# Standard video parameter
fps = 30

# 2. SET A MODERATE SCALE RANGE (adjust these for desired intensity)
# Effect Direction:
effect_direction = 'in' # Choose 'in' or 'out'

# Example for MEDIUM 'in' effect (Background Expands Moderately):
min_scale_medium_in = 1.0
max_scale_medium_in = 1.4  # Increased range compared to slow (e.g., 1.0 -> 1.4)

# Example for MEDIUM 'out' effect (Background Compresses Moderately):
min_scale_medium_out = 0.6 # Decreased min compared to slow (e.g., 0.6 -> 1.0)
max_scale_medium_out = 1.0

# Choose the scale range based on the selected direction
if effect_direction == 'in':
    min_scale = min_scale_medium_in
    max_scale = max_scale_medium_in
elif effect_direction == 'out':
    min_scale = min_scale_medium_out
    max_scale = max_scale_medium_out
else:
    print(f"Warning: Invalid effect_direction '{effect_direction}'. Defaulting to medium 'in'.")
    min_scale = 1.0
    max_scale = 1.4
    effect_direction = 'in'


# Focus Point (relative coordinates 0.0 to 1.0) - Center is usually best
focus_x_rel = 0.5
focus_y_rel = 0.5

# --- End Configuration ---

# --- Script Start ---
print("--- MEDIUM SPEED Simulated Dolly Zoom Video Generator ---") # <<< Updated title

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
        # Ensure values are within the valid range for uint8 before converting
        if img_original.max() > 255 or img_original.min() < 0:
             print("Clipping image values to [0, 255] before uint8 conversion.")
             img_original = np.clip(img_original, 0, 255)
        img_original = img_original.astype(np.uint8)
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
frame_size = (w, h)
# Use 'mp4v' for wider compatibility, especially on mobile/Android
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1', 'H264' if available
print(f"Attempting to create video writer for: {output_video_path}")

out = cv2.VideoWriter(output_video_path, fourcc, float(fps), frame_size)

if not out.isOpened():
    print(f"Error: Could not open video writer for path: '{output_video_path}'")
    print("Common issues: Check permissions, codec availability ('mp4v' is generally safe).")
    exit(1)

print("Video writer opened successfully.")
print(f"\nGenerating MEDIUM SPEED Simulated Dolly Zoom video...") # <<< Updated status
print(f"Direction: '{effect_direction}', Scale Range: [{min_scale:.2f} - {max_scale:.2f}]")
print(f"Duration: {duration_seconds}s @ {fps}fps ({total_frames} frames)")

start_time = time.time()
last_frame = None # Keep track of the last valid frame

try:
    # 5. Generate frames
    for i in range(total_frames):
        # Calculate linear progress (0.0 to 1.0)
        # Avoid division by zero if total_frames is 1
        progress = i / max(1, total_frames - 1) if total_frames > 1 else 0.0

        # Calculate current scale based on progress and direction
        if effect_direction == 'in':
            # Scale increases from min_scale to max_scale
            current_scale = min_scale + (max_scale - min_scale) * progress
        else: # 'out' or default
            # Scale decreases from max_scale to min_scale
            current_scale = max_scale - (max_scale - min_scale) * progress

        # Calculate the dimensions of the scaled image
        scaled_w = int(w * current_scale)
        scaled_h = int(h * current_scale)

        # --- Safety Check: Prevent invalid dimensions ---
        if scaled_w < 1 or scaled_h < 1:
            print(f"\nWarning: Calculated scale ({current_scale:.3f}) resulted in invalid dimensions ({scaled_w}x{scaled_h}) at frame {i}. Skipping frame or holding last frame.")
            if last_frame is not None:
                out.write(last_frame) # Write the previous frame again
            # If it's the very first frame and it's invalid, write a black frame
            elif i == 0:
                 black_frame = np.zeros((h, w, 3), dtype=np.uint8)
                 out.write(black_frame)
            continue # Skip processing for this frame

        # Resize the original image to the calculated scale
        # INTER_LINEAR is a good balance between speed and quality
        try:
            scaled_image = cv2.resize(img_original, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
            print(f"\nError during cv2.resize at frame {i}: {e}")
            print(f"Target dimensions: ({scaled_w}, {scaled_h}), Scale: {current_scale:.4f}")
            if last_frame is not None: out.write(last_frame)
            continue # Skip frame on resize error

        # Create the output frame (black canvas)
        final_frame = np.zeros((h, w, 3), dtype=np.uint8)

        # --- Logic for placing the scaled image onto the final frame ---
        if current_scale >= 1.0:
            # --- ZOOM IN (Crop the scaled image) ---
            # Calculate the top-left corner of the crop window in the scaled image
            # such that the focus point remains centered in the final frame.
            crop_x = int(focus_x_abs * current_scale - focus_x_abs)
            crop_y = int(focus_y_abs * current_scale - focus_y_abs)

            # Define the crop boundaries in the scaled image coordinates
            crop_x1 = crop_x
            crop_y1 = crop_y
            crop_x2 = crop_x + w
            crop_y2 = crop_y + h

            # --- Clipping to ensure crop boundaries are within the scaled image ---
            # Source coordinates (from scaled_image)
            src_x1 = np.clip(crop_x1, 0, scaled_w)
            src_y1 = np.clip(crop_y1, 0, scaled_h)
            src_x2 = np.clip(crop_x2, 0, scaled_w)
            src_y2 = np.clip(crop_y2, 0, scaled_h)

            # Destination coordinates (onto final_frame)
            # If the crop started outside the scaled image (negative crop_x/y),
            # the paste destination needs to be offset.
            dst_x1 = 0 if crop_x1 >= 0 else -crop_x1
            dst_y1 = 0 if crop_y1 >= 0 else -crop_y1
            # The width/height of the paste area is the valid intersection size
            valid_crop_w = src_x2 - src_x1
            valid_crop_h = src_y2 - src_y1
            dst_x2 = dst_x1 + valid_crop_w
            dst_y2 = dst_y1 + valid_crop_h

            # --- Final clipping for destination boundaries ---
            dst_x1 = np.clip(dst_x1, 0, w)
            dst_y1 = np.clip(dst_y1, 0, h)
            dst_x2 = np.clip(dst_x2, 0, w)
            dst_y2 = np.clip(dst_y2, 0, h)


            # Ensure there's a valid area to copy
            if (src_x2 > src_x1) and (src_y2 > src_y1) and \
               (dst_x2 > dst_x1) and (dst_y2 > dst_y1):
                # Extract the valid region from the scaled image
                cropped_part = scaled_image[src_y1:src_y2, src_x1:src_x2]
                # Ensure the extracted part has the expected shape for pasting
                if cropped_part.shape[0] == (dst_y2-dst_y1) and cropped_part.shape[1] == (dst_x2-dst_x1):
                    final_frame[dst_y1:dst_y2, dst_x1:dst_x2] = cropped_part
                # else: print warning about shape mismatch if needed

        else:
            # --- ZOOM OUT (Place smaller scaled image onto black canvas) ---
            # Calculate top-left corner for pasting the scaled image so focus point aligns
            paste_x = focus_x_abs - int(focus_x_abs * current_scale)
            paste_y = focus_y_abs - int(focus_y_abs * current_scale)

            # Define paste boundaries on the final_frame
            paste_x1 = paste_x
            paste_y1 = paste_y
            paste_x2 = paste_x + scaled_w
            paste_y2 = paste_y + scaled_h

            # --- Clipping to ensure paste boundaries are within the final frame ---
            # Destination coordinates (onto final_frame)
            dst_x1 = np.clip(paste_x1, 0, w)
            dst_y1 = np.clip(paste_y1, 0, h)
            dst_x2 = np.clip(paste_x2, 0, w)
            dst_y2 = np.clip(paste_y2, 0, h)

            # Source coordinates (from scaled_image)
            # If the paste started outside the frame (negative paste_x/y),
            # the source crop needs to be offset.
            src_x1 = 0 if paste_x1 >= 0 else -paste_x1
            src_y1 = 0 if paste_y1 >= 0 else -paste_y1
            # The width/height of the source crop area is the valid intersection size
            valid_paste_w = dst_x2 - dst_x1
            valid_paste_h = dst_y2 - dst_y1
            src_x2 = src_x1 + valid_paste_w
            src_y2 = src_y1 + valid_paste_h

             # --- Final clipping for source boundaries ---
            src_x1 = np.clip(src_x1, 0, scaled_w)
            src_y1 = np.clip(src_y1, 0, scaled_h)
            src_x2 = np.clip(src_x2, 0, scaled_w)
            src_y2 = np.clip(src_y2, 0, scaled_h)

            # Ensure there's a valid area to copy
            if (src_x2 > src_x1) and (src_y2 > src_y1) and \
               (dst_x2 > dst_x1) and (dst_y2 > dst_y1):
                # Extract the valid region from the scaled image
                part_to_paste = scaled_image[src_y1:src_y2, src_x1:src_x2]
                 # Ensure the extracted part has the expected shape for pasting
                if part_to_paste.shape[0] == (dst_y2-dst_y1) and part_to_paste.shape[1] == (dst_x2-dst_x1):
                    final_frame[dst_y1:dst_y2, dst_x1:dst_x2] = part_to_paste
                # else: print warning about shape mismatch if needed


        # Write the frame to the video file
        out.write(final_frame)
        last_frame = final_frame # Store the successfully generated frame

        # Print progress update periodically
        if (i + 1) % fps == 0 or (i + 1) == total_frames:
            elapsed_time = time.time() - start_time
            # Estimate remaining time (avoid division by zero)
            eta = (elapsed_time / (i + 1)) * (total_frames - (i + 1)) if (i + 1) > 0 and total_frames > (i+1) else 0
            print(f"Processed frame {i + 1}/{total_frames} ({(i+1)/total_frames*100:.1f}%) | Scale: {current_scale:.3f} | ETA: {eta:.1f}s", end='\r')

except Exception as e:
    print(f"\nError during frame generation or writing at frame {i}: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Ensure the progress line is cleared
    print("\n" + " " * 80) # Clear the progress line
    print("Releasing video writer...")
    out.release() # Release the video writer resources

end_time = time.time()
total_time = end_time - start_time

print(f"\nVideo generation finished in {total_time:.2f} seconds.")
# Verify the output file exists and has size
if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
     print(f"Video saved successfully to: {output_video_path}")
else:
     print(f"Error: Output video file '{output_video_path}' was NOT created or is empty. Check for errors above.")

print("--- Script End ---")
