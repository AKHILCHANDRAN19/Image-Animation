import cv2
import numpy as np
import os
import time # Added for potential progress delay if needed

# --- Configuration ---
image_path = "/storage/emulated/0/Download/input.png"
output_video_path = "/storage/emulated/0/Download/output_slow_zoom_animation.mp4"
duration_seconds = 10
fps = 30  # Frames per second

# --- !! KEY CHANGE FOR SLOWNESS !! ---
# This factor now controls how much zoom is achieved *by the end* of the 10s video.
# A smaller value means less zoom overall = slower pace.
# Try values like 1.1, 1.2, 1.5, etc.
# 1.0 would mean no zoom at all.
# 2.0 means it zooms to 2x by the end of 10 seconds.
end_zoom_factor = 1.2 # Example: Very subtle zoom to 1.2x over 10 seconds

# --- End Configuration ---

# 1. Check if input image exists
if not os.path.exists(image_path):
    print(f"Error: Input image not found at {image_path}")
    exit()

# 2. Load the input image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image file: {image_path}")
    exit()

# 3. Get image dimensions (height, width)
h, w = img.shape[:2]
print(f"Input image dimensions: Width={w}, Height={h}")

# 4. Setup Video Writer
frame_size = (w, h)  # Output video dimensions will match input image
total_frames = int(duration_seconds * fps)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 file
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

if not out.isOpened():
    print(f"Error: Could not open video writer for path: {output_video_path}")
    exit()

print(f"Generating SLOW zoom video: {output_video_path} ({duration_seconds}s @ {fps}fps)")
print(f"Target zoom factor at the end of {duration_seconds}s: {end_zoom_factor:.2f}x")
print(f"Total frames to generate: {total_frames}")

try:
    # 5. Generate frames with slow zoom animation
    # Create an array of scale factors from 1.0 (no zoom) down to 1.0 / end_zoom_factor
    # Since end_zoom_factor is small (e.g., 1.2), 1.0 / end_zoom_factor is close to 1.0 (e.g., 0.833)
    # This means the scale changes very little over the total_frames.
    start_scale = 1.0
    end_scale = 1.0 / end_zoom_factor
    scale_values = np.linspace(start_scale, end_scale, total_frames)

    # Keep track of the last successfully generated frame for fallback
    last_good_frame = None

    for i in range(total_frames):
        current_scale = scale_values[i]

        # Calculate the dimensions of the crop window
        crop_w = int(w * current_scale)
        crop_h = int(h * current_scale)

        # Ensure dimensions are at least 1 pixel
        crop_w = max(1, crop_w)
        crop_h = max(1, crop_h)

        # Calculate the top-left corner of the crop window (centered)
        center_x, center_y = w // 2, h // 2
        x1 = center_x - crop_w // 2
        y1 = center_y - crop_h // 2

        # Calculate bottom-right corner
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # --- Boundary checks (important!) ---
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2) # Use w, h as exclusive upper bounds for slicing
        y2 = min(h, y2)

        # Adjust crop dimensions based on boundary checks if needed
        final_crop_w = x2 - x1
        final_crop_h = y2 - y1

        # Ensure the final crop dimensions are valid before slicing
        if final_crop_w <= 0 or final_crop_h <= 0:
            print(f"Warning: Calculated crop dimensions invalid at frame {i} (scale={current_scale:.4f}).")
            # If we have a previous good frame, use it. Otherwise, use the original resized.
            if last_good_frame is not None:
                 print("Using last valid frame.")
                 zoomed_frame = last_good_frame
            else:
                 print("Using full image as fallback.")
                 zoomed_frame = cv2.resize(img, frame_size, interpolation=cv2.INTER_LINEAR)
                 last_good_frame = zoomed_frame # Store this as the first "good" frame
        else:
            # Crop the image
            cropped_region = img[y1:y2, x1:x2]

            # Resize the cropped region back to the original frame size
            # Use INTER_LINEAR or INTER_CUBIC for better resize quality
            zoomed_frame = cv2.resize(cropped_region, frame_size, interpolation=cv2.INTER_LINEAR)
            last_good_frame = zoomed_frame # Update last good frame

        # Write the frame to the video
        out.write(zoomed_frame)

        # Optional: Print progress less frequently for slow renders
        if (i + 1) % (fps * 2) == 0: # Print every 2 seconds
             print(f"Processed frame {i + 1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
             # Optional: Add a tiny sleep if processing is too fast for the system
             # time.sleep(0.001)


finally:
    # 6. Release the VideoWriter
    print("Releasing video writer...")
    out.release()

print(f"\nVideo saved successfully to: {output_video_path}")
