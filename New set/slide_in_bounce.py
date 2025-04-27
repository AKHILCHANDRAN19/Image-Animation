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
output_filename = 'slide_in_bounce_full_image.mp4' # Descriptive filename
output_path = os.path.join(image_folder, output_filename)
fps = 30.0           # Frames per second
total_duration_sec = 2.5 # Target total duration

# --- Animation Parameters ---
# Slide Direction ('left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top', 'topleft_diagonal', 'topright_diagonal', etc.)
slide_direction = 'left_to_right'

# Final Resting Position (Top-left corner of the image relative to canvas origin 0,0)
# For full image animation, often this is (0, 0) if canvas=image size
final_x = 0
final_y = 0

# Bounce Parameters
bounce_pixels = 30      # How many pixels to overshoot the final position during the bounce
num_bounces = 1.5       # How many full oscillations in the bounce-back phase (can be float)
bounce_height_factor = 0.6 # Multiplier for bounce_pixels during squash calc (adjusts intensity)
damping_factor = 5.0     # How quickly the bounce dampens out (higher = faster stop)
squash_factor_max = 0.85 # How much it squashes perpendicular to bounce axis (e.g., 0.85 = 85% size)

# --- Timing Allocation ---
# Divide the total duration between slide and bounce phases
slide_duration_sec = total_duration_sec * 0.6  # e.g., 60% of time for initial slide
bounce_duration_sec = total_duration_sec * 0.4 # e.g., 40% of time for bounce back

# --- End Configuration ---

# --- Load Image ---
print(f"Loading image from: {img_path}")
img_orig_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED) # Load potentially with alpha

if img_orig_raw is None:
    print(f"Error: Failed to load {img_path}. Check path and permissions."); exit()
print("Image loaded successfully.")

# --- Determine Canvas and Image Dimensions ---
img_h_orig, img_w_orig = img_orig_raw.shape[:2]
canvas_h, canvas_w = img_h_orig, img_w_orig # Canvas matches image size for full image animation
print(f"Image/Canvas Dimensions (HxW): {canvas_h}x{canvas_w}")

# --- Handle Transparency (Copied from your sample) ---
def get_image_components(img_to_process):
    h, w = img_to_process.shape[:2]
    # Check if image has 3 dimensions and 4 channels (BGRA)
    if len(img_to_process.shape) == 3 and img_to_process.shape[2] == 4:
        print("Detected BGRA image.")
        bgr = img_to_process[:, :, 0:3].copy() # Explicit copy
        # Normalize alpha channel to 0.0-1.0
        alpha_norm = img_to_process[:, :, 3].astype(np.float32) / 255.0
        # Create a 3-channel mask by merging the normalized alpha
        alpha_mask = cv2.merge([alpha_norm] * 3)
        return bgr, alpha_mask
    # Check if image has 3 dimensions (implicitly BGR)
    elif len(img_to_process.shape) == 3:
        print("Detected BGR image (no alpha). Assuming opaque.")
        bgr = img_to_process.copy() # Explicit copy
        # Create a fully opaque alpha mask (all ones)
        alpha_mask = np.ones((h, w, 3), dtype=np.float32)
        return bgr, alpha_mask
    # Check if image has 2 dimensions (grayscale)
    elif len(img_to_process.shape) == 2:
         print("Detected Grayscale image. Converting to BGR, assuming opaque.")
         # Convert grayscale to BGR (creates a new 3-channel image)
         bgr = cv2.cvtColor(img_to_process, cv2.COLOR_GRAY2BGR)
         # Create a fully opaque alpha mask
         alpha_mask = np.ones((h, w, 3), dtype=np.float32)
         return bgr, alpha_mask
    else:
        # Handle unexpected image shapes
        raise ValueError(f"Error: Unexpected image shape: {img_to_process.shape}")


# Get original components ONCE for resizing later
try:
    img_bgr_orig, img_alpha_mask_orig = get_image_components(img_orig_raw)
except Exception as e:
    print(f"Error processing image components: {e}")
    exit()


# --- Determine Start, Overshoot Coordinates based on direction ---
start_x, start_y = 0, 0
overshoot_x, overshoot_y = 0, 0
bounce_axis = 'none' # To determine squash direction

# --- Define slide start/overshoot --- (Using full image dimensions)
if slide_direction == 'left_to_right':
    start_x = -img_w_orig # Start completely off-screen left
    start_y = final_y
    overshoot_x = final_x + bounce_pixels
    overshoot_y = final_y
    bounce_axis = 'horizontal'
elif slide_direction == 'right_to_left':
    start_x = canvas_w # Start completely off-screen right
    start_y = final_y
    overshoot_x = final_x - bounce_pixels
    overshoot_y = final_y
    bounce_axis = 'horizontal'
elif slide_direction == 'top_to_bottom':
    start_x = final_x
    start_y = -img_h_orig # Start completely off-screen top
    overshoot_x = final_x
    overshoot_y = final_y + bounce_pixels
    bounce_axis = 'vertical'
elif slide_direction == 'bottom_to_top':
    start_x = final_x
    start_y = canvas_h # Start completely off-screen bottom
    overshoot_x = final_x
    overshoot_y = final_y - bounce_pixels
    bounce_axis = 'vertical'
# Add diagonal examples
elif slide_direction == 'topleft_diagonal':
    start_x = -img_w_orig
    start_y = -img_h_orig
    delta_x = final_x - start_x
    delta_y = final_y - start_y
    magnitude = np.sqrt(delta_x**2 + delta_y**2)
    norm_x = delta_x / magnitude if magnitude > 0 else 0
    norm_y = delta_y / magnitude if magnitude > 0 else 0
    overshoot_x = final_x + int(norm_x * bounce_pixels)
    overshoot_y = final_y + int(norm_y * bounce_pixels)
    bounce_axis = 'diagonal' # Squash might look odd, but possible
elif slide_direction == 'topright_diagonal':
    start_x = canvas_w
    start_y = -img_h_orig
    delta_x = final_x - start_x
    delta_y = final_y - start_y
    magnitude = np.sqrt(delta_x**2 + delta_y**2)
    norm_x = delta_x / magnitude if magnitude > 0 else 0
    norm_y = delta_y / magnitude if magnitude > 0 else 0
    overshoot_x = final_x + int(norm_x * bounce_pixels)
    overshoot_y = final_y + int(norm_y * bounce_pixels)
    bounce_axis = 'diagonal'
else:
    print(f"Error: Invalid slide_direction '{slide_direction}'. Defaulting to left_to_right.")
    slide_direction = 'left_to_right'
    start_x = -img_w_orig; start_y = final_y
    overshoot_x = final_x + bounce_pixels; overshoot_y = final_y
    bounce_axis = 'horizontal'


# --- Calculate Frame Counts ---
num_slide_frames = max(1, int(fps * slide_duration_sec))
num_bounce_frames = max(1, int(fps * bounce_duration_sec))
total_frames = num_slide_frames + num_bounce_frames
print(f"Total Duration: {total_duration_sec:.2f}s => {num_slide_frames} slide frames + {num_bounce_frames} bounce frames = {total_frames} total frames.")

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
background = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8) # Black background for canvas
start_time = time.time()

# Cache original center point for positioning deformed image
orig_center_x = final_x + img_w_orig // 2
orig_center_y = final_y + img_h_orig // 2

for frame_num in range(total_frames):
    canvas = background.copy()

    # Defaults for the current frame
    current_img_bgr = img_bgr_orig
    current_img_alpha_mask = img_alpha_mask_orig
    current_h, current_w = img_h_orig, img_w_orig
    pos_x = final_x # Default to final position unless changed below
    pos_y = final_y

    # --- Phase Logic ---
    if frame_num < num_slide_frames:
        # Phase 1: Sliding In (No deformation yet)
        phase = "Sliding"
        slide_progress = frame_num / (num_slide_frames - 1) if num_slide_frames > 1 else 1.0
        # Linear interpolation from start to overshoot
        pos_x = int(start_x + (overshoot_x - start_x) * slide_progress)
        pos_y = int(start_y + (overshoot_y - start_y) * slide_progress)
        img_to_draw_bgr = img_bgr_orig
        img_to_draw_alpha = img_alpha_mask_orig
        # Dimensions are original

    else:
        # Phase 2: Bouncing Back & Squashing/Stretching
        phase = "Bouncing"
        bounce_frame_num = frame_num - num_slide_frames
        bounce_progress = bounce_frame_num / (num_bounce_frames - 1) if num_bounce_frames > 1 else 1.0

        # Calculate bounce offset using a dampened sine wave (relative to FINAL position)
        angle = bounce_progress * num_bounces * math.pi
        dampening = math.exp(-damping_factor * bounce_progress)
        # Use bounce_pixels as the amplitude of the oscillation
        current_bounce_offset = int(bounce_pixels * math.sin(angle) * dampening) # Signed offset

        # Calculate current position based on bounce axis
        # We oscillate *around* the final position
        pos_x = final_x
        pos_y = final_y
        if bounce_axis == 'horizontal':
            pos_x = final_x + current_bounce_offset
        elif bounce_axis == 'vertical':
            pos_y = final_y + current_bounce_offset
        elif bounce_axis == 'diagonal': # Apply offset along the overshoot vector direction
            delta_overshoot_x = overshoot_x - final_x
            delta_overshoot_y = overshoot_y - final_y
            magnitude_overshoot = np.sqrt(delta_overshoot_x**2 + delta_overshoot_y**2)
            norm_ox = delta_overshoot_x / magnitude_overshoot if magnitude_overshoot > 0 else 0
            norm_oy = delta_overshoot_y / magnitude_overshoot if magnitude_overshoot > 0 else 0
            pos_x = final_x + int(norm_ox * current_bounce_offset)
            pos_y = final_y + int(norm_oy * current_bounce_offset)

        # --- Calculate squash/stretch ---
        # Squash is max when bounce offset is near zero (impact/rebound)
        # Use cos wave (1 at peaks, 0 at zero crossing) -> use 1-cos
        effective_bounce_amp = bounce_pixels * bounce_height_factor # Scale max squash effect
        squash_lerp = (1.0 - abs(math.cos(angle))) * dampening * effective_bounce_amp / bounce_pixels if bounce_pixels > 0 else 0.0
        squash_lerp = np.clip(squash_lerp, 0.0, 1.0) # Ensure it's between 0 and 1


        # Determine primary squash factor based on bounce axis
        primary_squash_factor = 1.0 - (1.0 - squash_factor_max) * squash_lerp
        secondary_squash_factor = 1.0 / primary_squash_factor if primary_squash_factor > 0.01 else 1.0

        if bounce_axis == 'horizontal':
            current_w_factor = primary_squash_factor   # Squash width
            current_h_factor = secondary_squash_factor # Stretch height
        elif bounce_axis == 'vertical':
            current_h_factor = primary_squash_factor   # Squash height
            current_w_factor = secondary_squash_factor # Stretch width
        else: # Diagonal or none - maybe just uniform scale? Or just don't squash? Let's do uniform for now.
             current_h_factor = primary_squash_factor
             current_w_factor = primary_squash_factor # Uniform squash for diagonal


        # Apply deformation and resize only if factors are significant
        if abs(current_h_factor - 1.0) > 0.005 or abs(current_w_factor - 1.0) > 0.005 :
            current_h = max(1, int(img_h_orig * current_h_factor))
            current_w = max(1, int(img_w_orig * current_w_factor))
            interpolation = cv2.INTER_LINEAR
            try:
                img_to_draw_bgr = cv2.resize(img_bgr_orig, (current_w, current_h), interpolation=interpolation)
                img_to_draw_alpha = cv2.resize(img_alpha_mask_orig, (current_w, current_h), interpolation=interpolation)
            except cv2.error as resize_err:
                 print(f"\nWarning: Resize error frame {frame_num}: {resize_err}. Target: {current_w}x{current_h}. Using original.")
                 current_h, current_w = img_h_orig, img_w_orig
                 img_to_draw_bgr = img_bgr_orig
                 img_to_draw_alpha = img_alpha_mask_orig

        else:
            # No significant deformation, use original size
            current_h, current_w = img_h_orig, img_w_orig
            img_to_draw_bgr = img_bgr_orig
            img_to_draw_alpha = img_alpha_mask_orig

        # Adjust top-left position to keep the *center* of the image aligned correctly during deformation
        # Current center should ideally match the calculated bounce position (pos_x, pos_y derived from final pos + offset)
        # Calculate where the center *would* be if we just placed the deformed image at (pos_x, pos_y)
        current_center_x = pos_x + current_w // 2
        current_center_y = pos_y + current_h // 2

        # Calculate the desired center based on the bounce logic applied to the original center
        desired_center_x = orig_center_x
        desired_center_y = orig_center_y
        if bounce_axis == 'horizontal':
             desired_center_x = orig_center_x + current_bounce_offset
        elif bounce_axis == 'vertical':
             desired_center_y = orig_center_y + current_bounce_offset
        elif bounce_axis == 'diagonal':
             desired_center_x = orig_center_x + int(norm_ox * current_bounce_offset)
             desired_center_y = orig_center_y + int(norm_oy * current_bounce_offset)


        # Adjust pos_x, pos_y to align the current center with the desired center
        pos_x += (desired_center_x - current_center_x)
        pos_y += (desired_center_y - current_center_y)


    # --- Drawing the Image (Using Robust Logic from Sample) ---
    # Calculate canvas ROI coordinates (clamped to canvas bounds)
    y_start_canvas = max(0, pos_y)
    y_end_canvas = min(canvas_h, pos_y + current_h)
    x_start_canvas = max(0, pos_x)
    x_end_canvas = min(canvas_w, pos_x + current_w)

    # Calculate corresponding image slice coordinates
    y_start_img = max(0, -pos_y) # How many pixels of the image top are off-canvas
    y_end_img = y_start_img + (y_end_canvas - y_start_canvas) # Match the height of the canvas ROI
    x_start_img = max(0, -pos_x) # How many pixels of the image left are off-canvas
    x_end_img = x_start_img + (x_end_canvas - x_start_canvas) # Match the width of the canvas ROI

    # Proceed only if there's a valid overlap area on both canvas and image
    if y_start_canvas < y_end_canvas and x_start_canvas < x_end_canvas and \
       y_start_img < y_end_img and x_start_img < x_end_img:

        h_roi = y_end_canvas - y_start_canvas
        w_roi = x_end_canvas - x_start_canvas
        h_img_slice = y_end_img - y_start_img
        w_img_slice = x_end_img - x_start_img

        # Ensure the calculated slice dimensions match the canvas ROI dimensions
        if h_roi == h_img_slice and w_roi == w_img_slice:
            try:
                # Extract the region of interest from the canvas
                roi = canvas[y_start_canvas : y_end_canvas, x_start_canvas : x_end_canvas]
                # Extract the corresponding slice from the (potentially resized) image BGR data
                img_slice = img_to_draw_bgr[y_start_img : y_end_img, x_start_img : x_end_img]
                # Extract the corresponding slice from the (potentially resized) alpha mask
                mask_slice = img_to_draw_alpha[y_start_img : y_end_img, x_start_img : x_end_img]

                # Calculate the inverse alpha mask (for background contribution)
                inv_alpha_mask = 1.0 - mask_slice

                # Blend the image slice and the canvas ROI using the alpha mask
                # Convert to float32 for accurate blending calculations
                blended_roi = cv2.addWeighted(img_slice.astype(np.float32) * mask_slice, 1.0,
                                              roi.astype(np.float32) * inv_alpha_mask, 1.0, 0.0)

                # Place the blended result back onto the canvas, converting back to uint8
                canvas[y_start_canvas:y_end_canvas, x_start_canvas:x_end_canvas] = np.clip(blended_roi, 0, 255).astype(np.uint8)

            except Exception as e:
                print(f"\nError during blending at frame {frame_num} ({phase}): {e}")
                # Optional: Add fallback (e.g., simple paste without blending) if needed
                pass # Continue to next frame
        else:
             # This case indicates a potential logic error in coordinate calculation
             print(f"\nWarning: Dimension mismatch frame {frame_num}. ROI:({h_roi}x{w_roi}), ImgSlice:({h_img_slice}x{w_img_slice}). Skipping draw.")

    # --- Write the frame ---
    try:
        video_writer.write(canvas)
    except Exception as e:
         print(f"\nError writing frame {frame_num} to video: {e}")
         break # Stop if writing fails

    # --- Progress Indicator ---
    if (frame_num + 1) % 5 == 0 or frame_num == total_frames - 1: # Update progress more often
        elapsed = time.time() - start_time
        percent_done = ((frame_num + 1) / total_frames) * 100
        print(f"\rProcessed frame {frame_num + 1}/{total_frames} ({percent_done:.1f}%) ({phase}) [{elapsed:.2f}s]", end="")


# --- Cleanup ---
video_writer.release()
# No cv2.destroyAllWindows() needed in Termux
end_time = time.time()
print("\n" + "-" * 30)
print("Animation finished.")
print(f"Video saved successfully to: {output_path}")
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print("-" * 30)
