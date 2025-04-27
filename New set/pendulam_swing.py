import cv2
import numpy as np
import os
import math
import time

# --- Configuration ---

# Paths (Hardcoded for specific Android/Termux location)
image_folder = '/storage/emulated/0/Download/' # Input and Output Folder
img_name = '1.png' # Specific image name in that folder
img_path = os.path.join(image_folder, img_name)

# Video Output Configuration
output_filename = 'pendulum_swing_output.mp4' # Name for the output video file
# Ensure output_path uses the same image_folder
output_path = os.path.join(image_folder, output_filename)

fps = 30.0           # Frames per second
total_duration_sec = 2.5 # Duration of the animation

# --- Animation Parameters ---
# Final Resting Position (Top-left corner of the image relative to canvas top-left)
final_x = 50
final_y = 50

# Pendulum Pivot Point (Relative to top-left of the canvas)
# Swing from the top-center
pivot_x_offset = 0 # Offset from canvas center horizontally
pivot_y_abs = -20   # Absolute Y position (negative means above the canvas top edge)

# Pendulum Physics Simulation
initial_angle_degrees = 80  # Starting swing angle (from vertical down)
num_swings = 3.0          # Total back-and-forth oscillations desired
damping_factor = 5.0      # How quickly the swing decays (higher = faster decay)
                          # A value around 5-7 usually looks decent for ~2-3 sec duration

# Optional: Image Tilt (Subtle effect)
max_tilt_degrees = 10     # Maximum tilt angle of the image during the swing
tilt_lag_factor = 0.8     # How much the tilt lags behind the pendulum angle (0=none, 1=matches angle phase)

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
    print("Please ensure '1.png' exists in the '/storage/emulated/0/Download/' folder.")
    exit()

img_orig_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if img_orig_raw is None:
    print(f"Error: Failed to load {img_path}. Check file integrity and permissions."); exit()
print("Image loaded successfully.")

# --- Determine Canvas and Image Dimensions ---
img_h_orig, img_w_orig = img_orig_raw.shape[:2]
# Make canvas slightly larger than image to accommodate swing if needed
canvas_extra_margin = 50 # Add some padding around the image's final area
canvas_h = img_h_orig + final_y + canvas_extra_margin
canvas_w = img_w_orig + final_x + canvas_extra_margin
print(f"Image Dimensions (HxW): {img_h_orig}x{img_w_orig}")
print(f"Canvas Dimensions (HxW): {canvas_h}x{canvas_w}")

# --- Handle Transparency ---
def get_image_components(img_to_process):
    """Separates image into BGR and 3-channel float32 alpha mask."""
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

try:
    img_bgr_orig, img_alpha_mask_orig = get_image_components(img_orig_raw)
except Exception as e:
    print(f"Error processing image components: {e}"); exit()

# --- Calculate Frame Counts & Pendulum Constants ---
total_frames = max(1, int(fps * total_duration_sec))
if total_duration_sec <= 0: total_duration_sec = 1/fps # Avoid division by zero

# Image center relative to its top-left corner
image_center_x_rel = img_w_orig / 2.0
image_center_y_rel = img_h_orig / 2.0

# Final center position of the image on the canvas
final_center_x = final_x + image_center_x_rel
final_center_y = final_y + image_center_y_rel

# Pivot point on the canvas
pivot_x = canvas_w / 2.0 + pivot_x_offset
pivot_y = pivot_y_abs # Use the absolute value provided

# Calculate pendulum arm length (distance from pivot to final center)
pendulum_length = math.hypot(final_center_x - pivot_x, final_center_y - pivot_y)
if pendulum_length < 1:
    print("Warning: Pendulum length is very small. Pivot and final position might be too close.")
    pendulum_length = max(1, pendulum_length) # Avoid division by zero later

# Physics constants for damped oscillation
initial_angle_radians = math.radians(initial_angle_degrees)
# Ensure non-zero duration for frequency/damping calculation
safe_duration = max(total_duration_sec, 1/fps)
angular_frequency = num_swings * 2 * math.pi / safe_duration
damping_coefficient = damping_factor / safe_duration

print(f"Total Duration: {total_duration_sec:.2f}s => {total_frames} total frames.")
print(f"Pendulum Pivot: ({pivot_x:.1f}, {pivot_y:.1f})")
print(f"Final Image Center: ({final_center_x:.1f}, {final_center_y:.1f})")
print(f"Pendulum Length: {pendulum_length:.1f}")
print(f"Initial Angle: {initial_angle_degrees:.1f} deg")
print(f"Damping Coefficient: {damping_coefficient:.2f}")
print(f"Angular Frequency: {angular_frequency:.2f} rad/s")


# --- Initialize Video Writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for .mp4
frame_size_writer = (canvas_w, canvas_h) # (width, height)
try:
    # Ensure the output directory exists (though we checked image_folder already)
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
    time_t = frame_num / fps
    progress = frame_num / (total_frames - 1) if total_frames > 1 else 1.0

    # --- Calculate Pendulum State ---
    if frame_num == total_frames - 1 and total_frames > 1:
        # *** FORCE FINAL FRAME STATE ***
        current_angle_pendulum = 0.0 # Ends vertical
        image_tilt_angle = 0.0       # No tilt at the end
        # Calculate position based on zero angle (should match final_center)
        current_center_x = pivot_x + pendulum_length * math.sin(current_angle_pendulum)
        current_center_y = pivot_y + pendulum_length * math.cos(current_angle_pendulum)
        # Force final position exactly to avoid rounding errors
        pos_x = final_x
        pos_y = final_y
        phase_desc = "Final Frame"

    else:
        # --- Calculate state for intermediate frames ---
        phase_desc = "Swinging"
        # Damped oscillation formula for angle (relative to vertical down)
        decay_factor = math.exp(-damping_coefficient * time_t)
        current_angle_pendulum = initial_angle_radians * decay_factor * math.cos(angular_frequency * time_t)

        # Calculate the image center's position based on the pendulum angle
        current_center_x = pivot_x + pendulum_length * math.sin(current_angle_pendulum)
        current_center_y = pivot_y + pendulum_length * math.cos(current_angle_pendulum)

        # Calculate Image Top-Left Position
        pos_x = int(round(current_center_x - image_center_x_rel))
        pos_y = int(round(current_center_y - image_center_y_rel))

        # Calculate Optional Image Tilt
        if max_tilt_degrees > 0:
            tilt_phase = angular_frequency * time_t - math.pi * tilt_lag_factor # Add lag
            # Tilt based on the cosine of the lagged phase, scaled by decay and max tilt
            # Use angle sign to keep tilt direction intuitive with swing direction
            tilt_direction_sign = -1 if current_angle_pendulum < 0 else 1
            image_tilt_angle = max_tilt_degrees * decay_factor * math.cos(tilt_phase) * tilt_direction_sign
        else:
            image_tilt_angle = 0.0


    # --- Perform Rotation (Tilt) ---
    apply_rotation = abs(image_tilt_angle) > 0.01 # Check if rotation is significant

    if apply_rotation:
        try:
            # Center of rotation is the image's own center
            rot_mat = cv2.getRotationMatrix2D(center=(image_center_x_rel, image_center_y_rel),
                                              angle=image_tilt_angle, # OpenCV uses degrees, positive=anti-clockwise
                                              scale=1.0)

            # Rotate BGR channels
            rotated_bgr = cv2.warpAffine(img_bgr_orig, rot_mat, (img_w_orig, img_h_orig),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)) # Fill border with black

            # Rotate Alpha mask separately
            alpha_ch = img_alpha_mask_orig[:,:,0] # Take one channel from the 3-channel mask
            rotated_alpha_single = cv2.warpAffine(alpha_ch, rot_mat, (img_w_orig, img_h_orig),
                                                  flags=cv2.INTER_LINEAR,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0) # Fill border with transparent
            rotated_alpha = cv2.merge([rotated_alpha_single] * 3) # Merge back to 3 channels for blending

        except cv2.error as warp_err:
             print(f"\nError during warpAffine frame {frame_num}: {warp_err}")
             rotated_bgr = img_bgr_orig.copy() # Fallback to unrotated copies
             rotated_alpha = img_alpha_mask_orig.copy()
             image_tilt_angle = 0.0 # Reset tilt if rotation failed
    else:
        # No rotation needed for this frame
        rotated_bgr = img_bgr_orig # Use original directly
        rotated_alpha = img_alpha_mask_orig
        image_tilt_angle = 0.0 # Ensure it's zero if not applied

    # Dimensions of the (potentially rotated) image are still the original
    current_h, current_w = img_h_orig, img_w_orig

    # --- Drawing the Image (Robust slicing logic) ---
    # Calculate the slice of the canvas to draw onto
    y_start_canvas = max(0, pos_y)
    y_end_canvas = min(canvas_h, pos_y + current_h)
    x_start_canvas = max(0, pos_x)
    x_end_canvas = min(canvas_w, pos_x + current_w)

    # Calculate the corresponding slice of the (rotated) image to use
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
            try:
                # Extract the region of interest (ROI) from the canvas
                roi = canvas[y_start_canvas : y_end_canvas, x_start_canvas : x_end_canvas]

                # Extract the corresponding slice from the rotated image and alpha mask
                img_slice = rotated_bgr[y_start_img : y_end_img, x_start_img : x_end_img]
                mask_slice = rotated_alpha[y_start_img : y_end_img, x_start_img : x_end_img]

                # Blend the image slice onto the ROI using the alpha mask
                inv_alpha_mask = 1.0 - mask_slice
                # Ensure consistent types for blending
                img_slice_float = img_slice.astype(np.float32)
                roi_float = roi.astype(np.float32)

                # Manual blending: (foreground * alpha) + (background * (1 - alpha))
                blended_roi_float = (img_slice_float * mask_slice) + (roi_float * inv_alpha_mask)

                # Place the blended ROI back onto the canvas, clipping and converting type
                canvas[y_start_canvas:y_end_canvas, x_start_canvas:x_end_canvas] = np.clip(blended_roi_float, 0, 255).astype(np.uint8)

            except Exception as e:
                # Print more detailed error info if slicing/blending fails
                print(f"\nError during blending/slicing frame {frame_num} ({phase_desc}): {e}")
                print(f"  Canvas ROI: y[{y_start_canvas}:{y_end_canvas}], x[{x_start_canvas}:{x_end_canvas}] (Shape: {roi.shape})")
                print(f"  Image Slice: y[{y_start_img}:{y_end_img}], x[{x_start_img}:{x_end_img}]")
                if 'img_slice' in locals(): print(f"  img_slice shape: {img_slice.shape}")
                if 'mask_slice' in locals(): print(f"  mask_slice shape: {mask_slice.shape}")
                print(f"  Rotated BGR Shape: {rotated_bgr.shape}, Rotated Alpha Shape: {rotated_alpha.shape}")
                # Continue to next frame if possible
        else:
             # Dimension mismatch case - do nothing
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
        angle_deg_display = math.degrees(current_angle_pendulum)
        # Use U+00B0 for degree symbol if terminal supports UTF-8
        print(f"\rProcessed frame {frame_num + 1}/{total_frames} ({percent_done:.1f}%) {phase_desc} Ang:{angle_deg_display:.1f}\xb0 Tilt:{image_tilt_angle:.1f}\xb0 Pos:({pos_x},{pos_y}) [{elapsed:.2f}s]", end="")


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
