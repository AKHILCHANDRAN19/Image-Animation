import cv2
import numpy as np
import os
import time

# --- Configuration ---
# Input image path
image_path = "/storage/emulated/0/Download/input.png" # <<< CHANGE THIS TO YOUR IMAGE PATH

# Output video path
output_dir = os.path.dirname(image_path)
output_filename = "output_unified_3D_zoom_swing_parallax.mp4" # <<< Changed filename
output_video_path = os.path.join(output_dir, output_filename)
os.makedirs(output_dir, exist_ok=True)

# Video Parameters
duration_seconds = 7
fps = 30
total_frames = int(duration_seconds * fps)

# --- 3D Zoom Parameters ---
start_camera_zoom = 1.0
end_camera_zoom = 1.4
zoom_parallax_intensity = 0.6

# --- Swing Motion Parameters ---
# Number of full back-and-forth swings during the video duration
num_swings = 3.0  # Can be non-integer for incomplete final swing

# Maximum horizontal distance the camera view shifts from center during a swing
max_swing_shift_x = 40 # Pixels - Increase for more pronounced horizontal swing

# Maximum vertical distance the camera view shifts from center during a swing
# Often smaller than horizontal for a natural feel
max_swing_shift_y = 15 # Pixels - Controls the up/down motion of the swing

# --- End Configuration ---

print("--- Unified 3D Zoom Parallax Effect (Warping with Swing) ---")
print(f"*** Simulating {num_swings} swings. Assumes bottom is foreground. ***")

# --- Load Input Image ---
img_original = cv2.imread(image_path, cv2.IMREAD_COLOR)

if img_original is None:
    print(f"Error: Could not load input image: {image_path}")
    exit(1)

h, w = img_original.shape[:2]
print(f"Input image dimensions: Width={w}, Height={h}")
output_frame_size = (w, h)

# --- Precompute Coordinate Grids (Optimization) ---
x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
x_coords_f = x_coords.astype(np.float32)
y_coords_f = y_coords.astype(np.float32)
center_x = w / 2.0
center_y = h / 2.0
dx = x_coords_f - center_x
dy = y_coords_f - center_y

# --- Setup Video Writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, float(fps), output_frame_size)

if not out.isOpened():
    print(f"Error: Could not open video writer for {output_video_path}")
    exit(1)

print("Starting frame generation using unified warping with swing...")
start_time = time.time()

# --- Generate Frames ---
for i in range(total_frames):
    progress = i / max(1, total_frames - 1) # 0.0 to 1.0

    # --- Calculate Current Camera State ---
    # 1. Overall Base Zoom Level
    current_camera_zoom = start_camera_zoom + (end_camera_zoom - start_camera_zoom) * progress

    # 2. Swinging Panning Position
    # The angle for the swing cycles multiple times based on num_swings
    swing_angle = progress * num_swings * 2 * np.pi

    # Calculate pan offsets using sine and cosine for smooth oscillation
    # Positive pan shifts the *sampling point* right/down, which moves the *view* left/up
    camera_pan_x = max_swing_shift_x * np.sin(swing_angle)
    # Using cosine for Y creates a slightly elliptical or figure-8 path relative to the center
    # depending on the phase. Using -cos makes it dip slightly as it swings sideways initially.
    # You can experiment: np.cos(swing_angle), np.sin(swing_angle + np.pi/2), etc.
    camera_pan_y = -max_swing_shift_y * np.cos(swing_angle) # Start dip


    # --- Calculate Per-Pixel Scaling Factor based on Depth Assumption (Y-position) ---
    normalized_y = y_coords_f / max(1, h - 1)
    zoom_amount = current_camera_zoom - 1.0
    parallax_zoom_boost = zoom_amount * normalized_y * zoom_parallax_intensity
    scale_factor_map = np.maximum(0.01, current_camera_zoom + parallax_zoom_boost)

    # --- Calculate Source Coordinates for cv2.remap ---
    # Incorporate both zoom (via scale_factor_map) and swing pan
    map_x = center_x + camera_pan_x + (dx / scale_factor_map)
    map_y = center_y + camera_pan_y + (dy / scale_factor_map)

    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # --- Apply Remapping ---
    final_frame = cv2.remap(
        img_original,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # --- Write Frame ---
    out.write(final_frame)

    # Print progress
    if (i + 1) % fps == 0 or (i + 1) == total_frames:
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total_frames - (i + 1)) if i+1 > 0 and total_frames > (i+1) else 0
        swing_x_disp = camera_pan_x # Get current displacement for display
        swing_y_disp = camera_pan_y
        print(f"Frame {i + 1}/{total_frames} | Zoom: {current_camera_zoom:.2f} | SwingX: {swing_x_disp:+.1f} | SwingY: {swing_y_disp:+.1f} | Time: {elapsed:.1f}s | ETA: {eta:.1f}s", end='\r')

# --- Cleanup ---
print("\n" + " " * 100) # Clear progress line
print("Releasing video writer...")
out.release()

end_time = time.time()
total_time = end_time - start_time
print(f"\nVideo generation finished in {total_time:.2f} seconds.")

if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
     print(f"Video saved successfully to: {output_video_path}")
     print(f"Effect: Unified Warping 3D Zoom + Swing (Zoom: {start_camera_zoom:.2f}->{end_camera_zoom:.2f}, Parallax: {zoom_parallax_intensity}, Swings: {num_swings})")
else:
     print(f"Error: Output video file '{output_video_path}' was NOT created or is empty.")

print("--- Script End ---")
