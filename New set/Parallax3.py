import cv2
import numpy as np
import os
import time

# --- Configuration ---
# Input image path
image_path = "/storage/emulated/0/Download/input.png" # <<< CHANGE THIS TO YOUR IMAGE PATH

# Output video path
output_dir = os.path.dirname(image_path)
output_filename = "output_unified_3D_zoom_parallax.mp4" # <<< Changed filename
output_video_path = os.path.join(output_dir, output_filename)
os.makedirs(output_dir, exist_ok=True)

# Video Parameters
duration_seconds = 7
fps = 30
total_frames = int(duration_seconds * fps)

# --- 3D Zoom Parameters ---
# Overall zoom factor at the end (e.g., 1.0 = no zoom, 1.5 = 50% zoom in)
start_camera_zoom = 1.0
end_camera_zoom = 1.4 # Zoom IN effect (use < 1.0 for zoom OUT)

# Parallax intensity during zoom: How much *more* the foreground (bottom) scales than background (top)
# 0 = all parts scale identically (normal zoom), > 0 = parallax effect
# Values around 0.3 - 0.8 often work well. This controls the *differential* scaling.
zoom_parallax_intensity = 0.6

# --- Optional Simultaneous Pan ---
# Add a subtle pan while zooming for more dynamism
max_camera_shift_x = 25 # Pixels
max_camera_shift_y = 10 # Pixels

# --- End Configuration ---

print("--- Unified 3D Zoom Parallax Effect (Warping) ---")
print("*** WARNING: Uses image warping. Assumes bottom is foreground. ***")

# --- Load Input Image ---
img_original = cv2.imread(image_path, cv2.IMREAD_COLOR)

if img_original is None:
    print(f"Error: Could not load input image: {image_path}")
    exit(1)

h, w = img_original.shape[:2]
print(f"Input image dimensions: Width={w}, Height={h}")
output_frame_size = (w, h) # Final video output size

# --- Precompute Coordinate Grids (Optimization) ---
# Create grids of (x, y) coordinates for the output frame dimensions
# These represent the destination pixel coordinates
x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

# Convert to float32 for calculations
x_coords_f = x_coords.astype(np.float32)
y_coords_f = y_coords.astype(np.float32)

# Calculate center coordinates
center_x = w / 2.0
center_y = h / 2.0

# Calculate vectors from the center to each pixel
dx = x_coords_f - center_x
dy = y_coords_f - center_y

# --- Setup Video Writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, float(fps), output_frame_size)

if not out.isOpened():
    print(f"Error: Could not open video writer for {output_video_path}")
    exit(1)

print("Starting frame generation using unified warping...")
start_time = time.time()

# --- Generate Frames ---
for i in range(total_frames):
    progress = i / max(1, total_frames - 1) # 0.0 to 1.0

    # --- Calculate Current Camera State ---
    # 1. Overall Base Zoom Level
    current_camera_zoom = start_camera_zoom + (end_camera_zoom - start_camera_zoom) * progress

    # 2. Optional Panning Position (e.g., sinusoidal)
    # Positive pan moves the viewpoint right/down, so the image content moves left/up
    camera_pan_x = max_camera_shift_x * np.sin(progress * 2 * np.pi)
    camera_pan_y = max_camera_shift_y * np.cos(progress * 2 * np.pi) # Use cosine for variation

    # --- Calculate Per-Pixel Scaling Factor based on Depth Assumption (Y-position) ---
    # normalized_y ranges from 0 (top) to 1 (bottom)
    normalized_y = y_coords_f / max(1, h - 1) # Avoid division by zero if h=1

    # Calculate the amount of zoom being applied at this frame
    zoom_amount = current_camera_zoom - 1.0 # Will be 0 at start, positive for zoom in

    # Calculate the differential scaling boost based on y-position and intensity
    # Pixels at the bottom (normalized_y=1) get the full intensity boost
    # Pixels at the top (normalized_y=0) get no boost
    parallax_zoom_boost = zoom_amount * normalized_y * zoom_parallax_intensity

    # Final scale factor for each pixel: base zoom + parallax boost
    # Clamp the scale factor to prevent it from becoming zero or negative
    scale_factor_map = np.maximum(0.01, current_camera_zoom + parallax_zoom_boost) # Ensure scale > 0

    # --- Calculate Source Coordinates for cv2.remap ---
    # To zoom *in* by scale_factor, we need to divide the distance from the center
    # by the scale_factor when finding the source pixel.
    # We also incorporate the camera pan here. Panning the camera right (positive pan_x)
    # means we need to sample pixels from further right in the source image.
    map_x = center_x + camera_pan_x + (dx / scale_factor_map)
    map_y = center_y + camera_pan_y + (dy / scale_factor_map)

    # Ensure maps are float32 for cv2.remap
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # --- Apply Remapping ---
    # cv2.remap looks up the color for each pixel (x_coords, y_coords) in the output
    # frame from the calculated (map_x, map_y) coordinates in the input (img_original).
    # BORDER_REPLICATE duplicates edge pixels, often looks better for zoom than black borders.
    final_frame = cv2.remap(
        img_original,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR, # Linear interpolation is usually a good balance
        borderMode=cv2.BORDER_REPLICATE
        # borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0) # Use this for black borders
    )

    # --- Write Frame ---
    out.write(final_frame)

    # Print progress
    if (i + 1) % fps == 0 or (i + 1) == total_frames:
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total_frames - (i + 1)) if i+1 > 0 and total_frames > (i+1) else 0
        print(f"Frame {i + 1}/{total_frames} | CamZoom: {current_camera_zoom:.2f} | Time: {elapsed:.1f}s | ETA: {eta:.1f}s", end='\r')

# --- Cleanup ---
print("\n" + " " * 80) # Clear progress line
print("Releasing video writer...")
out.release()

end_time = time.time()
total_time = end_time - start_time
print(f"\nVideo generation finished in {total_time:.2f} seconds.")

if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
     print(f"Video saved successfully to: {output_video_path}")
     print(f"Effect: Unified Warping 3D Zoom (Base Zoom: {start_camera_zoom:.2f}->{end_camera_zoom:.2f}, Parallax Intensity: {zoom_parallax_intensity})")
else:
     print(f"Error: Output video file '{output_video_path}' was NOT created or is empty.")

print("--- Script End ---")
