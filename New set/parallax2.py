import cv2
import numpy as np
import os
import time

# --- Configuration ---
# Input image path
image_path = "/storage/emulated/0/Download/input.png" # <<< CHANGE THIS TO YOUR IMAGE PATH

# Output video path
output_dir = os.path.dirname(image_path)
output_filename = "output_auto_parallax_feathered.mp4" # <<< Changed filename
output_video_path = os.path.join(output_dir, output_filename)
os.makedirs(output_dir, exist_ok=True)

# Video Parameters
# We'll use the input image dimensions unless final zoom crops it
duration_seconds = 10 # Increased duration slightly
fps = 30
total_frames = int(duration_seconds * fps)

# Parallax Parameters
background_factor = 0.05 # Top band moves least (less sensitive to feathering issues)
middle_factor = 0.3
foreground_factor = 0.7 # Bottom band moves most (might reduce factor slightly)
num_layers = 3 # Keep it simple, 3 is common: bg, mg, fg

# --- NEW: Feathering ---
# Height of the blend zone between layers (in pixels). Adjust based on image size/content.
feather_height = 40  # Start with ~3-5% of image height, e.g., 30-50px for 720p/1080p

# --- NEW: Subtle Zoom ---
# How much the overall scene zooms during the effect (1.0 = no zoom)
min_final_zoom = 0.98 # Slightly zoom out
max_final_zoom = 1.02 # Slightly zoom in

# Virtual Camera Movement
max_camera_shift_x = 60 # Pixels
max_camera_shift_y = 20 # Pixels

# --- End Configuration ---

print("--- IMPROVED Automatic Parallax Effect (Feathered Bands) ---")
print("*** WARNING: Uses feathered horizontal bands. Still an approximation. ***")

# --- Load Input Image ---
img_original = cv2.imread(image_path, cv2.IMREAD_COLOR)

if img_original is None:
    print(f"Error: Could not load input image: {image_path}")
    exit(1)

h, w = img_original.shape[:2]
print(f"Input image dimensions: Width={w}, Height={h}")
output_frame_size = (w, h) # The final output video dimensions

# --- Layer Separation with Feathering ---
print(f"Separating into {num_layers} feathered bands (feather={feather_height}px)...")
layers_bgra = [] # Store BGRA layer images
parallax_factors = np.linspace(background_factor, foreground_factor, num_layers) # Assign factors evenly
band_height = h // num_layers

# Ensure feather height isn't too large
if feather_height > band_height:
    print(f"Warning: Feather height ({feather_height}px) is larger than band height ({band_height}px). Reducing feather height.")
    feather_height = band_height // 2
if feather_height < 1: feather_height = 1 # Need at least 1 pixel

half_feather = feather_height // 2

# Convert original to BGRA *once*
img_original_bgra = cv2.cvtColor(img_original, cv2.COLOR_BGR2BGRA)
img_original_bgra[:, :, 3] = 255 # Ensure original alpha is fully opaque initially

for i in range(num_layers):
    # Create a full-size transparent canvas for the layer
    layer_img_bgra = np.zeros((h, w, 4), dtype=np.uint8)

    # Define the core y-range for this band
    core_start_y = i * band_height
    core_end_y = (i + 1) * band_height if i < num_layers - 1 else h

    # Define the actual pixel copy range (including potential feather overlap)
    copy_start_y = max(0, core_start_y - half_feather)
    copy_end_y = min(h, core_end_y + half_feather)

    # Copy the relevant horizontal slice from the original BGRA image
    layer_img_bgra[copy_start_y:copy_end_y, :] = img_original_bgra[copy_start_y:copy_end_y, :]

    # --- Create the Alpha Mask with Feathering ---
    alpha_mask = np.zeros((h, w), dtype=np.float32) # Use float for gradient

    # Top feather zone (fades in from top) - skip for the first layer (i=0)
    if i > 0:
        feather_start_y = core_start_y - half_feather
        feather_end_y = core_start_y + half_feather
        # Clamp to image bounds
        feather_start_y = max(0, feather_start_y)
        feather_end_y = min(h, feather_end_y)
        zone_h = feather_end_y - feather_start_y
        if zone_h > 0:
            gradient = np.linspace(0.0, 1.0, zone_h)[:, np.newaxis] # Vertical gradient 0->1
            alpha_mask[feather_start_y:feather_end_y, :] = np.maximum(alpha_mask[feather_start_y:feather_end_y, :], gradient)

    # Core opaque zone
    opaque_start_y = core_start_y + half_feather
    opaque_end_y = core_end_y - half_feather
    # Clamp to image bounds and ensure start < end
    opaque_start_y = max(0, opaque_start_y)
    opaque_end_y = min(h, opaque_end_y)
    if opaque_end_y > opaque_start_y:
        alpha_mask[opaque_start_y:opaque_end_y, :] = 1.0

    # Bottom feather zone (fades out to bottom) - skip for the last layer
    if i < num_layers - 1:
        feather_start_y = core_end_y - half_feather
        feather_end_y = core_end_y + half_feather
         # Clamp to image bounds
        feather_start_y = max(0, feather_start_y)
        feather_end_y = min(h, feather_end_y)
        zone_h = feather_end_y - feather_start_y
        if zone_h > 0:
            gradient = np.linspace(1.0, 0.0, zone_h)[:, np.newaxis] # Vertical gradient 1->0
            # Only apply if it doesn't overwrite the next layer's fade-in zone significantly
            # Use maximum to blend where zones might overlap slightly due to integer division
            alpha_mask[feather_start_y:feather_end_y, :] = np.maximum(alpha_mask[feather_start_y:feather_end_y, :], gradient)


    # Apply the float alpha mask (0.0-1.0) to the layer's alpha channel (0-255)
    layer_img_bgra[:, :, 3] = (alpha_mask * 255).astype(np.uint8)
    layers_bgra.append(layer_img_bgra)


print("Feathered layer separation complete.")

# --- Helper Function for Compositing (Same as before) ---
def overlay_image_alpha(background, overlay, x, y):
    """ Overlays overlay image (BGRA) onto background (BGR or BGRA) at (x, y) """
    b_h, b_w = background.shape[:2]
    if overlay.shape[2] != 4: return background # Safety check

    o_h, o_w = overlay.shape[:2]
    alpha = overlay[:, :, 3] / 255.0
    overlay_rgb = overlay[:, :, :3]

    y1, y2 = max(0, y), min(b_h, y + o_h)
    x1, x2 = max(0, x), min(b_w, x + o_w)
    o_y1, o_y2 = max(0, -y), min(o_h, b_h - y)
    o_x1, o_x2 = max(0, -x), min(o_w, b_w - x)

    if y1 >= y2 or x1 >= x2 or o_y1 >= o_y2 or o_x1 >= o_x2: return background

    # Ensure background is BGR for blending calculation
    if background.shape[2] == 4:
        bg_rgb = background[:, :, :3]
    else:
        bg_rgb = background

    bg_area = bg_rgb[y1:y2, x1:x2]
    overlay_area = overlay_rgb[o_y1:o_y2, o_x1:o_x2]
    alpha_mask = alpha[o_y1:o_y2, o_x1:o_x2, np.newaxis]

    try:
        composite = bg_area * (1.0 - alpha_mask) + overlay_area * alpha_mask
        # Place back into the original background (handling BGR or BGRA)
        if background.shape[2] == 4:
             background[y1:y2, x1:x2, :3] = composite.astype(np.uint8)
             # Also update alpha if background had alpha
             bg_alpha_area = background[y1:y2, x1:x2, 3] / 255.0
             new_alpha = bg_alpha_area * (1.0 - alpha_mask[...,0]) + alpha_mask[...,0]
             background[y1:y2, x1:x2, 3] = (new_alpha * 255).astype(np.uint8)
        else:
             background[y1:y2, x1:x2] = composite.astype(np.uint8)

    except ValueError as e:
         print(f"\nBlend Error: {e}, shapes: bg={bg_area.shape}, ov={overlay_area.shape}, alpha={alpha_mask.shape}")
         return background
    return background


# --- Setup Video Writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, float(fps), output_frame_size) # Use original size

if not out.isOpened():
    print(f"Error: Could not open video writer for {output_video_path}")
    exit(1)

print("Starting frame generation...")
start_time = time.time()

# --- Generate Frames ---
for i in range(total_frames):
    progress = i / max(1, total_frames - 1) # 0.0 to 1.0

    # Camera position using sine/cosine for smooth oscillation
    camera_shift_x = max_camera_shift_x * np.sin(progress * 2 * np.pi)
    camera_shift_y = max_camera_shift_y * np.cos(progress * 2 * np.pi) # Cosine for variation

    # Calculate current zoom level (oscillating)
    zoom_progress = (1 + np.sin(progress * 2 * np.pi)) / 2 # Oscillates 0 to 1
    current_zoom = min_final_zoom + (max_final_zoom - min_final_zoom) * zoom_progress

    # --- Create Canvas ---
    # Start with black BGR
    composed_frame = np.zeros((h, w, 3), dtype=np.uint8)

    # --- Composite Layers ---
    for layer_index in range(num_layers):
        layer_img = layers_bgra[layer_index]
        factor = parallax_factors[layer_index]

        layer_shift_x = int(-camera_shift_x * factor)
        layer_shift_y = int(-camera_shift_y * factor)

        paste_x = layer_shift_x
        paste_y = layer_shift_y

        composed_frame = overlay_image_alpha(composed_frame, layer_img, paste_x, paste_y)

    # --- Apply Final Zoom (Edge Handling) ---
    # Resize the composed frame according to the current zoom
    scaled_w = int(w * current_zoom)
    scaled_h = int(h * current_zoom)

    if scaled_w < 1 or scaled_h < 1: # Safety check
        final_frame_to_write = cv2.resize(composed_frame, output_frame_size, interpolation=cv2.INTER_LINEAR)
    else:
        scaled_frame = cv2.resize(composed_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        # Crop the scaled frame back to the original dimensions, centered
        crop_x = (scaled_w - w) // 2
        crop_y = (scaled_h - h) // 2

        # Ensure crop coordinates are valid
        crop_x1 = np.clip(crop_x, 0, scaled_w)
        crop_y1 = np.clip(crop_y, 0, scaled_h)
        crop_x2 = np.clip(crop_x + w, 0, scaled_w)
        crop_y2 = np.clip(crop_y + h, 0, scaled_h)

        paste_x1 = 0 if crop_x >= 0 else -crop_x
        paste_y1 = 0 if crop_y >= 0 else -crop_y
        paste_x2 = paste_x1 + (crop_x2 - crop_x1)
        paste_y2 = paste_y1 + (crop_y2 - crop_y1)

        final_frame_to_write = np.zeros((h, w, 3), dtype=np.uint8)
        if crop_x2 > crop_x1 and crop_y2 > crop_y1:
             try:
                 final_frame_to_write[paste_y1:paste_y2, paste_x1:paste_x2] = scaled_frame[crop_y1:crop_y2, crop_x1:crop_x2]
             except ValueError as e:
                  print(f"\nZoom Crop Error: {e}, trying resize fallback")
                  final_frame_to_write = cv2.resize(composed_frame, output_frame_size, interpolation=cv2.INTER_LINEAR)


    # --- Write Frame ---
    out.write(final_frame_to_write)

    # Print progress
    if (i + 1) % fps == 0 or (i + 1) == total_frames:
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total_frames - (i + 1)) if i+1 > 0 and total_frames > (i+1) else 0
        print(f"Frame {i + 1}/{total_frames} | Zoom: {current_zoom:.2f} | Time: {elapsed:.1f}s | ETA: {eta:.1f}s", end='\r')


# --- Cleanup ---
print("\n" + " " * 80) # Clear progress line
print("Releasing video writer...")
out.release()

end_time = time.time()
total_time = end_time - start_time
print(f"\nVideo generation finished in {total_time:.2f} seconds.")

if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
     print(f"Video saved successfully to: {output_video_path}")
     print(f"Used {num_layers} layers with feathering height {feather_height}px.")
else:
     print(f"Error: Output video file '{output_video_path}' was NOT created or is empty.")

print("--- Script End ---")
