import cv2
import numpy as np
import os
import time

# --- Configuration ---
# Input image path
image_path = "/storage/emulated/0/Download/input.png" # <<< CHANGE THIS TO YOUR IMAGE PATH

# Output video path
output_dir = os.path.dirname(image_path)
output_filename = "output_auto_parallax_basic.mp4" # <<< Changed filename
output_video_path = os.path.join(output_dir, output_filename)
os.makedirs(output_dir, exist_ok=True) # Create output dir if needed

# Video Parameters
# We'll use the input image dimensions
duration_seconds = 8 # Shorter duration might hide imperfections better
fps = 30
total_frames = int(duration_seconds * fps)

# Parallax Parameters
# How much each layer moves relative to the 'camera' movement.
# These factors determine the strength of the effect for each band.
background_factor = 0.1 # Top band moves least
middle_factor = 0.4 # Middle band moves moderately
foreground_factor = 0.9 # Bottom band moves most

# Define the virtual camera movement (e.g., horizontal pan + slight vertical)
# Max horizontal/vertical shift in pixels from the center
max_camera_shift_x = 50 # Adjust for intensity (pixels)
max_camera_shift_y = 15 # Adjust for intensity (pixels)

# Number of horizontal bands to create (3 is common: bg, mg, fg)
num_layers = 3

# --- End Configuration ---

print("--- BASIC Automatic Parallax Effect Video Generator ---")
print("*** WARNING: Using simple horizontal bands for layering. ***")
print("*** Results highly dependent on image content. May look unnatural. ***")

# --- Load Input Image ---
img_original = cv2.imread(image_path, cv2.IMREAD_COLOR) # Load as color

if img_original is None:
    print(f"Error: Could not load input image: {image_path}")
    exit(1)

h, w = img_original.shape[:2]
print(f"Input image dimensions: Width={w}, Height={h}")
frame_size = (w, h) # Output video will have same dimensions

# --- CRUDE Layer Separation (Horizontal Bands) ---
print(f"Separating image into {num_layers} horizontal bands...")
layers = [] # To store image data for each layer
layer_masks = [] # To store alpha masks for each layer
band_height = h // num_layers

# Convert original image to BGRA (adding an alpha channel)
# This is needed for the overlay function later. We'll make layers fully opaque initially.
img_original_bgra = cv2.cvtColor(img_original, cv2.COLOR_BGR2BGRA)

for i in range(num_layers):
    # Create a mask for the current band
    mask = np.zeros((h, w), dtype=np.uint8)
    start_y = i * band_height
    # Ensure the last band goes all the way to the bottom edge
    end_y = (i + 1) * band_height if i < num_layers - 1 else h
    mask[start_y:end_y, :] = 255 # Make the band area white in the mask
    layer_masks.append(mask)

    # Create the layer image: Black background with the band pixels copied
    # We need 4 channels (BGRA) for alpha compositing later
    layer_img_bgra = np.zeros((h, w, 4), dtype=np.uint8)

    # Copy pixels from original BGRA image where the mask is white
    # The mask needs to be broadcastable to 4 channels or used correctly
    # Use np.where or bitwise_and (less intuitive with alpha maybe)
    # Simpler: copy where mask is 255
    layer_img_bgra[mask == 255] = img_original_bgra[mask == 255]

    # Set the alpha channel of the layer based on the mask
    # Where the mask is 255, alpha is 255 (opaque). Where mask is 0, alpha is 0 (transparent).
    layer_img_bgra[:, :, 3] = mask

    layers.append(layer_img_bgra)

# Assign parallax factors (adjust if num_layers != 3)
# This example assumes 3 layers: bg, mg, fg
parallax_factors = [background_factor, middle_factor, foreground_factor]
if len(layers) != len(parallax_factors):
    print(f"Warning: Number of layers ({len(layers)}) doesn't match defined factors ({len(parallax_factors)}). Using linearly spaced factors.")
    # Create evenly spaced factors from min to max
    min_factor = 0.1
    max_factor = 0.9
    parallax_factors = np.linspace(min_factor, max_factor, num_layers)


print("Layer separation complete (using basic bands).")

# --- Helper Function for Compositing with Alpha ---
# (Same function as before)
def overlay_image_alpha(background, overlay, x, y):
    """ Overlays overlay image with alpha channel onto background at (x, y) """
    b_h, b_w = background.shape[:2]
    # Overlay must have 4 channels (BGRA)
    if overlay.shape[2] != 4:
        print("Error: Overlay image must have 4 channels (BGRA) for alpha blending.")
        return background # Return original background on error

    o_h, o_w = overlay.shape[:2]

    # Get alpha mask and RGB part
    alpha = overlay[:, :, 3] / 255.0 # Normalize alpha channel (0.0 - 1.0)
    overlay_rgb = overlay[:, :, :3]

    # Calculate boundaries for overlaying, handling off-screen placement
    y1, y2 = max(0, y), min(b_h, y + o_h)
    x1, x2 = max(0, x), min(b_w, x + o_w)

    # Corresponding region in the overlay image needs to be calculated
    o_y1, o_y2 = max(0, -y), min(o_h, b_h - y)
    o_x1, o_x2 = max(0, -x), min(o_w, b_w - x)

    # Check if there is any actual overlap to blend
    if y1 >= y2 or x1 >= x2 or o_y1 >= o_y2 or o_x1 >= o_x2:
        return background # No overlap, return original background

    # Select the overlapping areas
    bg_area = background[y1:y2, x1:x2]
    overlay_area = overlay_rgb[o_y1:o_y2, o_x1:o_x2]
    alpha_mask = alpha[o_y1:o_y2, o_x1:o_x2, np.newaxis] # Add channel dim for broadcasting

    # Perform alpha blending: B = B*(1-alpha) + O*alpha
    try:
        composite = bg_area * (1.0 - alpha_mask) + overlay_area * alpha_mask
        background[y1:y2, x1:x2] = composite.astype(np.uint8)
    except ValueError as e:
         print(f"\nError during blending: {e}")
         print(f"BG area shape: {bg_area.shape}, Overlay area shape: {overlay_area.shape}, Alpha mask shape: {alpha_mask.shape}")
         # This might happen if calculation of overlap regions is incorrect
         # As a fallback, just return the background without blending this part
         return background

    return background


# --- Setup Video Writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' for good compatibility
out = cv2.VideoWriter(output_video_path, fourcc, float(fps), frame_size)

if not out.isOpened():
    print(f"Error: Could not open video writer for {output_video_path}")
    print("Check permissions and codec availability ('mp4v' is usually safe).")
    exit(1)

print("Starting frame generation...")
start_time = time.time()

# --- Generate Frames ---
for i in range(total_frames):
    progress = i / max(1, total_frames - 1) # Linear progress: 0.0 to 1.0

    # Calculate current camera 'position' (e.g., smooth sinusoidal motion)
    # Oscillates between -max_shift and +max_shift using sine/cosine
    camera_shift_x = max_camera_shift_x * np.sin(progress * 2 * np.pi)
    camera_shift_y = max_camera_shift_y * np.cos(progress * 2 * np.pi) # Add vertical motion


    # --- Create Canvas for the current frame ---
    # Start with a black background (3 channels BGR)
    final_frame = np.zeros((h, w, 3), dtype=np.uint8)

    # --- Composite Layers (Back to Front) ---
    # Layer 0 (top band) is assumed background, Layer N-1 (bottom band) is foreground
    for layer_index in range(num_layers):
        layer_img_bgra = layers[layer_index]
        factor = parallax_factors[layer_index]

        # Calculate shift for this layer (opposite to camera movement)
        layer_shift_x = int(-camera_shift_x * factor)
        layer_shift_y = int(-camera_shift_y * factor)

        # The 'overlay_image_alpha' function takes the top-left corner (x, y)
        # Since our layers are full-frame with transparency, the top-left is just the shift
        paste_x = layer_shift_x
        paste_y = layer_shift_y

        # Overlay the shifted layer onto the final frame
        final_frame = overlay_image_alpha(final_frame, layer_img_bgra, paste_x, paste_y)


    # --- Write Frame ---
    out.write(final_frame)

    # Print progress
    if (i + 1) % fps == 0 or (i + 1) == total_frames:
        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (total_frames - (i + 1)) if i+1 > 0 and total_frames > (i+1) else 0
        print(f"Frame {i + 1}/{total_frames} | Time: {elapsed:.1f}s | ETA: {eta:.1f}s", end='\r')


# --- Cleanup ---
print("\n" + " " * 80) # Clear the progress line
print("Releasing video writer...")
out.release()

end_time = time.time()
total_time = end_time - start_time
print(f"\nVideo generation finished in {total_time:.2f} seconds.")

if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
     print(f"Video saved successfully to: {output_video_path}")
     print("NOTE: Remember the layering was automatic and very basic (horizontal bands).")
else:
     print(f"Error: Output video file '{output_video_path}' was NOT created or is empty. Check for errors above.")

print("--- Script End ---")
