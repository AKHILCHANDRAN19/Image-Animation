import os
import glob
import numpy as np
from PIL import Image
from moviepy.video.VideoClip import VideoClip

def zoom_in_effect(image_path, duration=5, zoom_factor=1.5, fps=25):
    """
    Creates a MoviePy VideoClip that zooms in on the given image.
    The effect starts with the original image and gradually zooms into a larger version,
    cropping the center to maintain the original dimensions.
    
    :param image_path:  Full path to the source image file.
    :param duration:    Duration (in seconds) of the zoom-in animation.
    :param zoom_factor: Final zoom factor at the end of the animation (e.g., 1.5 means 150%).
    :param fps:         Frames per second for the generated clip.
    :return:            A MoviePy VideoClip.
    """
    # Load the image using Pillow
    original_image = Image.open(image_path)
    w, h = original_image.size
    # Convert the Pillow Image to a NumPy array for pixel manipulation
    img_np = np.array(original_image)

    def make_frame(t):
        # Calculate progress as a fraction of total duration (0.0 to 1.0)
        progress = t / duration
        # Start at 1.0 and increase to zoom_factor over time
        current_zoom = 1.0 + (zoom_factor - 1.0) * progress
        
        # Compute new scaled dimensions
        new_w = int(w * current_zoom)
        new_h = int(h * current_zoom)
        
        # Guard against non-positive dimensions (edge case)
        if new_w <= 0 or new_h <= 0:
            new_w, new_h = w, h
        
        # Resize the image using a high-quality filter
        img_resized = Image.fromarray(img_np).resize((new_w, new_h), Image.LANCZOS)
        
        # Crop the center back to the original size (w x h)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        right = left + w
        bottom = top + h
        img_cropped = img_resized.crop((left, top, right, bottom))
        
        # Return the cropped frame as a NumPy array
        return np.array(img_cropped)
    
    # Create a MoviePy VideoClip from the frame function,
    # then set the duration and frames per second using with_duration and with_fps.
    clip = VideoClip(make_frame, duration=duration).with_duration(duration).with_fps(fps)
    return clip

if __name__ == "__main__":
    # Full path to the Download folder (update if needed)
    downloads_folder = "/storage/emulated/0/Download"
    
    # Image file extensions to search for
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    
    # Collect all image files from the Download folder
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(downloads_folder, ext)))
    
    if not image_files:
        print("No image files found in the Download folder.")
    else:
        print(f"Found {len(image_files)} images. Processing...")
    
    # Process each image file: create a zoom-in animation and save as an MP4 in the Download folder
    for img_path in image_files:
        try:
            # Generate the zoom-in clip for each image
            clip = zoom_in_effect(
                image_path=img_path,
                duration=5,      # 5-second animation
                zoom_factor=1.5, # End at 150% of the original size
                fps=25
            )
            
            # Create an output filename (e.g., IMG-XXXX_zoomin.mp4)
            base_name, _ = os.path.splitext(os.path.basename(img_path))
            output_video_path = os.path.join(downloads_folder, base_name + "_zoomin.mp4")
            
            # Write the animated clip to a video file using libx264 codec
            print(f"Creating video for: {img_path}")
            clip.write_videofile(output_video_path, codec="libx264")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print("All done!")
