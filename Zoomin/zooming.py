import os
import glob
import numpy as np
from PIL import Image
from moviepy.video.VideoClip import VideoClip

def zoom_out_effect(image_path, duration=5, zoom_factor=1.5, fps=25):
    """
    Creates a MoviePy VideoClip that zooms out on the given image.

    :param image_path:  Full path to the source image file.
    :param duration:    Duration (in seconds) of the zoom-out animation.
    :param zoom_factor: How much larger the image starts relative to its original size.
                        e.g. 1.5 => 150% of the original dimensions at t=0.
    :param fps:         Frames per second for the generated clip.
    :return:            A MoviePy VideoClip that can be written to a video file.
    """
    # Load the image with Pillow
    original_image = Image.open(image_path)
    w, h = original_image.size
    
    # Convert the Pillow Image to a NumPy array for pixel manipulation
    img_np = np.array(original_image)

    def make_frame(t):
        # t goes from 0 to 'duration'
        progress = t / duration  # fraction of the animation from 0.0 to 1.0
        
        # Interpolate the zoom level from 'zoom_factor' down to 1.0
        current_zoom = zoom_factor - (zoom_factor - 1.0) * progress
        
        # Compute new scaled dimensions
        new_w = int(w * current_zoom)
        new_h = int(h * current_zoom)
        
        # Avoid zero or negative dimensions (edge case)
        if new_w <= 0 or new_h <= 0:
            new_w, new_h = w, h
        
        # Resize using a high-quality filter
        img_resized = Image.fromarray(img_np).resize((new_w, new_h), Image.LANCZOS)
        
        # Crop the center to the original size (w x h)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        right = left + w
        bottom = top + h
        img_cropped = img_resized.crop((left, top, right, bottom))
        
        return np.array(img_cropped)
    
    # Create a MoviePy VideoClip from the frame function,
    # then chain the new with_duration and with_fps methods.
    clip = VideoClip(make_frame, duration=duration).with_duration(duration).with_fps(fps)
    return clip

if __name__ == "__main__":
    # Adjust this path to your specific Download folder location
    downloads_folder = "/storage/emulated/0/Download"
    
    # Image file extensions to look for
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    
    # Gather all matching image paths from the Download folder
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(downloads_folder, ext)))
    
    if not image_files:
        print("No image files found in the Download folder.")
    else:
        print(f"Found {len(image_files)} images. Processing...")
    
    # Process each image: create a zoom-out animation and save to MP4
    for img_path in image_files:
        try:
            # Generate the zoom-out clip for each image
            clip = zoom_out_effect(
                image_path=img_path,
                duration=5,      # 5-second animation
                zoom_factor=1.5, # Start at 150% of original size
                fps=25
            )
            
            # Construct output filename (e.g. original_name_zoomout.mp4)
            base_name, _ = os.path.splitext(os.path.basename(img_path))
            output_video_path = os.path.join(downloads_folder, base_name + "_zoomout.mp4")
            
            # Write the animation to a video file
            print(f"Creating video for: {img_path}")
            clip.write_videofile(output_video_path, codec="libx264")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print("All done!")
