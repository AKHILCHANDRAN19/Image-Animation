import os
import glob
import math
import numpy as np
from PIL import Image
from moviepy.video.VideoClip import VideoClip

def circular_motion_effect(image_path, duration=5, scale_factor=1.3, fps=25):
    """
    Creates a MoviePy VideoClip that applies a circular motion animation on the given image.
    
    The image is first scaled up by 'scale_factor' to provide extra margin.
    Then, a crop window of the original image size is moved along a circular path.
    The circular path is centered in the scaled image, with a radius determined such that
    the crop window always remains within the scaled image.
    
    :param image_path:   Full path to the source image file.
    :param duration:     Duration (in seconds) of the circular motion animation.
    :param scale_factor: Factor by which the image is scaled (>= 1) to allow room for motion.
    :param fps:          Frames per second for the generated clip.
    :return:             A MoviePy VideoClip with the circular motion animation.
    """
    # Load the original image using Pillow
    original_image = Image.open(image_path)
    w, h = original_image.size

    # Compute new dimensions based on the scaling factor
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # Pre-scale the image to the larger size
    scaled_image = original_image.resize((new_w, new_h), Image.LANCZOS)

    # Center of the scaled image
    center_x = new_w / 2
    center_y = new_h / 2

    # Maximum radius to ensure the crop window (of size w x h) stays fully within the scaled image.
    max_radius = min((new_w - w) / 2, (new_h - h) / 2)

    def make_frame(t):
        # Calculate progress as a fraction of the total duration (0.0 to 1.0)
        progress = t / duration
        # Compute angle in radians (full circle: 0 to 2π)
        angle = 2 * math.pi * progress
        # Determine the center of the crop window along the circular path
        cx = center_x + max_radius * math.cos(angle)
        cy = center_y + max_radius * math.sin(angle)
        # Calculate top-left coordinates of the crop window
        left = int(cx - w / 2)
        top = int(cy - h / 2)
        right = left + w
        bottom = top + h
        # Crop the scaled image to the original dimensions
        cropped = scaled_image.crop((left, top, right, bottom))
        # Return the cropped frame as a NumPy array
        return np.array(cropped)

    # Create a MoviePy VideoClip using the make_frame function
    clip = VideoClip(make_frame, duration=duration).with_duration(duration).with_fps(fps)
    return clip

if __name__ == "__main__":
    # Define the full path to your Download folder
    downloads_folder = "/storage/emulated/0/Download"
    # Define image file extensions to search for
    extensions = ["*.jpg", "*.jpeg", "*.png"]

    # Gather all matching image file paths from the Download folder
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(downloads_folder, ext)))

    if not image_files:
        print("No image files found in the Download folder.")
    else:
        print(f"Found {len(image_files)} images. Processing...")

    # Process each image: create a circular motion animation and save as an MP4
    for img_path in image_files:
        try:
            # Generate the circular motion clip for each image
            clip = circular_motion_effect(
                image_path=img_path,
                duration=5,        # 5-second animation
                scale_factor=1.3,  # Scale up to 130% for enough room to move along a circle
                fps=25
            )
            # Construct output filename (e.g., original_name_circular_motion.mp4)
            base_name, _ = os.path.splitext(os.path.basename(img_path))
            output_video_path = os.path.join(downloads_folder, base_name + "_circular_motion.mp4")
            print(f"Creating video for: {img_path}")
            clip.write_videofile(output_video_path, codec="libx264")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("All done!")
