import os
import glob
import numpy as np
from PIL import Image
from moviepy.video.VideoClip import VideoClip

def slide_bottom_effect(image_path, duration=5, scale_factor=1.2, fps=25):
    """
    Creates a MoviePy VideoClip that applies a slide_bottom animation on the given image.
    
    The image is first scaled up by 'scale_factor' to provide extra vertical pixels.
    Then a crop window of the original size (w x h) is moved vertically from the bottom edge
    (y_offset = new_h - h) to the top edge (y_offset = 0) over the duration.
    The horizontal crop is centered.
    
    :param image_path:   Full path to the source image file.
    :param duration:     Duration (in seconds) of the sliding animation.
    :param scale_factor: Factor by which the image is scaled (>= 1).
    :param fps:          Frames per second for the generated clip.
    :return:             A MoviePy VideoClip with the slide_bottom animation.
    """
    # Load the original image using Pillow
    original_image = Image.open(image_path)
    w, h = original_image.size

    # Compute new dimensions based on the scaling factor
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # Pre-scale the image to the larger size
    scaled_image = original_image.resize((new_w, new_h), Image.LANCZOS)

    # Compute horizontal offset to center the crop window
    x_offset = (new_w - w) // 2

    def make_frame(t):
        # Calculate progress from 0.0 to 1.0 over the duration
        progress = t / duration
        # For slide_bottom: at t=0, y_offset = new_h - h (bottom), and at t=duration, y_offset = 0 (top).
        y_offset = int((new_h - h) * (1 - progress))
        # Crop the scaled image to the original dimensions (w x h)
        cropped = scaled_image.crop((x_offset, y_offset, x_offset + w, y_offset + h))
        # Return the cropped frame as a NumPy array
        return np.array(cropped)

    # Create a MoviePy VideoClip using the frame function,
    # then set the duration and fps using with_duration and with_fps.
    clip = VideoClip(make_frame, duration=duration).with_duration(duration).with_fps(fps)
    return clip

if __name__ == "__main__":
    # Define the full Download folder path
    downloads_folder = "/storage/emulated/0/Download"
    # File extensions to search for
    extensions = ["*.jpg", "*.jpeg", "*.png"]

    # Gather all matching image file paths from the Download folder
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(downloads_folder, ext)))

    if not image_files:
        print("No image files found in the Download folder.")
    else:
        print(f"Found {len(image_files)} images. Processing...")

    # Process each image: create a slide_bottom animation and save as an MP4
    for img_path in image_files:
        try:
            # Generate the slide_bottom clip for each image
            clip = slide_bottom_effect(
                image_path=img_path,
                duration=5,       # 5-second animation
                scale_factor=1.2, # Scale up to 120% to allow vertical slide
                fps=25
            )
            # Construct output filename (e.g., original_name_slide_bottom.mp4)
            base_name, _ = os.path.splitext(os.path.basename(img_path))
            output_video_path = os.path.join(downloads_folder, base_name + "_slide_bottom.mp4")
            print(f"Creating video for: {img_path}")
            clip.write_videofile(output_video_path, codec="libx264")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("All done!")
