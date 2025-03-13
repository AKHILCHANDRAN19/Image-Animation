import os
import glob
import numpy as np
from PIL import Image
from moviepy.video.VideoClip import VideoClip

def slide_left_to_right_effect(image_path, duration=5, scale_factor=1.2, fps=25):
    """
    Creates a MoviePy VideoClip that applies a slide left-to-right animation
    on the given image.

    The image is first scaled up by 'scale_factor' to ensure extra width is available.
    Then a crop window of the original image size is moved horizontally from left (0)
    to right (new_w - w) over the animation duration, while vertically centering the crop.

    :param image_path:   Full path to the source image file.
    :param duration:     Duration (in seconds) of the sliding animation.
    :param scale_factor: Factor by which the image is scaled for the slide effect (>= 1).
    :param fps:          Frames per second for the generated clip.
    :return:             A MoviePy VideoClip with the slide animation.
    """
    # Load the original image using Pillow
    original_image = Image.open(image_path)
    w, h = original_image.size

    # Compute new dimensions based on the scaling factor
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # Pre-scale the image once to the larger size
    scaled_image = original_image.resize((new_w, new_h), Image.LANCZOS)

    # Compute vertical offset to center the crop window vertically
    y_offset = (new_h - h) // 2

    def make_frame(t):
        # Calculate progress (from 0.0 to 1.0)
        progress = t / duration
        # Compute horizontal offset: at t=0, offset is 0; at t=duration, offset is (new_w - w)
        x_offset = int((new_w - w) * progress)
        # Crop the scaled image to the original size (w x h)
        cropped = scaled_image.crop((x_offset, y_offset, x_offset + w, y_offset + h))
        # Return the cropped frame as a NumPy array
        return np.array(cropped)

    # Create a MoviePy VideoClip using the make_frame function,
    # then use with_duration and with_fps to set the animation properties.
    clip = VideoClip(make_frame, duration=duration).with_duration(duration).with_fps(fps)
    return clip

if __name__ == "__main__":
    # Define the full Download folder path
    downloads_folder = "/storage/emulated/0/Download"
    # Define image file extensions to search for
    extensions = ["*.jpg", "*.jpeg", "*.png"]

    # Gather all image file paths from the Download folder
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(downloads_folder, ext)))

    if not image_files:
        print("No image files found in the Download folder.")
    else:
        print(f"Found {len(image_files)} images. Processing...")

    # Process each image file: create a slide left-to-right animation and save as an MP4
    for img_path in image_files:
        try:
            # Generate the slide left-to-right clip for each image
            clip = slide_left_to_right_effect(
                image_path=img_path,
                duration=5,        # 5-second animation
                scale_factor=1.2,  # Scale up to 120% to allow horizontal shift
                fps=25
            )
            # Create an output filename (e.g., original_name_slide_left_to_right.mp4)
            base_name, _ = os.path.splitext(os.path.basename(img_path))
            output_video_path = os.path.join(downloads_folder, base_name + "_slide_left_to_right.mp4")
            print(f"Creating video for: {img_path}")
            clip.write_videofile(output_video_path, codec="libx264")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print("All done!")
