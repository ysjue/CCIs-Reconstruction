"""
OpenCV capture images with RealSense camera
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
# import argparse


# create a directory to save captured images 
def create_image_directory():
    now = datetime.datetime.now()
    dir_str = now.strftime("%Y-%m-%d-%H%M%S")
    try:
        if not(os.path.isdir(dir_str)):
            os.makedirs(os.path.join(dir_str))
    except OSError as e:
        print("Can't make the directory")
        raise
    return dir_str



def main():
    # parser = argparse.ArgumentParser(description='RealSense Camera Calibration')
    # parser.add_argument('mode', type=int, help='0 - 640x480, 1 - 1920x1080')
    # args = parser.parse_args()

    # if args.mode == 0:
    #     IMAGE_WIDTH = 640
    #     IMAGE_HEIGHT = 480
    # else:
    # DEPTH_IMAGE_WIDTH = 640
    # DEPTH_IMAGE_HEIGHT = 480
    # IMAGE_WIDTH = 1920
    # IMAGE_HEIGHT = 1080
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    image_dir = create_image_directory()

    print("press SPACE to capture an image or press Esc to exit...")

    image_counter = 0
    

    try:
        while(True):
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # display the captured image
            cv2.imshow('Color Images',color_image)
            cv2.imshow('Depth Images',depth_image)
            pressedKey = (cv2.waitKey(1) & 0xFF)

            # handle key inputs
            if pressedKey == 27:
                break
            elif pressedKey == 32:
                cv2.imwrite(os.path.join(image_dir, str(image_counter) + 'color'+'.jpg'), color_image)
                cv2.imwrite(os.path.join(image_dir, str(image_counter) + 'depth'+'.png'), depth_image)
                print('Image caputured - ' + os.path.join(image_dir, str(image_counter) + '.jpg'))

                image_counter+=1
    finally:
        # Stop streaming
        pipeline.stop()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
