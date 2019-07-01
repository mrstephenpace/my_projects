"""camera.py

This code implements the Camera class, which encapsulates code to
handle IP CAM, USB webcam or the Jetson onboard camera.  The Camera
class is further extend to take either a video or an image file as
input.
"""


import time
import logging
import threading

import numpy as np
import cv2



def open_cam_usb(dev, width, height):
    """Open a USB webcam.

    We want to set width and height here, otherwise we could just do:
        return cv2.VideoCapture(dev)
    """

    # MODIFIED TO USE USB CAMERA ON LINUX COMPUTER  

    #gst_str = ('v4l2src device=/dev/video{} ! '
    #           'video/x-raw, width=(int){}, height=(int){}, '
    #           'format=(string)RGB ! videoconvert ! '
    #           'appsink').format(dev, width, height)
    #width = 960
    #height = 720

    #return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    cap = cv2.VideoCapture(dev)

    #cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    #cap.set(cv2.CAP_PROP_FPS, 20)
    
    #while(True):
        # Capture frame-by-frame
    #    ret, frame = cap.read()
            
        # Display the resulting frame
    #    cv2.imshow('USB camera',frame)
    #        if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break

    return cap



def grab_img(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """
    while cam.thread_running:
        _, cam.img_handle = cam.cap.read()
        if cam.img_handle is None:
            logging.warning('grab_img(): cap.read() returns None...')
            break
    cam.thread_running = False


class Camera():
    """Camera class which supports reading images from theses video sources:

    1. Video file
    2. Image (jpg, png, etc.) file, repeating indefinitely
    3. RTSP (IP CAM)
    4. USB webcam
    5. Jetson onboard camera
    """

    def __init__(self, args):
        self.args = args
        self.is_opened = False
        self.thread_running = False
        self.img_handle = None
        self.img_width = 0
        self.img_height = 0
        self.cap = None
        self.thread = None

    def open(self):
        """Open camera based on command line arguments."""
        assert self.cap is None, 'Camera is already opened!'
        args = self.args
        
        self.cap = open_cam_usb(
            args.video_dev,
            args.image_width,
            args.image_height
        )

        if self.cap != 'OK':
            if self.cap.isOpened():
                # Try to grab the 1st image and determine width and height
                _, img = self.cap.read()
                if img is not None:
                    self.img_height, self.img_width, _ = img.shape
                    self.is_opened = True

    def start(self):
        assert not self.thread_running
        self.thread_running = True
        self.thread = threading.Thread(target=grab_img, args=(self,))
        self.thread.start()

    def stop(self):
        self.thread_running = False
        self.thread.join()

    def read(self):
        return self.img_handle

    def release(self):
        assert not self.thread_running
        if self.cap != 'OK':
            self.cap.release()
