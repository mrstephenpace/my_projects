"""people_detector.py

This code is modified from JK Jung's camera_tf_trt.py script
https://github.com/jkjung-avt/tf_trt_models/blob/master/camera_tf_trt.py

by S.D. Pace <paceste1@gmail.com>
"""

import sys
import time
import logging
import argparse
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

from utils.camera import Camera
from utils.od_utils import read_label_map, build_trt_pb, load_trt_pb, detect
from utils.visualization import BBoxVisualization


# Constants
DEFAULT_MODEL = 'ssd_mobilenet_v1_coco_people'
DEFAULT_LABELMAP = 'data/people_label_map.pbtxt'
WINDOW_NAME = 'People Detector'
WINDOW_NAME2 = 'People Detector Camera 2'
BBOX_COLOR = (255, 0, 0)  # red


def parse_args():
    """Parse input arguments."""
    desc = ('This script captures and displays live camera video, '
            'and does a real-time person detector with TF-TRT model '
            'on Jetson TX2/TX1')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--n_cameras', dest='n_cameras',
                        help='number of usb cameras',
                        default=1, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        default='--usb',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [0]',
                        default=0, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [1280]',
                        default=1280, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [720]',
                        default=720, type=int)
    parser.add_argument('--model', dest='model',
                        help='tf-trt object detecion model '
                        '[{}]'.format(DEFAULT_MODEL),
                        default=DEFAULT_MODEL, type=str)
    parser.add_argument('--build', dest='do_build',
                        help='re-build TRT pb file (instead of using'
                        'the previously built version)',
                        action='store_true')
    parser.add_argument('--labelmap', dest='labelmap_file',
                        help='[{}]'.format(DEFAULT_LABELMAP),
                        default=DEFAULT_LABELMAP, type=str)
    parser.add_argument('--confidence', dest='conf_th',
                        help='confidence threshold [0.3]',
                        default=0.5, type=float)
    parser.add_argument('--output', dest='output_file',
                    	help='path to saved output video file')
    args = parser.parse_args()
    return args


def open_display_window(width, height):
    """Open the cv2 window for displaying images with bounding boxeses."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'People Detector')


def draw_help_and_fps(img, fps):
    """Draw help message and fps number at top-left corner of the image."""
    help_text = "'Esc' to Quit, 'H' for FPS & Help"
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA

    fps_text = 'FPS: {:.1f}'.format(fps)
    cv2.putText(img, help_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, help_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    cv2.putText(img, fps_text, (11, 50), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 50), font, 1.0, (240, 240, 240), 1, line)
    return img

def show_bounding_boxes(img, box, conf, cls, cls_dict):
    """Draw detected bounding boxes on the original image."""
    font = cv2.FONT_HERSHEY_DUPLEX
    for bb, cf, cl in zip(box, conf, cls):
        cl = int(cl)
        y_min, x_min, y_max, x_max = bb[0], bb[1], bb[2], bb[3]
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), BBOX_COLOR, 2)
        txt_loc = (max(x_min, 5), max(y_min-3, 20))
        cls_name = cls_dict.get(cl, 'CLASS{}'.format(cl))
        txt = '{} {:.2f}'.format(cls_name, cf)
        cv2.putText(img, txt, txt_loc, font, 0.8, BBOX_COLOR, 1)    
    return img


#def loop_and_detect(cam, tf_sess, conf_th, vis, od_type):
    """Loop, grab images from camera, and do object detection.

    # Arguments
      cam: the camera object (video source).
      tf_sess: TensorFlow/TensorRT session to run SSD object detection.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
#    show_fps = True
#    full_scrn = False
#    fps = 0.0
#    tic = time.time()
#    writer = None
#    args = parse_args()
#    while True:
#        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            # Check to see if the user has closed the display window.
            # If yes, terminate the while loop.
#            break

#        img = cam.read()
#        if img is not None:
#            box, conf, cls = detect(img, tf_sess, conf_th, od_type=od_type)
#            img = vis.draw_bboxes(img, box, conf, cls)
#            if show_fps:
#                img = draw_help_and_fps(img, fps)
#            cv2.imshow(WINDOW_NAME, img)
#            toc = time.time()
#            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
#            fps = curr_fps if fps == 0.0 else (fps*0.9 + curr_fps*0.1)
#            tic = toc

          # check if the video writer is None
#        if writer is None and args.output_file is not None:
            # initialize our video writer
#            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#            writer = cv2.VideoWriter(args.output_file, fourcc, curr_fps,(img.shape[1], img.shape[0]), True)
        
#        if writer is not None:
            # write the output frame to disk
#            writer.write(img)   
                
#        key = cv2.waitKey(1)
#        if key == 27:  # ESC key: quit program
#            if writer is not None:
#                writer.release()
#            break
#        elif key == ord('H') or key == ord('h'):  # Toggle help/fps
#            show_fps = not show_fps
#        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
#            full_scrn = not full_scrn
#            set_full_screen(full_scrn)


def main():
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Ask tensorflow logger not to propagate logs to parent (which causes
    # duplicated logging)
    logging.getLogger('tensorflow').propagate = False

    args = parse_args()
    logger.info('called with args: %s' % args)

    # build the class (index/name) dictionary from labelmap file
    logger.info('reading label map')
    cls_dict = read_label_map(args.labelmap_file)

    pb_path = './ssd_mobilenet_v1_coco_people/{}_trt.pb'.format(args.model)
    log_path = './logs/{}_trt'.format(args.model)
    print(log_path)
    if args.do_build:
        logger.info('building TRT graph and saving to pb: %s' % pb_path)
        build_trt_pb(args.model, pb_path)

    logger.info('opening camera device/file')
    
    #cam1 = Camera(args)
   
    #cam1.open()
    #if not cam1.is_opened:
    #    sys.exit('Failed to open camera #1!')

    #cam1.start()  # ask the camera to start grabbing images


    logger.info('loading TRT graph from pb: %s' % pb_path)
    trt_graph = load_trt_pb(pb_path)

    logger.info('starting up TensorFlow session')
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config, graph=trt_graph)

    od_type = 'ssd'

    cam2 = cv2.VideoCapture(1)
    
    logger.info('starting to loop and detect')
    vis = BBoxVisualization(cls_dict)
    
    show_fps1 = True
    fps1 = 0.0
    tic1 = time.time()
    while True:
        ret2, frame2 = cam2.read()
        if ret2:
            box, conf, cls = detect(frame2, tf_sess, args.conf_th, od_type=od_type)
            frame2 = vis.draw_bboxes(frame2, box, conf, cls)
            if show_fps1:
                frame2 = draw_help_and_fps(frame2, fps1)
            cv2.imshow(WINDOW_NAME2, frame2)
            toc1 = time.time()
            curr_fps1 = 1.0 / (toc1 - tic1)
            # calculate an exponentially decaying average of fps number
            fps1 = curr_fps1 if fps1 == 0.0 else (fps1*0.9 + curr_fps1*0.1)
            tic1 = toc1

        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('H') or key == ord('h'):  # Toggle help/fps
            show_fps1 = not show_fps1



    # grab image and do object detection (until stopped by user)
    
    #open_display_window(cam1.img_width, cam1.img_height)
    #open_display_window(args.image_width, args.image_height)
    #loop_and_detect(cam1, tf_sess, args.conf_th, vis, od_type=od_type)
    #loop_and_detect(cam2, tf_sess, args.conf_th, vis, od_type=od_type)

    
    logger.info('cleaning up')
    #cam1.stop()  # terminate the sub-thread in camera

    tf_sess.close()
    
    #cam1.release()
    cam2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
