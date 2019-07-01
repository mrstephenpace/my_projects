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
#import tensorflow.contrib.tensorrt as trt

#from utils.camera import Camera
from utils.od_utils import read_label_map, build_trt_pb, load_trt_pb, detect
from utils.visualization import BBoxVisualization


# Constants
DEFAULT_MODEL = 'ssd_mobilenet_v1_coco_people'
DEFAULT_LABELMAP = 'data/people_label_map.pbtxt'
WINDOW_NAME1 = 'People Detector Camera 1'
WINDOW_NAME2 = 'People Detector Camera 2'
BBOX_COLOR = (255, 0, 0)  # red
# DEFAULT camera width and height is 640 x 480



def parse_args():
    """Parse input arguments."""
    desc = ('This script captures and displays live camera video from 2 usb'
            'cameras and does real-time person detections with TF-TRT model '
            'on NVIDIA Jetson TX2/TX1')
    parser = argparse.ArgumentParser(description=desc)
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


def draw_help_and_fps(img, fps):
    """Draw help message and fps number at top-left corner of the image."""
    help_text = "'Esc' to Quit, 'F' for FPS"
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

        logger.info('loading TRT graph from pb: %s' % pb_path)
    trt_graph = load_trt_pb(pb_path)

    logger.info('starting up TensorFlow session')
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config, graph=trt_graph)

    od_type = 'ssd'

    logger.info('opening cameras')
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)
    
    logger.info('starting to loop and detect')
    vis = BBoxVisualization(cls_dict)
    
    show_fps1 = True
    fps1 = 0.0
    show_fps2 = True
    fps2 = 0.0
    tic1 = time.time()
    tic2 = time.time()
    
    # grab images and do object detections (until stopped by user)
    while True:
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        if ret1 and ret2:
            box1, conf1, cls1 = detect(frame1, tf_sess, args.conf_th, od_type=od_type)
            box2, conf2, cls2 = detect(frame2, tf_sess, args.conf_th, od_type=od_type)

            frame1 = vis.draw_bboxes(frame1, box1, conf1, cls1)
            frame2 = vis.draw_bboxes(frame2, box2, conf2, cls2)
            if show_fps1:
                frame1 = draw_help_and_fps(frame1, fps1)
            cv2.imshow(WINDOW_NAME1, frame1)
            toc1 = time.time()
            curr_fps1 = 1.0 / (toc1 - tic1)
            # calculate an exponentially decaying average of fps number
            fps1 = curr_fps1 if fps1 == 0.0 else (fps1*0.9 + curr_fps1*0.1)
            tic1 = toc1

            if show_fps2:
                frame2 = draw_help_and_fps(frame2, fps2)
            cv2.imshow(WINDOW_NAME2, frame2)
            toc2 = time.time()
            curr_fps2 = 1.0 / (toc2 - tic2)
            # calculate an exponentially decaying average of fps number
            fps2 = curr_fps2 if fps2 == 0.0 else (fps2*0.9 + curr_fps2*0.1)
            tic2 = toc2

        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fps
            show_fps1 = not show_fps1


    logger.info('cleaning up and closing cameras')

    tf_sess.close()
    
    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
