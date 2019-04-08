import sys, os, argparse, json
import cv2
import time, datetime, pytz
from darkflow.net.build import TFNet
from imutils.video import FPS

# zmq import stuff
import zmq, imagezmq
from collections import defaultdict
from ColConfig import ColConfig

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,
	help="path to output video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("--remote_camera", required=False, type=bool, default=False,
    help="camera feed is remote (True) from another device")
ap.add_argument("-s", "--show_camera", required=False, type=bool, default=False,
	help="show live camera feed and FPS stats during run time and end")
ap.add_argument("--save_images", type=bool, default=False,
    help="save images of detected objects")
args = vars(ap.parse_args())

options = {"model": "cfg/yolo.cfg", 
            "load": "bin/yolo.weights",
            "verbalise": False,
            "threshold": args["confidence"],
            "gpu": 0.8,
            }

tfnet = TFNet(options)

# disable Tensorflow GPU info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2', '3'} to disable Tensorflow GPU debugging info

# set timezone
tz = pytz.timezone('US/Central')

if args["remote_camera"]:
    # set up zmq stuff for remote camera
    config = ColConfig()
    port = config.getConfig(ColConfig.SUB)[ColConfig.SUB_PORT]
    bind_address = "tcp://192.168.1.171:{}".format(port) # 'tcp://*:5555'
    #bind_address = 'tcp://192.168.1.171:5555'
    image_hub = imagezmq.ImageHub()
    sender_image_counts = defaultdict(int)  # dict for counts by sender
    print("Subscribe Video at ", bind_address)
else:
    # use usb camera
    cap = cv2.VideoCapture(-1)

# initialize the pointer to output video file
writer = None

image_count = 0
first_image = True

try:
    while(True):  # receive images until Ctrl-C is pressed
        if args["remote_camera"]:
            sent_from, frame = image_hub.recv_image() #read images from remote camera stream
            sender_image_counts[sent_from] += 1
            image_hub.send_reply(b"OK")  # REP reply 
        else:
            ret, frame = cap.read() #read images from usb camera stream
        
        if first_image:
            fps = FPS().start()  # start FPS timer after first image is received
            first_image = False
        fps.update()
        
        # get current time and display in camera frame
        clock_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, clock_time, (5, 25),	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        result = tfnet.return_predict(frame)
        
        #print(len(result))
        for i in range(len(result)):
            # extract the bounding box coordinates
            (x, y) = (result[i]["topleft"]["x"], result[i]["topleft"]["y"])
            (w, h) = (result[i]["bottomright"]["x"], result[i]["bottomright"]["y"])
            cv2.rectangle(frame, (x, y), (w, h), (128, 128, 0), 2)
            
            # extract the label and confidence
            label = result[i]["label"]
            confidence = result[i]["confidence"]
            text = "{}: {:.4f}".format(label, confidence)
            cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

            # save image if it is a car or truck
            if label is 'car' or 'truck' and args["save_images"] is True:
                #print(label, image_count)
                folder = 'images/'
                filename = '{}{}.jpg'.format(label,image_count)
                #print(filename)
                cv2.imwrite(os.path.join(folder,filename),frame[y:h, x:w])
            
        # check if the video writer is None
        if writer is None and args["output"] is not None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,(frame.shape[1], frame.shape[0]), True)
            
        if writer is not None:
            # write the output frame to disk
            writer.write(frame)

        

        image_count += 1  # global count of all images received
        cv2.imshow('USB Camera', frame)  # display camera stream
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
                

except (KeyboardInterrupt, SystemExit):
    pass  # Ctrl-C was pressed to end program; FPS stats computed below

finally:
    # stop the timer and display FPS information
    print()
    print('Test Program: ', __file__)
    print('Total Number of Images received: {:,g}'.format(image_count))
    if first_image:  # never got images from any RPi
        sys.exit()
    fps.stop()
   
    print('Elasped time: {:,.2f} seconds'.format(fps.elapsed()))
    print('Approximate FPS: {:.2f}'.format(fps.fps()))

if writer is not None:
    writer.release()
cv2.destroyAllWindows()
sys.exit()
