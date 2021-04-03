import cv2
import sys
from random import randint

# video source, toggle T-F to use webcam or not
use_webcam = True
video_path = 0
if not use_webcam:
    video_path = "videos/race.mp4" # example path to video

# version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".") 
tracker_types = ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "MOSSE", "CSRT"] 
tracker_type = tracker_types[6]
def createTrackerByName(tracker_type):
    if tracker_type == "BOOSTING": 
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == "MIL": 
        tracker = cv2.legacy.TrackerMIL_create()
    if tracker_type == "KCF": 
        tracker = cv2.legacy.TrackerKCF_create()
    if tracker_type == "TLD":
        tracker = cv2.legacy.TrackerTLD_create()
    if tracker_type == "MEDIANFLOW":
        tracker = cv2.legacy.TrackerMedianFlow_create()
    if tracker_type == "MOSSE": 
        tracker = cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "CSRT": 
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in tracker_types:
          print(t)
    return tracker

# test capture
video = cv2.VideoCapture(video_path) 
if not video.isOpened():
    print("Error loading video.")
    sys.exit()
success, frame = video.read() 
if not success:
    print("Not possible to read the file.")
    sys.exit()

# select multiple regions of interest of the frame
bboxes, colors = [], []
while True:
    bbox = cv2.selectROI("Select an object, press ENTER to agree or press 'q' to quit", frame, False) 
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    k = cv2.waitKey(0) & 0xff
    if k == ord("q"): 
        break
multiTracker = cv2.legacy.MultiTracker_create() 
for bbox in bboxes:
    multiTracker.add(createTrackerByName(tracker_type), frame, bbox)
print(bboxes)
# tracking
while video.isOpened():
    ok, frame = video.read()
    if not ok:
        break
    timer = cv2.getTickCount() 
    ok, boxes = multiTracker.update(frame) 
    fps = cv2.getTickFrequency() / (cv2.getTickCount()-timer)
    if ok:
        for i, newbox in enumerate(boxes): 
            (x, y, w, h) = [int(v) for v in newbox] 
            cv2.rectangle(frame, (x, y), (x+w, y+h), colors[i], 2, 1)
    else:
        cv2.putText(frame, "Fail to track object", (100,80), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)
    cv2.putText(frame, "Tracker: " + tracker_type, (100,20), cv2.FONT_HERSHEY_SIMPLEX, .75, (50,170,50), 2)
    cv2.putText(frame, "FPS: " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, .75, (50,170,50), 2)
    cv2.imshow("Video, press 'q' to quit", frame)
    if cv2.waitKey(1) == ord("q"):
        break

