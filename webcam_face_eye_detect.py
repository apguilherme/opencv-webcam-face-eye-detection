import cv2

def video_detect():
    face_classifier = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
    eyes_classifier = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")
    video = cv2.VideoCapture(0)
    while True:
        connected, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10, minSize=(60,60))
        for x, y, w, h in faces:
            face = cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            face_region = face[y:y+h, x:x+w]
            gray_face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            eyes = eyes_classifier.detectMultiScale(gray_face_region,  scaleFactor=1.1, minNeighbors=10, minSize=(30,30))
            for x1, y1, w1, h1 in eyes: 
                cv2.rectangle(face_region, (x1,y1), (x1+w1, y1+h1), (0,0,255), 2)
        cv2.imshow("Video, press 'q' to quit", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_detect()
