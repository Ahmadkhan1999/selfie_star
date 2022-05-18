import cv2
import datetime
cap_0 = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
while True:
    retu , frame0 = cap_0.read()
    original_frame = frame0.copy()
    gray = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, 1.3 ,5)
    for x,y,w,h in face:
        cv2.rectangle(frame0, (x,y), (x+w,y+h), (0, 255, 255), 2)
        face_roi = frame0[y:y+h, x:x+w]
        gray_roi = gray[y:y+h, x:x+w]
        smile = smile_cascade.detectMultiScale(gray_roi, 1.3, 25)
        for x1, y1, w1, h1 in smile:
            cv2.rectangle(face_roi, (x1,y1),(x1+w1, y1+h1), (0, 0, 255),2)
            time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            file_name = f'selfie-{time_stamp}.png'
            cv2.imwrite(file_name,original_frame)
    cv2.imshow('frame', frame0)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
