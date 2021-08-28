


# NOTE :

# 1. This Project is for only webcam face detection
# 2. We can see in further projects , which is the extension of this where we can see to detect multiple faces in a clip.
# 3. Also we will see how  to detect faces in an image and  clip.


# STEP 1 : Install OpenCV
# Steps to Install : File > Setting > Project > Python Interpreter > "+" sign > Search  "opencv python" > click Install
# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# STEP 2 : Import libraries

import cv2
from random import randrange as r

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# STEP 3 : Import HARR Cascade which contains tons of trained data which stored in a variable for further use

trained_data = cv2.CascadeClassifier("file.xml")

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# I use third party app for webcam purpose , you can inserty 1 or 0 instead video_path
# # STEP 4 : Open video camera

video_path = "https://192.168.1.201:8080/video"
capture = cv2.VideoCapture(video_path,cv2.CAP_DSHOW)
capture.open(video_path)
print("Your Webcam is :\nTrue : Open \nFalse:Close\n",capture.isOpened())

while (capture.isOpened()):

    ret,frame = capture.read()
    frame = cv2.resize(frame,(800,600))

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# STEP 5 : Grayscale Image

    grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# STEP 6 : Find out the Coordinates of the Gray scale image to recognise the Face
#          and then store in to a variable for further use.

    find_coordinates = trained_data.detectMultiScale(grayscale)

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# STEP 7 : Draw A rectangle around face to show the detected area .

    for x,y,w,h in find_coordinates:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (r(0, 255), r(0, 255), r(0, 255)), 8)
        cv2.imshow("final output",frame)

    key = cv2.waitKey(1) & 0xFF
    if key ==27:
        print("Congratulation ....\nYou have Successfully Detecting the Face")
        break

capture.release()
cv2.destroyAllWindows()
