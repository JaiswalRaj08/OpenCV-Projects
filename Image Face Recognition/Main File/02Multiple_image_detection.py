# NOTE :

# 1. This Project is for only multiple faces in  image detection
# 2. Also we will see how to use webcam to detect faces.
# 3. In previous we have seen for single image


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

# STEP 4 : Import Images

image_path = "C:/Users/HOME/Desktop/GIthub/Pycharm/OpenCV Projects/Image Face Recognition/Images/UCL2015.jpg"

single_image = cv2.imread(image_path)


cv2.imshow("original Image",single_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# STEP 5 : Grayscale Image

grayscale_image = cv2.cvtColor(single_image,cv2.COLOR_BGR2GRAY)


cv2.imshow("Gray scale Image",grayscale_image)
print(" Successfully Converted into Gray scale ")
cv2.waitKey(0)
cv2.destroyAllWindows()

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# STEP 6 : Find out the Coordinates of the Gray scale image to recognise the Face
#          and then store in to a variable for further use.


find_coordinates = trained_data.detectMultiScale(grayscale_image)

# print coordinates of face found in an image as [[x ,y ,w(width) ,h(height)]] in an array

print(find_coordinates)

# since we are getting multiple list of arrays , which shows multiple faces are exists in an image

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# STEP 7 : Draw A rectangle around face to show the detected area .

# prepare for loop to reduce codes and iterate the list of coordinates to detect face

for x,y,w,h in find_coordinates:
    cv2.rectangle(single_image,(x,y),((x+w),(y+h)), (r(0,255),r(0,255),r(0,255)),8)

cv2.imshow("final output",single_image)
print("Congratulation ....\nYou have Successfully Detected the Face")

cv2.waitKey(0)
cv2.destroyAllWindows()
