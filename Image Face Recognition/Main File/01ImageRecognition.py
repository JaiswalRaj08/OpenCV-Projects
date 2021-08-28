
# NOTE :

# 1. This Project is for only one image detection
# 2. We can see in further projects , which is the extension of this where we can see to detect multiple faces.
# 3. Also we will see how to use webcam to detect faces.


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

image_path = "C:/Users/HOME/Desktop/GIthub/Pycharm/OpenCV Projects/Image Face Recognition/Images/messi.jpg"

single_image = cv2.imread(image_path)
single_image = cv2.resize(single_image,(1200,800))


cv2.imshow("original Image",single_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# STEP 5 : Grayscale Image

grayscale_image = cv2.cvtColor(single_image,cv2.COLOR_BGR2GRAY)
grayscale_image = cv2.resize(grayscale_image,(1200,800))


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


# *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

# STEP 7 : Draw A rectangle around face to show the detected area .

x,y,w,h = find_coordinates[0]
cv2.rectangle(single_image,(x,y),((x+w),(y+h)), (r(0,255),r(0,255),r(0,255)),8)

cv2.imshow("final output",single_image)
print("Congratulation ....\nYou have Successfully Detected the Face")

cv2.waitKey(0)
cv2.destroyAllWindows()

