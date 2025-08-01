# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 02:46:14 2025

@author: 1215m
"""
import cv2

videoPATH = r"street.mp4"
video = cv2.VideoCapture(videoPATH)
face_cascade = cv2.CascadeClassifier("face_detector.xml")

def blur_face(img):
    final_img = img.copy() #Clear Frame
    blur_img = img.copy() #Blur Frame
    rectangle = face_cascade.detectMultiScale(blur_img, scaleFactor=1.2, minNeighbors=5)
    blur_img = cv2.blur(blur_img, (101, 101))
    for (x, y, w, h) in rectangle:
        final_img[y:y+h, x:x+w] = blur_img[y:y+h, x:x+w] #Overlay Blured Frame over Clear Frame on detected area
        pass
    
    return final_img

# Check if camera opened successfully
if (video.isOpened()== False):
    print("Error opening video file")

vfps = video.get(cv2.CAP_PROP_FPS)
vwidth = video.get(cv2.CAP_PROP_FRAME_WIDTH)
vheight = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Read until video is completed
while(video.isOpened()):
    
    # Capture frame-by-frame
    ret, frame = video.read()
    if ret == True:
        alterFrame = blur_face(frame) #Blur Current Frame
        
        # Display the resulting frame
        cv2.imshow('Frame', alterFrame)
        
    # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# Break the loop
    else:
        break

cv2.destroyAllWindows()
video.release()
