import math
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import time


cap = cv2.VideoCapture(0)
detector= HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
imageSize = 300

folder="Data/A"
counter=0

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

while(True):
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        
        imgCropShape = imgCrop.shape
        
        imgWhite[0:imgCrop.shape[0],0:imgCrop.shape[1]] = imgCrop
        
        aspectRatio = h/w
        if aspectRatio > 1:
            k = imageSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imageSize))
            imgResizeShape = imgResize.shape 
            wGap = math.ceil(imageSize - wCal) // 2
            imgWhite[:,wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(img)
            print(prediction, index)
            
        else:
            k = imageSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imageSize, hCal))
            imgResizeShape = imgResize.shape 
            hGap = math.ceil(imageSize - hCal) // 2
            imgWhite[hGap:hCal+hGap,:] = imgResize
        
        
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
        
        
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
        
        