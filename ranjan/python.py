import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os


mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

offset = 20
imageSize = 300

folder = "Data/A"
counter = 0


if not os.path.exists(folder):
    os.makedirs(folder)
    print(f"Created directory: {folder}")

print("Press 's' to save an image, 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(imgRGB)

    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            
            h_img, w_img, _ = img.shape
            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w_img))
                y_list.append(int(lm.y * h_img))

            x, y = min(x_list), min(y_list)
            w, h = max(x_list) - x, max(y_list) - y

            
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            imgWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255
            
            
            y1 = max(0, y - offset)
            y2 = min(h_img, y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(w_img, x + w + offset)
            
            imgCrop = img[y1:y2, x1:x2]

            
            if imgCrop.size != 0:
                imgCropShape = imgCrop.shape
                aspectRatio = h / w

                try:
                    if aspectRatio > 1:
                        
                        k = imageSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imageSize))
                        wGap = math.ceil((imageSize - wCal) / 2)
                        
                       
                        imgWhite[:, wGap:wCal + wGap] = imgResize

                    else:
                        
                        k = imageSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imageSize, hCal))
                        hGap = math.ceil((imageSize - hCal) / 2)
                        
                        
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    cv2.imshow("ImageCrop", imgCrop)
                    cv2.imshow("ImageWhite", imgWhite)

                except Exception as e:
                    print(f"Error resizing: {e}")

    
    cv2.imshow("Image", img)
    
    key = cv2.waitKey(1)
    
    
    if key == ord("s"):
        counter += 1
        filename = f'{folder}/Image_{time.time()}.jpg'
        
        
        if 'imgWhite' in locals():
            cv2.imwrite(filename, imgWhite)
            print(f"Saved: {filename} (Count: {counter})")
        else:
            print("No hand detected to save!")
            
            
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
