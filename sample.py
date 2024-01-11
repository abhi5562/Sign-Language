import cv2
import numpy as np
import os

if not os.path.exists("data"):
    os.makedirs("data")
    os.makedirs("data/train")
    os.makedirs("data/test")
    os.makedirs("data/train/hello")
    os.makedirs("data/train/good")
    os.makedirs("data/train/i_love_you")
    os.makedirs("data/train/yes")
    os.makedirs("data/train/no")

mode = "train"
directory = "data/" + mode + "/"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    height = int(cap.get(4))
    width = int(cap.get(3))

    # Getting count of existing images
    count = {"hello": len(os.listdir(directory+"hello")),
             "good": len(os.listdir(directory+"good")),
             "i_love_you": len(os.listdir(directory+"i_love_you")),
             "yes": len(os.listdir(directory+"yes")),
             "no": len(os.listdir(directory+"no"))}
    
    font_style = cv2.FONT_HERSHEY_PLAIN
    # Printing the count in each set to the screen
    cv2.putText(frame, "MODE : "+mode, (width-200, 50), font_style, 1, (0, 0, 255), 1)
    cv2.putText(frame, "IMAGE COUNT", (width-200, 70), font_style, 1, (0, 0, 255), 1)
    cv2.putText(frame, "hello : "+str(count['hello']), (width-200, 90), font_style, 1, (0, 0, 255), 1)
    cv2.putText(frame, "good : "+str(count['good']), (width-200, 110), font_style, 1, (0, 0, 255), 1)
    cv2.putText(frame, "i_love_you : "+str(count['i_love_you']), (width-200, 130), font_style, 1, (0, 0, 255), 1)
    cv2.putText(frame, "yes : "+str(count['yes']), (width-200, 150), font_style, 1, (0, 0, 255), 1)
    cv2.putText(frame, "no : "+str(count['no']), (width-200, 170), font_style, 1, (0, 0, 255), 1)



    # cv2.putText(frame, "Hello Abhi!",10,20,font_style,1,())

    cv2.imshow("frame",frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()