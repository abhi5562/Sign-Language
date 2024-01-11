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
    os.makedirs("data/test/hello")
    os.makedirs("data/test/good")
    os.makedirs("data/test/i_love_you")
    os.makedirs("data/test/yes")
    os.makedirs("data/test/no")

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
    
    # Printing the count in each set to the screen
    font_style = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(frame, "MODE : "+mode, (width-200, 50), font_style, 1, (0, 0, 255), 1)
    cv2.putText(frame, "IMAGE COUNT", (width-200, 70), font_style, 1, (0, 0, 255), 1)
    cv2.putText(frame, "hello(0) : "+str(count['hello']), (width-200, 90), font_style, 1, (0, 0, 255), 1)
    cv2.putText(frame, "good(1) : "+str(count['good']), (width-200, 110), font_style, 1, (0, 0, 255), 1)
    cv2.putText(frame, "i_love_you(2) : "+str(count['i_love_you']), (width-200, 130), font_style, 1, (0, 0, 255), 1)
    cv2.putText(frame, "yes(3) : "+str(count['yes']), (width-200, 150), font_style, 1, (0, 0, 255), 1)
    cv2.putText(frame, "no(4) : "+str(count['no']), (width-200, 170), font_style, 1, (0, 0, 255), 1)

    # Coordinates of the ROI (Region of intrest)
    x1 = 20
    y1 = 20
    x2 = int(width/2) - 200
    y2 = int(height/2) + 200

    # Drawing the Boundary box of ROI
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1)

    # Extracting and Resizing ROI
    roi = frame[y1+1:y2,x1+1:x2]
    roi = cv2.resize(roi,(64,64))

    #Display Frame
    cv2.imshow("Frame",frame)

    # Convert to grey-scale
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("ROI",roi)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break
    elif key == ord("0"):
        cv2.imwrite(directory + "hello/" + str(count['hello']) + ".jpg",roi)
    elif key == ord("1"):
        cv2.imwrite(directory + "good/" + str(count['good']) + ".jpg",roi)
    elif key == ord("2"):
        cv2.imwrite(directory + "i_love_you/" + str(count['i_love_you']) + ".jpg",roi)
    elif key == ord("3"):
        cv2.imwrite(directory + "yes/" + str(count['yes']) + ".jpg",roi)
    elif key == ord("4"):
        cv2.imwrite(directory + "no/" + str(count['no']) + ".jpg",roi)

cap.release()
cv2.destroyAllWindows()