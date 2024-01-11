import numpy as np
from keras.models import model_from_json
import operator
import cv2

#Loading the model
json_file = open("model-bw.json","r")
model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(model_json)
loaded_model.load_weights("model-bw.h5")
print("Loaded weights from the model")

cap = cv2.VideoCapture(0)

categories = {0:"hello",
              1:"good",
              2:"i_love_you",
              3:"yes",
              4:"no"}

while True:
    ret, frame = cap.read()

    height = int(cap.get(4))
    width = int(cap.get(3))

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

    # Convert to grey-scale
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("Test Image",test_image)

    result = loaded_model.predict(test_image.reshape(1,64,64,1))
    prediction = {"good": result[0][0],
                  "hello": result[0][1],
                  "i_love_you": result[0][2],
                  "no": result[0][3],
                  "yes": result[0][4]}
    
    predictions = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

    print(predictions[0])

    #Display Frame
    cv2.putText(frame, predictions[0][0],(500,500),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),3)
    cv2.imshow("Frame",frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release
cv2.destroyAllWindows()
