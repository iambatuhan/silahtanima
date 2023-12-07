import cv2
import numpy as np
import winsound
import time
net=cv2.dnn.readNet("C:/Users/90537/yolov3_training_2000.weights","C:/Users/90537/yolov3_testing.cfg")
classes=["Silah Tespit Edildi"]
output_layers=net.getUnconnectedOutLayersNames()
cap=cv2.VideoCapture(0)
pTime = 0
fps = 0
while True:
    ret,frame=cap.read()
    if not ret:
        print("Hata")
        break
    cTime = time.time()
    if cTime -pTime > 0:
        fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    height,width,channels=frame.shape
    blob=cv2.dnn.blobFromImage(frame,1/255.0,(416,416),(0,0,0),True,crop=False)
    net.setInput(blob)
    outs=net.forward(output_layers)
    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores=detection[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.5:
                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)
                x=int(center_x-w/2)
                y=int(center_y/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                winsound.Beep(2000, 1000)
    index=cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in index:
            x,y,w,h=boxes[i]
            label=str(classes[class_ids[i]])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            
            cv2.putText(frame,label, (x,y+60), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255),3)
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1) &  0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
    