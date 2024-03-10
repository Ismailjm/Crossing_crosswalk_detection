import cv2
import torch
from tracker import *
import numpy as np
import time 

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cap=cv2.VideoCapture('crossing.mp4')
cv2.namedWindow('FRAME')
tracker = Tracker()

crosswalk = [(393,350),(414,441),(502,440),(479,350)]
moukhalifoun = set()
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(700,500))
    result = model(frame)
    dimensions=[]
    for index, row in result.pandas().xyxy[0].iterrows():
        x1=int(row['xmin'])
        y1=int(row['ymin'])
        x2=int(row['xmax'])
        y2=int(row['ymax'])
        b=str(row['name'])
        # show only persons
        if 'person' in b:
            dimensions.append([x1,y1,x2,y2])
    boxes_ids=tracker.update(dimensions)
    cv2.polylines(frame,[np.array(crosswalk,np.int32)], True,(0,255,0),3)
    for box_id in boxes_ids:
        x,y,w,h,id=box_id
        cv2.rectangle(frame,(x ,y),(w,h),(255,0,255),2)
        cv2.putText(frame,str(id),(x,y),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
        results=cv2.pointPolygonTest(np.array(crosswalk,np.int32),(int(w/2),int(h)),False)    
        if results < 0 :
            moukhalifoun.add(id)
            # filename = f'{str(id)}.jpg'
            now = time.localtime()
            filename = f"{str(id)}_{now.tm_year}{now.tm_mon}{now.tm_mday}_{now.tm_hour}{now.tm_min}{now.tm_sec}.jpg"
            cv2.imwrite(filename, frame)
    cv2.imshow('FRAME',frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
    
    
