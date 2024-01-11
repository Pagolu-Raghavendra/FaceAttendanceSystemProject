import os
import pickle
import math
import time
import cvzone
import numpy as np
import cv2
import face_recognition
from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

confidence = 0.6

global cls

model = YOLO("../models/n_version_1_3.pt")
classNames = ["fake", "real"]

prev_frame_time = 0
new_frame_time = 0

id=-1
imgStudent=[]
from datetime import datetime
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,
                              {
                                  'databaseURL':"https://attendance-system-6e2e6-default-rtdb.firebaseio.com/",
                                  "storageBucket":"attendance-system-6e2e6.appspot.com"
                              }
                              )
bucket = storage.bucket()
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

imgBackground = cv2.imread('Resources/bgpic.png')

folderModePath='Resources/Modes'
modePath=os.listdir(folderModePath)
imgModeList=[]
for path in modePath:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

file=open('EncodeFile.p','rb')
encodeListKnownWithIds=pickle.load(file)
file.close()
encodeListKnown,student_id=encodeListKnownWithIds

modeType=0
counter=0

while True:
    success, img = cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame=face_recognition.face_encodings(imgS,faceCurFrame)

    imgBackground[562:562+720,940:940+1280]=img
    imgBackground[100:100 + 1378, 2270:2270 + 935] = imgModeList[modeType]
    if faceCurFrame:
        if counter==0:
            new_frame_time = time.time()
            success, img = cap.read()
            results = model(img, stream=True, verbose=False)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                    w, h = x2 - x1, y2 - y1
                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Class Name
                    cls = int(box.cls[0])
                    if conf > confidence:
                        if classNames[cls] == 'real':
                                if counter==0:
                                    cvzone.putTextRect(imgBackground,"Loading",(1390,1000),scale=9,thickness=5,colorT=(255,255,255),colorR=(0,0,0))
                                    cv2.imshow('Face Attendance', imgBackground)
                                    cv2.waitKey(1)
                                    counter = 1
                                    #modeType = 1
                                    #imgBackground[100:100 + 1378, 2270:2270 + 935] = imgModeList[modeType]
                        else:
                            cvzone.putTextRect(imgBackground, "Loading", (1390, 1000), scale=9, thickness=5,
                                               colorT=(255, 255, 255), colorR=(0, 0, 0))
                            cv2.imshow('Face Attendance', imgBackground)
                            cv2.waitKey(1)
                            counter=0
                            modeType=4
                            imgBackground[100:100 + 1378, 2270:2270 + 935] = imgModeList[modeType]
        if counter != 0:
            if counter == 1:
                new_frame_time = time.time()
                success, img = cap.read()
                results = model(img, stream=True, verbose=False)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Bounding Box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                        w, h = x2 - x1, y2 - y1
                        # Confidence
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        # Class Name
                        cls = int(box.cls[0])
                        if conf > confidence:
                            if classNames[cls] == 'real':
                                success, img = cap.read()
                                imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
                                imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                                faceCurFrame = face_recognition.face_locations(imgS)
                                encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

                                #imgBackground[562:562 + 720, 940:940 + 1280] = img
                                #imgBackground[100:100 + 1378, 2270:2270 + 935] = imgModeList[modeType]

                                for encoface, faceloc in zip(encodeCurFrame, faceCurFrame):
                                    matches = face_recognition.compare_faces(encodeListKnown, encoface)
                                    faceDis = face_recognition.face_distance(encodeListKnown, encoface)
                                    matchIndex = np.argmin(faceDis)
                                    print(faceloc)
                                    if matches[matchIndex]:
                                        id = student_id[matchIndex]
                                stu_id = id
                                studentInfo = db.reference(f'Students/{id}').get()

                                print(studentInfo)
                                blob = bucket.get_blob(f'Images/{id}.jpg')
                                if blob!=None:
                                    array = np.frombuffer(blob.download_as_string(), np.uint8)
                                    imgStudent = cv2.imdecode(array, cv2.COLOR_BGR2RGB)

                                    datetimeObject = datetime.strptime(studentInfo['last_attendance_time'],"%Y-%m-%d %H:%M:%S")
                                    durationElapsed = (datetime.now()-datetimeObject).total_seconds()
                                    print(durationElapsed)

                                    if durationElapsed>57600:#57600
                                        ref=db.reference(f'Students/{id}')
                                        studentInfo['total_attendance']+=1
                                        ref.child('total_attendance').set(studentInfo['total_attendance'])
                                        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                                        modeType=1
                                    else:
                                        modeType=3
                                        counter=0
                                        imgBackground[100:100 + 1378, 2270:2270 + 935] = imgModeList[modeType]
                                        c = 0
                                        for i in range(35):
                                            c += 1
                                        if (c >= 35):
                                            modeType = 0
                                            imgBackground[100:100 + 1378, 2270:2270 + 935] = imgModeList[modeType]
                                            modeType=3
                            else:
                                counter=0
                                modeType=0
                                imgBackground[100:100 + 1378, 2270:2270 + 935] = imgModeList[modeType]

            if modeType!=3:
                if 7<counter<14:
                    modeType=2
                    imgBackground[100:100 + 1378, 2270:2270 + 935] = imgModeList[modeType]
                if counter<=7:
                    if studentInfo!=None:
                        modeType = 1
                        imgBackground[100:100 + 1378, 2270:2270 + 935] = imgModeList[modeType]
                        cv2.putText(imgBackground,str(studentInfo['class']),(2630,1220), cv2.FONT_HERSHEY_DUPLEX,2,(0,0,0),thickness=5)
                        cv2.putText(imgBackground, str(stu_id), (2600, 1060), cv2.FONT_HERSHEY_DUPLEX, 1.7, (0, 0, 0),
                                    thickness=5)
                        (w,h),_=cv2.getTextSize(studentInfo['name'],cv2.FONT_HERSHEY_DUPLEX,1,1)
                        offset=(935-w)//2
                        cv2.putText(imgBackground, str(studentInfo['name']), (2100+offset, 930), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0),
                                    thickness=5)

                        imgBackground[375:375+413,2520:2520+413]=imgStudent


                counter+=1

                if counter>=14:
                    counter=0
                    modeType=0
                    studentInfo=[]
                    imgStudent=[]
                    imgBackground[100:100 + 1378, 2270:2270 + 935] = imgModeList[modeType]

    else:
        modeType=0
        counter=0

    cv2.namedWindow('Face Attendance', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Face Attendance', cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Face Attendance', imgBackground)
    cv2.waitKey(1)