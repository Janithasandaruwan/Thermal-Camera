import serial
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import cv2
import face_recognition
import os
from datetime import datetime
#-*- coding: utf-8 -*-

path = 'Images'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


encodeList = []
for img in images:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encodeList.append(encode)

cap = cv2.VideoCapture(0)



class sensor:

    def __init__(self):

        #For Thermal camera
        ser = serial.Serial('COM4', 57600, bytesize=serial.EIGHTBITS, stopbits=serial.STOPBITS_ONE, timeout=1) #Open the com port
        values = bytearray([0xAD, 0x02, 0x0D, 0xD0, 0x4E, 0x00]) #Hex command for obtain the data packet
        ser.write(values) #Send the command
        all_bytes=ser.read(576).hex()
        chunks,chunk_size = len(all_bytes),len(all_bytes)//(len(all_bytes)//2)
        self.data_list=np.array([all_bytes[i:i+chunk_size] for i in range(0,chunks,chunk_size)]) #creat the array with all data


    def human_temp(self):
        t_ambient = ((int(self.data_list[11], 16) * 256 + int(self.data_list[12], 16)) - 27315) / 100
        self.forehead_temp = round(self.max_temp*(273/256) + 1.68,2)
        #print(t_ambient)
        #print(self.max_temp)
        #print(self.forehead_temp)

    def object_temp(self):
        idx = 0
        raw_data = [0] * 512
        raw_temp = [0] * 256
        for i in range(9):
            if i == 0:
                offset = 15
                data_cnt = 49
            elif i == 8:
                offset = 2
                data_cnt = 29
            else:
                offset = 2
                data_cnt = 62
            reverse = True
            offset = 64 * i + offset
            for j in range(offset, offset + data_cnt):
                raw_data[idx] = self.data_list[j]
                idx = idx + 1

        if idx >= 512:
            idx = 0
            for i in range(0, 512, 2):
                if (i + 1) < 512:
                    temp = ((int(raw_data[i], 16) * 256 + int(raw_data[i + 1], 16)) - 27315) / 100
                    raw_temp[idx] = float(temp)
                    idx = idx + 1
        temp_array = np.array(raw_temp)
        self.grid_array = temp_array.reshape(16,16)
        self.max_temp = np.max(self.grid_array)
        #print(self.grid_array)

    def map_data(self):
        ax = sns.heatmap(self.grid_array,xticklabels=False, yticklabels=False, linewidth=0,vmin=20,vmax=40, cmap="YlOrRd",annot=True,cbar=False)
        #plt.show()

    def detect_face(self):
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeList, encodeFace)
            faceDis = face_recognition.face_distance(encodeList, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                self.object_temp()
                self.human_temp()
                name = classNames[matchIndex] + " T=" + str(self.forehead_temp)

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.rectangle(img, (x1, y2), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.QT_FONT_NORMAL, 1, (0,0,255), 1)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)

if __name__ == "__main__":

    while True:
        module = sensor()
        module.detect_face()


