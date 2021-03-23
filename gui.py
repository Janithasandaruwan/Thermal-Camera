import serial
import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime
import pygame
import time


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

pygame.init()
white = (255, 255, 255)
screen_width = 480
screen_height = 320
display_surface = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Human Temperature')
img1 = 'Back/1.jpg'
img2 = 'Back/2.jpg'
img3 = 'Back/3.jpg'
img4 = 'Back/wall.jpg'
image = pygame.image.load(img1)

font1 = pygame.font.SysFont('bahnschrift', 20)
font2 = pygame.font.SysFont('malgungothic', 60)
font3 = pygame.font.SysFont('bahnschrift', 40)
iter = 0
temp = []
result = ""
color = (0,0,0)
dateTime = ""
class sensor:
    def __init__(self):

        #For Thermal camera
        ser = serial.Serial('COM4', 57600, bytesize=serial.EIGHTBITS, stopbits=serial.STOPBITS_ONE, timeout=1) #Open the com port
        values = bytearray([0xAD, 0x02, 0x0D, 0xD0, 0x4E, 0x00]) #Hex command for obtain the data packet
        ser.write(values) #Send the command
        all_bytes=ser.read(576).hex()
        chunks,chunk_size = len(all_bytes),len(all_bytes)//(len(all_bytes)//2)
        self.data_list=np.array([all_bytes[i:i+chunk_size] for i in range(0,chunks,chunk_size)]) #creat the array with all data
        self.switch = False


    def human_temp(self):
        self.object_temp()
        #self.t_ambient = ((int(self.data_list[11], 16) * 256 + int(self.data_list[12], 16)) - 27315) / 100

        check_temp = self.max_temp
        if check_temp < 32.5:
            self.forehead_temp = round(self.max_temp * (273 / 256) + 1.68, 1)
        elif 32.5 <= check_temp < 35:
            self.forehead_temp = round(self.max_temp * (57 / 256) + 29.13, 1)
        elif check_temp > 35:
            self.forehead_temp = round(self.max_temp * (223 / 256) + 6.58, 1)

        return self.forehead_temp

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


    def detect_face(self):
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        self.user_name = ""
        self.user_temp = ""
        self.color = (0, 255, 0)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeList, encodeFace)
            faceDis = face_recognition.face_distance(encodeList, encodeFace)
            matchIndex = np.argmin(faceDis)
            value = np.min(faceDis)

            if value < 0.5:
                self.user_name = classNames[matchIndex]

            else:
                self.user_name = "Unknown User"
            self.switch = True

    def display(self):

        text3 = font1.render(dateTime, True, (255, 255, 255))
        textRect3 = text3.get_rect()
        textRect3.center = (screen_width // 2, 290)
        display_surface.blit(text3, textRect3)

        text2 = font2.render(result, True, color)
        textRect2 = text2.get_rect()
        textRect2.center = (383, 135)
        display_surface.blit(text2, textRect2)

        text1 = font3.render(self.user_name, True, (255, 255, 255))
        textRect1 = text1.get_rect()
        textRect1.center = (screen_width // 2, 33)
        display_surface.blit(text1, textRect1)





if __name__ == "__main__":

    try:
        while True:
            module = sensor()
            display_surface.fill(white)
            display_surface.blit(image, (0, 0))

            module.detect_face()
            module.display()

            if module.switch == True:
                iter +=1
                module.human_temp()
                temp.append(module.forehead_temp)

                if iter ==3:
                    final_temp = round((temp[0] + temp[1] + temp[2])/3,1)
                    result = str(final_temp) + u'\N{DEGREE SIGN}'+"C"
                    x = int(float(final_temp))
                    dateTime = datetime.now().strftime("%H:%M:%S %p | %d/%b/%Y")
                    if x < 37.5:

                        color = (0,255,0)
                        image = pygame.image.load(img2)
                    else:
                        color = (255,0,0)
                        image = pygame.image.load(img3)

            else:
                iter = 0
                temp = []
                result = ""
                color = (0,0,0)
                dateTime = ""
                image = pygame.image.load(img1)


            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            pygame.display.update()
    except:
        print("Check the connections of Thermal sensor")


