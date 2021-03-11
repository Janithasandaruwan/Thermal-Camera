import cv2
import face_recognition
import os
import numpy as np

path = "Images" #Path of the folder that contain images
images = [] #Create a list of all images that imports
image_names = [] #Get the names of images

#Grab list of images in that folder
img_list = os.listdir(path)

#Import images one by one and get its names
for nme in img_list:
    current_img = cv2.imread(f'{path}/{nme}') #Open the images using its path and name
    images.append(current_img) #Append images to images array
    image_names.append(os.path.splitext(nme)[0]) #Append image names to image_name array


#Find encodings of each images
def find_encodings(images):
    encode_list = []
    for img in images:
        # We are getting the images as BGR but Dlib understand it as RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

#Encodings of the known images
encode_img = find_encodings(images)

#Match this known images encode with web cam images
#ir_cam = cv2.VideoCapture(1)
original_cam = cv2.VideoCapture(1)

while True:
    success, img1 = original_cam.read()
    #success, img2 = ir_cam.read()
    #Reduce the size of image for the higher speed processing
    reduce_img = cv2.resize(img1,(0,0),None,0.25,0.25)
    #Find the encodings of web cam images
    reduce_img = cv2.cvtColor(reduce_img, cv2.COLOR_BGR2RGB)
    #In webcam images can detect multiple faces
    #So we have to find the locations of these faces
    #And then these locations for encoding
    faces_inFrame = face_recognition.face_locations(reduce_img)
    encode_frame = face_recognition.face_encodings(reduce_img,faces_inFrame)

    #Finding the matches by iterate through all the faces found in current frame
    #And compare all these faces with all known encodings

    for encodeFace, face_location in zip(encode_frame,faces_inFrame):
        matches = face_recognition.compare_faces(encode_img,encodeFace)
        #To find how similar these images for best match
        face_distance = face_recognition.face_distance(encode_img,encodeFace)
        #face_distance[0] is a list and lowest value is the best match
        match_index = np.argmin(face_distance)

        #Display bounding box around faces and display their name
        if matches[match_index]:
            #Display the upper case name of the image
            name = image_names[match_index].upper()
            y1,x2,y2,x1 = face_location
            #Since we scale down our image to 0.25,In order to revive actual values multiply by 4
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img1, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img1, name, (x1 + 6, y2 - 6), cv2.QT_FONT_NORMAL, 1, (255, 0, 255), 1)

    cv2.imshow("Web cam",img1)
    cv2.waitKey(1)

















