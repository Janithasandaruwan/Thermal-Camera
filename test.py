import cv2
import face_recognition

#Load the images and convert it to RGB
#We are getting the images as BGR but Dlib understand it as RGB
original_img = face_recognition.load_image_file('Images/Elon Musk.jpg')
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
test_img = face_recognition.load_image_file('Images/Bill Gates.jpg')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

#Finding faces in our image

#Detect the faces in image
face_loc1 = face_recognition.face_locations(original_img)[0]
#Find encodings of the image
encode_img1 = face_recognition.face_encodings(original_img)[0]
cv2.rectangle(original_img, (face_loc1[3], face_loc1[0]), (face_loc1[1], face_loc1[2]), (255, 0, 255), 2)

face_loc2 = face_recognition.face_locations(test_img)[0]
encode_img2 = face_recognition.face_encodings(test_img)[0]
cv2.rectangle(test_img, (face_loc2[3], face_loc2[0]), (face_loc2[1], face_loc2[2]), (255, 0, 255), 2)

#Comparing these two faces and finding distance between them
results = face_recognition.compare_faces([encode_img1], encode_img2)
#Find the best match when there are many faces
face_distance = face_recognition.face_distance([encode_img1], encode_img2)
print(results, face_distance)
cv2.putText(test_img, f'{results} {round(face_distance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Elon Musk', original_img)
cv2.imshow('Elon Test', test_img)
cv2.waitKey(0)