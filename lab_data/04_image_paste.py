import face_recognition
from PIL import Image, ImageDraw
import numpy
import math

image_path = '../data/ElonMusk.jpg'
mask_image_path = '../data/mask.png'

face_image_np = face_recognition.load_image_file(image_path)
face_image = Image.fromarray(face_image_np)
draw = ImageDraw.Draw(face_image)

mask_image = Image.open(mask_image_path)
mask_image = mask_image.resize((math.sqrt(math.pow(face_landmark['chin'][1][0] - face_landmark['chin'][15][0] , 2) + math.pow(face_landmark['chin'][1][1] - face_landmark['chin'][15][1] , 2) ),  math.sqrt(math.pow(face_landmark['nose_bridge'][1][0] - face_landmark['chin'][8][0] , 2) + math.pow(face_landmark['nose_bridge'][1][1] - face_landmark['chin'][8][1] , 2) )))  #마스크 사이즈를 리사이즈함

face_image.paste(mask_image, (right-left,bottom), mask_image) #마스크 사진에서 마스크만 추출하고 나머지 영역을 투명으로 바꿔줌
face_image.show()