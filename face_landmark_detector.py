#얼굴 랜드마크 추출
import face_recognition
from PIL import Image, ImageDraw
import math

face_image_path = 'data/ElonMusk.jpg'

face_image_np = face_recognition.load_image_file(face_image_path) #배열 정보
face_locations = face_recognition.face_locations(face_image_np)
face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)

face_landmark_image = Image.fromarray(face_image_np)
draw = ImageDraw.Draw(face_landmark_image)

print('==  ==')
print(face_landmarks)

for face_landmark in face_landmarks:
    print('== face_landmark ==')
    print(face_landmark)

for face_landmark in face_landmarks:
    for feature_name, points in face_landmark.items():
        print(feature_name, points)
        for point in points:
            draw.point(point)

#face_landmark_image.show()


########
#image_path = 'data/ElonMusk.jpg'
mask_image_path = 'data/mask.png'

#face_image_np = face_recognition.load_image_file(face_image_path)
#face_image = Image.fromarray(face_image_np)
#draw = ImageDraw.Draw(face_image)

A = int(math.sqrt(math.pow(face_landmark['chin'][15][0]- face_landmark['chin'][1][0], 2) + math.pow(face_landmark['chin'][15][1]-face_landmark['chin'][1][1],2)))+2*(face_landmark['left_eyebrow'][0][0]-face_landmark['chin'][0][0])
B = int(math.sqrt(math.pow(face_landmark['nose_bridge'][0][0] - face_landmark['chin'][8][0], 2) + math.pow(face_landmark['nose_bridge'][0][1] - face_landmark['chin'][8][1], 2)))

mask_image = Image.open(mask_image_path)
mask_image = mask_image.resize((A,B))  #마스크 사이즈를 리사이즈함

#face_landmark_image.paste(mask_image,(face_landmark['chin'][0][0]-(face_landmark['left_eyebrow'][0][0]-face_landmark['chin'][0][0]),face_landmark['nose_bridge'][0][1]),mask_image) #마스크 사진에서 마스크만 추출하고 나머지 영역을 투명으로 바꿔줌
#face_landmark_image.show()

#-(face_landmark['left_eyebrow'][0][0]-face_landmark['chin'][0][0]) 밑에서 두번째 줄 코드 x좌표에서 삭제함


#########이거 안됌 다시!!!!!!! 다시 코드 짜야함 !!!!!!!!(기울어진 사진에 적용하는 코드)
gradient= (face_landmark['nose_bridge'][3][1]-face_landmark['nose_bridge'][0][1])/(face_landmark['nose_bridge'][0][0]-face_landmark['nose_bridge'][3][0])
mask_image=mask_image.rotate(gradient)
face_landmark_image.paste(mask_image,(face_landmark['chin'][0][0]-(face_landmark['left_eyebrow'][0][0]-face_landmark['chin'][0][0]),face_landmark['nose_bridge'][0][1]),mask_image) #마스크 사진에서 마스크만 추출하고 나머지 영역을 투명으로 바꿔줌
face_landmark_image.show()