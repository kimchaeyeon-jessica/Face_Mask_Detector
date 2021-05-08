import face_recognition
from PIL import Image, ImageDraw

image_path = 'data/without_mask/1.jpg'
mask_image_path = 'data/mask.png'

face_image_np=face_recognition.load_image_file(image_path)

face_locations = face_recognition.face_locations(face_image_np, model='hog0')
#face_location 기능 제공, hog라는 학습 되어 있는 모델 (그냥 재료를 넣으면 처리 결과가 튀어나옴 마법의 상자)
#얼굴이 포함된 이미지를 넣으면 얼굴의 좌표를 return 해줌

#print(face_locations) #리스트로 되어 있음 -> 얼굴이 여러 개여도 리스트로 모두 보여줌

face_image = Image.fromarray(face_image_np)
draw = ImageDraw.Draw(face_image) #얼굴 이미지를 그리기 위한 기능을 연결 시켜줌, Draw변수를 이용해서 face image에 그릴 수 있음

for face_location in face_locations:
    top = face_location[0]
    right = face_location[1]
    bottom = face_location[2]
    left = face_location[3]
    draw.rectangle(((left, top),(right, bottom)), outline=(255,0,0), width=4)
    #얼굴 영역 표시 (빨간 굵기 4 네모로 표시)

face_image_np = face_recognition.load_image_file(image_path)
face_image = Image.fromarray(face_image_np)
draw = ImageDraw.Draw(face_image)

mask_image = Image.open(mask_image_path)
mask_image = mask_image.resize((right-left,bottom-top-35))  #마스크 사이즈를 리사이즈함

face_image.paste(mask_image, (left,top+40), mask_image) #마스크 사진에서 마스크만 추출하고 나머지 영역을 투명으로 바꿔줌
face_image.show()