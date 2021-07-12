#04. keras_train_model.py
import tensorflow as tf
import matplotlib.pyplot as plt

#모의고사 문제집 느낌 #훈련 단계
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../data/',
    validation_split=0.2,
    subset='training', #subset을 트레이닝해줘서 80퍼센트를 가져옴
    seed=123,
    image_size=(224,224),
    batch_size=16 #16개짜리 랜덤그룹으로 쪼개서 한 번 학습할 때 드는 연산량을 줄임 (연산 작업량을 줄임)
)

#수능 본시험 느낌 #평가 단계
valid_dataset = tf.dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../data/',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(224,224),
    batch_size=16
)

#Sequential:이어주는 역할
resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

rc_train_dataset = train_dataset.map(lambda x, y:(resize_and_crop(x),y))
rc_valid_dataset = valid_dataset.map(lambda x, y:(resize_and_crop(x),y))

#모델 생성
#MobileNet:모바일용으로 개발하여 연산량 적음
model = tf.keras.applications.MobileNet(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'#이미지넷(ImageNet)데려옴(최상위 모델 와...)
)

model.trainable = False

model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalAveragePooling2D(), #풀링: 연산량 감소하게 함 에버러지 풀링:대표값을 평균을 내어 연산량 감소하게 함(그러면서 의미는 포함하고 있음)
    tf.keras.layers.Dense(1) #값이 하나로 나오게 함
])

#모델 학습
learning_rate = 0.0001
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), #crossentropy(분류 문제들)이 2개 짜리일 때 (지금처럼 withmask/without~로 binary일 때) #잠재력
    optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),#loss 값을 보고 다시 tuning을 통해 좋은 방향으로 계속 발전) lr=learning rate->속도(작으면 느려지고 크면 빨라짐)
    metrics=['accuracy'] #정확성을 봄
)

print(model.summary())

history = model.fit(
    rc_train_dataset,
    epochs=2,
    validation_data=rc_valid_dataset)
print(history)