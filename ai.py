# ai.py
import tensorflow as tf
import numpy as np
import os

resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224, width=224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255) #정규화시킴
])


# 학습 데이터 로드 -데이터 수집, 가공은 이미 했기때문에 로드만 해줌!
def load_data():
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'data/',
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(224, 224),
        batch_size=16
    ) #모의고사 문제

    valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'data/',
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(224, 224),
        batch_size=16
    ) #시험 문제

    rc_train_dataset = train_dataset.map(lambda x, y: (resize_and_crop(x), y))
    rc_valid_dataset = valid_dataset.map(lambda x, y: (resize_and_crop(x), y))

    return rc_train_dataset, rc_valid_dataset


# 모델 생성
#저장된 모델이 있으면 가져오고 없으면 그때 생성하게 함
def create_model():
    if os.path.exists('models/mymodel'):
        model = tf.keras.models.load_model('models/mymodel') #마이모델 모델 있다면 그대로 불러옴

        model.layers[0].trainable = False   #학습이 가능하게 만들것인가
        model.layers[2].trainable = True    #없어도 상관 없는데 불러와서 더 학습할 때 (중간 상태 저장해놓고 이어하기)를 위해서 씀
    else:
        model = tf.keras.applications.MobileNet(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )

        model.trainable = False

        model = tf.keras.Sequential([
            model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1)
        ])

        learning_rate = 0.001
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
            metrics=['accuracy']
        )

        train_dataset, valid_dataset = load_data()
        train_model(model, 2, train_dataset, valid_dataset, True) #epochs 나중에 바꿔도 됨
    return model


# 모델 학습
def train_model(model, epochs, train_dataset, valid_dataset, save_model): #(학습할)모델,몇번할지,학습을 위한 트레인데이터셋,,저장할지말지 결정하는 모델로 5개의 인자 받아옴
    history = model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset)
    if save_model:
        model.save('models/mymodel')
    return history  #학습 내역 반환 (나중에 필요할 수도 있으니깐)


# 학습된 모델로 예측
def predict(model, image):
    rc_image = resize_and_crop(np.array([image])) #np: 여러 이미지 동시에 예측가능하게 설계돼서 하나만 넣어도 np배열로 만들어서 넣어줘야 함
    result = model.predict(rc_image)   #최종 가공된 데이터를 rc_image에 넣어줌
    if result[0] > 0.5:
        return 1   #without_mask
    else:
        return 0  #with_mask


if __name__ == '__main__':
    train_dataset, valid_dataset = load_data()
    model = create_model()
    train_model(model, 2, train_dataset, valid_dataset, True) #학습 누적된 모델 생성
    #print(model.summary())


#인공지능의 학습 과정 4가지는 항상 이렇다 하지만 그 안의 코드는 라이브러리에 따라, 모델에 따라 사람마다 모두 다 달라질 수 있다.