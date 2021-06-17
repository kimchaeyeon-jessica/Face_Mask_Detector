# 03. keras_create_model.py
import tensorflow as tf

model = tf.keras.applications.VGG16()
print(model.summary())
#인공신경망 모델 종류
#모델을 다양하게 불러다가 학습 데이터(는 이미 있으니깐)넣어볼 수 있음