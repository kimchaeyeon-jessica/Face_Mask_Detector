# 02. keras_load_data.py
import tensorflow as tf
import matplotlib.pyplot as plt

#tensorflow 내부의 keras 기능 중 preprocessing(전처리)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../data/', #
    validation_split=0.2, #모의고사랑 본고사를 구분하는 작업
    subset='training',
    seed=123, #대부분 현재시간 값을 많이 활용함 (계속 해서 새로 생성됨)
    image_size=(224,224),
    batch_size=16, #연산량
)

valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    '../data/',
    validation_split=0.2,
    subseb='validation',
    seed=123,
    image_size=(224,224),
    batch_size=16
)

resize_and_crop = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomCrop(height=224,width=224),
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
])

rc_train_dataset = train_dataset.map(lambda x,y: (resize_and_crop(x),y))
rc_valid_dataset = valid_dataset.map(lambda x,y: (resize_and_crop(x),y))

print(rc_train_dataset) #인공지능의 학습데이터로 쓰이게 됨
print(rc_valid_dataset)

print(train_dataset.class_names)

plt.figure(0)
plt.title('train_dataset')
for images,labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(train_dataset.class_name[labels[i]])
        plt.axis('off')

plt.figure(1)
plt.title('valid_dataset')
for images, labels in valid_dataset.take(1):
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(valid_dataset.class_names[labels[i]])
        plt.axis('off')

plt.show()

