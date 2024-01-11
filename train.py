from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers.legacy import Adam

train = ImageDataGenerator(rescale=1./255,
                           shear_range=0.2,
                           zoom_range=0.2,
                           horizontal_flip=True)

test = ImageDataGenerator(rescale=1./255)

train_set = train.flow_from_directory('data/train',
                                      target_size=(64,64),
                                      class_mode='categorical',
                                      color_mode='grayscale')

test_set = test.flow_from_directory('data/test',
                                      target_size=(64,64),
                                      class_mode='categorical',
                                      color_mode='grayscale')

model = Sequential()

model.add(Conv2D(input_shape=(64,64,1),filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))


model.add(Flatten())

model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=5,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])

model.fit(train_set, epochs=20, batch_size=64, validation_data=test_set)

model_json = model.to_json()

with open("model-bw.json",'w') as json_file:
    json_file.write(model_json)

model.save_weights('model-bw.h5')

