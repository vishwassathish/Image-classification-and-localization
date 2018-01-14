from NNfunctions import get_dataset , predict_bnd_box 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

NB_TRAIN = 3000
x_master, y_master, z_master = get_dataset() 

x_train = x_master[0:NB_TRAIN]
y_train = y_master[0:NB_TRAIN]

x_val = x_master[NB_TRAIN:]
y_val = y_master[NB_TRAIN:]



model = Sequential()


model.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation = 'relu'))

model.summary()

#model.load_weights('bnd_box.h5', by_name=True)


#model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size = 50, epochs = 20, validation_data = [x_val, y_val])

model.save_weights('bnd_box.h5')

predict_bnd_box('M:/5th Sem/AI/cnn/VOC2010Dataset/JPEGImages/2007_000187.jpg', model)