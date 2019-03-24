import numpy as np
import scipy.io as sio
from keras.layers import Dense, Dropout, Conv2D, Flatten, Input
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical

# This script should only be referenced for the RecursiveNet architecture,          ######
# the HIKNet.py file should be referenced for all else related to the project       ######
# (pre-processing, functions, etc.).                                                ######

labels = sio.loadmat('../../Data/labels.mat')
labels = labels['label_impact_noimpact']
labels_1d = labels
labels = to_categorical(labels, num_classes=2)

data = sio.loadmat('../../Data/data.mat')
data = data['data']
data = np.transpose(data, axes=(0, 2, 1))
data = np.expand_dims(data, -1)

sample_num = len(data[:, 0, 0, 0])
measurement_num = len(data[0, :, 0, 0])
time_num = len(data[0, 0, :, 0])
mean_values = np.mean(data, axis=2)
variance_values = np.var(data, axis=2)


# Standardizes the data
for m in range(0, measurement_num):
    for t in range(0, time_num):
        data[:, m, t, :] = data[:, m, t, :] - mean_values[:, m, :]
        data[:, m, t, :] = data[:, m, t, :] / variance_values[:, m, :]

# Splits the data
train_split = 0.7
eval_split = 0.15
test_split = 0.15
# Train set
train_x = data[0:int(np.floor(train_split*sample_num)), :, :]
train_y = labels[0:int(np.floor(0.7*sample_num))]
# Evaluation set
eval_x = data[int(np.floor(train_split*sample_num)):int(np.floor((train_split+eval_split)*sample_num)), :, :]
eval_y = labels[int(np.floor(train_split*sample_num)):int(np.floor((train_split+eval_split)*sample_num))]
# Test set
test_x = data[int(np.floor((train_split+eval_split)*sample_num)):, :, :]
test_y = labels[int(np.floor((train_split+eval_split)*sample_num)):]
test_y_1D = labels_1d[int(np.floor((train_split+eval_split)*sample_num)):]


# Model Architecture
input = Input((measurement_num, time_num, 1))

x1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last', input_shape=(6, 199, 1))(input)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x1)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Dropout(0.25)(x)

x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(concatenate([x, x2]))
x = Dropout(0.25)(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(concatenate([x, x1]))
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
output = Dense(2, activation='sigmoid')(x)

model = Model(inputs=[input], outputs=[output])
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopper = EarlyStopping(patience=5, verbose=1)
check_pointer = ModelCheckpoint(filepath='net_1.hdf5', verbose=1, save_best_only=True)
model.fit(train_x, train_y, batch_size=32, epochs=5, shuffle='true',
          callbacks=[early_stopper, check_pointer], validation_data=(test_x, test_y))

# Loads best loss epoch model
loaded_model = load_model('net_1_weights.hdf5')
# Evaluates the loaded model
evaluation = loaded_model.evaluate(eval_x, eval_y, verbose=0)
print('Evaluation Metrics: ', loaded_model.metrics_names[0], evaluation[0], loaded_model.metrics_names[1], evaluation[1])
# Makes the predictions from the loaded model
predictions = loaded_model.predict(eval_x)
