import numpy as np
import csv
import scipy.io as sio
import pdb
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling2D, Input, GlobalMaxPooling2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Sequential, Model, load_model
from keras.losses import binary_crossentropy
from keras.utils import to_categorical
from keras.optimizers import Adam, RMSprop
from keras.layers.core import Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.io import loadmat



def load_dataset():

  labels = sio.loadmat('../../Data/labels.mat')
  labels = labels['label_impact_noimpact']
  data = sio.loadmat('../../Data/data.mat')
  data = data['data']

  # labels = sio.loadmat('../../Data/labels_dirty.mat')
  # labels = labels['labels_dirty']
  # data = sio.loadmat('../../Data/data_dirty.mat')
  # data = data['data_dirty']

  return data, labels


def normalize_features(data):
  mean = np.mean(data,axis=1)
  std = np.std(data,axis=1)
  data_normalized = np.zeros(data.shape)

  for t in range(data.shape[1]):
    data_normalized[:,t,:] = (data[:,t,:] - mean)/std

  return data_normalized


def split_dataset_train_test(data,labels,train_split = 0.7,test_split = 0.3):

  # Train set
  sample_num = len(data[:, 1, 1])
  train_x = data[0:int(np.floor(train_split*sample_num)), :, :]
  train_y = labels[0:int(np.floor(0.7*sample_num))]
  print('train_x shape: ' + str(train_x.shape))
  print('train_y shape: ' + str(train_y.shape))

  # Test set
  test_x = data[int(np.floor(train_split*sample_num)):, :, :]
  test_y = labels[int(np.floor(train_split*sample_num)):]
  print('test_x shape: ' + str(test_x.shape))
  print('test_y shape: ' + str(test_y.shape))

  print('Data shape: ' + str(data.shape))
  #labels = to_categorical(labels, num_classes=2)
  print('Labels shape: ' + str(type(labels)))

  sample_num = len(data[:, 1, 1])
  print('Sample Number: ' + str(sample_num))

  return train_x, train_y, test_x, test_y

def fit_model1(train_x, train_y, eval_x, eval_y, n_epochs = 10, batch_size = 32):

  # model taken directly from PerceptionNet paper


  n_timesteps, n_features = train_x.shape[1], train_x.shape[2]

  model = Sequential()

  # Layer 1
  model.add(Conv1D(filters=48, kernel_size=15, activation='relu', input_shape=(n_timesteps,n_features)))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Dropout(0.2))
  print('Layer 1 output shape:' + str(model.output_shape))


  # Layer 2
  model.add(Conv1D(filters=96, kernel_size=15, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Dropout(0.2))
  layer2_output_shape = model.output_shape
  print('Layer 2 output shape:' + str(layer2_output_shape))

  # reshape output for Conv2D filter
  model.add(Reshape((layer2_output_shape[1],layer2_output_shape[2],-1)))

  # Layer 3
  model.add(Conv2D(filters=96, kernel_size=(3,15), strides=(3,1), activation = 'relu'))
  #model.add(Conv2D(filters=96, kernel_size=(3,15), strides=(3,1), activation = 'relu'))
  model.add(GlobalAveragePooling2D())
  model.add(Dropout(0.2))
  print('Layer 3 output shape:' + str(model.output_shape))

  # output layer
  model.add(Dense(units=1, activation='sigmoid'))
  model.summary()

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  early_stopper = EarlyStopping(patience=5, verbose=1)
  check_pointer = ModelCheckpoint(filepath='net_4_finalLayer.hdf5', verbose=1, save_best_only=True)
  results = model.fit(train_x, train_y,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    callbacks=[early_stopper, check_pointer],
                    validation_data=(eval_x, eval_y))
  model = load_model('net_4_finalLayer.hdf5')
  return model, results


def fit_model2(train_x, train_y, eval_x, eval_y, n_epochs = 10, batch_size = 32):

  # model taken directly from PerceptionNet paper


  n_timesteps, n_features = train_x.shape[1], train_x.shape[2]

  model = Sequential()

  # Layer 1
  model.add(Conv1D(filters=48, kernel_size=15, activation='relu', input_shape=(n_timesteps,n_features)))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Dropout(0.2))
  print('Layer 1 output shape:' + str(model.output_shape))


  # Layer 2
  model.add(Conv1D(filters=96, kernel_size=15, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Dropout(0.2))
  layer2_output_shape = model.output_shape
  print('Layer 2 output shape:' + str(layer2_output_shape))

  # reshape output for Conv2D filter
  model.add(Reshape((layer2_output_shape[1],layer2_output_shape[2],-1)))

  # Layer 3
  model.add(Conv2D(filters=96, kernel_size=(3,15), strides=(3,1), activation = 'relu'))
  #model.add(Conv2D(filters=96, kernel_size=(3,15), strides=(3,1), activation = 'relu'))
  model.add(GlobalMaxPooling2D())
  model.add(Dropout(0.2))
  print('Layer 3 output shape:' + str(model.output_shape))

  # output layer
  model.add(Dense(units=1, activation='sigmoid'))
  model.summary()

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  early_stopper = EarlyStopping(patience=5, verbose=1)
  check_pointer = ModelCheckpoint(filepath='net_4_finalLayer.hdf5', verbose=1, save_best_only=True)
  results = model.fit(train_x, train_y,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    callbacks=[early_stopper, check_pointer],
                    validation_data=(eval_x, eval_y))
  model = load_model('net_4_finalLayer.hdf5')
  return model, results



def fit_model3(train_x, train_y, eval_x, eval_y, n_epochs = 10, batch_size = 32):

  # model taken directly from PerceptionNet paper


  n_timesteps, n_features = train_x.shape[1], train_x.shape[2]

  model = Sequential()

  # Layer 1
  model.add(Conv1D(filters=48, kernel_size=15, activation='relu', input_shape=(n_timesteps,n_features)))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Dropout(0.2))
  print('Layer 1 output shape:' + str(model.output_shape))


  # Layer 2
  model.add(Conv1D(filters=96, kernel_size=15, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Dropout(0.2))
  layer2_output_shape = model.output_shape
  print('Layer 2 output shape:' + str(layer2_output_shape))

  # reshape output for Conv2D filter
  model.add(Reshape((layer2_output_shape[1],layer2_output_shape[2],-1)))

  # Layer 3
  model.add(Conv2D(filters=96, kernel_size=(3,15), strides=(3,1), activation = 'relu'))
  #model.add(Conv2D(filters=96, kernel_size=(3,15), strides=(3,1), activation = 'relu'))
  model.add(Flatten())
  model.add(Dense(250))
  model.add(Dropout(0.2))
  print('Layer 3 output shape:' + str(model.output_shape))

  # output layer
  model.add(Dense(units=1, activation='sigmoid'))
  model.summary()

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  early_stopper = EarlyStopping(patience=5, verbose=1)
  check_pointer = ModelCheckpoint(filepath='net_4_finalLayer.hdf5', verbose=1, save_best_only=True)
  results = model.fit(train_x, train_y,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    callbacks=[early_stopper, check_pointer],
                    validation_data=(eval_x, eval_y))
  model = load_model('net_4_finalLayer.hdf5')
  return model, results

def evaluate_model(model, train_x, train_y, test_x, test_y):

  _, accuracy_test = model.evaluate(test_x, test_y)
  _, accuracy_train = model.evaluate(train_x, train_y)  

  return accuracy_test, accuracy_train


def compute_roc_curve(model, test_x, test_y):
  y_pred = model.predict(test_x).ravel()
  fpr, tpr, thresholds = roc_curve(test_y,y_pred)

  area_under_curve = auc(fpr, tpr)

  print('area under ROC curve:' + str(area_under_curve))

  plt.figure()
  plt.plot(fpr,tpr)
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.show()

def compute_pr_curve(model, test_x, test_y):
  y_pred = model.predict(test_x).ravel()
  precision, recall, thresholds = precision_recall_curve(test_y, y_pred)
  area_under_curve = auc(recall,precision)
  print('area under PR curve:' + str(area_under_curve))
  plt.figure()
  plt.plot(recall,precision)
  plt.ylim((0,1))
  plt.xlabel('recall')
  plt.ylabel('precision')
  plt.title('PR curve')
  plt.show()

def compute_performance_metrics(model,test_x,test_y):
  y_pred = model.predict_classes(test_x)
  TP = 0
  FP = 0
  TN = 0
  FN = 0

  for i in range(len(y_pred)): 
      if test_y[i]==y_pred[i]==1:
         TP += 1
      if y_pred[i]==1 and test_y[i]!=y_pred[i]:
         FP += 1
      if test_y[i]==y_pred[i]==0:
         TN += 1
      if y_pred[i]==0 and test_y[i]!=y_pred[i]:
         FN += 1

  sensitivity = TP/(TP+FN)
  specificity = TN/(TN+FP)
  accuracy = (TP+TN)/(TP+TN+FP+FN)
  precision = TP/(TP+FP)

  return TP, FP, TN, FN, sensitivity, specificity, accuracy, precision

def run_experiment(n_loops = 1, n_epochs = 100, batch_size = 32):

  # load and process data
  data, labels = load_dataset()
  data = normalize_features(data)
  #train_x, train_y, eval_x, eval_y, test_x, test_y = split_dataset(data,labels)
  train_x, train_y, test_x, test_y = split_dataset_train_test(data,labels)
  eval_x, eval_y = test_x, test_y


  sensitivity_1 = np.array([])
  specificity_1 = np.array([])
  accuracy_1 = np.array([])
  precision_1 = np.array([])

  sensitivity_2 = np.array([])
  specificity_2 = np.array([])
  accuracy_2 = np.array([])
  precision_2 = np.array([])

  sensitivity_3 = np.array([])
  specificity_3 = np.array([])
  accuracy_3 = np.array([])
  precision_3 = np.array([])



  for r in range(n_loops):

    # train model 1
    print('model1')
    model, results = fit_model1(train_x, train_y, eval_x, eval_y, n_epochs = n_epochs, batch_size = batch_size)

    TP, FP, TN, FN, sensitivity, specificity, accuracy, precision = compute_performance_metrics(model,test_x,test_y)

    sensitivity_1 = np.append(sensitivity_1,sensitivity)
    specificity_1 = np.append(specificity_1,specificity)
    accuracy_1 = np.append(accuracy_1,accuracy)
    precision_1 = np.append(precision_1,precision)

    # train model 2
    print('model2')
    model, results = fit_model2(train_x, train_y, eval_x, eval_y, n_epochs = n_epochs, batch_size = batch_size)

    TP, FP, TN, FN, sensitivity, specificity, accuracy, precision = compute_performance_metrics(model,test_x,test_y)

    sensitivity_2 = np.append(sensitivity_2,sensitivity)
    specificity_2 = np.append(specificity_2,specificity)
    accuracy_2 = np.append(accuracy_2,accuracy)
    precision_2 = np.append(precision_2,precision)

    # train model 3
    print('model3')
    model, results = fit_model3(train_x, train_y, eval_x, eval_y, n_epochs = n_epochs, batch_size = batch_size)

    TP, FP, TN, FN, sensitivity, specificity, accuracy, precision = compute_performance_metrics(model,test_x,test_y)

    sensitivity_3 = np.append(sensitivity_3,sensitivity)
    specificity_3 = np.append(specificity_3,specificity)
    accuracy_3 = np.append(accuracy_3,accuracy)
    precision_3 = np.append(precision_3,precision)


  metrics1 = (sensitivity_1,specificity_1,accuracy_1,precision_1)
  metrics2 = (sensitivity_2,specificity_2,accuracy_2,precision_2)
  metrics3 = (sensitivity_3,specificity_3,accuracy_3,precision_3)



  return metrics1, metrics2, metrics3

metrics1, metrics2, metrics3 = run_experiment(n_loops = 10)

sensitivity1 = np.mean(metrics1[0])
specificity1 = np.mean(metrics1[1])
accuracy1 = np.mean(metrics1[2])
precision1 = np.mean(metrics1[3])

sensitivity2 = np.mean(metrics2[0])
specificity2 = np.mean(metrics2[1])
accuracy2 = np.mean(metrics2[2])
precision2 = np.mean(metrics2[3])

sensitivity3 = np.mean(metrics3[0])
specificity3 = np.mean(metrics3[1])
accuracy3 = np.mean(metrics3[2])
precision3 = np.mean(metrics3[3])



print('sensitivity1:' + str(sensitivity1))
print('specificity1:' + str(specificity1))
print('accuracy1:' + str(accuracy1))
print('precision1:' + str(precision1))

print('sensitivity2:' + str(sensitivity2))
print('specificity2:' + str(specificity2))
print('accuracy2:' + str(accuracy2))
print('precision2:' + str(precision2))

print('sensitivity3:' + str(sensitivity3))
print('specificity3:' + str(specificity3))
print('accuracy3:' + str(accuracy3))
print('precision3:' + str(precision3))



accuracy_data = np.asarray([metrics1[2],metrics2[2],metrics3[2]])

np.savetxt('results2.txt',accuracy_data)

fig,ax = plt.subplots()
ax.set_title('Accuracy with different final layer')
ax.boxplot(accuracy_data.T,showfliers=False)
plt.show()

pdb.set_trace()


