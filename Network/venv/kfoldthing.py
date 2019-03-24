import numpy as np
from sklearn.model_selection import KFold
import csv
import scipy.io as sio
import pdb
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling2D, Input
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
  test_x = data[int(np.floor((train_split)*sample_num)):, :, :]
  test_y = labels[int(np.floor((train_split)*sample_num)):]
  print('test_x shape: ' + str(test_x.shape))
  print('test_y shape: ' + str(test_y.shape))

  print('Data shape: ' + str(data.shape))
  #labels = to_categorical(labels, num_classes=2)
  print('Labels shape: ' + str(type(labels)))

  sample_num = len(data[:, 1, 1])
  print('Sample Number: ' + str(sample_num))

  return train_x, train_y, test_x, test_y

def fit_model(train_x, train_y, eval_x, eval_y, n_epochs = 2, batch_size = 32):

  # model taken directly from PerceptionNet paper


  n_timesteps, n_features = train_x.shape[1], train_x.shape[2]

  model = Sequential()

  # Layer 1
  model.add(Conv1D(filters=48, kernel_size=15, activation='relu', input_shape=(n_timesteps,n_features)))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Dropout(0.4))
  print('Layer 1 output shape:' + str(model.output_shape))


  # Layer 2
  model.add(Conv1D(filters=96, kernel_size=15, activation='relu'))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Dropout(0.4))
  layer2_output_shape = model.output_shape
  print('Layer 2 output shape:' + str(layer2_output_shape))

  # reshape output for Conv2D filter
  model.add(Reshape((layer2_output_shape[1],layer2_output_shape[2],-1)))

  # Layer 3
  model.add(Conv2D(filters=96, kernel_size=(3,15), strides=(3,1), activation = 'relu'))
  #model.add(Conv2D(filters=96, kernel_size=(3,15), strides=(3,1), activation = 'relu'))
  model.add(GlobalAveragePooling2D())
  model.add(Dropout(0.4))
  print('Layer 3 output shape:' + str(model.output_shape))

  # output layer
  model.add(Dense(units=1, activation='sigmoid'))
  model.summary()

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  early_stopper = EarlyStopping(patience=5, verbose=1)
  check_pointer = ModelCheckpoint(filepath='net_4.hdf5', verbose=1, save_best_only=True)
  results = model.fit(train_x, train_y,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    callbacks=[early_stopper, check_pointer],
                    validation_data=(eval_x, eval_y))
  model = load_model('net_4.hdf5')
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


def kfold_function(k):
    data, labels = load_dataset()
    data = normalize_features(data)
    kfold = KFold(n_splits = k)
    data_train = []
    data_test = []
    labels_train = []
    labels_test = []
    for train, test in kfold.split(data):
        data_train.append(data[train])
        labels_train.append(labels[train])
        data_test.append(data[test])
        labels_test.append(labels[test])


    TP = np.zeros(k)
    FP = np.zeros(k)
    TN = np.zeros(k)
    FN = np.zeros(k)
    sensitivity = np.zeros(k)
    specificity = np.zeros(k)
    accuracy = np.zeros(k)
    precision = np.zeros(k)
    for i in range(0,k):
        model = fit_model(data_train[i], labels_train[i], data_test[i], labels_test[i])
        TP[i], FP[i], TN[i], FN[i], sensitivity[i], specificity[i], accuracy[i], precision[i] = compute_performance_metrics(model, data_test[i], labels_test[i])

    TP_final = np.mean(TP)
    FP_final = np.mean(FP)
    TN_final = np.mean(TN)
    FN_final = np.mean(FN)
    sense_final = np.mean(sensitivity)
    spec_final = np.mean(specificity)
    acc_final = np.mean(accuracy)
    prec_final = np.mean(precision)

    print('TP_final: ' + str(TP_final))
    print('FP_final: ' +str(FP_final))
    print('TN_final: ' +str(TN_final))
    print('FN_final: ' + str(FN_final))
    print('sense_final: ' + str(sense_final))
    print('spec_final: ' + str(spec_final))
    print('acc_final: ' + str(acc_final))

    return TP_final, FP_final, TN_final, FN_final, sense_final, spec_final, acc_final, prec_final



def run_experiment(n_loops = 1, n_epochs = 100, batch_size = 32):

  # load and process data
  data, labels = load_dataset()
  data = normalize_features(data)
  #train_x, train_y, eval_x, eval_y, test_x, test_y = split_dataset(data,labels)
  train_x, train_y, test_x, test_y = split_dataset_train_test(data,labels)
  eval_x, eval_y = test_x, test_y

  scores_test = np.array([])
  scores_train = np.array([])
  sensitivity_all = np.array([])
  specificity_all = np.array([])
  accuracy_all = np.array([])
  precision_all = np.array([])

  for r in range(n_loops):

    # train a model
    model, results = fit_model(train_x, train_y, eval_x, eval_y, n_epochs = n_epochs, batch_size = batch_size)


    accuracy_test, accuracy_train = evaluate_model(model, train_x, train_y, test_x, test_y)

    scores_test = np.append(scores_test,accuracy_test)
    scores_train = np.append(scores_train,accuracy_train)

    TP, FP, TN, FN, sensitivity, specificity, accuracy, precision = compute_performance_metrics(model,test_x,test_y)

    sensitivity_all = np.append(sensitivity_all,sensitivity)
    specificity_all = np.append(specificity_all,specificity)
    accuracy_all = np.append(accuracy_all,accuracy)
    precision_all = np.append(precision_all,precision)

  return np.mean(scores_test), np.mean(scores_train), np.mean(sensitivity_all), np.mean(specificity_all), np.mean(accuracy_all), np.mean(precision_all)


#scroes_test, scores_train, sensitivity, specificity, accuracy, precision = run_experiment(n_loops = 5)

#print('sensitivity:' + str(sensitivity))
#print('specificity:' + str(specificity))
#print('accuracy:' + str(accuracy))
#print('precision:' + str(precision))

kfold_function(4)
