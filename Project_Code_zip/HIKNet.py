import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.layers import Dense, Dropout, Conv2D, Conv1D, MaxPooling1D, GlobalAveragePooling2D
from keras.models import Sequential, load_model
from keras.layers.core import Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def load_dataset():
    labels = sio.loadmat('../../Data/labels.mat')
    labels = labels['label_impact_noimpact']
    data = sio.loadmat('../../Data/data.mat')
    data = data['data']

    return data, labels


def normalize_features(data):
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    data_normalized = np.zeros(data.shape)

    for t in range(data.shape[1]):
        data_normalized[:, t, :] = (data[:, t, :] - mean) / std

    return data_normalized


def split_dataset(data, labels, train_split=0.7):
    # Train set
    sample_num = len(data[:, 1, 1])
    train_x = data[0:int(np.floor(train_split * sample_num)), :, :]
    train_y = labels[0:int(np.floor(train_split * sample_num))]

    # Evaluation set
    eval_x = data[int(np.floor(train_split * sample_num)):, :, :]
    eval_y = labels[int(np.floor(train_split * sample_num)):]

    return train_x, train_y, eval_x, eval_y


def fit_model(train_x, train_y, eval_x, eval_y, n_epochs=10, batch_size=32):
    n_timesteps, n_features = train_x.shape[1], train_x.shape[2]

    model = Sequential()

    # Hidden layer 1
    model.add(Conv1D(filters=48, kernel_size=15, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))

    # Hidden layer 2
    model.add(Conv1D(filters=96, kernel_size=15, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))

    # Reshapes hidden layer 2 output for the Conv2D filter
    layer2_output_shape = model.output_shape
    model.add(Reshape((layer2_output_shape[1], layer2_output_shape[2], -1)))

    # Hidden Layer 3
    model.add(Conv2D(filters=96, kernel_size=(3, 15), strides=(3, 1), activation='relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.4))

    # Output layer
    model.add(Dense(units=1, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopper = EarlyStopping(patience=5, verbose=1)
    check_pointer = ModelCheckpoint(filepath='HIKNet.hdf5', verbose=1, save_best_only=True)
    results = model.fit(train_x, train_y,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        callbacks=[early_stopper, check_pointer],
                        validation_data=(eval_x, eval_y))
    model = load_model('HIKNet.hdf5')

    return model, results


def evaluate_model(model, train_x, train_y, test_x, test_y):
    _, accuracy_test = model.evaluate(test_x, test_y)
    _, accuracy_train = model.evaluate(train_x, train_y)

    return accuracy_test, accuracy_train


def compute_roc_curve(model, test_x, test_y):
    y_pred = model.predict(test_x).ravel()
    fpr, tpr, thresholds = roc_curve(test_y, y_pred)
    area_under_curve = auc(fpr, tpr)
    print('Area under ROC curve:' + str(area_under_curve))

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.show()


def compute_pr_curve(model, test_x, test_y):
    y_pred = model.predict(test_x).ravel()
    precision, recall, thresholds = precision_recall_curve(test_y, y_pred)
    area_under_curve = auc(recall, precision)
    print('Area under PR curve:' + str(area_under_curve))

    plt.figure()
    plt.plot(recall, precision)
    plt.ylim((0, 1))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.show()


def compute_performance_metrics(model, test_x, test_y):
    y_pred = model.predict_classes(test_x)
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if test_y[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and test_y[i] != y_pred[i]:
            FP += 1
        if test_y[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and test_y[i] != y_pred[i]:
            FN += 1

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)

    return TP, FP, TN, FN, sensitivity, specificity, accuracy, precision


def k_fold_function(k):
    data, labels = load_dataset()
    data = normalize_features(data)
    kfold = KFold(n_splits=k)
    data_train = []
    data_test = []
    labels_train = []
    labels_test = []
    for train, test in kfold.split(data):
        data_train.append(data[train])
        labels_train.append(labels[train])
        data_test.append(data[test])
        labels_test.append(labels[test])

    TP_k = np.zeros(k)
    FP_k = np.zeros(k)
    TN_k = np.zeros(k)
    FN_k = np.zeros(k)
    sensitivity_k = np.zeros(k)
    specificity_k = np.zeros(k)
    accuracy_k = np.zeros(k)
    precision_k = np.zeros(k)
    for i in range(0, k):
        model = fit_model(data_train[i], labels_train[i], data_test[i], labels_test[i])
        TP_k[i], FP_k[i], TN_k[i], FN_k[i], sensitivity_k[i], specificity_k[i], accuracy_k[i], precision_k[i] = compute_performance_metrics(model, data_test[i], labels_test[i])

    mean_TP = np.mean(TP_k)
    mean_FP = np.mean(FP_k)
    mean_TN = np.mean(TN_k)
    mean_FN = np.mean(FN_k)
    mean_sensitivity = np.mean(sensitivity_k)
    mean_specificity = np.mean(specificity_k)
    mean_accuracy = np.mean(accuracy_k)
    mean_precision = np.mean(precision_k)

    print('Mean TP: ' + str(mean_TP))
    print('Mean FP: ' + str(mean_FP))
    print('Mean TN: ' + str(mean_TN))
    print('Mean FN: ' + str(mean_FN))
    print('Mean sensitivity: ' + str(mean_sensitivity))
    print('Mean specificity: ' + str(mean_specificity))
    print('Mean accuracy: ' + str(mean_accuracy))
    print('Mean precision: ' + str(mean_precision))

    return mean_TP, mean_FP, mean_TN, mean_FN, mean_sensitivity, mean_specificity, mean_accuracy, mean_precision


def run_experiment(n_loops=1, n_epochs=100, batch_size=32):
    # Loads and pre-processes data
    data, labels = load_dataset()
    data = normalize_features(data)
    train_x, train_y, eval_x, eval_y = split_dataset(data, labels)

    scores_test = np.array([])
    scores_train = np.array([])
    sensitivity_all = np.array([])
    specificity_all = np.array([])
    accuracy_all = np.array([])
    precision_all = np.array([])

    for r in range(n_loops):
        # Trains a model
        model, results = fit_model(train_x, train_y, eval_x, eval_y, n_epochs=n_epochs, batch_size=batch_size)
        accuracy_test, accuracy_train = evaluate_model(model, train_x, train_y, eval_x, eval_y)
        scores_test_n = np.append(scores_test, accuracy_test)
        scores_train_n = np.append(scores_train, accuracy_train)

        _, _, _, _, sensitivity_n, specificity_n, accuracy_n, precision_n = compute_performance_metrics(model, eval_x, eval_y)

        sensitivity_all = np.append(sensitivity_all, sensitivity_n)
        specificity_all = np.append(specificity_all, specificity_n)
        accuracy_all = np.append(accuracy_all, accuracy_n)
        precision_all = np.append(precision_all, precision_n)

    metrics_n = (sensitivity_all, specificity_all, accuracy_all, precision_all)

    return scores_test_n, scores_train_n, metrics_n


# k_fold_function(3)

scores_test, scores_train, metrics = run_experiment(n_loops=10)
sensitivity = np.mean(metrics[0])
specificity = np.mean(metrics[1])
accuracy = np.mean(metrics[2])
precision = np.mean(metrics[3])

print('Sensitivity:' + str(sensitivity))
print('Specificity:' + str(specificity))
print('Accuracy:' + str(accuracy))
print('Precision:' + str(precision))

compute_pr_curve(model, test_x, test_y)