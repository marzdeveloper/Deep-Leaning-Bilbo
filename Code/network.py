__author__ = 'Daniele Marzetti'

import numpy as np
from tensorflow.keras.layers import Conv1D, GlobalMaxPool1D, concatenate, Dropout, Dense, Embedding, LSTM, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

max_epoch = 10
batch_size = 512
EMBEDDING_DIMENSION = 128
NUM_CONV_FILTERS = 60
max_features = 38         #?

path = "/content/drive/MyDrive/Cyber Security/"
out_path = "/content/drive/MyDrive/Cyber Security/Bilbo/"

def create_model(MAX_STRING_LENGTH, MAX_INDEX):
    net = {}
    net['input'] = Input((MAX_STRING_LENGTH, ), dtype='int32', name='input')

    ########################
    #          CNN         #
    ########################

    net['embeddingCNN'] = Embedding(output_dim=EMBEDDING_DIMENSION,
                                    input_dim=MAX_INDEX,
                                    input_length=MAX_STRING_LENGTH,
                                    name='embeddingCNN')(net['input'])

    # Parallel Convolutional Layer

    net['conv2'] = Conv1D(NUM_CONV_FILTERS, 2, name='conv2')(net['embeddingCNN'])

    net['conv3'] = Conv1D(NUM_CONV_FILTERS, 3, name='conv3')(net['embeddingCNN'])

    net['conv4'] = Conv1D(NUM_CONV_FILTERS, 4, name='conv4')(net['embeddingCNN'])

    net['conv5'] = Conv1D(NUM_CONV_FILTERS, 5, name='conv5')(net['embeddingCNN'])

    net['conv6'] = Conv1D(NUM_CONV_FILTERS, 6, name='conv6')(net['embeddingCNN'])

    # Global max pooling

    net['pool2'] = GlobalMaxPool1D(name='pool2')(net['conv2'])

    net['pool3'] = GlobalMaxPool1D(name='pool3')(net['conv3'])

    net['pool4'] = GlobalMaxPool1D(name='pool4')(net['conv4'])

    net['pool5'] = GlobalMaxPool1D(name='pool5')(net['conv5'])

    net['pool6'] = GlobalMaxPool1D(name='pool6')(net['conv6'])

    net['concatcnn'] = concatenate([net['pool2'], net['pool3'], net['pool4'
                                   ], net['pool5'], net['pool6']], axis=1,
                                   name='concatcnn')

    net['dropoutcnnmid'] = Dropout(0.5, name='dropoutcnnmid')(net['concatcnn'])

    net['densecnn'] = Dense(NUM_CONV_FILTERS, activation='relu', name='densecnn')(net['dropoutcnnmid'])

    net['dropoutcnn'] = Dropout(0.5, name='dropoutcnn')(net['densecnn'])

    ########################
    #         LSTM         #
    ########################

    net['embeddingLSTM'] = Embedding(output_dim=max_features,
                                     input_dim=256,
                                     input_length=MAX_STRING_LENGTH,
                                     name='embeddingLSTM')(net['input'])

    net['lstm'] = LSTM(256, name='lstm')(net['embeddingLSTM'])

    net['dropoutlstm'] = Dropout(0.5, name='dropoutlstm')(net['lstm'])

    ########################
    #    Combine - ANN     #
    ########################

    net['concat'] = concatenate([net['dropoutcnn'], net['dropoutlstm']], axis=-1, name='concat')

    net['dropoutsemifinal'] = Dropout(0.5, name='dropoutsemifinal')(net['concat'])

    net['extradense'] = Dense(100, activation='relu', name='extradense')(net['dropoutsemifinal'])

    net['dropoutfinal'] = Dropout(0.5, name='dropoutfinal')(net['extradense'])

    net['output'] = Dense(1, activation='sigmoid', name='output')(net['dropoutfinal'])

    model = Model(net['input'], net['output'])
    return model

def train_eval_test(MAX_STRING_LENGTH, MAX_INDEX, start_fold, end_fold):
    for fold in range(start_fold, end_fold):
        #Get fold by csv
        x_train = np.genfromtxt(path + "x_train" + str(fold) + ".csv", delimiter=',')
        x_test = np.genfromtxt(path + "x_test" + str(fold) + ".csv", delimiter=',')
        y_train = np.genfromtxt(path + "y_train" + str(fold) + ".csv", delimiter=',')
        y_test = np.genfromtxt(path + "y_test" + str(fold) + ".csv", delimiter=',')

        model = None
        model = create_model(MAX_STRING_LENGTH, MAX_INDEX)
        earlystop = EarlyStopping(monitor='loss', patience=3)
        best_save = ModelCheckpoint('bestmodel.hdf5', save_best_only=True, save_weights_only=False, monitor='val_loss',
                                    mode='min')
        model.compile(optimizer='ADAM', loss=BinaryCrossentropy(),
                      metrics=[BinaryAccuracy(), AUC(), Precision(), Recall()])
        model.summary()

        history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=max_epoch, callbacks=[earlystop, best_save], validation_split=0.1)

        fig1 = plt.figure(1)
        plt.title('Loss')
        plt.plot(history.history["val_loss"], 'r', label='Validation Loss')
        plt.plot(history.history["loss"], 'b', label='Training Loss')
        plt.legend(loc="upper right")
        #x = list(range(len(loss_train)+1, 1))
        plt.grid(True)
        fig1.savefig(out_path + "loss" + str(fold) + ".png")
        plt.show()
        plt.close(fig1)

        fig2 = plt.figure(2)
        plt.title('Accuracy')
        plt.plot(history.history["val_binary_accuracy"], 'r', label='Validation Accuracy')
        plt.plot(history.history["binary_accuracy"], 'b', label='Training Accuracy')
        plt.legend(loc="lower right")
        plt.grid(True)
        fig2.savefig(out_path + "accuracy" + str(fold) + ".png")
        plt.show()
        plt.close(fig2)

        best_model = load_model('bestmodel.hdf5')
        metrics = best_model.evaluate(x= x_test, y= y_test, batch_size= batch_size, return_dict = True)
        try:
            df = pd.read_csv(out_path + "metrics.csv", index_col=[0])
            df = df.append(pd.Series(metrics, name=str(fold)))
            df.to_csv(out_path + "metrics.csv")
        except:
            pd.DataFrame(metrics, index=[str(fold)]).to_csv(out_path + "metrics.csv")
        predicted = np.round(best_model.predict(x=x_test, batch_size=batch_size), decimals=0)
        metrics1 = classification_report(y_test, predicted, output_dict=True, target_names=['alexa', 'dga'])
        try:
            df1 = pd.read_csv(out_path + "metrics1.csv", index_col=[0])
            df1 = df1.append(pd.DataFrame(metrics1))
            df1.to_csv(out_path + "metrics1.csv")
        except:
            pd.DataFrame(metrics1).to_csv(out_path + "metrics1.csv")
        np.savetxt(out_path + 'confusion_matrix' + str(fold) + '.csv', confusion_matrix(y_test, predicted), delimiter=',',  fmt='%i')
