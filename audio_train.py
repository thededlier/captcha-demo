import os
import librosa   #for audio processing
import numpy as np
from scipy.io import wavfile #for audio processing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot

import warnings
warnings.filterwarnings("ignore")

def build_model():
    K.clear_session()

    inputs = Input(shape=(8000,1))

    #First Conv1D layer
    conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    #Second Conv1D layer
    conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    #Third Conv1D layer
    conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    #Fourth Conv1D layer
    conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    #Flatten layer
    conv = Flatten()(conv)

    #Dense Layer 1
    conv = Dense(256, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    #Dense Layer 2
    conv = Dense(128, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    outputs = Dense(len(labels), activation='softmax')(conv)

    return Model(inputs, outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dataset', help='Where to look for the training image dataset', type=str)
    # parser.add_argument('--validate-dataset', help='Where to look for the validation image dataset', type=str)
    parser.add_argument('--output-model-name', help='Where to save the trained model', type=str)
    args = parser.parse_args()

    if args.train_dataset is None:
        print("Please specify the path to the training data set")
        exit(1)

    if args.output_model_name is None:
        print("Please specify a name for the trained model")
        exit(1)

    captcha_symbols = None
    with open(args.symbols) as symbols_file:
        captcha_symbols = symbols_file.readline()

    labels=os.listdir(args.train_dataset)

    all_wave = []
    all_label = []
    for label in labels:
        print(label)
        text = label.split('_')[0]
        samples, sample_rate = librosa.load(train_audio_path + '/' + label, sr = 16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if(len(samples)== 8000) :
            all_wave.append(samples)
            all_label.append(text)


    le = LabelEncoder()
    y=le.fit_transform(all_label)
    classes= list(le.classes_)


    y=np_utils.to_categorical(y, num_classes=len(labels))
    all_wave = np.array(all_wave).reshape(-1,8000,1)

    x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)

    model = build_model()

    model.summary()

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
    mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

if __name__ == '__main__':
    main()
