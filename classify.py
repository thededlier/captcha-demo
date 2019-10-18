#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os
import cv2
import numpy
import string
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--audio-model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    with tf.device('/cpu:0'):
        with open(args.output, 'w') as output_file:
            json_file = open(args.model_name+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights(args.model_name+'.h5')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])

            json_file = open(args.audio_model_name+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            audio_model = keras.models.model_from_json(loaded_model_json)
            audio_model.load_weights(args.audio_model_name+'.h5')
            audio_model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])

            for x in os.listdir(args.captcha_dir):
                if x.endswith('.mp3'):
                    # Save mp3 as a temp spectogram
                    sample, sr = librosa.load(os.path.join(args.captcha_dir, x))
                    plt.figure(figsize=(1.28, 0.64), dpi = 100)
                    plt.axis('off')
                    plt.axes([0., 0., 1., 1., ], frameon=False, xticks=[], yticks=[])
                    mel_spec = librosa.feature.melspectrogram(y = sample, sr = sr)
                    librosa.display.specshow(librosa.power_to_db(mel_spec, ref = numpy.max))
                    plt.savefig('temp.png', bbox_inches=None, pad_inches=0)
                    plt.close()

                    # load image and preprocess it
                    raw_data = cv2.imread('temp.png')
                    rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
                    rgb_data = cv2.equalizeHist(rgb_data)
                    rgb_data = numpy.expand_dims(rgb_data, axis=3)
                    image = numpy.array(rgb_data) / 255.0
                    (c, h, w) = image.shape
                    # import pdb; pdb.set_trace()
                    reshaped_image = image.reshape([-1, c, h, w])
                    prediction = audio_model.predict(reshaped_image)
                    pred = decode(captcha_symbols, prediction)
                else:
                    # load image and preprocess it
                    raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
                    # rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2GRAY)
                    rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                    image = numpy.array(rgb_data) / 255.0
                    (c, h, w) = image.shape
                    reshaped_image = image.reshape([-1, c, h, w])
                    prediction = model.predict(reshaped_image)
                    pred = decode(captcha_symbols, prediction)

                output_file.write(x + "," + pred + "\n")

                print('Classified ' + x)

if __name__ == '__main__':
    main()
