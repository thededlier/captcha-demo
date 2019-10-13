#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import numpy
import string
import random
import argparse
import tensorflow as tf
import tensorflow.keras as keras
import speech_recognition as sr
import os
from pydub import AudioSegment

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
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

            for x in os.listdir(args.captcha_dir):
                if x.ends_with('.mp3'):
                    # convert wav to mp3
                    ORIGINAL_AUDIO_FILE = x
                    sound = AudioSegment.from_mp3(ORIGINAL_AUDIO_FILE)
                    sound.export('test.wav', format="wav")
                    # use the audio file as the audio source

                    AUDIO_FILE = os.path.join(os.getcwd(), "test.wav")

                    r = sr.Recognizer()
                    with sr.AudioFile(AUDIO_FILE) as source:
                        audio = r.record(source)  # read the entire audio file

                    # recognize speech using IBM Speech to Text
                    IBM_USERNAME = "80144216-f25e-4f01-87d2-e7f44bcf773a"  # IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
                    IBM_PASSWORD = "3EW2ew_9Gx01zwKlxfLYs1XBy4k0hKEkeyJSYjv2A5O-"  # IBM Speech to Text passwords are mixed-case alphanumeric strings
                    try:
                        print("IBM Speech to Text thinks you said " + r.recognize_ibm(audio, username=IBM_USERNAME, password=IBM_PASSWORD))
                    except sr.UnknownValueError:
                        print("IBM Speech to Text could not understand audio")
                    except sr.RequestError as e:
                        print("Could not request results from IBM Speech to Text service; {0}".format(e))
                    continue

                # load image and preprocess it
                raw_data = cv2.imread(os.path.join(args.captcha_dir, x))
                rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                image = numpy.array(rgb_data) / 255.0
                (c, h, w) = image.shape
                image = image.reshape([-1, c, h, w])
                prediction = model.predict(image)
                output_file.write(x + ", " + decode(captcha_symbols, prediction) + "\n")

                print('Classified ' + x)

if __name__ == '__main__':
    main()
