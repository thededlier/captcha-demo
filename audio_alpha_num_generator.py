# Generate audio as separate alphabets and numbers in different variations

#!/usr/bin/env python3

import os
import numpy
import random
import string
import argparse
import pyttsx3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--is-test', help='Whether the data generated is for train or test', type=str)
    args = parser.parse_args()

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating audio with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    engine = pyttsx3.init(driverName='sapi5', debug=True)

    if args.is_test == 'yes':
        voice_speeds = [125 + 10, 150 + 5, 175 + 9, 200 + 15]
    else:
        voice_speeds = [100, 125, 150, 175, 200, 225, 250, 300]

    for i in range(len(captcha_symbols)):
        for voice_speed in voice_speeds:
            alphanum = captcha_symbols[i]
            audio_path = os.path.join(args.output_dir, alphanum+'.wav')
            version = 1
            while os.path.exists(os.path.join(args.output_dir, alphanum + '_' + str(version) + '.wav')):
                version += 1
            for voice in engine.getProperty('voices'):
                audio_path = os.path.join(args.output_dir, alphanum + '_' + str(version) + '.wav')

                print(audio_path + '=' + alphanum)

                engine.setProperty('voice', voice.id)
                engine.setProperty('rate', voice_speed)
                engine.save_to_file(alphanum, audio_path)
                engine.runAndWait()
                version += 1
    engine.stop()
if __name__ == '__main__':
    main()
