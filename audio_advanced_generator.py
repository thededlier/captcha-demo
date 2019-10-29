#!/usr/bin/env python3

# Generates captchas using gtts only
# For voice variations use audio_alpha_num_generator.py
import os
import numpy
import random
import string
import argparse
import pyttsx3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.count is None:
        print("Please specify the captcha count to generate")
        exit(1)

    if args.output_dir is None:
        print("Please specify the captcha output directory")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "}")

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    engine = pyttsx3.init(driverName='sapi5', debug=True)
    voices = engine.getProperty('voices')
    for i in range(args.count):
        captcha_text = ''.join([random.choice(captcha_symbols) for j in range(args.length)])
        audio_path = os.path.join(args.output_dir, captcha_text+'.wav')
        if os.path.exists(audio_path):
            version = 1
            while os.path.exists(os.path.join(args.output_dir, captcha_text + '_' + str(version) + '.wav')):
                version += 1
            audio_path = os.path.join(args.output_dir, captcha_text + '_' + str(version) + '.wav')

        print(captcha_text)

        engine.setProperty('voice', random.choice(voices).id)
        # engine.setProperty('rate', voice_speed)
        engine.save_to_file(captcha_text, audio_path)
        engine.runAndWait()

    engine.stop()

if __name__ == '__main__':
    main()
