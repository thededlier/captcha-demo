# Convers audio file to spectogram

import os
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import argparse

def create_spectogram(path, spec_dir, audio_file):
    x, sr = librosa.load(os.path.join(path, audio_file))
    plt.figure(figsize=(1.28, 0.64), dpi = 100)
    plt.axis('off')
    plt.axes([0., 0., 1., 1., ], frameon=False, xticks=[], yticks=[])
    mel_spec = librosa.feature.melspectrogram(y = x, sr = sr)
    librosa.display.specshow(librosa.power_to_db(mel_spec, ref = np.max))
    plt.savefig(os.path.join(spec_dir, os.path.splitext(audio_file)[0]+'.png'), bbox_inches=None, pad_inches=0)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-dir', help='Audio path', type=str)
    args = parser.parse_args()

    if args.audio_dir is None:
        print("Please specify the audio directory")
        exit(1)

    spec_dir = args.audio_dir + '_spec'

    if not os.path.exists(spec_dir):
        print("Creating output directory " + spec_dir)
        os.makedirs(spec_dir)

    for audio_file in os.listdir(args.audio_dir):
        print('Preprocessing : ' + audio_file)
        create_spectogram(args.audio_dir, spec_dir, audio_file)

if __name__ == '__main__':
    main()
