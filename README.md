# CNN Based Captcha Breaker Project

Required dependencies: python-captcha, opencv, python-tensorflow (CPU or GPU)


## Generating captchas

```
./generate.py --width 128 --height 64 --length 4 --symbols symbols.txt --count 3200 --scramble --output-dir test
```

This generates 3200 128x64 pixel captchas with 4 symbols per captcha, using the set of symbols in the `symbols.txt` file.
The captchas are stored in the folder `test`, which is created if it doesn't exist. The names of the captcha images are scrambled.

Without the `--scramble` option, the name of the image is the captcha text.

To train and validate a neural network, we need two sets of data: a big training set, and a smaller validation set.
The network is trained on the training set, and tested on the validation set, so it is very important that there are no images that are in both sets.

## Training the neural network

```
./train.py --width 128 --height 64 --length 4 --symbols symbols.txt --batch-size 32 --epochs 5 --output-model test.h5 --train-dataset training_data --validate-dataset validation_data
```

Train the neural network for 5 epochs on the data specified. One epoch is one pass through the full dataset.

For the initial problem of captcha recognition, we train the network for 2 epochs.

The dataset size for the initial training is 20000 images, and the validation set size is 4000 images.

## Running the classifier

```
./classify.py  --model-name test --captcha-dir ~/Downloads/validation_data/ --output ~/Downloads/stuff.txt --symbols symbols.txt
```

With `--model-name test` the classifier script will look for a model called `test.json` with weights `test.h5` in the current directory, and load the model up.

The classifier runs all the images in `--captcha-dir` through the model, and saves the file names and the model's guess at captcha contained in the image in the `--output` file.
