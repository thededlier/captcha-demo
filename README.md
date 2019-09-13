# CNN Based Captcha Breaker Project

Required dependencies: python-captcha, opencv, python-tensorflow (CPU or GPU)


## Generating captchas

```
./generate.py --width 128 --height 64 --length 4 --symbols symbols.txt --count 3200 --output-dir test
```

This generates 3200 128x64 pixel captchas with 4 symbols per captcha, using the set of symbols in the `symbols.txt` file.
The captchas are stored in the folder `test`, which is created if it doesn't exist

To train and validate a neural network, we need two sets of data: a big training set, and a smaller validation set.
The network is trained on the training set, and tested on the validation set, so it is very important that there are no images that are in both sets.

## Training the neural network

```
./train.py --width 128 --height 64 --length 4 --symbols symbols.txt --batch-size 32 --epochs 5 --output-model test.h5 --train-dataset training_data --validate-dataset validation_data
```

Train the neural network for 5 epochs on the data specified. One epoch is one pass through the full dataset.
For the initial problem of captcha recognition, we train the network for 100 epochs.

The dataset for the initial training should be 128000 images, and the validation set should be 12800 images.

