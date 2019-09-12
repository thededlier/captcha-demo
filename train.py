import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import cv2
import argparse
import tensorflow as tf
import tensorflow.keras as keras

# Build a Keras model given some parameters
def create_model(captcha_length, captcha_num_symbols, input_shape=(128, 64, 3), model_depth=5, module_size=2):
  model = keras.models.Sequential()
  for i, module_length in enumerate([module_size] * model_depth):
    for j in range(module_length):
      model.add(keras.layers.Conv2D(32*2**min(i, 3), (3, 3), input_shape=input_shape, padding='same'))
      model.add(keras.layers.BatchNormalization())
      model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))

  model.add(keras.layers.Flatten())
  for x in range(captcha_length):
    model.add(keras.layers.Dense(captcha_num_symbols, activation='softmax'))

  return model

# A Sequence represents a dataset for training in Keras
# In this case, we have a folder full of images
# Elements of a Sequence are *batches* of images, of some size batch_size
class ImageSequence(keras.utils.Sequence):
    def __init__(self, directory_name, batch_size, steps, captcha_length, captcha_symbols, captcha_width, captcha_height):
        self.directory_name = directory_name
        self.batch_size = batch_size
        self.steps = steps
        self.captcha_length = captcha_length
        self.captcha_symbols = captcha_symbols
        self.captcha_width = captcha_width
        self.captcha_height = captcha_height

        file_list = os.listdir(self.directory_name)
        self.files = dict(zip(map(lambda x: x.split('.')[0], file_list), file_list))
        self.used_files = []

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.captcha_height, self.captcha_width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.captcha_symbols), dtype=np.uint8) for i in range(self.captcha_length)]

        for i in range(self.batch_size):
            random_image_label = random.choice(self.files.keys())
            random_image_file = self.files[random_image_label]

            # We've used this image now, so we can't repeat it in this iteration
            self.used_files.append(self.files.pop(random_image_label))

            # We have to scale the input pixel values to the range [0, 1] for
            # Keras so we divide by 255 since the image is 8-bit RGB
            X[i] = np.array(cv2.imread(random_image_file)) / 255.0

            for j, ch in enumerate(random_image_label):
                y[j][i, :] = 0
                y[j][i, self.captcha_symbols.find(ch)] = 1

        return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--symbols', help='How many different symbols to use in captchas', type=int)
    parser.add_argument('--output-model', help='Where to save the trained model', type=str)
    parser.add_argument('--input-model', help='Where to look for the input model to continue training', type=str)
    parser.add_argument('--iterations', help='How many training iterations to do', type=int)
    args = parser.parse_args()

    if args.width is None:
        print("Please specify the captcha image width")
        exit(1)

    if args.height is None:
        print("Please specify the captcha image height")
        exit(1)

    if args.length is None:
        print("Please specify the captcha length")
        exit(1)

    if args.iterations is None:
        print("Please specify the number of training iterations to do")
        exit(1)

    if args.output_model is None:
        print("Please specify the path to save the trained model")
        exit(1)

    model = None

    if args.input_model is not None:
      pass
    else:
      model = create_model(args.length, args.symbols)

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.summary()

    # model.fit(train_images, train_labels, epochs=args.iterations)

    # test_loss, test_acc = model.evaluate(test_images, test_labels)

    # print("Model trained to " + str(test_acc * 100) + "% accuracy")

if __name__ == '__main__':
    main()

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)

# def decode(y):
    # y = np.argmax(np.array(y), axis=2)[:,0]
    # return ''.join([captcha_symbols[x] for x in y])

# data = CaptchaSequence(captcha_symbols, batch_size=10, steps=2)
# X, y = data[0]
# imgplot = plt.imshow(X[0])
# plt.title(decode(y))
# plt.show()

# for ix in range(len(data)):
  # X, y = data[ix]
  # image = X[0]
  # print(image.shape)
  # plt.imshow(image)
  # plt.show()
  # image_transposed = np.uint8(np.transpose(image, (2, 0, 1)))
  # print(image_transposed.shape)
  # cv2.imwrite("data/captcha-"+str(decode(y))+".png", image)
