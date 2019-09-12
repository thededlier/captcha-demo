# import tensorflow as tf
# import tensorflow.keras.backend
# from tensorflow.keras.utils import Sequence

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)

# class CaptchaSequence(Sequence):
    # def __init__(self, captcha_symbols, batch_size, steps, captcha_length=4, width=128, height=64):
        # self.captcha_symbols = captcha_symbols
        # self.batch_size = batch_size
        # self.steps = steps
        # self.captcha_length = captcha_length
        # self.width = width
        # self.height = height
        # self.num_captcha_symbols = len(captcha_symbols)
        # self.generator = ImageCaptcha(width=width, height=height)

    # def __len__(self):
        # return self.steps

    # def __getitem__(self, idx):
        # X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        # y = [np.zeros((self.batch_size, self.num_captcha_symbols), dtype=np.uint8) for i in range(self.captcha_length)]
        # for i in range(self.batch_size):
            # random_str = ''.join([random.choice(self.captcha_symbols) for j in range(self.captcha_length)])
            # X[i] = np.array(self.generator.generate_image(random_str)) / 255.0
            # for j, ch in enumerate(random_str):
                # y[j][i, :] = 0
                # y[j][i, self.captcha_symbols.find(ch)] = 1
        # return X, y

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
