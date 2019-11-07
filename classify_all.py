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

# This code is designed for project 3 in scalable computing. I modified the original code to be able to process audio spectrums.
# The audio spectrums are converted to be gray scale, but the images of captchas remain the RGB mode.
# In this combination, the score is the highest(472/1000).
# The training numbers of audio captchas and image captchas both are 200,000. (the numbers of validation both are 50,000.)
# The batch sizes of audio captchas and image captchas are 100 and 256, the epochs both are 20. By Tung-Te Lin, 18th October 2019.

def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

def main():

    symbols_file = open('symbols.txt', 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    #with tf.device('/cpu:0'):
    with tf.device('/device:GPU:0'):
        with open('output.csv', 'wb') as output_file:

            # This part is processing audio spectrums, the mode is gray scale.
            # I have ever tried to use RGB mode to process audio spectrums, but the result is worse then gray scale mode.
            json_file = open('audio8.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights('audio8.h5')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])
            for x in os.listdir('test_figure/'):
                #image = cv2.imread(os.path.join('test_figure/', x))
                image = cv2.imread(os.path.join('test_figure/', x),0)   # reading mode is gray scale.
                (h, w) = image.shape
                image = numpy.reshape(image,(h,w,1))                    # reshaping the array, then it can be processed in keras.
                #rgb_data = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                image = numpy.array(image) / 255.0
                (c, h, w) = image.shape
                image = image.reshape([-1, c, h, w])
                prediction = model.predict(image)
                filename, extension = os.path.splitext(x)
                x = filename + '.mp3'                                   # Changing the file extension to be mp3, because spectrums are png.
                line = x + "," + decode(captcha_symbols, prediction) + "\n"
                output_file.write(line.encode())

                print('Classified ' + x)

            # This part is processing image captchas, the code is the same with the code in project 2.
            # I have ever tried to use gray scale mode to process image captchas, but the result is worse then RGB mode.             
            json_file = open('image8.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = keras.models.model_from_json(loaded_model_json)
            model.load_weights('image8.h5')
            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                          metrics=['accuracy'])
            for x in os.listdir('test_image/'):
                #image = cv2.imread(os.path.join('test_image/', x),0)
                #image = cv2.cvtColor(raw_data, cv2.COLOR_BGR2RGB)
                image = cv2.imread(os.path.join('test_image/', x))
                #(h, w) = image.shape
                #image = numpy.reshape(image,(h,w,1))
                image = numpy.array(image) / 255.0
                (c, h, w) = image.shape
                image = image.reshape([-1, c, h, w])
                prediction = model.predict(image)
                line = x + "," + decode(captcha_symbols, prediction) + "\n"
                output_file.write(line.encode())

                print('Classified ' + x)



if __name__ == '__main__':
    main()
