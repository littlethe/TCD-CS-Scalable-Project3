import os
from pydub import AudioSegment
import subprocess
from glob import glob
from shutil import copyfile
import librosa as lr
import scipy
import numpy as np
import matplotlib.pyplot as plt

# This code is used to create audio spectrums from audio captchas. By Tung-Te Lin 18th October 2019

origin_dir = '19309523-project3'
test_audio_dir = 'test_audio'
test_image_dir = 'test_image'
test_figure_dir = 'test_figure'

training_figure_dir = 'training_figure'
training_audio_dir = 'training_audio'

validation_figure_dir = 'validation_figure'
validation_audio_dir = 'validation_audio'


# This part is classifying the image and audio into different directory, then it is easy to be processed.
image_files = glob(origin_dir + '/*.png')
audio_files = glob(origin_dir + '/*.mp3')

for path in image_files:
    filename = os.path.basename(path)
    new_path = test_image_dir+'/'+filename
    copyfile(path, new_path)

for path in audio_files:
    filename = os.path.basename(path)
    filename, extension = os.path.splitext(filename)
    new_path = test_audio_dir + '/' + filename + '.wav'
    subprocess.call(['ffmpeg', '-i', path,new_path]) # convert wav to mp3


# This part is generating the audio spectrums from the audio files in testing directory.
print('generating test figure...')
wav_files = glob(test_audio_dir + '/*.wav')

for path in wav_files:
    filename = os.path.basename(path)
    filename, extension = os.path.splitext(filename)
    
    audio_file, audio_frequency = lr.load(path)
    time = np.arange(0,len(audio_file))/audio_frequency
    fig, ax = plt.subplots(figsize=(20, 2))
    ax.plot(time, audio_file)
    new_path = test_figure_dir + '/' + filename + '.png'
    plt.savefig(new_path)
    plt.clf()
    plt.close()

    
# This part is generating the audio spectrums from the audio files in training directory.
print('generating training figure...')
training_wav = glob(training_audio_dir + '/*.wav')

i = 0

for path in training_wav:
    filename = os.path.basename(path)
    filename, extension = os.path.splitext(filename)  
    audio_file, audio_frequency = lr.load(path)
    time = np.arange(0,len(audio_file))/audio_frequency
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.plot(time, audio_file)
    new_path = training_figure_dir + '/' + filename + '.png'
    plt.gray()
    plt.savefig(new_path)
    plt.clf()
    plt.close()
    i += 1
    if i % 1000 == 0:
        print(i)


# This part is generating the audio spectrums from the audio files in validation directory.
print('generating validation figure...')
validation_wav = glob(validation_audio_dir + '/*.wav')

i = 0

for path in validation_wav:
    filename = os.path.basename(path)
    filename, extension = os.path.splitext(filename)  
    audio_file, audio_frequency = lr.load(path)
    time = np.arange(0,len(audio_file))/audio_frequency
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.plot(time, audio_file)
    new_path = validation_figure_dir + '/' + filename + '.png'
    plt.gray()
    plt.savefig(new_path)
    plt.clf()
    plt.close()
    i += 1
    if i % 1000 == 0:
        print(i)

input("Press Enter to continue...")
