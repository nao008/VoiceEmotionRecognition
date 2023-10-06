#data/song/ & data/speech/のwavファイルをスペクトログラム画像に変換しdata/train及びdata/valに保存する

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import glob
import cv2
import random
import os
#ファイル名取得

files1 = glob.glob("/data/song/**/*.wav")
files2 = glob.glob("/data/speech/**/*.wav")
files3 = files1 + files2

trainfiles = random.sample(files3, int(len(files3)*0.8))
valfiles = list(set(files3) - set(trainfiles))
files = [trainfiles, valfiles]

# ディレクトリの用意
train_directory_path = "data/train"
val_directory_path = "data/val"
for i in range(8):
    if not os.path.exists(f'{train_directory_path}/{i+1}'):
        os.makedirs(f'{train_directory_path}/{i+1}')
    if not os.path.exists(f'{val_directory_path}/{i+1}'):
        os.makedirs(f'{val_directory_path}/{i+1}')


for hoge in files:
    for file in hoge:
        #ファイル名からラベルを取得
        label = file[-17]
        filename = file[-24:-4]

        #保存先のフォルダ情報
        if hoge == trainfiles:
            folder = "train"
        elif hoge == valfiles:
            folder = "val"
        
        # 音声ファイルの読み込み
        sample_rate, samples = wavfile.read(file)
        
        # FFTをかけて周波数スペクトルを取得
        frequencies, times, spec = spectrogram(samples, fs=sample_rate)
        
        
        # # スペクトログラム画像の描画
        Z = 10. * np.log10(spec)
        # plt.imshow(Z, origin='lower', aspect='auto', cmap='viridis')
        # plt.xlabel('Time (sec)')
        # plt.ylabel('Frequency (Hz)')
        # plt.colorbar()
        # plt.show()
        cv2.imwrite(f'data/{folder}/{label}/{filename}.png', Z)