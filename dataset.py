#data/song/ & data/speech/のwavファイルをスペクトログラム画像に変換しdata/img/に保存する

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram
import glob
import cv2
#ファイル名取得

files1 = glob.glob("/home/naoya/Desktop/github/VoiceEmotionRecognition/data/song/**/*.wav")
files2 = glob.glob("/home/naoya/Desktop/github/VoiceEmotionRecognition/data/speech/**/*.wav")
files = files1 + files2

for file in files:
    #ファイル名からラベルを取得
    # label = file[-17]
    filename = file[-24:-4]
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
    cv2.imwrite(f'/home/naoya/Desktop/github/VoiceEmotionRecognition/data/img/{filename}.png', Z)