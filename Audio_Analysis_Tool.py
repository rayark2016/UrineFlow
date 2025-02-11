import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.io import wavfile
from scipy.signal.windows import hann
from IPython.display import Audio

class Audio_Analysis_Tool:
    def __init__(self):
        """初始化工具類"""
        self.df = None

    def Read_Csv(self, Path):
        """讀取音訊數據 (CSV 文件)"""
        self.df = pd.read_csv(Path, header=None, names=["Data"])
        return self.df

    def Read_Wav(self, Path_):
        """讀取音訊數據 (WAV 文件)"""
        fs, x = wavfile.read(Path_, mmap=False)
        x = x.astype(float)
        x /= 32768.0
        if x.ndim > 1:
            x = x[:, 0]
        return fs, x

    def Frequency(self, fs, Data):
        """傅立葉轉換 + 窗函數"""
        Windowed = Data * hann(len(Data))
        FFT = np.fft.fft(Windowed)
        Freq = np.fft.fftfreq(len(FFT), d=1.0/fs)
        FFT = FFT[:len(FFT)//2]
        Freq = Freq[:len(Freq)//2]
        FFT_db = 20 * np.log10(np.abs(FFT))

        Oct_Frac = 1/48
        Oct_Freq, Oct_Amp = self.Oct_Smooth(Freq, FFT_db, Oct_Fraction=Oct_Frac)
        return Oct_Freq, Oct_Amp

    def Oct_Smooth(self, Freq, FFT_db, Base_Freq=20, Oct_Fraction=1/48):
        """倍頻程平滑"""
        Oct_Bands = []
        Current_Freq = Base_Freq * (2 ** (-Oct_Fraction / 2))

        while Current_Freq < max(Freq):
            Next_Freq = Current_Freq * (2 ** Oct_Fraction)
            Oct_Bands.append((Current_Freq, Next_Freq))
            Current_Freq = Next_Freq

        oct_frequencies = []
        oct_amplitudes = []

        for (low, high) in Oct_Bands:
            indices = np.where((Freq >= low) & (Freq < high))[0]
            if len(indices) > 0:
                mean_freq = np.mean(Freq[indices])
                mean_amplitude = np.mean(FFT_db[indices])
                oct_frequencies.append(mean_freq)
                oct_amplitudes.append(mean_amplitude)

        return np.array(oct_frequencies), np.array(oct_amplitudes)

    def Calcu_STD(self, Data, n=4):
        """標準差過濾"""
        mean_value = np.mean(Data)
        std_dev = np.std(Data)
        threshold_upper = mean_value + n * std_dev
        threshold_lower = mean_value - n * std_dev
        STD_Data = np.where((Data > threshold_upper) | (Data < threshold_lower), 0, Data)
        return STD_Data

    def Band_Pass_Filter(self, data, low_cutoff, high_cutoff, fs, order=4):
        """帶通濾波器Filter"""
        nyquist = 0.5 * fs
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = signal.butter(order, [low, high], btype='band', analog=False)
        y = signal.filtfilt(b, a, data)
        return y
