import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
data = pd.read_csv('ecg_data.csv')
ecg = data['Signal'].values
fs = 360
def bandpass_filter(signal, low, high, fs, order=4):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)
ecg_filt = bandpass_filter(ecg, 0.5, 40, fs)
peaks,_=find_peaks(ecg_filt,height=np.mean(ecg_filt)*0.6,distance=0.25*fs) 
plt.figure(figsize=(12,4)) 
plt.plot(ecg_filt,label='Labeled ECG',color='b') 
plt.plot(peaks,ecg_filt[peaks],'ro',label='R-peaks') 
plt.title('ECG Signal with R-peaks') 
plt.xlabel('Sample Index') 
plt.ylabel('Amplitute(mv)') 
plt.legend() 
plt.grid(True) 
plt.tight_layout() 
plt.savefig("images/r_peak_detection.png")
plt.show()