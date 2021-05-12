import librosa
import numpy as np

def add_white_noise(data, noisefactor=0.09, sr=16000):
    noise = noisefactor*np.random.randn(len(data))
    aug_data = data + noise
    return aug_data

def time_shift(data, max_shift=None, sr=16000):
    if max_shift==None: max_shift=int(0.3*len(data))
    rshift = np.random.randint(max_shift)
    if np.random.uniform() > 0.5: rshift = -rshift
    aug_data = np.roll(data, rshift)
    if rshift > 0:
        aug_data[:rshift] = 0
    else:
        aug_data[rshift:] = 0
    return aug_data

def add_tone(data, max_num_tones=5, minf=220, maxf=8000, sr=16000):
    rnum = np.random.randint(1, max_num_tones+1)
    freq = np.random.randint(minf, maxf, size=rnum)
    rphase = np.random.uniform(-np.pi, np.pi, size=rnum)
    tone = librosa.tone(freq[0], sr, len(data), rphase[0])
    for i in range(1, rnum):
        tone = tone + librosa.tone(freq[i], sr, len(data), rphase[i])
    aug_data = data + tone
    return aug_data

def add_clicks(data, num_shots=10, sr=16000):
    rtimes = np.random.randint(0, len(data), num_shots)/sr
    r = 0.5*np.random.uniform()
    rdur = min(r, r*len(data)/sr)
    rfreq = np.random.randint(220, 8000)
    clicks = librosa.clicks(times=rtimes, sr=sr, click_duration=rdur, 
                    length=len(data), click_freq=rfreq)
    aug_data = data + clicks
    return aug_data

def augment_audio(data, sr=16000):
    if np.random.randn() > 0.5:
        data = add_white_noise(data, sr=sr)
    if np.random.randn() > 0.5:
        data = add_tone(data, sr=sr)
    if np.random.randn() > 0.5:
        data = add_clicks(data, sr=sr)
    if np.random.randn() > 0.5:
        data = time_shift(data, sr=sr)
    return data


