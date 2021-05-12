import os
import re
import copy

import librosa
import numpy as np
import sklearn.cluster as sklearn_cluster
import sklearn.metrics as sklearn_metrics
import tensorflow as tf


# parsing the xml for timestamps
def parse_icsi_xml(path):
    """returns a list of [starttime, endtime, participant]"""
    transcript = []
    with open(path, 'r') as f:
        data = f.read()
    segments = re.findall("<segment.*", data)
    for temp in segments:
        s1 = temp.find("\"", temp.index("starttime")) + 1
        s2 = temp.find("\"", s1)
        t1 = temp.find("\"", temp.index("endtime")) + 1
        t2 = temp.find("\"", t1)
        p1 = temp.find("\"", temp.index("participant")) + 1
        p2 = temp.find("\"", p1)
        if (s2-s1>0 and t2-t1>0 and p2-p1>0):
            transcript.append([ 
                float(temp[s1: s2]), float(temp[t1: t2]), temp[p1: p2]
            ])
    return transcript

# make a dict mapping of file name and transcript
def gather_transcripts(icsi_segments_dir):
    """returns a dict mapping of filename to transcript"""
    segment_fns = sorted(os.listdir(icsi_segments_dir))
    transcripts = {}
    for segxml_fn in segment_fns:
        name = segxml_fn.split('.')[0]
        transcript = parse_icsi_xml(f"{icsi_segments_dir}/{segxml_fn}")
        if transcripts.get(name, False):
            transcripts[name].extend(transcript)
        else:
            transcripts[name] = transcript
    for name in transcripts.keys(): 
        transcripts[name].sort()
    return transcripts

def remove_overlaps(transcript, min_len=2):
    """ removes all the overlapping regions 
        min_len: minimum segment len in seconds"""
    transcript = copy.deepcopy(transcript)
    transcript.sort()
    i = 0
    j = 1
    e = transcript[0][1]
    while j < len(transcript):
        if transcript[j][0] < e:
            transcript[i][1] = min(round(transcript[j][0] - 0.001, 3), transcript[i][1])
            transcript[j][0] = round(e + 0.001, 3)
            j+=1
        else:
            i+=1
            e = transcript[i][1]
            j=i+1
    
    i=0
    while i < len(transcript):
        if transcript[i][0] + min_len > transcript[i][1]:
            transcript.pop(i)
        else:
            i+=1
    return transcript

def make_speaker_map(transcripts):
    """returns a mapping of speakers to timestamps and audio file name"""
    spk_dict = {}
    for k in transcripts.keys():
        for seg in transcripts[k]:
            spk = seg[-1]
            if spk_dict.get(spk, False):
                spk_dict[spk].extend([seg[:-1]+[k]])
            else:
                spk_dict[spk] = [seg[:-1]+[k]]
    return spk_dict

def split_segments(spk_map, dur, min_num=10):
    """ splits the segments greater than dur into of lenth dur,
        speakers with less than min_num segments are dropped """
    spk_map = copy.deepcopy(spk_map)
    for k in [*spk_map.keys()]:
        i=0
        while i < len(spk_map[k]):
            if spk_map[k][i][1] - spk_map[k][i][0] > dur:
                e = spk_map[k][i][1]
                s = spk_map[k][i][0] + dur
                b = spk_map[k][i][2]
                spk_map[k][i][1] = s
                while s + dur <= e:
                    i+=1
                    spk_map[k].insert(i, [s, s + dur, b])
                    s += dur
            elif spk_map[k][i][0] + dur > spk_map[k][i][1]:
                spk_map[k].pop(i)
                i-=1
            i+=1

    for k in [*spk_map.keys()]:
        if len(spk_map[k]) < min_num:
            spk_map.pop(k)
    return spk_map

def extract_mfcc(signal_segment, sr=16000, n_mfcc=13, n_fft=512):
    hop = int(0.010 * sr) # 10ms
    win = int(0.025 * sr) # 25ms
    mfcc = librosa.feature.mfcc(signal_segment, sr, n_mfcc= n_mfcc, hop_length= hop, win_length= win, n_fft=n_fft, window= "ham")
    del_mfcc = librosa.feature.delta(mfcc)
    ddel_mfcc = librosa.feature.delta(del_mfcc)
    return np.vstack([mfcc, del_mfcc, ddel_mfcc])

load_segments_cache = {}
data_cache = {}
def load_segments(segments, audio_dir, featextr_fn, cache_dir= None, sr=22050):
    """ featextr_fn : function to extract features takes 
                        in audio data and sample rate 
        array of [[start, stops, file_name],...]"""
    segments = np.asarray(segments)
    fnames = np.unique(segments[:, -1])
    starts = segments[:,0].astype(float)
    stops  = segments[:,1].astype(float)
    mfccs = []
    for fname in fnames:

        if load_segments_cache.get(fname, False) or os.path.isfile(f"{cache_dir}/{fname}.npy"):
            # if not load_segments_cache.get(fname, False):
                # load_segments_cache[fname] = f"{cache_dir}/{fname}.npy" # if load_segments_cache got erased on restart
            # x = np.load(load_segments_cache[fname], allow_pickle=True)
            x = data_cache[fname]
        else:
            x, _ = librosa.load(f"{audio_dir}/{fname}/{fname}.interaction.wav", sr)
            if cache_dir is not None:
                load_segments_cache[fname] = f"{cache_dir}/{fname}.npy"
                data_cache[fname] = x
                # np.save(load_segments_cache[fname], x)
            
        args = np.where(segments[:,-1] == fname)
        for start, stop in zip(starts[args], stops[args]):
            assert len(x) > int(np.round(start * sr)), f"{start} {stop} {len(x)/sr} {fname}"
            seg = x[int(np.round(start * sr)): int(np.round(stop * sr))]
            mfcc = featextr_fn(seg, sr)
            mfccs.append(mfcc.T) # shape: (time_steps, F)
    return mfccs

# callback to calculate NMI and purity scores
class NMI_purity_callback(tf.keras.callbacks.Callback):
    def __init__(self, data_generator, loggingfn, period=100, verbose=1):
        super(NMI_purity_callback, self).__init__()
        self.loggingfn = loggingfn
        self.data_generator = data_generator
        self.period = period
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        self.data_generator.reset()
        if (epoch+1) % self.period == 0:
            embeddings = []; true_labels = []
            for batch in self.data_generator.generator():
                embds = self.model.predict(batch)
                embeddings.extend(embds)
                true_labels.extend(batch['labels'])
            embeddings = np.asarray(embeddings)
            true_labels = np.asarray(true_labels)
            predicted_labels = sklearn_cluster.KMeans().fit_predict(embeddings)
            NMI_score = sklearn_metrics.normalized_mutual_info_score(true_labels, predicted_labels)
            contingency_matrix = sklearn_metrics.cluster.contingency_matrix(true_labels, predicted_labels)
            Purity_score = np.sum(np.amax(contingency_matrix, axis=0))/np.sum(contingency_matrix)
            self.loggingfn(NMI_score, Purity_score, epoch)
            if self.verbose == 1: tf.print(f"\nepoch: {epoch} nmi: {NMI_score} purity: {Purity_score}\n")

    
    def set_model(self, model):
        self.model = model

