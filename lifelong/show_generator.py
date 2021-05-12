import numpy as np
import threading
import queue
import tensorflow as tf
from copy import copy
from helper_funcs import *

global FILLABEL
global data_cache

data_cache = {}
FILLABEL = "non-sp/overlap"

def fillin(small_transcripts):
    """
    fills in the overlap and non-speach segments into the transcript.
    the inteverls in the given transcript are assumed to have no overlaps.
    """
    global FILLABEL

    small_transcripts = copy.deepcopy(small_transcripts)
    for k in small_transcripts.keys():
        i = 0
        fname, split_num, show_len = k.split('.')
        show_len = int(show_len)
        split_num = int(split_num)
        start = split_num*show_len
        while(i < len(small_transcripts[k])):
            if start < small_transcripts[k][i][0]:
                small_transcripts[k].insert(i, [start, small_transcripts[k][i][0], FILLABEL])
                i+=1
            start = small_transcripts[k][i][1]
            i+=1
        if small_transcripts[k][-1][1] != start + show_len:
            small_transcripts[k].append([small_transcripts[k][-1][1], start + show_len, FILLABEL])
    return small_transcripts
            

def bin_search(i,j, target, small_transcript):
    """searches for the interval in small_transcript where the target occurs"""
    if i > j: return -1

    mid = (i+j)//2
    t1, t2 = small_transcript[mid][0:2]
    
    if t1 <= target <= t2:
        return mid
    elif target < t1:
        return bin_search(i, mid-1, target, small_transcript)
    elif t2 < target:
        return bin_search(mid+1, j, target, small_transcript)


def majority_label(cur_segments, start, end):
    """returns the majority label in the given segments"""
    global FILLABEL
    max_dur = 0
    max_label = FILLABEL
    for seg in cur_segments:
        s,t,label = seg
        dur = min(end, float(t)) - max(start, float(s))
        if dur > max_dur:
            max_dur = dur
            max_label = label
    return max_label


def assign_label(interval, small_transcript):
    start, end = interval
    s = bin_search(0, len(small_transcript)-1, start, small_transcript)
    t = bin_search(0, len(small_transcript)-1, end, small_transcript)

    if s==-1 or t==-1: 
        text = f"s:{s} t:{t} st:{start} ed:{end}"
        raise ValueError(text)

    if s == t:
        return small_transcript[s][-1]
    else:
        cur_segments = small_transcript[s:t+1]
        label = majority_label(cur_segments, start, end)
        return label


def create_token_to_int(small_transcripts):
    """returns: a dict mapping of labels to int"""
    labels = set()
    for transcript in small_transcripts.values():
        for row in transcript: labels.add(row[-1])
    return {l:i for i,l in enumerate(sorted(labels))}


def interval_loader(intervals, fname, audio_dir, featextr_fn, cache_dir=None, sr=22050):
    """ featextr_fn : function to extract features takes in audio data and sample rate 
        intervals: array of [[start, stops],...]"""
    # global data_cache

    if cache_dir is not None:
        if data_cache.get(fname, False):
            x = np.load(f"{cache_dir}/{fname}.npy", allow_pickle=True)
        else:
            x, _ = librosa.load(f"{audio_dir}/{fname}/{fname}.interaction.wav", sr)
            data_cache[fname] = True
            np.save(f"{cache_dir}/{fname}.npy", x)
    else:
        x, _ = librosa.load(f"{audio_dir}/{fname}/{fname}.interaction.wav", sr)
    
    
    # x, _ = librosa.load(f"{audio_dir}/{fname}/{fname}.interaction.wav", sr)

    mfccs = []
    for start, stop in intervals:
        assert len(x) > int(np.round(start * sr)), f"{start} {stop} {len(x)/sr} {fname}"
        seg = x[int(np.round(start * sr)): int(np.round(stop * sr))]
        mfcc = featextr_fn(seg, sr)
        mfccs.append(mfcc.T) # shape: (time_steps, F)
    return mfccs

class show_generator(threading.Thread):

    def __init__(self, small_transcripts, featextr_fn, audio_dir, cache_dir=None, sr=22050, prefetch_buffer_size=3, max_threads=10):
        """ featextr_fn : function to extract features takes in audio data and sample rate 
            show_len: length of the audio in minutes
            batch_size: number of such audio segments in one batch, 
            small_transcripts: a dict like object with <fname>.<split_num>.<show_len(in sec)> as keys"""

        threading.Thread.__init__(self)

        self.daemon = True
        self.queue = queue.Queue(prefetch_buffer_size)
        
        self.small_transcripts = fillin(copy.deepcopy(small_transcripts))
        self.small_keys = [*self.small_transcripts.keys()]
        self.token_to_int = create_token_to_int(self.small_transcripts)

        self.continue_generation = True
        self.max_threads = max_threads
        self.active_threads = 0

        self.interval_loader = lambda intervals, fname: interval_loader(
            intervals, fname, audio_dir, featextr_fn, cache_dir, sr
        )

        self.start() # start prefetching
        # self.D = self._sequencial_make_batch()['feats'].shape[-1]

    def _sequencial_make_batch(self):
        """
        returns: the features and labels for the current small_transcript in
        the dict self.small_transcripts. the next time this function is called
        the next small_transcript would be processed.
        """
        batch = {'feats':[], 'labels':[]}

        # sample batch_size shows
        key = self.small_keys.pop(0)
        small_transcript = self.small_transcripts.pop(key)
        fname, split_num, show_len = key.split('.')
        show_len = int(show_len)
        split_num = int(split_num)
        offset = split_num*show_len

        intervals = [[offset + i*2, offset + (i+1)*2] for i in range(show_len//2)]
        feats = self.interval_loader(intervals, fname)
        labels = [assign_label(interval, small_transcript) for interval in intervals]
        labels = [self.token_to_int[l] for l in labels]

        # print([f.shape for f in batch['feats']])
        batch['feats']  = np.stack(feats)
        batch['labels'] = np.asarray(labels)
        return batch

    def _prefetch_worker(self):
        self.active_threads += 1
        try:
            batch = self._sequencial_make_batch()
            self.queue.put(batch) 
        except:
            pass
        self.active_threads -= 1

    def run(self):
        """ starts prefetching in the background """
        while len(self.small_transcripts) > 0:
            # while self.active_threads < self.max_threads:
            #     threading.Thread(target=self._prefetch_worker, daemon=True).start()
            batch = self._sequencial_make_batch()
            self.queue.put(batch)  
        self.queue.put(None)

    def dict_generator(self):
        """ yeilds a dict {"feats":... , "labels":...} for each batch"""
        while True:
            # while self.queue.empty(): pass 
            batch = self.queue.get()
            if batch != None: yield batch
            else: break

    def usual_generator(self):
        """ yeilds (feats, labels) for each batch"""
        while True:
            batch = self.queue.get()
            if batch != None: yield batch['feats'], batch['labels']
            else: break

    def get_int_to_token(self):
        """ returns the reverse mapping for token_to_int dict """
        int_to_token = {self.token_to_int[k]: k for k in self.token_to_int.keys()}
        return int_to_token
