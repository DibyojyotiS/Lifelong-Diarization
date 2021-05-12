import os
import re
import copy

import librosa
import numpy as np


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


# remove the overlapping regions of speech
def remove_overlaps(transcript, min_len=2):
    """ removes all the overlapping regions 
        transcript: list 
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


# split the shows into handleable sizes
def split_transcripts(transcripts, show_len=10):
    """
    gives the corresponging transcript if the audio recording were to be 
    split into segments of show_len minutes
    transcripts: a dict with each fname as key
    returns: a dict like object with <fname>.<split_num>.<show_len(in sec)> as keys
    """

    show_len = show_len * 60
    small_shows = {}
    
    for name in transcripts.keys(): 
        transcript = transcripts[name] # [starttime, endtime, participant]
        max_time = transcript[-1][1]
        n = int(max_time/show_len) #of full small shows

        divs = [(i*show_len, (i+1)*show_len) for i in range(n)]
        # if (n+1)*show_len - max_time < show_len/4: 
        #     divs.append((max_time-show_len, max_time))

        # make transcript for each div
        p = 0
        N = len(transcript)
        for k, div in enumerate(divs):
            s, e = div
            key = f"{name}.{k}.{show_len}"

            while(p < N and s < e and transcript[p][0] < e):
                
                seg = copy.deepcopy(transcript[p])
                if seg[0] < s : seg[0] = s
                if seg[1] - s > show_len: seg[1] = s = e
                else: p+=1

                if not small_shows.get(key, False):
                    small_shows[key] = [seg]
                else:
                    small_shows[key].append(seg)

    return small_shows
            

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


def extract_mfcc(signal_segment, sr=16000, n_mfcc=13, n_fft=512):
    hop = int(0.010 * sr) # 10ms
    win = int(0.025 * sr) # 25ms
    mfcc = librosa.feature.mfcc(signal_segment, sr, n_mfcc= n_mfcc, hop_length= hop, win_length= win, n_fft=n_fft, window= "ham")
    del_mfcc = librosa.feature.delta(mfcc)
    ddel_mfcc = librosa.feature.delta(del_mfcc)
    return np.vstack([mfcc, del_mfcc, ddel_mfcc])
