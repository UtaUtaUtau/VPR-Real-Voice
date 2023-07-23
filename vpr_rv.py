import pyworld as pw
import numpy as np
from numba import njit, float64, optional
import soundfile as sf
import json
from argparse import ArgumentParser
import traceback
import os
import scipy.interpolate as interp
import zipfile as zf
import logging
import librosa
from copy import deepcopy
logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO, datefmt='%x %a %X')

silences = ['pau', 'sil', 'Sil']

f0_floor = pw.default_f0_floor 
f0_ceil = 1760

class LabelLine:
    def __init__(self, start, end, phone, isHTK):
        div = 10000000 if isHTK else 1
        self.start = start / div
        self.end = end / div
        self.phone = phone

    def duration(self):
        return self.end - self.start

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{self.start}\t{self.end}\t{self.phone}'

    def __mul__(self, other):
        return LabelLine(self.start * other, self.end * other, self.phone)

    def __rmul__(self, other):
        return LabelLine(self.start * other, self.end * other, self.phone)

    def __imul__(self, other):
        self.start *= other
        self.end *= other
    
    @staticmethod
    def from_line(line, isHTK):
        spl = line.strip().split(maxsplit=2)
        return LabelLine(float(spl[0]), float(spl[1]), spl[2], isHTK)

def get_label(fname, args):
    if args.lab:
        return args.lab
    if os.path.exists(fname + '.txt'):
        return fname + '.txt', False
    elif os.path.exists(fname + '.lab'):
        return fname + '.lab', True
    else:
        labf = input('Drag and drop label (Audacity .txt or HTK .lab): ').strip('"')
        _, ext = os.path.splitext(labf)
        while ext not in ['.wav', '.lab']:
            labf = input('Drag and drop label (Audacity .txt or HTK .lab): ').strip('"')
            _, ext = os.path.splitext(labf)
        return labf, ext == '.lab'

@njit(float64(float64[:], optional(float64), optional(float64)))
def _jit_base_frq(f0, f0_min, f0_max):
    q = 0
    avg_frq = 0
    tally = 0
    N = len(f0)

    if f0_min is None:
        f0_min = f0_floor

    if f0_max is None:
        f0_max = f0_ceil
    
    for i in range(N):
        if f0[i] >= f0_min and f0[i] <= f0_max:
            if i < 1:
                q = f0[i+1] - f0[i]
            elif i == N - 1:
                q = f0[i] - f0[i-1]
            else:
                q = (f0[i+1] - f0[i-1]) / 2
            weight = 2 ** (-q * q)
            avg_frq += f0[i] * weight
            tally += weight

    if tally > 0:
        avg_frq /= tally
    return avg_frq

def base_frq(f0, f0_min=None, f0_max=None):
    """Get average F0 with a stronger bias on flatter areas. 

    Parameters
    ----------
    f0 : list or ndarray
        Array of F0 values.

    f0_min : float, optional
        Lower F0 limit.

    f0_max : float, optional
        Upper F0 limit.

    Returns
    -------
    float
        Average F0.
    """
    return _jit_base_frq(f0, f0_min, f0_max)

def load_json(file):
    with open(file, 'r', encoding='utf8') as f:
        return json.load(f)

def remove_zeros(y):
    x = np.arange(y.size)
    nonzero_y = y[y != 0]
    nonzero_x = x[y != 0]

    nonzeros = interp.interp1d(nonzero_x, nonzero_y, bounds_error=False, fill_value=(nonzero_y[0], nonzero_y[-1]))
    return nonzeros(x)

try:
    parser = ArgumentParser(description='idk man Real Voice but for V6.')
    parser.add_argument('wave', help='The sound file. Supported formats are anything soundfile supports.')
    parser.add_argument('--lab', '-L', help='The label (Audacity .txt or HTK .lab format). Finds a label of the same filename in the same folder if not specified.')
    parser.add_argument('--bpm', '-B', default=120, type=float, help='The BPM to use. Default: 120')

    args, _ = parser.parse_known_args()
    
    vpr = load_json('base.json')
    jpn2voc = load_json('jpn2voc.json')
    note_base = load_json('note_base.json')

    tps = 480 * args.bpm / 60

    wavf = args.wave
    fname, _ = os.path.splitext(wavf)
    labf, isHTK = get_label(fname, args)

    logging.info('Reading label')
    lab = []
    with open(labf, encoding='utf8') as f:
        for l in f.readlines():
            lab.append(LabelLine.from_line(l, isHTK))
            

    logging.info('Converting lengths to VOCALOID ticks')
    for i in lab:
        i *= tps
    
    for i in range(len(lab)):
        if i > 0:
            lab[i].start = lab[i-1].end
        lab[i].start = int(round(lab[i].start))
        lab[i].end = int(round(lab[i].end))

    for i in range(len(lab)):
        lab_len = lab[i].duration()
        if lab_len == 0:
            lab[i].start -= 1
            lab[i-1].end -= 1
    
    logging.info('Reading sound file')
    x, fs = sf.read(args.wave)
    
    if x.ndim > 1:
        x = np.mean(x, axis=1)
        
    x_len = x.size / fs
    track_len = int(x_len * tps)

    logging.info('Extracting pitch')
    
    pit, pit_t = pw.harvest(x, fs)
    pit = remove_zeros(pit)

    logging.info('Extracting loudness')
    
    sample_hop = int(fs * pw.default_frame_period / 1000)
    sample_period = sample_hop / fs
    S, _ = librosa.magphase(librosa.stft(x, hop_length=sample_hop))
    dyn = librosa.feature.rms(S=S).flatten() + 0.000001
    dyn /= np.max(dyn)
    dyn_t = np.arange(dyn.size) * sample_period

    logging.info('Converting seconds to VOCALOID ticks')
    ticks_t = np.arange(track_len) / tps
    pit_l = interp.Akima1DInterpolator(pit_t, pit)
    dyn_l = interp.Akima1DInterpolator(dyn_t, dyn)

    pit_ticks = pit_l(np.minimum(ticks_t, pit_t[-1]))
    dyn_ticks = dyn_l(np.minimum(ticks_t, dyn_t[-1]))

    logging.info('Writing to VPR sequence')

    vpr['tracks'][0]['parts'][0]['duration'] = track_len
    vpr['masterTrack']['tempo']['global']['value'] = int(100 * args.bpm)
    vpr['masterTrack']['tempo']['events'][0]['value'] = int(100 * args.bpm)
    
    notes = vpr['tracks'][0]['parts'][0]['notes']
    controllers = vpr['tracks'][0]['parts'][0]['controllers']

    # controllers.append({'name': 'dynamics', 'events': []})
    controllers.append({'name': 'pitchBendSens', 'events': []})
    controllers.append({'name': 'pitchBend', 'events': []})

    vpr['tracks'][0]['volume']['events'] = []
    dyn = vpr['tracks'][0]['volume']['events'] # controllers[0]['events']
    pbs = controllers[0]['events']
    pbn = controllers[1]['events']

    mean_note = 0
    tally = 0
    for l in lab:
        if l.phone not in silences:
            s = l.start
            e = l.end
            base_note = int(round(librosa.hz_to_midi(base_frq(pit_ticks[s:e]))))
            mean_note += base_note
            tally += 1
            base_hz = librosa.midi_to_hz(base_note)

            pbn_note = 12 * (np.log2(pit_ticks[s:e]) - np.log2(base_hz))
            pbs_note = int(np.ceil(np.max(np.abs(pbn_note))))
            pbn_note = (8192 * pbn_note / pbs_note).clip(-8192, 8191).astype(np.int32)

            p = l.phone

            new_note = deepcopy(note_base)
            new_note['lyric'] = l.phone
            new_note['phoneme'] = p
            new_note['pos'] = s
            new_note['duration'] = l.duration()
            new_note['number'] = base_note
            notes.append(new_note)

            pbs.append({'pos': s, 'value': pbs_note})

            for i in range(pbn_note.size):
                if i == 0:
                    pbn.append({'pos': s+i, 'value': int(pbn_note[i])})
                else:
                    if pbn_note[i-1] != pbn_note[i]:
                        pbn.append({'pos': s+i, 'value': int(pbn_note[i])})

    mean_note = int(mean_note / tally)
    vpr['tracks'][0]['lastScrollPositionNoteNumber'] = mean_note
                        
    voc_dyn = (200 * np.log10(dyn_ticks)).astype(np.int32)
    for i in range(voc_dyn.size):
        if i == 0:
            dyn.append({'pos': i, 'value': int(voc_dyn[i])})
        else:
            if voc_dyn[i-1] != voc_dyn[i]:
                dyn.append({'pos': i, 'value': int(voc_dyn[i])})
    
    logging.info('Compressing')
    if os.path.exists('output.vpr'):
        os.remove('output.vpr')

    with open('output.json', 'w', encoding='utf8') as f:
        json.dump(vpr, f, indent=2)
        
    with zf.ZipFile('output.vpr', 'x', compression=zf.ZIP_DEFLATED, compresslevel=9) as zipf:
        zipf.writestr('Project/sequence.json', json.dumps(vpr))
        zipf.writestr('Project/Audio/', '')

    os.system('pause')
except Exception as e:
    for i in traceback.format_exception(e.__class__, e, e.__traceback__):
        print(i, end='')
    os.system('pause')
