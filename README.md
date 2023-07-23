# VPR-Real-Voice
idk man [Real Voice by Hataori](https://github.com/hataori-p/real-voice) but V6.

# Information
this was just a funny scripting thing i made out of boredom and out of just giving up because oomf showed me this one thing that exports midi with vocaloid pitchbends and it was just so weird that i had to make it myself. don't expect the best out of it.

i didn't even plan to do V6 purely but here are some amazing reasons why i used the V6 format:
1. vprs are just zip files with the sequence as a JSON file so that was an easy reverse engineer.
2. it was the only vocaloid version i had installed.

also some funny caveats because there's always some. not all vocaloid configs are created equal so you might be disappointed after labeling something and converting it to a vpr and then BOOM MikuV3_Ori- i mean a random voicebank of yours decides that the consonants are gonna have wacky timing because the configs are doo doo water. so yeah have fun :)

# How to use
1. You need Python 3.8 or higher possibly. Use your favorite method of getting it.
2. Install these libraries: numpy pyworld numba soundfile scipy librosa
3. Clone the repository somewhere.
4. Label your sound file and stuff with either Audacity (save as `.txt`) or vLabeler or wavesurfer. If the formatting is HTK make sure the extension is `.lab`
   - Audacity format is tab-separated and stores the start and end points as decimals which is equivalent to seconds
   - HTK format is space-separated and stores the start and end points as integers with a really weird unit of time. Something like 10 HTK units = 1 microsecond.
5. Save your label with the proper file extension explained above and with the same filename as your sound file.
6. Drag and drop the sound file over the script and wait.
   - OR you can run it through terminal. I'll just put the help text it spits out here too.
```
usage: vpr_rv.py [-h] [--lab LAB] [--bpm BPM] wave

idk man Real Voice but for V6.

positional arguments:
  wave               The sound file. Supported formats are anything soundfile supports.

optional arguments:
  -h, --help         show this help message and exit
  --lab LAB, -L LAB  The label (Audacity .txt or HTK .lab format). Finds a label of the same filename in the same
                     folder if not specified.
  --bpm BPM, -B BPM  The BPM to use. Default: 120
```
