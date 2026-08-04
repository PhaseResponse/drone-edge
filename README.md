# drone-edge
Edge deployment model and pipeline. 

```
          ●
         /|\
        / | \
       ●  |  ●
          ●   
 
   Vertex + 3 base mics, synchronized and equally spaced.

  [4-mic] --> [fusion] -> [Spectrogram] --> [TFLite model] --> [threshold] --> drone / no-drone
                                                                    ^
                                                                    |
                                                           [level to threshold] 
       
```

Pipeline: sliding_windows → combine_channels → samples_to_spectrogram → [TFLite model] → is_drone

[level_to_threshold] is callable at any time — converts dial position (0-10) to a threshold value for is_drone, for easy field calibration.
Threshold currently set at knob level = 6 which translates to DETECTION_THRESHOLD = 0.974, based on validation set knee of PR-curve.

## python

```
python/realtime_pipeline.py 
```

Internal parameters that the model was trained on (do not change):

N_FFT = 1024 sample window

N_OVERLAP = 512 samples

FS = 16000 Hz sampling rate

N_SEGMENT = N_FFT * 20 = samples per segment = 1.28 seconds/segment

Functions:

level_to_threshold - set threshold for is_drone

sliding_windows - extract segment of size N_SEGMENT  for spectrogram

combine_channels - supports 1 to N channels, tested on 1 and 4 channels

samples_to_spectrogram - configured for drone harmonics

is_drone - based on DETECTION_THRESHOLD

## models

```
Trained models available in person :) 
```

Input shape  [ 1  193  39 1 ]

Output shape [ 1 ]

