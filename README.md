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

[level_to_threshold] can be called between detections — converts dial position (0-10) to a threshold value for is_drone, for easy field calibration.

## python

```
python/realtime_pipeline.py 
```
Safe to change:

DETECTION_THRESHOLD: Drone / no-drone detection threshold, currently set at knob level = 6 which translates to 0.974, based on validation set knee of PR-curve.

N_FFT_HOPS: Input buffer hop size in units of fft window size (which is set in model to 1024 samples), default hop = 4*1024 [samples] which translates to 256ms @ 16kHz.

The other parameters are fixed to the trained model, do not change. _See parameter descriptions in python code comments_.

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

## Timing and Memory

Current model configurations translate to a latency of 20 [segments] * 1024 [samples] @16kHz = 1.28s.  
Model inference and spectrogram add a negligible 1.38ms (CPU profiling).

TFLite model size 424KB.

