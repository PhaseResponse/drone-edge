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

Pipeline: sliding_windows → combine_channels_beamform → samples_to_spectrogram → [TFLite model] → is_drone

[level_to_threshold] can be called between detections — converts dial position (0-10) to a threshold value for is_drone, for easy field calibration.

## python

```
python/realtime_pipeline.py 
python/beamform.py
```
### Safe to change

```
DETECTION_THRESHOLD    # Drone / no-drone detection threshold, currently set at knob level = 6 which translates to 0.996, based on validation set knee of PR-curve.

N_FFT_HOPS             # Input buffer hop size in units of fft window size (which is set in model to 1024 samples), default hop = 4*1024 [samples] which translates to 256ms @ 16kHz. 

# Parameters for phase correlation  
HIGHPASS_CUTOFF = 30   # microphone noise floor [Hz]
MAX_LAG = 5            # max delay between channels [samples]

# Normalization parameters for mic channel fusion  
TARGET_RMS             # Target RMS level (relative to full scale)  
MAX_GAIN               # Gain multiplier  
COMP_THRESHOLD         # Threshold to begin compression
```

### Do not change
The other parameters are fixed to the trained model. _See parameter descriptions in python code comments_.

### Functions

```
level_to_threshold     # set threshold for is_drone

sliding_windows        # extract segment of size N_SEGMENT for spectrogram [NOT INCLUDED, implement for target HW]

combine_channels_beamform  # supports 1 to N channels, tested on 1 and 4 channels

samples_to_spectrogram # configured for drone harmonics

is_drone               # based on DETECTION_THRESHOLD
```

## models

```
Trained models available in person :) 

Input shape  [ 1  193  39 1 ]

Output shape [ 1 ]
```

## Timing and Memory

Current model configurations translate to a latency of 20 * 1024 [samples] @16kHz = 1.28s.    
Model inference and spectrogram add 1.38ms per call.  
Phase alignment adds ~700ms per call (before optimization/parallelization).    

TFLite model size 424KB. TFLite conversion reduces model size by ~8x compared to the Pytorch float model <sup>*</sup>.  

| Model                        | Processing time   | Memory    |
|------------------------------|-------------------|-----------|
| Batear                       |   0.85 ms         |   -       |
| CNN (pytorch float)          |   1.38 ms         | 3.35 MB   |
| CNN (pytorch quantized       |   1.37 ms         | 1.77 MB   |
| CNN (TFLite)                 |     -             |  424 KB   |  
| Naive channel fusion         |   0.09 ms         |   -       |
| Phase aligned channel fusion | 722.53 ms         |   -       |

_CNN timing includes spectrogram computation.  Timing measured on CPU._  
_<sup>*</sup> Pytorch model memory estimated by torchinfo, includes model weights, input tensor, and forward-pass activations._

## classification results
Batear, a classical signal processing-based algorithm, is used as benchmark [2].  

PR curves show that CNN mAP 0.998 substantially outperforms classical mAP on the same 4-channel test set. CNN f1-score is also significantly higher than the classical algorithm, although the latter is evaluated at default threshold while the CNN threshold is tuned on a validation set.  
Channel fusion raises CNN mAP by ~0.40, from a mean of ~0.60 across individual channels to ~1.00, where mAP is further increased by phase alignment compared to a naive sum.  At the operating point of 0% false alarms, phase alignment raises CNN hit rate from 53.6% to 94.7%, while FA rate at max recall stays unchanged at 9.1%.  The classical algorithm also benefits from phase alignment.    
CNN results may shift with additional application-specific target HW data and scenarios.  

<img src="images/pr_curve_4ch_tdoa_naive_v1.png" width="100%">

## References
[2] Batear by TN, founder of batear.io: https://github.com/batear-io/batear

