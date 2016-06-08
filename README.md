# speechvis

This project provides a set of tools for segmenting, analyzing, and visualizing recorded speech. In particular, the goal
is to treat segmented speech from the Warlaumont IVFCR corpus as high-dimensional time-series, and provide a library of
analyses to reveal the structure within these high-dimensional data.

### Features

* Parse LENA ITS files to extract segment boundaries and labels for entire recordings.
* Plot speaker counts, durations, intervals, and volubility for parsed recordings.
* Segment WAV file recordings to save segments as individual, categorized files.
* Generate and plot acoustic analyses for individual segments.
  * Waveform
  * Power Over Time and Power Over Frequency
  * Spectrogram and Formant Frequency Detection
  * Mel-Frequency Filter Bank and Mel-Frequency Cepstral Coefficients

### Examples

#### Parse a recording and display speaker segment counts

_Follow the normal process for importing data from the used LENA recorder to your LENA
workstation. Use the LENA software export to save the ITS and WAV files._

From a terminal at the root of the speechvis library:

`ipython -i speechvis.py`

In the python interpreter:

```
root = <new root location>
ids.append(<new recording id>)
recording = Recording(root, ids[-1])
plot_speaker_counts(recording)
```

#### Display a Principal Components decomposition of the MFCCs for a speaker

In the python interpreter:

```
recording = Recording(root, ids[<Rec #>])
plot_mfcc_pca(recording, 'CHN') # Change CHN to any LENA speaker category
```
