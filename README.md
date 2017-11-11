# Speaker-Gender-Classification
Classification of speaker's gender based on MFCC features

Gaétan Ramet

## Overview

This project is about gender classification of speakers. We use different Machine learning algorithms to predict the gender of speakers based on the MFCCs in small audio files.

## Data

The data used for training and testing comes from the 'dev-clean' dataset of [Librispeech](http://www.openslr.org/12/). Download the dataset, then extract the archive and copy the folder beside the notebook.

## Dependencies

This project make use of a few python libraries :

- [numpy](http://www.numpy.org/) 
- [pysoundfiles](https://github.com/bastibe/PySoundFile) for sound extraction
- [python_speech_features](https://github.com/jameslyons/python_speech_features) for MFCCs extraction
- [Sickit-learn](http://scikit-learn.org/stable/) for Machine learning algorithms
- [Tensorflow](https://www.tensorflow.org/) for Neural networks

Make sure to download and install the necessary libraries before running the notebook.

## Project

Results are presented in the notebook, a few functions are written in the lib module for readability.
