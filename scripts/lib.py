import os, sys
import numpy as np
import soundfile as sf # pip install pysoundfile
import python_speech_features as speech_lib # pip install python_speech_features
from sklearn.metrics import classification_report


def split_train_test_speakers(speakers, ratio, seed=0):
    """Separates speakers into training(size: 1-ratio) and testing(size: ratio) sets """
    
    np.random.seed(seed)
    ids = np.random.permutation(speakers.shape[0])
    
    id_train = ids[:int(speakers.shape[0]*ratio)]
    id_test = ids[int(speakers.shape[0]*ratio):]
    
    return speakers[id_train], speakers[id_test]

def split_train_test_data(x, y, ratio, seed=0):
    """Separates data x, y into training(size: 1-ratio) and testing(size: ratio) sets """
    
    np.random.seed(seed)
    ids = np.random.permutation(y.shape[0])
    
    id_train = ids[:int(y.shape[0]*ratio)]
    id_test = ids[int(y.shape[0]*ratio):]
    
    return y[id_train], x[id_train], y[id_test], x[id_test]

def create_dataset(metadata,mfcc_num=13):
    """Creates dataset from metadata with format [speaker_id, speaker_gendeer]"""
    
    dataset = np.ndarray(shape=(0,mfcc_num))
    gender_vector = np.ndarray(shape=(0,1))
    for speaker_id, gender in metadata:
        print("loading data from speaker ",int(speaker_id))
        speaker_folder = os.path.join('./dev-clean/LibriSpeech/dev-clean/',str(int(speaker_id)))
        for root, dirs, files in os.walk(speaker_folder):
            for name in files:
                if name.endswith(".flac"):
                    filepath = os.path.join(root,name)
                    with open(filepath, 'rb') as f:
                    
                        signal, samplerate = sf.read(f)

                        MFCCs = speech_lib.mfcc(signal,samplerate,winlen=0.060,winstep=0.03,numcep=mfcc_num,
                     nfilt=mfcc_num,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,
         ceplifter=22,appendEnergy=False)
                        #mean value of each MFCC over the sample
                        mean_mfcc = np.expand_dims(np.mean(MFCCs,axis=0),axis=1).T
                        dataset = np.append(dataset,mean_mfcc,axis=0)
                        gender_vector = np.append(gender_vector,np.expand_dims(gender*np.ones(mean_mfcc.shape[0]),axis=1),axis=0)
                        
    return dataset, gender_vector


def normalize_data(train_data, test_data):
    """Normalize training data to have mean 0 and variance 1, then apply the same treatment to test data"""

    means = np.mean(train_data,axis=0)
    stds = np.std(train_data,axis=0)
    
    normalized_train_data = (train_data-means)/stds
    normalized_test_data = (test_data-means)/stds
    
    return normalized_train_data, normalized_test_data

def test_classifier(classifier, test_x, test_y):
    """Test a classifier on the test_x data towards the test_y labels and print the classification report"""
    accuracy = classifier.score(test_x,test_y)

    print("Test accuracy : ", accuracy)

    predictions = classifier.predict(test_x)

    male_as_female = np.sum(np.logical_and(test_y==1,predictions==0))
    female_as_male = np.sum(np.logical_and(test_y==0,predictions==1))
    print("{:d} males classified as females out of {:.0f}, {:.3f} %".format(male_as_female, np.sum(test_y==1), 100*male_as_female/np.sum(test_y==1)))
    print("{:d} females classified as males out of {:.0f}, {:.3f} %".format(female_as_male, np.sum(test_y==0), 100*female_as_male/np.sum(test_y==0)))

    print(classification_report(test_y, predictions))

def one_hot_convert(vector, num_classes=None):
    """ (From https://stackoverflow.com/questions/29831489/numpy-1-hot-array)
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result
