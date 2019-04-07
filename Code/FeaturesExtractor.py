import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
from python_speech_features import mfcc
from python_speech_features import delta


class FeaturesExtractor:
    def __init__(self):
        pass
       
    def extract_features(self, audio_path):
        """
        Extract voice features including the Mel Frequency Cepstral Coefficient (MFCC)
        from an audio using the python_speech_features module, performs Cepstral Mean
        Normalization (CMS) and combine it with MFCC deltas and the MFCC double
        deltas.
     
        Args: 	    
            audio_path (str) : path to wave file without silent moments. 
        Returns: 	    
            (array) : Extracted features matrix. 	
        """
        rate, audio  = read(audio_path)
        mfcc_feature = mfcc(# The audio signal from which to compute features.
                            audio,
                            # The samplerate of the signal we are working with.
                            rate,
                            # The length of the analysis window in seconds. 
                            # Default is 0.025s (25 milliseconds)
                            winlen       = 0.05,
                            # The step between successive windows in seconds. 
                            # Default is 0.01s (10 milliseconds)
                            winstep      = 0.01,
                            # The number of cepstrum to return. 
                            # Default 13.
                            numcep       = 5,
                            # The number of filters in the filterbank.
                            # Default is 26.
                            nfilt        = 30,
                            # The FFT size. Default is 512.
                            nfft         = 512,
                            # If true, the zeroth cepstral coefficient is replaced 
                            # with the log of the total frame energy.
                            appendEnergy = True)
    
        
        mfcc_feature  = preprocessing.scale(mfcc_feature)
        deltas        = delta(mfcc_feature, 2)
        double_deltas = delta(deltas, 2)
        combined      = np.hstack((mfcc_feature, deltas, double_deltas))
        return combined