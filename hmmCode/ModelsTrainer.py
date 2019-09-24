import os
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor
from hmmlearn import hmm


import pydub
import subprocess
import speech_recognition as sr
from pydub import AudioSegment
from subprocess import Popen, PIPE
from pydub.silence import split_on_silence, detect_nonsilent

warnings.filterwarnings("ignore")


class ModelsTrainer:

    def __init__(self, females_files_path, males_files_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.features_extractor    = FeaturesExtractor()







    def ffmpeg_silence_eliminator(self, input_path, output_path):
        """
        Eliminate silence from voice file using ffmpeg library.
        Args:
            input_path  (str) : Path to get the original voice file from.
            output_path (str) : Path to save the processed file to.
        Returns:
            (list)  : List including True for successful authentication, False otherwise and a percentage value
                      representing the certainty of the decision.
        """
        # filter silence in mp3 file
        filter_command = ["ffmpeg", "-i", input_path, "-af", "silenceremove=1:0:0.05:-1:1:-36dB", "-ac", "1", "-ss", "0","-t","90", output_path, "-y"]
        out = subprocess.Popen(filter_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out.wait()
        
        with_silence_duration = os.popen("ffprobe -i '" + input_path + "' -show_format -v quiet | sed -n 's/duration=//p'").read()
        no_silence_duration   = os.popen("ffprobe -i '" + output_path + "' -show_format -v quiet | sed -n 's/duration=//p'").read()
        
        # print duration specs
        try:
            print("%-32s %-7s %-50s" % ("ORIGINAL SAMPLE DURATION",         ":", float(with_silence_duration)))
            print("%-23s %-7s %-50s" % ("SILENCE FILTERED SAMPLE DURATION", ":", float(no_silence_duration)))
        except:
            print("WaveHandlerError: Cannot convert float to string", with_silence_duration, no_silence_duration)
    
        # convert file to wave and read array
        load_command = ["ffmpeg", "-i", output_path, "-f", "wav", "-" ]
        p            = Popen(load_command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        data         = p.communicate()[0]
        audio_np     = np.frombuffer(data[data.find(b'\x00data')+ 9:], np.int16)
        
        # delete temp silence free file, as we only need the array
        #os.remove(output_path)
        return audio_np, no_silence_duration
    





    def process(self):
        females, males = self.get_file_paths(self.females_training_path,
                                             self.males_training_path)
        # collect voice features
        female_voice_features = self.collect_features(females)
        male_voice_features   = self.collect_features(males)
        # generate gaussian mixture models
        females_gmm = hmm.GaussianHMM(n_components=3)
        males_gmm   = hmm.GaussianHMM(n_components=3)
        ubm         = hmm.GaussianHMM(n_components=3)
        # fit features to models
        females_gmm.fit(female_voice_features)
        males_gmm.fit(male_voice_features)
        ubm.fit(np.vstack((female_voice_features, male_voice_features)))
        # save models
        self.save_gmm(females_gmm, "females")
        self.save_gmm(males_gmm,   "males")
        self.save_gmm(males_gmm,   "ubm")


    def get_file_paths(self, females_training_path, males_training_path):
        # get file paths
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) ]
        return females, males

    def collect_features(self, files):
        """
    	Collect voice features from various speakers of the same gender.

    	Args:
    	    files (list) : List of voice file paths.

    	Returns:
    	    (array) : Extracted features matrix.
    	"""
        features = np.asarray(())
        # extract features for each speaker
        for file in files:
            print("%5s %10s" % ("PROCESSNG ", file))
            self.ffmpeg_silence_eliminator(file, file.split('.')[0] + "_without_silence.wav")
        
            # extract MFCC & delta MFCC features from audio
            try: 
                vector    = self.features_extractor.extract_features(file.split('.')[0] + "_without_silence.wav")
                # stack the features
                if features.size == 0:  features = vector
                else:                   features = np.vstack((features, vector))           
            except : pass
            os.remove(file.split('.')[0] + "_without_silence.wav")
        return features

    def save_gmm(self, gmm, name):
        """ Save Gaussian mixture model using pickle.

            Args:
                gmm        : Gaussian mixture model.
                name (str) : File name.
        """
        filename = name + ".hmm"
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print ("%5s %10s" % ("SAVING", filename,))


if __name__== "__main__":
    models_trainer = ModelsTrainer("TrainingData/females", "TrainingData/males")
    models_trainer.process()
