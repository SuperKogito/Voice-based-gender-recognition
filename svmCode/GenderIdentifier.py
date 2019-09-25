import os
import pickle
import warnings
import numpy as np
from FeaturesExtractor import FeaturesExtractor
from hmmlearn import hmm

warnings.filterwarnings("ignore")

import pydub
import subprocess
import speech_recognition as sr
from pydub import AudioSegment
from subprocess import Popen, PIPE
from pydub.silence import split_on_silence, detect_nonsilent

from sklearn.svm import SVC


class GenderIdentifier:

    def __init__(self, females_files_path, males_files_path, females_model_path, males_model_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.error                 = 0
        self.total_sample          = 0
        self.features_extractor    = FeaturesExtractor()
        # load models
        self.females_gmm = pickle.load(open(females_model_path, 'rb'))
        self.males_gmm   = pickle.load(open(males_model_path, 'rb'))
        
        # svm
        self.X_train = np.vstack((self.females_gmm, self.males_gmm))
        self.y_train = np.hstack(( -1 * np.ones(self.females_gmm.shape[0]), np.ones(self.males_gmm.shape[0])))
        self.clf = SVC(kernel = 'rbf', probability=True)
        self.clf.fit(self.X_train, self.y_train)
        
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
        files = self.get_file_paths(self.females_training_path, self.males_training_path)
        # read the test directory and get the list of test audio files
        for file in files:
            self.total_sample += 1
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))

            #self.ffmpeg_silence_eliminator(file, file.split('.')[0] + "_without_silence.wav")

            # extract MFCC & delta MFCC features from audio
            try: 
                # vector = self.features_extractor.extract_features(file.split('.')[0] + "_without_silence.wav")
                vector = self.features_extractor.extract_features(file)
                print("S1")
                # generate gaussian mixture models
                spk_gmm = hmm.GaussianHMM(n_components=16)      
                print("S2")

                # fit features to models
                spk_gmm.fit(vector)
                print("S3")
                
                self.spk_vec = spk_gmm.means_
                print(self.clf.predict(self.spk_vec))
                if sum(self.clf.predict(self.spk_vec)) > 0 : sc =  1
                else                                       : sc = -1
                genders = {-1: "female", 1: "male"}
                winner = genders[sc]
                expected_gender = file.split("/")[1][:-1]
                print(expected_gender)
                
                print("%10s %6s %1s" % ("+ EXPECTATION",":", expected_gender))
                print("%10s %3s %1s" %  ("+ IDENTIFICATION", ":", winner))

                if winner != expected_gender: self.error += 1
                print("----------------------------------------------------")

    
            except : print("Error")
            # os.remove(file.split('.')[0] + "_without_silence.wav")
            
            
        accuracy     = ( float(self.total_sample - self.error) / float(self.total_sample) ) * 100
        accuracy_msg = "*** Accuracy = " + str(round(accuracy, 3)) + "% ***"
        print(accuracy_msg)  
        


    def get_file_paths(self, females_training_path, males_training_path):
        # get file paths
        females = [ os.path.join(females_training_path, f) for f in os.listdir(females_training_path) ]
        males   = [ os.path.join(males_training_path, f) for f in os.listdir(males_training_path) ]
        files   = females + males
        return files

    def identify_gender(self, vector):
        ubm_score = self.ubm.score(vector)
        # female hypothesis scoring
        is_female_log_likelihood = self.females_gmm.score(vector) / ubm_score
        # male hypothesis scoring
        is_male_log_likelihood = self.males_gmm.score(vector) / ubm_score

        print("%10s %5s %1s" % ("+ FEMALE SCORE",":", str(round(is_female_log_likelihood, 3))))
        print("%10s %7s %1s" % ("+ MALE SCORE", ":", str(round(is_male_log_likelihood,3))))

        if is_male_log_likelihood > is_female_log_likelihood: winner = "male"
        else                                                : winner = "female"
        return winner


if __name__== "__main__":
    gender_identifier = GenderIdentifier("TestingData/females", "TestingData/males", "females.hmm", "males.hmm")
    gender_identifier.process()
