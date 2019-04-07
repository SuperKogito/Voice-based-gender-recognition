import os
import pickle
import warnings
import numpy as np
from sklearn.mixture import GMM
from FeaturesExtractor import FeaturesExtractor

warnings.filterwarnings("ignore")


class ModelsTrainer:

    def __init__(self, females_files_path, males_files_path):
        self.females_training_path = females_files_path
        self.males_training_path   = males_files_path
        self.features_extractor    = FeaturesExtractor()

    def process(self):
        females, males = self.get_file_paths(self.females_training_path,
                                             self.males_training_path)
        # collect voice features
        female_voice_features = self.collect_features(females)
        male_voice_features   = self.collect_features(males)
        # generate gaussian mixture models
        females_gmm = GMM(n_components = 16, n_iter = 200, covariance_type='diag', n_init = 3)
        males_gmm   = GMM(n_components = 16, n_iter = 200, covariance_type='diag', n_init = 3)
        # fit features to models
        females_gmm.fit(female_voice_features)
        males_gmm.fit(male_voice_features)
        # save models
        self.save_gmm(females_gmm, "females")
        self.save_gmm(males_gmm,   "males")

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
            # extract MFCC & delta MFCC features from audio
            vector    = self.features_extractor.extract_features(file)
            # stack the features
            if features.size == 0:  features = vector
            else:                   features = np.vstack((features, vector))
        return features

    def save_gmm(self, gmm, name):
        """ Save Gaussian mixture model using pickle.

            Args:
                gmm        : Gaussian mixture model.
                name (str) : File name.
        """
        filename = name + ".gmm"
        with open(filename, 'wb') as gmm_file:
            pickle.dump(gmm, gmm_file)
        print ("%5s %10s" % ("SAVING", filename,))


if __name__== "__main__":
    models_trainer = ModelsTrainer("TrainingData/females", "TrainingData/males")
    models_trainer.process()
