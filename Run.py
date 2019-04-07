import os
import urllib.request


if __name__== "__main__":
    # download dataset
    print("# Download dataset zip file")
    zip_url  = "http://www.openslr.org/resources/45/ST-AEDS-20180100_1-OS.tgz"
    urllib.request.urlretrieve(zip_url, 'SLR45.tgz')

    # extract and manage dataset files
    print("# Mange and organize files")
    os.system('python3 Code/DataManager.py')

    # train gender gmm models
    print("# Train gender models")
    os.system('python3 Code/ModelsTrainer.py')

    # test system and recognise/identify speakers gender
    print(" # Identify genders")
    os.system('python3 Code/GenderIdentifier.py')
