"""
Create a database containing the hashcodes of the songs stored 
in the specified folder (.wav files only). 
The database is saved as a pickle file as a list of dictionaries.
Each dictionary has two keys 'song' and 'hashcodes', corresponding 
to the name of the song and to the hashcodes used as signature for 
the matching algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from algorithm import *


# ----------------------------------------------
# Run the script
# ----------------------------------------------
if __name__ == '__main__':

    folder = './samples/'

    # 1: Load the audio files
    import os
    audiofiles = os.listdir(folder) 
    audiofiles = [item for item in audiofiles if item[-4:] =='.wav'] #liste des morceaux sans le ./samples/ au début de chaque morceaux
    

    # 2: Set the parameters of the encoder
    encoder = Encoding()
    
    # 3: Construct the database
    database = []
    for item in audiofiles :
        fs, s = read('./samples/' + item) 
        encoder.process(fs, s[:]) 
        hash = encoder.processsuit()
        database.append(hash)

    # 4: Save the database
    with open('songs.pickle', 'wb') as handle:
        pickle.dump(database, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('songs.pickle', 'wb') as handle:
        for lign in handle :
            print(lign)
    

