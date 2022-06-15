"""
Description
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from time import time
from scipy.io.wavfile import read
from algorithm import *

# ----------------------------------------------
# Run the script
# ----------------------------------------------
if __name__ == '__main__':
    tours = 100
    with open('songs.pickle', 'rb') as handle:
        database = pickle.load(handle)
    f=0
    temps = time()
    for i in range(tours):
        encoder = Encoding()
        
        songs = [item for item in os.listdir('./samples') if item[:-4] != '.wav']
        song = random.choice(songs)
        # print('Selected song: ' + song[:-4])
        filename = './samples/' + song

        fs, s = read(filename)
        tmin = int(50*fs) # We select an extract starting at 50s ...
        duration = int(10*fs) # ... which lasts 10s

        encoder.process(fs, s[tmin:tmin + duration])
        hashes = encoder.hash

        m=0
        for morceau in database:
            matching = Matching(morceau['hashcodes'],hashes)
            # matching.display_histogram(nom = morceau['song'])
            if matching.offset != [] :
                test = Counter(matching.offset).most_common(1)[0][1]
                if test > m :
                    m = test
                    morcea = morceau['song']
        if morcea == song :
            f+=1
    print(f"L'algorithme a correctement reconnu {f}/{tours} échantillons en un temps moyen de {(time() - temps)/tours} secondes par échantillon")
        # print(morcea==song)





