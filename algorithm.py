import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import spectrogram
from skimage.feature import peak_local_max

class Encoding:

    """  
    The fingerprint is created through the following steps
    - compute the spectrogram of the audio signal
    - extract local maxima of the spectrogram
    - create hashes using these maxima

    """

    def __init__(self):

        """
        Initialize in the constructor all the parameters required for
        creating the signature of the audio files. These parameters include for
        instance:
        - the window selected for computing the spectrogram
        - the size of the temporal window 
        - the size of the overlap between subsequent windows
        - etc.
   
        All these parameters should be kept as attributes of the class.
        """
        self.window = "gaussian"
        self.size = 500 #128 samples  voire 500 car marche mieux
        self.overlap = 400   #32 voire 400 car marche mieux

    def process(self, fs, s):

        """
        This function takes as input a sampled signal s and the sampling
        frequency fs and returns the fingerprint (the hashcodes) of the signal.
        The fingerprint is created through the following steps
        - spectrogram computation
        - local maxima extraction
        - hashes creation

        Implement all these operations in this function. Keep as attributes of
        the class the spectrogram, the range of frequencies, the anchors, the 
        list of hashes, etc.

        Each hash can conveniently be represented by a Python dictionary 
        containing the time associated to its anchor (key: "t") and a numpy 
        array with the difference in time between the anchor and the target, 
        the frequency of the anchor and the frequency of the target 
        (key: "hash")
        Parameters
        ----------
        fs: int
           sampling frequency [Hz]
        s: numpy array
           sampled signal
        """
        self.fs = fs
        self.s = s
        self.freq, self.times, self.spectre = spectrogram(s, fs, noverlap = self.overlap, window=(self.window, self.size), nperseg = self.size)
        u=self.spectre.reshape((self.spectre.shape[0]*self.spectre.shape[1]))
        #on va maintenant ne garder que les coef correspondant à 90% de l'énergie
        idx=np.flip(u.argsort())  
        f=u[idx]**2       #la liste ordonnée des énergies, par ordre décroissant
        nrj=np.sum(f)     #nrj totale du signal
        cumul = np.cumsum(f)       #nrj cumulée jusque là (si [1,2,3] renvoie [1,3,8])
        u[idx[cumul>0.9*nrj]]=0     #met à 0 tous les coefficients qui sont pas nécessaires pour atteindre 90%
      #   print(np.sum(u**2)/nrj) #bien 0.9 donc on a bien gardé 90% de l'énergie du signal
      #   print(np.sum(u!=0), len(u)) #beaucoup plus court qu'avant
        u=u.reshape((self.spectre.shape[0],self.spectre.shape[1]))
      #   print(len(s)*10**-4)
        self.spectre = u   #u est la matrice à 2 dim fréquence (ordonnée) temps (abscisses) où apparaissent les coefficients correspondant à 90% de l'énergie
        self.lign, self.col = peak_local_max(np.abs(self.spectre), min_distance = 30, num_peaks = int(len(s)*(10**-4)),
          exclude_border=False).T #on garde un nombre de pics proportionnel à longueur de l'extrait, et de sorte à avoir quelques milliers de pics pour un morceau
      #   print(len(self.lign))
           
        self.hash = []
        deltaf = 1000 #de 1000 à 5000 hz 
        deltat = 0.25 #de 1 à 2 s
        for ancre in range(len(self.lign)):
           t_a, f_a = self.times[self.col[ancre]], self.freq[self.lign[ancre]]
           for i in range(len(self.lign)):
              if i != ancre:
                 f_i, t_i = self.freq[self.lign[i]], self.times[self.col[i]], 
                 if 0 < abs(t_i - t_a) < deltat and abs(f_i - f_a) < deltaf:
                    v_ia = np.array([t_i - t_a, f_a, f_i ])
                    self.hash.append({"t" : t_a, "hash" : v_ia })
      #   print(f'morceau a {len(self.hash)}') 
                     
    def display_spectrogram(self): #spectrogramme du signal audio
      plt.scatter(self.times[self.col], self.freq[self.lign], s=2)
      plt.show()

class Matching:

    """
    hashes1: list of dictionaries
       hashes extracted as fingerprints for the first audiofile. Each hash 
       is represented by a dictionary containing the time associated to
       its anchor (key: "t") and a numpy array with the difference in time
       between the anchor and the target, the frequency of the anchor and
       the frequency of the target (key: "hash")

    hashes2: list of dictionaries

    matching: numpy array
       absolute times of the hashes that match together

    offset: numpy array
       time offsets between the matches
    """

    def __init__(self, hashes1, hashes2):

        """
        Implement a code establishing correspondences between the hashes of
        both files. Once the correspondences computed, construct the 
        histogram of the offsets between hashes. Finally, search for a criterion
        based on the histogram that allows to determine if both audio files 
        match

        hashes1: list of dictionaries
           hashes extracted as fingerprint for the first audiofile. Each hash 
           is represented by a dictionary containing the time associated to
           its anchor (key: "t") and a numpy array with the difference in time
           between the anchor and the target, the frequency of the anchor and
           the frequency of the target

        hashes2: list of dictionaries
        """
        self.hashes1 = hashes1
        self.hashes2 = hashes2
        self.matching = []
        self.offset = []
        for d1 in hashes1 : #morceau
           for d2 in hashes2 :  #extrait 
              if (d1['hash'] == d2['hash']).all():
                 self.matching.append([d1['t'], d2['t']])
                 self.offset.append(d1['t']- d2['t'])

    def display_scatterplot(self):
        X = np.array([self.matching[k][0] for k in range(len(self.matching))])
        Y = np.array([self.matching[k][1] for k in range(len(self.matching))])
        plt.scatter(X,Y, s= 3)
        plt.title ("scatterplot of the times associated to the hashes matching")
        plt.show()

    def display_histogram(self, nom = None): #display l'histo des offsets
        plt.hist(self.offset, bins= 200)
        plt.title(f'Offset histogram {nom}')
        plt.show()


if __name__ == '__main__': 
    encoder1 = Encoding()
    encoder2 = Encoding()
    extr = Encoding()

    fs1, s1 = read('./samples/Frisk - Au.Ra.wav') 
    encoder1.process(fs1, s1[:]) 

   #  encoder1.display_spectrogram()   
   #  fs2, s2 = read('./samples/Dark Alley Deals - Aaron Kenny.wav')  #on compare le morceau avec un autre 
   #  encoder2.process(fs2, s2[:]) 
   
   #  encoder2.display_spectrogram()
   #  extr.process(fs1, s1[1000000:1720000] )

   #when it matches
    matching1 = Matching(encoder1.hash, extr.hash)
    matching1.display_scatterplot()
    matching1.display_histogram()

   #when it doesn't match
   #  matching2 = Matching(encoder2.hash, extr.hash)
   #  matching2.display_scatterplot()
   #  matching2.display_histogram()