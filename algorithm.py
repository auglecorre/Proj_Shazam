"""
Algorithm implementation
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import read
from scipy.signal import spectrogram
from skimage.feature import peak_local_max

# ----------------------------------------------------------------------------
# Create a fingerprint for an audio file based on a set of hashes
# ----------------------------------------------------------------------------


class Encoding:

    """  
    Class implementing the procedure for creating a fingerprint 
    for the audio files

    The fingerprint is created through the following steps
    - compute the spectrogram of the audio signal
    - extract local maxima of the spectrogram
    - create hashes using these maxima

    """

    def __init__(self):

        """
        Class constructor

        To Do
        -----

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
        self.overlap = 32   #32 voire 400 car marche mieux


        


    def process(self, fs, s):

        """

        To Do
        -----

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
        #on essaie maintenant de ne garder que les coef correspondant à 90% de l'énergie
        idx=np.flip(u.argsort())  
        f=u[idx]**2       #la liste ordonnée des énergies, par ordre décroissant
        nrj=np.sum(f)     #nrj totale du signal            
        cumul = np.cumsum(f)       #nrj cumulée jusque là (si [1,2,3] renvoie [1,3,8])
        u[idx[cumul>0.9*nrj]]=0     #met à 0 tous les coefficients qui sont pas nécessaires pour atteindre 90%
      #   print(np.sum(u**2)/nrj) #bien 0.9 donc on a bien gardé 90% de l'énergie du signal
      #   print(np.sum(u!=0))  #beaucoup plus court qu'avant
        u=u.reshape((self.spectre.shape[0],self.spectre.shape[1]))
        self.spectre = u   #u est la matrice à 2 dim fréquence (ordonnée) temps (abscisses) où apparaissent les coefficients correspondant à 90% de l'énergie
        self.lign, self.col = peak_local_max(np.abs(self.spectre), min_distance = 50, exclude_border=False).T  #50 ou 5 pour min distance? à déterminer
        
#50 pour delta t et delta f, proposé par Xavier et Boris
#les delta du hash doivent être déterminés par nous même, pour qu'ils soient optimaux

    def display_spectrogram(self):
      plt.scatter(self.times[self.col], self.freq[self.lign], s=2)
      plt.show()
      """
        Display the spectrogram of the audio signal
        """
    def processsuit(self):
       self.hash = []   #il faut que pour chacune des ancres(ti,fi) on lui associe l'ensemble des points vérifiant ce qu'ils disent
                           #pour le hash, 3 grandeurs à chaque fois
       nb_anchor = int(len(self.lign)/10) #on prend un point ancre tout les dix points
       delta_t = 5*min(self.times[self.col]) #18,8s
       delta_f = 5*min(self.freq[self.lign]) #720 Hz
       for k in range(nb_anchor) :
          i_a = np.random.randint(len(self.col)) #on choisi au hazard l'indice de la liste des indices 
          t_a, f_a = np.array([self.times[self.col[i_a]], self.freq[self.lign[i_a]] ]) #c_a=(t_a,f_a)
          for i in range(len(self.lign)) : 
             f_i, t_i = self.times[self.col[i]], self.freq[self.lign[i]]
             if 0 < t_i - t_a < delta_t and np.abs(f_i - f_a) < delta_f and i!=i_a : 
                v_ia = np.array([t_i - t_a, f_a, f_i ])
                self.hash.append({"t" : t_a, "hash" : v_ia })
       print(self.hash) #problème : on a des f_i qui ne sont pas présents dans le spectrogramme
       return(self.hash) 



# ----------------------------------------------------------------------------
# Compares two set of hashes in order to determine if two audio files match
# ----------------------------------------------------------------------------

class Matching:

    """
    Compare the hashes from two audio files to determine if these
    files match

    Attributes
    ----------

    hashes1: list of dictionaries
       hashes extracted as fingerprints for the first audiofile. Each hash 
       is represented by a dictionary containing the time associated to
       its anchor (key: "t") and a numpy array with the difference in time
       between the anchor and the target, the frequency of the anchor and
       the frequency of the target (key: "hash")

    hashes2: list of dictionaries
       hashes extracted as fingerprint for the second audiofile. Each hash 
       is represented by a dictionary containing the time associated to
       its anchor (key: "t") and a numpy array with the difference in time
       between the anchor and the target, the frequency of the anchor and
       the frequency of the target (key: "hash")

    matching: numpy array
       absolute times of the hashes that match together

    offset: numpy array
       time offsets between the matches
    """

    def __init__(self, hashes1, hashes2):

        """
        Class constructor

        Compare the hashes from two audio files to determine if these
        files match

        To Do
        -----

        Implement a code establishing correspondences between the hashes of
        both files. Once the correspondences computed, construct the 
        histogram of the offsets between hashes. Finally, search for a criterion
        based on the histogram that allows to determine if both audio files 
        match

        Parameters
        ----------

        hashes1: list of dictionaries
           hashes extracted as fingerprint for the first audiofile. Each hash 
           is represented by a dictionary containing the time associated to
           its anchor (key: "t") and a numpy array with the difference in time
           between the anchor and the target, the frequency of the anchor and
           the frequency of the target

        hashes2: list of dictionaries
           hashes extracted as fingerprint for the second audiofile. Each hash 
           is represented by a dictionary containing the time associated to
           its anchor (key: "t") and a numpy array with the difference in time
           between the anchor and the target, the frequency of the anchor and
           the frequency of the target
        """


        self.hashes1 = hashes1
        self.hashes2 = hashes2
        self.matching = []
        self.offset = []
        for d1 in hashes1 : 
           for d2 in hashes2 :
              if d1['hash'] == d2['hash'] : 
                 self.matching.append([d1['t'], d2['t']])
                 #self.offset.append(np.abs(d1['t']- d2['t'])) #l'offset correspond au décalage de temps entre deux matchs ? ou entre les deux chanson ? 
        for i in len(self.matching) :
           delta_t = np.abs(self.matching[i+1][0] - self.matching[i][0]) #ou [i][1] et [i+1][1]
           self.offset.append(delta_t)
        self.matching = np.array(self.matching)
        self.offset = np.array(self.offset)


             
    def display_scatterplot(self):

        """
        Display through a scatterplot the times associated to the hashes
        that match
        """
        X = np.array([self.matching[k][0] for k in range(len(self.matching))])
        Y = np.array([self.matching[k][1] for k in range(len(self.matching))])
        plt.scatter(X,Y)
        plt.title ("scatterplot of the times associated to the hashes matching")
        plt.show()


    def display_histogram(self):

        """
        Display the offset histogram
        """

        plt.histogram(self.offset)
        plt.title("Offset histogram")
        plt.show()



# ----------------------------------------------
# Run the script
# ----------------------------------------------
'''if __name__ == '__main__':

    encoder = Encoding()
    fs, s = read('./samples/Lucid Haze - Amulets.wav')
    encoder.process(fs, s[:])   #900000
    encoder.display_spectrogram() #display_anchors=True
    hash = encoder.processsuit()'''

if __name__ == '__main__': 

    encoder = Encoding()
    fs1, s1 = read('./samples/Lucid Haze - Amulets.wav') 
    fs2, s2 = read('./samples/Frisk - Au.Ra.wav')  #on compare le morceau avec un autre 
    encoder.process(fs1, s1[:]) 
    encoder.process(fs2, s2[:])  
    hash1 = encoder.processsuit()
    hash2 = encoder.processsuit()
    matching = Matching(hash1, hash2)
    matching.display_scatterplot()

