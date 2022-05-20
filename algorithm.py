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
        self.size = 500 #128 samples  voire 500
        self.overlap = 400   #32 voire 400


        


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
        idx=np.flip(u.argsort())
        f=u[idx]**2
        nrj=np.sum(f)
        cumul = np.cumsum(f)
        u[idx[cumul>0.9*nrj]]=0
      #   print(np.sum(u**2)/nrj) #bien 0.9
      #   print(np.sum(u!=0))  #beaucoup plus court qu'avant
        u=u.reshape((self.spectre.shape[0],self.spectre.shape[1]))
        self.spectre=u
        self.lign, self.col = peak_local_max(np.abs(self.spectre), min_distance = 40, exclude_border=False).T  #50 ou 5
        
#50 pour delta t et delta f
#les delta du hash doivent être déterminés par nous même, pour qu'ils soient optimaux

    def display_spectrogram(self):
      plt.scatter( self.times[self.col], self.freq[self.lign], s=2)
      plt.show()
      """
        Display the spectrogram of the audio signal
        """

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

        # Insert code here

             
    def display_scatterplot(self):

        """
        Display through a scatterplot the times associated to the hashes
        that match
        """
    
        # Insert code here


    def display_histogram(self):

        """
        Display the offset histogram
        """

        # Insert code here



# ----------------------------------------------
# Run the script
# ----------------------------------------------
if __name__ == '__main__':

    encoder = Encoding()
    fs, s = read('./samples/Lucid Haze - Amulets.wav')
    encoder.process(fs, s[:])   #900000
    encoder.display_spectrogram() #display_anchors=True





