'''
Author: Matthew McGonagle

Create .wav audio files containing tones whose durations match the interval lengths of the different iterates
leading to the Cantor set.
'''

# The use of sine waves for building tones in a .wav file using the wave.py module comes from the 
# following answer on stack overflow:
# https://stackoverflow.com/questions/33879523/python-how-can-i-generate-a-wav-file-with-beeps/33913403#33913403
# However, we use numpy arrays for our waveforms instead of regular python lists.

import wave
import numpy as np
import matplotlib.pyplot as plt

class ToneManipulator:
    '''
    Class for adding tones to numpy array and converting numpy array to compact data for .wav audio file. 
    Only deals with single channel .wav files and .wav files that have 2 byte integer values for each frame.

    Members
    -------
    self.framerate : Integer
        The number of frames (elements of array) per second.
    '''    

    def __init__(self, framerate):
        '''
        Initializer
        
        Records the desired framerate. This is used to calculate desired number of frames for a given duration
        in seconds.

        Parameters
        ----------
        self: 
            Self-reference to instance.
        framerate: Integer.
            The number of frames per second for the desired .wav file.
        '''

        self.framerate = framerate

    def createZeroSeries(self, duration):
        '''
        Create a numpy array of zeroes that represents a given amount of time (in seconds).
        
        Parameters
        ---------- 
        
        self : 
            Self-reference to class instance.
        duration : Float
            Duration of time in seconds that array represents. This is used to calculate the number
            of frames in the array.
        '''

        nframes = int(self.framerate * duration)
        return np.zeros(nframes)

    def addWave(self, tSeries, tStart, frequency, amplitude, duration):
        '''
        Add a wave (giving a certain tone) into a waveform time series. It will only add in the part of the wave
        that actually fits into the waveform time series.

        Parameters
        ----------
        self : 
            Self-reference to class instance.
        tSeries : Numpy array.
            A numpy array for the waveform that the new wave (new tone) will be added to.
        tStart : Float
            The time that the new wave should start (0.0 is the 0th element of tSeries).
        frequency : Float
            The frequency of the wave in Hz (periods per second).
        amplitude : Float
            The amplitude to give the new wave.
        duration : Float
            How long to make the wave in seconds. This will be converted to the correct number of frames.
            If the wave won't fit into the time series, then the part that doesn't fit will be cut off and 
            not added in.
        '''

        # If the start of the wave is before 0.0, then cut off the part of the wave that happens 
        # before 0.0. If there is nothing left of the wave, then just exit.

        if tStart < 0.0:
           duration += tStart
           tStart = 0.0
           if duration < 0.0:
              return 

        # Now set up necessary parameters.

        angularSpeed = 2.0 * np.pi * frequency / self.framerate
        nframes = int(self.framerate * duration)
        frameStart = int(self.framerate * tStart)

        if frameStart > len(tSeries):
            return

        endFrame = min(len(tSeries), frameStart + nframes)

        wave = amplitude * np.sin(angularSpeed * np.arange(endFrame - frameStart))
        tSeries[frameStart:endFrame] += wave

    def convertToWaveData(self, tSeries):
        '''
        Take a numpy array, normalize the values to the 2 byte integer values for .wav file,
        and then compactify it into a proper byte array.
        
        Parameters
        ----------
        self : 
            Self-reference to class instance.
        tSeries : Numpy array.
            The waveform to create into a data form appropriate to put into .wav file.
        '''
      
        # First normalize 

        seriesMax = np.amax(tSeries)
        seriesMin = np.amin(tSeries)
        mid = (seriesMax + seriesMin) / 2.0
        data = (tSeries - mid) / (seriesMax - seriesMin) * 2.0

        # Multiply by max

        sampleWidth = 2 # Restrict to using 2 byte integers for .wav file.
        maxsize = 2**(8 * sampleWidth - 1) - 1 # Max value for signed integer. 

        data *= maxsize

        # Convert to Integer

        data = data.astype(np.int16)
        data = data.tobytes()
        return data

    def _addCantorLevel(self, tSeries, tStarts, duration, levelFreqs, amp):
        '''
        Private function for doing recursion of adding Cantor levels. Difference between
        public function addCantorLevel is that _addCantorLevel uses an array of starting
        times.
        Members
        -------
        self : 
            Self-reference to class instance.
        tSeries : Numpy array.
            Waveform to add Cantor tones to.
        duration : Float
            The duration of each interval of this cantor level in seconds.
        levelFreqs: Array
            The frequencies for this level and the following levels.
        amp : Float
            The amplitude to use at each level.
        ''' 

        # If there are no more frequencies to add, then we stop.

        if len(levelFreqs) < 1:
            return

        # Frequency for this level is just the first frequency in levelFreqs.
        freq = levelFreqs[0]

        nextStarts = []
        nextduration = duration / 3.0
        
        # As we add in waves for each starting point of this level, we also find
        # the starting times of the next level.
        for start in tStarts:        
            self.addWave(tSeries, start, freq, amp, duration)  
            nextStarts.append(start)
            nextStarts.append(start + 2 * nextduration)

        # Now get the next level; only pass tail of levelFreqs.

        self._addCantorLevel(tSeries, nextStarts, nextduration, levelFreqs[1:], amp)

    def addCantorTones(self, tSeries, tStart, duration, levelFreqs, amplitude):
        '''
        Public function to add cantor tones into a waveform. It will only add in the part of the Cantor Tones
        that actually fits into the waveform time series.

        Members
        -------
        self : 
            Self-reference to class instance.
        tSeries : Numpy Array.
            Waveform that cantor tones are added to.
        duration : Float
            The length that the cantor tones last overall in seconds. If the duration does not fit into
            the time series then the amount that doesn't fit will be cut off. 
        levelFreqs: Array
            An array of frequencies for the tones at each level. The length of the array determines the
            number of levels to add.
        amplitude : Float
            The amplitude to use for all of the tones.
        '''
        tStarts = np.array([tStart])
        self._addCantorLevel(tSeries, tStarts, duration, levelFreqs, amplitude) 

# Determine our framerate based on some basic considerations.        

minFramesPerPeriod = 10 # We want atleast 10 frames for each period.
maxFreq = 790 # Hz, i.e. periods per second.
framerate = minFramesPerPeriod * maxFreq 
print("Using framerate = ", framerate)

# Parameters for .wav file.

nchannels = 1 
sampleWidth = 2

# First we will create a .wav file playing notes in a C chord (as found on a guitar) in progression. 
# So we set up variables to hold information for each tone.
# Reference for frequencies of notes are https://www.seventhstring.com/resources/notefrequencies.html 

chordFreqs = [261.6, 329.6, 392.0, 523.3, 659.3] # Frequencies of C chord on guitar.
tStarts = [0.0, 0.5, 1.0, 1.5, 2.0]
durations = [4.5, 3.5, 2.5, 1.5, 0.5]

manip = ToneManipulator(framerate)
waveform = manip.createZeroSeries(durations[0])

for freq, tStart, duration in zip(chordFreqs, tStarts, durations):

    print("Adding frequency ", freq, " to C chord waveform.")
    manip.addWave(waveform, tStart = tStart, frequency = freq, amplitude = 1.0, duration = duration)  

data = manip.convertToWaveData(waveform)

# Open a .wav file and write in the data.

wave_writer = wave.open('2018-01-05-output/cchord.wav', 'w')
wave_writer.setnchannels(nchannels)
wave_writer.setsampwidth(sampleWidth)
wave_writer.setframerate(framerate)
wave_writer.writeframesraw(data)
wave_writer.close()

# Graph what the waveform looks like

plt.plot(waveform)
plt.show()

# Now let's try set of Cantor tones; the frequency to use at each level will be a frequency 
# from the guitar C chord.

duration = 5.0 # Now let the .wav file last 5 seconds.
waveform = manip.createZeroSeries(duration)
manip.addCantorTones(waveform, tStart = 0, duration = duration, levelFreqs = chordFreqs, amplitude = 1.0) 
data = manip.convertToWaveData(waveform)

# Write the .wav file.

wave_writer = wave.open('2018-01-05-output/cantor.wav', 'w')
wave_writer.setnchannels(nchannels)
wave_writer.setsampwidth(sampleWidth)
wave_writer.setframerate(framerate)
wave_writer.writeframesraw(data)
wave_writer.close()

# Graph the waveform.

plt.plot(waveform)
plt.show()

# Now let's put in a series of Cantor tones. We will use the chord progression from the beginning of the song "House of the Rising Sun" by The Animals as described in the guitar tabs contained at https://tabs.ultimate-guitar.com/tab/the_animals/house_of_the_rising_sun_tabs_45131.

duration = 3.0 # Each cantor progression will last 3 seconds.

chords = [[220.0, 349.2, 440.0, 523.3, 659.3], # Am chord on guitar.
          [261.6, 329.6, 392.0, 523.3, 659.3], # C chord on guitar.
          [293.7, 440.0, 587.3, 740.0], # D chord on guitar.
          [349.2, 440.0, 523.3, 698.5], # F chord on guitar
          [220.0, 349.2, 440.0, 523.3, 659.3], # Am chord on guitar.
          [164.8, 246.9, 329.6, 415.3, 493.9, 659.3], # E chord on guitar.
          [220.0, 349.2, 440.0, 523.3, 659.3], # Am chord on guitar.
          [164.8, 246.9, 329.6, 415.3, 493.9, 659.3]] # E chord on guitar.

# Open the .wav file to write.

wave_writer = wave.open('2018-01-05-output/cantorProgression.wav', 'w')
wave_writer.setnchannels(nchannels)
wave_writer.setsampwidth(sampleWidth)
wave_writer.setframerate(framerate)

# For each list of chord frequencies in chords, add an arrangement of Cantor tones for the chord.

for chord in chords:

    # Re-zero our waveform.
    waveform = manip.createZeroSeries(duration)

    # We will be adding waveform to the end of the .wav file, so tStart is just 0.0 seconds.
    manip.addCantorTones(waveform, tStart = 0.0, duration = duration, levelFreqs = chord, amplitude = 1.0) 
    data = manip.convertToWaveData(waveform)
    wave_writer.writeframesraw(data)

wave_writer.close()
