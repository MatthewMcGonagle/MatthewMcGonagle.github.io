import wave
import numpy as np
import matplotlib.pyplot as plt

# Based on answer on stackoverflow:
# https://stackoverflow.com/questions/33879523/python-how-can-i-generate-a-wav-file-with-beeps/33913403#33913403

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
        self.framerate = framerate

    def createZeroSeries(self, duration):

        nframes = int(self.framerate * duration)
        return np.zeros(nframes)

    def addWave(self, tSeries, tStart, frequency, amplitude, duration):

        angularSpeed = 2.0 * np.pi * frequency / self.framerate
        nframes = int(self.framerate * duration)
        frameStart = int(self.framerate * tStart)

        wave = amplitude * np.sin(angularSpeed * np.arange(nframes))
        tSeries[frameStart:frameStart + nframes] += wave

    def convertToWaveData(self, tSeries):
      
        # First normalize 

        seriesMax = np.amax(tSeries)
        seriesMin = np.amin(tSeries)
        mid = (seriesMax + seriesMin) / 2.0
        data = (tSeries - mid) / (seriesMax - seriesMin) * 2.0

        # Multiply by max
        sampleWidth = 2 # Restrict to using 2 byte integers for .wav file.
        self.maxsize = 2**(8 * sampleWidth - 1) - 1 # Max value for signed integer. 

        data *= self.maxsize

        # Convert to Integer
        data = data.astype(np.int16)
        data = data.tobytes()
        return data

    def _addCantorLevel(self, tSeries, tStarts, duration, levelFreqs, amp):

        if len(levelFreqs) < 1:
            return

        freq = levelFreqs[0]

        nextStarts = []
        nextduration = duration / 3.0
        for start in tStarts:        
            self.addWave(tSeries, start, freq, amp, duration)  
            nextStarts.append(start)
            nextStarts.append(start + 2 * nextduration)

        self._addCantorLevel(tSeries, nextStarts, nextduration, levelFreqs[1:], amp)

    def addCantorTones(self, tSeries, tStart, duration, levelFreqs, amplitude):
        tStarts = np.array([tStart])
        self._addCantorLevel(tSeries, tStarts, duration, levelFreqs, amplitude) 
        
minFramesPerPeriod = 10
maxFreq = 790
framerate = minFramesPerPeriod * maxFreq 
print("Using framerate = ", framerate)

nchannels = 1
sampleWidth = 2
duration = 3

# Reference for frequencies of notes are https://www.seventhstring.com/resources/notefrequencies.html 
chordFreqs = [261.6, 329.6, 392.0, 523.3, 659.3] # Frequencies of C chord on guitar.
tStarts = [0.0, 0.5, 1.0, 1.5, 2.0]
durations = [5.0, 3.5, 2.5, 1.5, 0.5]

manip = ToneManipulator(framerate)
waveform = manip.createZeroSeries(durations[0])

for freq, tStart, duration in zip(chordFreqs, tStarts, durations):

    print("Adding frequency ", freq)
    manip.addWave(waveform, tStart = tStart, frequency = freq, amplitude = 1.0, duration = duration)  

data = manip.convertToWaveData(waveform)

wave_writer = wave.open('2018-01-05-output/cchord.wav', 'w')
wave_writer.setnchannels(nchannels)
wave_writer.setsampwidth(sampleWidth)
wave_writer.setframerate(framerate)
wave_writer.writeframesraw(data)
wave_writer.close()

plt.plot(waveform)
plt.show()

duration = 5.0
# framerate = 44.1 * 10**3 # CD quality 
manip = ToneManipulator(framerate)
waveform = manip.createZeroSeries(duration)
manip.addCantorTones(waveform, tStart = 0, duration = duration, levelFreqs = chordFreqs, amplitude = 1.0) 
data = manip.convertToWaveData(waveform)

wave_writer = wave.open('2018-01-05-output/cantor.wav', 'w')
wave_writer.setnchannels(nchannels)
wave_writer.setsampwidth(sampleWidth)
wave_writer.setframerate(framerate)
wave_writer.writeframesraw(data)
wave_writer.close()

plt.plot(waveform)
plt.show()

# C-F-G chord progression reference is https://www.uberchord.com/blog/5-popular-common-guitar-chord-progressions-song-writers/
chords = [[261.6, 329.6, 392.0, 523.3, 659.3], # C Chord on Guitar
          [174.6, 261.6, 349.2, 440.0, 523.3, 349.2], # F Chord on Guitar
          [196.0, 246.9, 293.7, 392.0, 587.3, 784.0]] # G Chord on Guitar 
waveform = manip.createZeroSeries(duration * 3)
for start, chord in zip(duration * np.arange(3), chords):
    manip.addCantorTones(waveform, tStart = start, duration = duration, levelFreqs = chord, amplitude = 1.0) 
data = manip.convertToWaveData(waveform)

wave_writer = wave.open('2018-01-05-output/cantorProgression.wav', 'w')
wave_writer.setnchannels(nchannels)
wave_writer.setsampwidth(sampleWidth)
wave_writer.setframerate(framerate)
wave_writer.writeframesraw(data)
wave_writer.close()

plt.plot(waveform)
plt.show()
