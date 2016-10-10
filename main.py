"""
                Tentative Document
    ------------------------------------------
    A code which will train itself from songs in .wav format.
    Planning to learn Signal/Audio processing using python. Try feature extraction.
    Next any song given as input, user will be given the output(predicted genre).
    Planning to use Classification algorithms for predicting and training data. Also supervised and unsupervised learning algorithms.
    Need to find a module or an api for fetching songs from youtube at a fixed frequency or same quality.

"""

# ignore warnings.
import warnings
warnings.filterwarnings('ignore')

# Audio processing
from scipy.io import wavfile

# maths and sci libraries.
import numpy as np
import scipy as sp

# for dict and all.
import collections

# for plotting.
import matplotlib.pyplot as plt

# signal processing
from scipy.io                     import wavfile
from scipy                        import stats, signal
from scipy.fftpack                import fft

from scipy.signal                 import lfilter, hamming
from scipy.fftpack.realtransforms import dct
from scikits.talkbox              import segment_axis
from scikits.talkbox.features     import mfcc


# encoding purpose.
from base64 import b64decode

#pandas for csvs.
import pandas as pd

# import stft
import stft


class Classifier:
    def __init__(self):
        # create the audio files path.
        self.audio_file = collections.defaultdict(dict)
        # initialise the audio songs as (genre, path).
        # where path is path of the current .wav audio.
        self.audio_file["rock"]["path"] = r"tomydeepestego.wav"

        self.bark = [100, 200, 300, 400, 510, 630, 770, 920,
                1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150,
                3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]

        self.eq_loudness = np.array(
            [[55, 40, 32, 24, 19, 14, 10, 6, 4, 3, 2,
              2, 0, -2, -5, -4, 0, 5, 10, 14, 25, 35],
             [66, 52, 43, 37, 32, 27, 23, 21, 20, 20, 20,
              20, 19, 16, 13, 13, 18, 22, 25, 30, 40, 50],
             [76, 64, 57, 51, 47, 43, 41, 41, 40, 40, 40,
              39.5, 38, 35, 33, 33, 35, 41, 46, 50, 60, 70],
             [89, 79, 74, 70, 66, 63, 61, 60, 60, 60, 60,
              59, 56, 53, 52, 53, 56, 61, 65, 70, 80, 90],
             [103, 96, 92, 88, 85, 83, 81, 80, 80, 80, 80,
              79, 76, 72, 70, 70, 75, 79, 83, 87, 95, 105],
             [118, 110, 107, 105, 103, 102, 101, 100, 100, 100, 100,
              99, 97, 94, 90, 90, 95, 100, 103, 105, 108, 115]])

        self.loudn_freq = np.array(
            [31.62, 50, 70.7, 100, 141.4, 200, 316.2, 500,
             707.1, 1000, 1414, 1682, 2000, 2515, 3162, 3976,
             5000, 7071, 10000, 11890, 14140, 15500])

    def periodogram(x, win, Fs=None, nfft=1024):
        if Fs == None:
            Fs = 2 * np.pi

        U = np.dot(win.conj().transpose(), win)  # compensates for the power of the window.
        Xx = fft((x * win), nfft)  # verified
        P = Xx * np.conjugate(Xx) / U

        # Compute the 1-sided or 2-sided PSD [Power/freq] or mean-square [Power].
        # Also, compute the corresponding freq vector & freq units.

        # Generate the one-sided spectrum [Power] if so wanted
        if nfft % 2 != 0:
            select = np.arange((nfft + 1) / 2)  # ODD
            P_unscaled = P[select, :]  # Take only [0,pi] or [0,pi)
            P[1:-1] = P[1:-1] * 2  # Only DC is a unique point and doesn't get doubled
        else:
            select = np.arange(nfft / 2 + 1)  # EVEN
            P = P[select]  # Take only [0,pi] or [0,pi) # todo remove?
            P[1:-2] = P[1:-2] * 2

        P /= 2 * np.pi

        return P

    def nextpow2(num):
        n = 2
        i = 1
        while n < num:
            n *= 2
            i += 1
        return i

    def audio_length(self):
        samplerate, wavedata = wavfile.read(self.audio_file["rock"]["path"])
        self.audio_file["rock"]["wavedata"] = wavedata
        self.audio_file["rock"]["samplerate"] = samplerate
        number_of_samples = wavedata.shape[0]
        print samplerate, wavedata.shape[0]
        # song length : number of samples/samplerate.
        print "Audio length: " + str(number_of_samples / samplerate) + " seconds"

    def wavedata_mean(self):
        self.audio_file["rock"]["wavedata"] = np.mean(self.audio_file["rock"]["wavedata"], axis=1)

    @staticmethod
    def zero_crossing_rate(wavedata, block_length, sample_rate):
        # Number of blocks required.
        num_blocks = int(np.ceil(len(wavedata) / block_length))

        # Timestamps for the beginning of the blocks.
        timestamps = (np.arange(0, num_blocks - 1) * (block_length / float(sample_rate)))

        zcr = []
        for i in range(0, num_blocks - 1):
            start = i * block_length
            stop = np.min([(start + block_length - 1), len(wavedata)])
            zc = 0.5 * np.mean(np.abs(np.diff(np.sign(wavedata[start:stop]))))
            zcr.append(zc)

        return np.asarray(zcr), np.asarray(timestamps)

    @staticmethod
    def root_mean_square(wavedata, sample_rate, block_length=1024):
        num_blocks = int(np.ceil(len(wavedata) / block_length))

        timestamps = (np.arange(0, num_blocks - 1) * (block_length / float(sample_rate)))

        rms = []

        for i in range(0, num_blocks - 1):
            start = i * block_length
            stop = np.min([(start + block_length - 1), len(wavedata)])
            rms_seg = np.sqrt(np.mean(wavedata[start:stop]**2))
            rms.append(rms_seg)
        return np.asarray(rms), np.asarray(timestamps)

    """ Spectral features. """
    """ Spectral centroid : centre of gravity.
        Tells the frequency around which most of the signal energy is concentrated.
        Tells how dark/bright the sound is."""

    @staticmethod
    def spectral_centroid(wavedata, sample_rate, window_size=1024):
        magnitude_spectrum = stft.spectrogram(wavedata, window_size)
        timebins, freqbins = np.shape(magnitude_spectrum)
        timestamps = (np.arange(0, timebins - 1) * (timebins / float(sample_rate)))
        spec_centroid = []

        for t in range(timebins - 1):
            power_spectrum = np.abs(magnitude_spectrum[t]) ** 2

            sc_t = np.sum(power_spectrum * np.arange(1, freqbins + 1)) / np.sum(power_spectrum)

            spec_centroid.append(sc_t)

        return np.nan_to_num(np.asarray(spec_centroid)), np.asarray(timestamps)

    """ Sprectral rolloff :
        Nth percentile of the power spectral distribution,
        where N is 85% or 95%. The rolloff point is the frequency below which
        the N% of the magnitude distribution is concentrated.
        Used to distinguish voice speech from unvoiced.
        Unvoiced has a high proportion of energy contained in the high-frequency range of the spectrum.
        - fraction of bins in the power spectrum at which 85%(N%) of the power is at lower frequencies.
    """

    @staticmethod
    def spectral_rolloff(wavedata, window_size, sample_rate, k=0.85):
        # convert into frequency domain using short term fourier transform.
        magnitude_spectrum = stft.spectrogram(wavedata, window_size)
        time_bins, freq_bins = np.shape(magnitude_spectrum)
        power_spectrum = np.abs(magnitude_spectrum) ** 2

        # create timestamps
        timestamps = (np.arange(0, time_bins - 1) * (time_bins / float(sample_rate)))

        spec_rolloff = []

        spectral_sum = np.sum(power_spectrum, axis=1)

        for t in range(time_bins - 1):
            # find frequency-bin indices where cummulative sum of all bins is higher than k-percent of the sum of all bins.
            # minimum index = rolloff.
            spec_rolloff_temp = np.where(np.cumsum(power_spectrum[t, :]) >= k * spectral_sum[t])[0][0]
            spec_rolloff.append(spec_rolloff_temp)

        spec_rolloff = np.asarray(spec_rolloff).astype(float)

        spec_rolloff = (spec_rolloff / freq_bins) * (sample_rate / 2.0)

        return spec_rolloff, np.asarray(timestamps)

    """ Spectral flux : squared diff in frequency distribution of two successive time frames."""
    """ Helps in measuring rate of local change in the spectrum"""

    @staticmethod
    def spectral_flux(wavedata, window_size, sample_rate):
        magnitude_spectrum = stft.spectrogram(wavedata, window_size)
        time_bins, freq_bins = np.shape(magnitude_spectrum)

        # create timestamps.
        timestamps = (np.arange(0, time_bins - 1) * (time_bins / float(sample_rate)))

        spec_flux = np.sqrt(sp.sum(np.diff(np.abs(magnitude_spectrum)) ** 2, axis=1)) / freq_bins

        return spec_flux[1:], np.asarray(timestamps)

    """ MFCC : coefficients that collectively make up an MFC.
        They are derived from a type of cepstral representation of the audio ( a nonlinear spectrum of a spectrum).
        In MFC the frequency bands are equally spaced on the mel scale, which approximates the human auditory systems response more closely than the linear spaced frequency bands.
    """

    @staticmethod
    def MFCC_Cal(input_data):
        # apply pre-filtering.

        # params
        nwin = 256
        nfft = 1024
        fs = 16000
        nceps = 13

        # pre-emphasis factor
        prefac = 0.97

        over = nwin - 160

        filtered_data = lfilter([1., -prefac], 1, input_data)

        # compute the spectrum amplitude by windowing with a hamming window.
        windows = hamming(256, sym=0)
        framed_data = segment_axis(filtered_data, nwin, over) * windows

        magnitude_spectrum = np.abs(fft(framed_data, nfft, axis=-1))

        # filter the signal in the spectral domain with a triangular filter-bank,
        # whose filters are approximately linearly spaced on the mel scale, and have equal bandwidth in the mel scale/

        lowfreq = 133.33
        linsc = 200 / 3
        logsc = 1.0711703
        fs = 44100

        nlinfilt = 13
        nlogfilt = 27

        # total filters
        nfilt = nlinfilt + nlogfilt

        # Compute the filter bank.
        # compute start/middle/end points of the triangular filters in spectral.

        # domain.
        freqs = np.zeros(nfilt + 2)
        freqs[:nlinfilt] = lowfreq + np.arange(nlinfilt) * linsc

        freqs[nlinfilt:] = freqs[nlinfilt - 1] * logsc ** np.arange(1, nlogfilt + 3)

        heights = 2. / (freqs[2:] - freqs[0:-2])

        # compute filterbank coeff (in fft domain, in bins)
        filterbank = np.zeros((nfilt, nfft))

        # FFT bins (in Hz)
        nfreqs = np.arange(nfft) / (1. * nfft) * fs

        for i in range(nfilt):
            low = freqs[i]
            cen = freqs[i + 1]
            hi = freqs[i + 2]

            lid = np.arange(np.floor(low * nfft / fs) + 1,
                            np.floor(cen * nfft / fs) + 1, dtype=np.int)

            rid = np.arange(np.floor(cen * nfft / fs) + 1,
                            np.floor(hi * nfft / fs) + 1, dtype=np.int)

            lslope = heights[i] / (cen - low)
            rslope = heights[i] / (hi - cen)

            filterbank[i][lid] = lslope * (nfreqs[lid] - low)
            filterbank[i][rid] = rslope * (hi - nfreqs[rid])

            # filter the spectrum through the triangle filterbank.

            mspec = np.log10(np.dot(magnitude_spectrum, filterbank.T))

            # Use the DCT to 'compress' the coefficients (spectrum -> cepstrum domain)
            MFCCs = dct(mspec, type=2, norm='ortho', axis=-1)[:, :nceps]

            return MFCCs, mspec, magnitude_spectrum

    """ Rhythm patterns :
        Describe modulation amplitudes for a range of modulation frequencies on "critical bands" of the human auditory range.
        Two step process.
        1) The specific loudness sensation in different frequency bands is computed, grouping the resulting frequency bands to psycho-acoustically motivated critical-bands.
            -> This results in human loudness sensation. (Sonogram)
        2) The spectrum is transformed into a time-invariant representation based on modulation frequency, which is achieved by applying DFT. resulting in amplitude modulations of the loudness in individual critical bands.
        Amplitude modulations having different effects on human hearing sensation depending on their frequency, the most significant of which, referred to as fluctuation strength, is most intense is 4Hz and decreasing towards 15Hz.

    """

    @staticmethod
    def spectrograph(self, wave_data, sample_rate):
        # parameters.
        skip_leading_fadeout = 1
        step_width = 3

        segment_size = 2 ** 18
        fft_window_size = 1024  # for 44100 Hz

        # required precalculations.

        duration = wave_data.shape[0] / sample_rate
        # calculate frequency values on y-axis (for bark scale calculation)
        freq_axis = float(sample_rate) / fft_window_size * np.arange(0, (fft_window_size / 2) + 1)
        mod_freq_res = 1 / (float(segment_size) / sample_rate)
        mod_freq_axis = mod_freq_res * np.arange(257)  # modulation frequencies along.
        # x-axis from index 1 to 257
        fluct_curve = 1 / (mod_freq_axis / 4 + 4 / mod_freq_axis)

        skip_seg = skip_leading_fadeout
        seg_pos = np.array([1, segment_size])

        if (skip_leading_fadeout > 0) or (step_width > 1):
            if duration < 45:
                step_width = 1
                skip_seg = 0
            else:
                seg_pos += segment_size * skip_seg

        wavsegment = wave_data[seg_pos[0] - 1: seg_pos[1]]

        wavsegment = 0.0875 * wavsegment * (2 ** 15)

        n_iter = wavsegment.shape[0] / fft_window_size * 2 - 1
        w = np.hanning(fft_window_size)

        spectrograph = np.zeros((fft_window_size / 2 + 1), n_iter)

        idx = np.arange(fft_window_size)

        # stepping through the wave segment, building spectrum for each window.
        for i in range(n_iter):
            spectrograph[:, i] = self.periodogram(x=wavsegment[idx], win=w)
            idx += fft_window_size / 2
        Pxx = spectrograph
        return Pxx

    def bark_scale(self, Pxx, sample_rate, fft_window_size=1024):
        # calculate bark-filterbank
        loudn_bark = np.zeros((self.eq_loudness.shape[0], len(self.bark)))

        i = 0
        j = 0

        for bsi in self.bark:

            while j < len(self.loudn_freq) and bsi > self.loudn_freq[i]:
                j += 1
            j -= 1

            if np.where(self.loudn_freq == bsi)[0].size != 0:
                loudn_bark[:, i] = self.eq_loudness[:, np.where(self.loudn_freq == bsi)][:, 0, 0]
            else:
                w1 = 1 / np.abs(self.loudn_freq[j] - bsi)
                w2 = 1 / np.abs(self.loudn_freq[j + 1] - bsi)
                loudn_bark[:, i] = (self.eq_loudness[:, j] * w1 + self.eq_loudness[:, j + 1] * w2) / (w1 + w2)
            i += 1

        # Apply bark-filter
        matrix = np.zeros((len(self.bark), Pxx.shape[1]))

        barks = self.bark[:]
        barks.insert(0, 0)
        freq_axis = float(sample_rate) / fft_window_size * np.arange(0, (fft_window_size / 2) + 1)
        for i in range(len(barks) - 1):
            matrix[i] = np.sum(Pxx[((freq_axis >= barks[i]) & (freq_axis < barks[i + 1]))], axis=0)

        return matrix

    """ Spectral Masking :
        occlusion of a quiet sound by a louder soung when both sounds are present
        simultaneously and have similar frequencies.
        Masking can be categorized as :
        1) simultaneous masking : two sounds active simultaneously.
        2) post-masking : a sound closely following it (100-200ms).
        3) pre-masking : a sound preceding it.

    """

    def spectral_masking(self, matrix):
        n_bark_bands = len(self.bark)

        Const_spread = np.zeros((n_bark_bands, n_bark_bands))

        for i in range(n_bark_bands):
            Const_spread[i, :] = 10 ** ((15.81 + 7.5 * ((i - np.arange(n_bark_bands)) + 0.474) - 17.5 * (
            1 + ((i - np.arange(n_bark_bands)) + 0.474) ** 2) ** 0.5) / 10)
        spread = Const_spread[:matrix.shape[0], :]
        matrix = np.dot(spread, matrix)

        # map to decibel scale
        matrix[np.where(matrix < 1)] = 1
        matrix = 10 * np.log10(matrix)

    """ Phon Scale : Equal loudness curves (Phon)

        relationship between sound pressure level in decibel and perceived hearing sensation.
        each loudness contours for 3,20, 40, 60, 80, 100 phon"""

    def phon_mapping(self, matrix):
        phon = [3, 20, 40, 60, 80, 100, 101]
        n_bands = matrix.shape[0]
        t = matrix.shape[1]

        # DB to Phon bark-scale-limit table!

        # introducing 1 level more with level(1) being infinite to avoid (levels-1) producing errors like division by 0.

        table_dim = n_bands
        cbv = np.concatenate((np.tile(np.inf, (table_dim, 1)), self.loudn_freq[:, 0:n_bands].transpose()), 1)

        phons = phon[:]
        phons.insert(0, 0)
        phons = np.asarray(phons)

        # init lowest level = 2
        levels = np.tile(2, (n_bands, t))

        for lev in range(1, 6):
            db_thislev = np.tile(np.asarray([cbv[:, lev]]).transpose(), (1, t))
            levels[np.where(matrix > db_thislev)] = lev + 2
        # the matrix 'levels' stores the correct Phon level for each datapoint
        cbv_ind_hi = np.ravel_multi_index(dims=(table_dim, 7), multi_index=np.array(
            [np.tile(np.array([range(0, table_dim)]).transpose(), (1, t)), levels - 1]), order='F')

        cbv_ind_lo = np.ravel_multi_index(dims=(table_dim, 7), multi_index=np.array(
            [np.tile(np.array([range(0, table_dim)]).transpose(), (1, t), levels - 2)]), order='F')

        # interpolation factor % OPT : pre-calc diff
        ifac = (matrix[:, 0:t] - cbv.transpose().ravel()[cbv_ind_lo]) / (
        cbv.transpose().ravel()[cbv_ind_hi] - cbv.transpose().ravel()[cbv_ind_lo])

        # keeps the upper phon value
        ifac[np.where(levels == 2)] = 1
        ifac[np.where(levels == 8)] = 1

        matrix[:, 0:t] = phons.transpose().ravel()[levels - 2] + (
        ifac * (phons.transpose().ravel()[levels - 1] - phons.transpose().ravel()[levels - 2]))

        return matrix

    # Transform to Sone scale.

    # Sone : 1, 2, 4, 8, 16, 32, 64. Phon : 40, 50, 60, 70, 80, 90, 100

    @staticmethod
    def sone_scale(matrix):
        idx = np.where(matrix >= 40)
        not_idx = np.where(matrix < 40)

        matrix[idx] = 2 ** ((matrix[idx] - 40) / 10)
        matrix[not_idx] = (matrix[not_idx] / 40) ** 2.642

        return matrix

    def calc_statistical_features(matrix):
        result = np.zeros((matrix.shape[0], 7))
        result[:, 0] = np.mean(matrix, axis=1)
        result[:, 1] = np.var(matrix, axis=1)
        result[:, 2] = sp.stats.skew(matrix, axis=1)
        result[:, 3] = sp.stats.kurtosis(matrix, axis=1)
        result[:, 4] = np.median(matrix, axis=1)
        result[:, 5] = np.min(matrix, axis=1)
        result[:, 6] = np.max(matrix, axis=1)

        result = np.nan_to_num(result)

        return result

    """ Rhythm patterns : calculate fluctuation patterns from scaled spectrum. """

    def rhythm_patterns(self, matrix):
        fft_size = 2 ** (self.nextpow2(matrix.shape[1]))

        rhythm_pattrn = np.zeros((matrix.shape[0], fft_size), dtype=np.complex128)

        # calculate fourier transform for each bark scale
        for b in range(0, matrix.shape[0]):
            rhythm_pattrn[b, :] = fft(matrix[b, :], fft_size)

        # normalize results
        rhythm_pattrn /= 256

        # take first 60 values of fft result including DC component.
        feature_part_xaxis_rp = range(0, 60)

        rp = np.abs(rhythm_pattrn[:, feature_part_xaxis_rp])

        # histogram
        rh = np.sum(np.abs(rhythm_pattrn[:, feature_part_xaxis_rp]), axis=0)

        return rp, rh

    # Modulation variation descriptors : measures variation over the critical frequency bands for a specific modulation frequency derived from rhythm patterns.

    def mvd(self, rp):
        return self.calc_statistical_features(rp.transpose())
