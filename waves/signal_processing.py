"""

Code for processing waveforms and extracting features. 

"""
import numpy as np
from librosa import zero_crossings as zc
from scipy.integrate import simps

def get_signal_start_end(waveform, threshold=0.1):
    """
    
    Gets indices of the signal start and end defined by a floating threshold.

    Parameters
    ----------
    waveform : array-like
        Voltage time series of the waveform
    threshold : float
        Floating threshold that defines signal start and end

    Returns
    -------
    start_index : int
        Array index of signal start 
    end_index : int
        Array index of signal signal end 

    """
    if threshold<0 or threshold>1:
        raise ValueError('Threshold must be between 0 and 1')

    max_amp = np.max(waveform)
    start_index, end_index = \
        np.nonzero(waveform > threshold*max_amp)[0][[0, -1]]
        
    return start_index, end_index


def get_rise_time(waveform):
    """
    
    Get rise time of signal, which is time from low threshold to peak.

    Note: Current implementation will take MAX amplitude, so even if the max
    amplitude appears later in the waveform, which will result in a large
    rise time, that is somewhat unrealistic when you look at the waveform.
    
    Parameters
    ----------
    waveform : array-like
        Voltage time series of the waveform.

    Returns
    -------
    rise_time : float
        Signal rise time.

    """
    max_amp = np.max(waveform)
    peak_time = np.argmax(waveform)/10 # NOTE: time of signal peak in us 
    imin, imax = get_signal_start_end(waveform)
    start_time = imin/10  # Note: converts index location to a start time (us)
    end_time = imax/10    # Note: converts index location to an end time (us)
    rise_time = peak_time - start_time
   
    return rise_time

def get_duration(waveform):
    """
    
    Get duration of signal as determined by set thresholds.

    Parameters
    ----------
    waveform : array-like
        Voltage time series of the waveform.
       
    Returns
    -------
    duration : float
        Signal duration.

    """
    imin, imax = get_signal_start_end(waveform)
    start_time = imin/10  # Note: converts index location to a start time (us)
    end_time = imax/10    # Note: converts index location to an end time (us)    
    duration = end_time-start_time
    
    return duration


def get_peak_freq(waveform, dt=10**-7, low=None, high=None):
    """
    
    Gets peak frequency of signal.
    
    Parameters
    ----------
    waveform : array-like
        Voltage time series of the waveform.
    dt : float
        Time between samples (s) (also inverse of sampling rate).
    low : int, optional
        Low pass filter threshold. The default is None.
    high : int, optional
        High pass filter threshold. The default is None.
        
    Returns
    -------
    peak_freq : float
        Frequency of maximum FFT power in Hz
        
    """
    w, z = fft(dt, waveform, low_pass=low, high_pass=high)
    max_index = np.argmax(z)
    peak_freq = w[max_index]

    return peak_freq

def get_freq_centroid(waveform, dt=10**-7, low=None, high=None):
    """
    
    Get frequency centroid of signal. By doing fft first then computing.

    Parameters
    ----------
    waveform : array-like
        Voltage time series of the waveform.
    dt : TYPE, optional
        Time between samples (s) (also inverse of sampling rate). The default 
        is 10**-7.
    low : int, optional
        Low pass filter threshold. The default is None.
    high : int, optional
        High pass filter threshold. The default is None.

    Returns
    -------
    freq_centroid : float
        Frequency centroid of signal.

    """
    w, z = fft(dt, waveform, low_pass=low, high_pass=high)
    freq_centroid = np.sum(z*w)/np.sum(z)
    
    return freq_centroid

def fft(dt, y, low_pass=None, high_pass=None):
    '''
    Performs FFT

    dt (float): Sampling rate
    y (array-like): Voltage time-series
    low_pass (float): Optional variable for a low band pass filter in the same units of w
    high_pass (float): Optional variable for a high band pass filter in the same units of w

    returns:
    w (array-like): Frequency
    z (array-liike): Power
    '''
    z = np.abs(np.fft.fft(y))
    w = np.fft.fftfreq(len(z), dt)
    w = w[np.where(w>=0)] # NOTE: Gets positive frequencies from spectrum
    z = z[np.where(w>=0)] # NOTE: Gets positive frequencies from spectrum

    if low_pass is not None:
        z = z[np.where(w > low_pass)]
        w = w[np.where(w > low_pass)]
    if high_pass is not None:
        z = z[np.where(w < high_pass)]
        w = w[np.where(w < high_pass)]

    return w, z

def wave2vec(dt, waveform, lower, upper, dims, FFT_units, upsample=10001):
    '''
    dt: time spacing in seconds
    waveform (array-like): 1D array of waveform
    lower: lower bound on FFT in Hz
    upper: upper bound on FFT in Hz
    dims (int): dimension of vector

    This is a helper function which takes a single waveform and casts it as a vector.
    Upsampling is nessecary to ensure the whole FFT is integrated.
    '''
    feature_vector = []
    w, z = fft(dt, waveform, low_pass=lower, high_pass= upper) # NOTE: verified works
    w = w/FFT_units

    upsampled_w = np.linspace(lower, upper, upsample)/FFT_units # NOTE: 10000 is a good number of samples
    upsampled_z = np.interp(upsampled_w, w, z)
    dw=upsampled_w[1]-upsampled_w[0]

    interval_width = int(len(upsampled_z)/dims) # NOTE: range of index that is integrated over
    true_bounds = []

    for j in range(dims):
        subinterval = upsampled_z[j*interval_width: (j+1)*interval_width]
        sub_int_mass = simps(subinterval) # NOTE: area under sub interval
        feature_vector.append(sub_int_mass) # single waveform as a vector, unnormalized

        true_bounds.append(lower/FFT_units+j*interval_width*dw)

    # NOTE: Calculate bounds and frequency spacing
    true_upper_bound = (j+1)*interval_width*dw+lower/FFT_units # NOTE: true upper bound in kHz, not exact due to numerical considerations
    spacing = interval_width*dw # NOTE: kHz

    if (upper/FFT_units-true_upper_bound)/(upper/FFT_units-lower/FFT_units)>.01:
        raise ValueError('Increase upsampling number')
        return None
    return feature_vector/np.sum(feature_vector), np.array(true_bounds), spacing


def extract_Sause_vect(waveform=[], dt=10**-7, threshold=.1,low_pass=None,high_pass=None):
    '''
    waveform (array-like): Voltage time series describing the waveform
    dt(float): Spacing between time samples (s)
    energy (float): energy of the waveform as calculated by the AE software
    low_pass (float): lower bound on bandpass (Hz)
    high_pass (float): upper bound on bandpass (Hz)

    return:
    vect (array-like): feature vector extracted from a waveform according to Moevus2008
    '''
    if waveform == []:
        raise ValueError('An input is missing')

    imin, imax = get_signal_start_end(waveform)

    risingpart = waveform[imin:np.argmax(waveform)] #Note: grabs the rising portion of the waveform
    fallingpart = waveform[np.argmax(waveform):imax] #Note: grabs the falling portion of the waveform

    average_freq= get_average_freq(waveform, dt=dt, threshold=threshold)
    #rise_freq = get_average_freq(risingpart, dt=dt, threshold=threshold)
    #reverb_freq = get_average_freq(fallingpart, dt=dt, threshold=threshold)
    rise_freq=0
    reverb_freq=0
    
    # Nick edited to add low pass and high pass filters
    w, z = fft(dt, waveform[imin:imax],low_pass=low_pass,high_pass=high_pass)
    freq_centroid = get_freq_centroid(w,z)

    peak_freq = get_peak_freq(waveform[imin:imax])
    wpf = np.sqrt(freq_centroid*peak_freq)

    pp1 = get_partial_pow(waveform[imin:imax], lower_bound=0, upper_bound=150*10**3)
    pp2 = get_partial_pow(waveform[imin:imax], lower_bound=150*10**3, upper_bound=300*10**3)
    pp3 = get_partial_pow(waveform[imin:imax], lower_bound=300*10**3, upper_bound=450*10**3)
    pp4 = get_partial_pow(waveform[imin:imax], lower_bound=450*10**3, upper_bound=600*10**3)
    pp5 = get_partial_pow(waveform[imin:imax], lower_bound=600*10**3, upper_bound=900*10**3)
    pp6 = get_partial_pow(waveform[imin:imax], lower_bound=900*10**3, upper_bound=1200*10**3)

    #feature_vector = [pp1, pp2]
    #feature_vector = [average_freq, reverb_freq, freq_centroid, rise_freq, peak_freq, wpf, pp1, pp2, pp3]

    feature_vector = [average_freq, reverb_freq, rise_freq, peak_freq, freq_centroid, wpf, pp1, pp2, pp3, pp4, pp5, pp6]

    return feature_vector

def get_counts(waveform, threshold=0.1):
    '''
    Gets number of counts in AE signal, equiv to number of zero crossings

    waveform (array-like): voltage time-series of the waveform
    threshold (float): Floating threshold that defines the start and end of signal

    return
    counts (int): Average frequency of signal in Hz
    '''
    imin, imax = get_signal_start_end(waveform)
    cut_signal = waveform[imin:imax]
    num_zero_crossings = len(np.nonzero(zc(cut_signal)))
    return num_zero_crossings

def get_average_freq(waveform, dt=10**-7, threshold=0.1):
    '''
    Gets average frequency defined as the number of zero crossings
    divided by the length of the signal according to Moevus2008

    waveform (array-like): voltage time-series of the waveform
    dt (float): spacing between time samples (s)

    threshold (float): Floating threshold that defines the start and end of signal

    return
    average_frequency (float): Average frequency of signal in Hz
    '''
    imin, imax = get_signal_start_end(waveform, threshold=threshold)
    cut_signal = waveform[imin:imax]
    num_zero_crossings = len(np.nonzero(zc(cut_signal)))

    return num_zero_crossings/(len(cut_signal)*dt)

def get_partial_pow(waveform=[], lower_bound=None, upper_bound=None, dt = 10**-7):
    '''
    Gets partial power of signal from waveform from f_0 to f_1

    :param waveform: (array-like) Voltage time series of the waveform
    :param lower_bound: (float) Lower bound of partial power in Hz
    :param upper_bound: (float) Upper bound of partial power in Hz
    :param dt: (float) Time between samples (s) (also inverse of sampling rate)

    :return pow: (float) Partial power
    '''
    if lower_bound is None or upper_bound is None:
        raise ValueError('Partial power bounds not defined')

    w, z = fft(dt, waveform)
    total_pow = np.sum(z)

    pow = np.sum(z[np.where((w>=lower_bound) & (w<upper_bound))])
    partial_pow = pow/total_pow

    return partial_pow