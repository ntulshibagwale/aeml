"""
ae_functions

Utility functions for various AE tasks / plotting. Many functions are copied
from Caelin Muir Github, minor edits to fit into my scripts.

Nick Tulshibagwale

Updated: 2022-05-04

"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
from .ae_measure2 import *
#from ae_measure2 import *
import librosa
import cv2
from sklearn.decomposition import PCA
    
def create_figure(suptitle, columns, rows, width=20, height=10,
                  suptitle_font_size=24, default_font_size=10,
                  title_font_size=12, axes_font_size=12, tick_font_size=10,
                  legend_font_size=10, w_space=0.25, h_space=0.25):
    """
    
    Create a gridspec figure, so more flexibility with subplots.

    Parameters
    ----------
    suptitle : string
        Master title.
    columns : int
        Subplot columns.
    rows : int
        Subplot rows.
    width : int, optional
        Figure width. The default is 20.
    height : int, optional
        Figure height. The default is 10.
    suptitle_font_size : int, optional
        Master title size. The default is 24.
    default_font_size : TYPE, optional
        The default is 10.
    title_font_size : int, optional
        Individual subplot title size. The default is 12.
    axes_font_size : int, optional
        Font size for x and y labels. The default is 12.
    tick_font_size : int, optional
        The default is 10.
    legend_font_size : int, optional
        The default is 10.
    w_space : float, optional
        Distance between subplots horizontally. The default is 0.25.
    h_space : TYPE, optional
        Distance between subplots vertically. The default is 0.25.

    Returns
    -------
    fig : Matplotlib object
        The figure handle.
    spec2 : Matplotlib object
        Used for adding custom sized subplots ; fig.add_subplot(spec2[0,0]).

    """
    fig = plt.figure(figsize=(width,height))
    
    # Create subplot grid -> used for subplots
    spec2 = gridspec.GridSpec(ncols = columns, nrows = rows, figure = fig,
                              wspace = w_space,hspace = h_space)
    
    # Master Figure Title
    fig.suptitle(suptitle,fontsize=suptitle_font_size)
    
    # General plotting defaults    
    plt.rc('font', size=default_font_size)     # controls default text size
    plt.rc('axes', titlesize=title_font_size)  # fontsize of the title
    plt.rc('axes', labelsize=axes_font_size)   # fontsize of the x and y labels
    plt.rc('xtick', labelsize=tick_font_size)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=tick_font_size)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=legend_font_size)# fontsize of the legend
    
    return fig, spec2

def flatten(t): # flattens out a list of lists (CHECK)
    return [item for sublist in t for item in sublist]

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

def get_folder_pickle_files(path):
    """
    
    Given directory path, will return all pickle files in directory.

    Parameters
    ----------
    path : string
        Path to directory of interest.

    Returns
    -------
    files : list
        List of pickle files
    
    """
    files = [pickle_file for pickle_file in os.listdir(path) \
             if pickle_file.endswith(".pkl")]
    
    return files

def visualize_clustering_results_in_2D_PCA(channels_feature_vec,
                                           channels_labels,title,names):
    """
    
    For visualization purposes, project feature vector for a given channel
    down to 2 principal components and label red and blue to correspond
    to the spectral clustering results. Red should be fiber and blue should 
    be matrix (There are edge cases where it gets accidentally swapped due 
    to the # of points clustered). Cycle through various principal 
    components.

    Parameters
    ----------
    channels_feature_vec : array-like
        For 4 channels, feature vectors, list.
    channels_labels : array-like
        For 4 channels, labels, list.
    title : string
        Title of plot.
    names : list of strings
        For 4 channels, channel names.

    Returns
    -------
    None.

    """
    for x_axis in range(2):
        for y_axis in range(x_axis+1,2):
            fig,spec2 = create_figure(f'PC {x_axis+1} vs. PC {y_axis+1} |' + title, \
                                   4,1,width=20,w_space=0.6, h_space=0.5,height=5,\
                                      axes_font_size=15,tick_font_size=14,suptitle_font_size=20,\
                                     legend_font_size=14) 
            for idx, feature_vec in enumerate(channels_feature_vec):
                pca = PCA(n_components=2)
                ax = fig.add_subplot(spec2[0,idx])
                X_reduced = pca.fit_transform(feature_vec) 
                X_reduced_matrix,X_reduced_fiber= split_array(X_reduced,channels_labels[idx])               
                
                ax.scatter(X_reduced_matrix[:,x_axis],X_reduced_matrix[:,y_axis],color='blue')
                ax.scatter(X_reduced_fiber[:,x_axis],X_reduced_fiber[:,y_axis],color='red')
                
                ax.set_xlabel(f'PC {x_axis+1}')
                ax.set_xlim([-0.15, 0.20])
                ax.set_ylim([-0.15, 0.15])
                ax.set_ylabel(f'PC {y_axis+1}')
                ax.set_title(names[idx])

    plt.show()
    
def calc_avg_std_diff_for_mismatched_events(channels_feature_vec_1,
                                            labels_1,channels_feature_vec_2,
                                            labels_2,names,title):
    """
    
    For two different channels feature vectors, compare each channel and see 
    if there are any mismatches. Record the mismatches in a vector, then avg
    and std to see the difference between the feature vectors.

    Parameters
    ----------
    channels_feature_vec_1 : TYPE
        DESCRIPTION.
    labels_1 : TYPE
        DESCRIPTION.
    channels_feature_vec_2 : TYPE
        DESCRIPTION.
    labels_2 : TYPE
        DESCRIPTION.
    names : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.

    Returns
    -------
    fv_diff_avg_all_channels : TYPE
        DESCRIPTION.
    fv_diff_std_all_channels : TYPE
        DESCRIPTION.

    """
    fv_diff_for_mismatched_labels_all_channels = [] 

    for idx,channel in enumerate(channels_feature_vec):
        
        # Obtain labels for single channel
        label_1 = labels_1[idx] # original
        label_2 = labels_2[idx] # reconstructed
        
        # Obtain feature vectors for single channel
        channel_feature_vec_1 = channels_feature_vec_1[idx]
        channel_feature_vec_2 = channels_feature_vec_2[idx]
        
        fv_diff_for_mismatched_labels = []
        for i, event in enumerate(channel_feature_vec_1): # loop through all events
            # If this event was clustered differently between the two channels
            # append the DIFFERENCE in the two partial power vectors to the list
            if label_1[i] != label_2[i]: # mismatch
                # calculate difference in feature vectors
                fv_difference = np.array(np.abs(channel_feature_vec_1 - channel_feature_vec_2))
                # append difference
                fv_diff_for_mismatched_labels.append(fv_difference[i])   

        fv_diff_for_mismatched_labels_all_channels.append(fv_diff_for_mismatched_labels)
    
    fig,spec2 = create_figure(f'Average Difference in Cluster Results | {title} ', \
                   1,len(labels_1),width=15,w_space=0.5, h_space=0.5,height=10,\
                      axes_font_size=10,tick_font_size=10,suptitle_font_size=20,\
                     legend_font_size=14) 
        
    fv_diff_avg_all_channels = []
    fv_diff_std_all_channels = []
    for idx, fv_diff in enumerate(fv_diff_for_mismatched_labels_all_channels):
        fv_diff = np.array(fv_diff) # convert to numpy array
        fv_diff_avg = np.average(fv_diff,axis=0)
        fv_diff_std = np.std(fv_diff,axis=0)

        fv_diff_avg_all_channels.append(fv_diff_avg)
        fv_diff_std_all_channels.append(fv_diff_std)
        print(f"Channel {names[idx]} (v{idx}) Mismatch Events = {len(fv_diff)}")    
        
        # Print the avg and std for the given channel
        ax = fig.add_subplot(spec2[idx,0]) 
        ax.set_title(f'{names[idx]} | v{idx}')
        print(fv_diff_avg.shape)
        plot_feature_vec(ax,fv_diff_avg,0.15,color='orange')
        ax.set_ylim([0,0.04])
        x=np.arange(1,27)
        plt.errorbar(x,fv_diff_avg,yerr=fv_diff_std,fmt='none',elinewidth=1,capsize=1)
        ax.text(0,.8,f'# Samples: {len(fv_diff)}',horizontalalignment='left', # num mismatch events 
                                        verticalalignment='bottom',fontsize=13,
                                        transform = ax.transAxes, weight='bold')
        
    return fv_diff_avg_all_channels, fv_diff_std_all_channels
    
def plot_signal(ax,signal,dt,sig_len):
    """
    
    Plot raw event signal waveform. 
    
    """
    duration = sig_len*dt*10**6 # convert to us
    time = np.linspace(0,duration,sig_len) # discretization of signal time
    if type(signal) is list:
        for idx,sig in enumerate(signal):
            ax.plot(time,sig)
    else:
        ax.plot(time,signal)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (us)')
    ax.set_xlim([0,duration])

    return ax

def label_match_rate(labels,labels_reduced):
    """
    
    Provide percentage match between two labels sets.
    
    """
    channels_match_rate = []
    for idx,label in enumerate(labels):
        label_reduced = labels_reduced[idx]
        
        counter = 0
        for i in range(len(label)):
            if label[i] == label_reduced[i]:
                counter = counter + 1

        percent_match = counter / len(label) 
        
        if percent_match < 0.5: 
            percent_match = 1-percent_match
            
        channels_match_rate.append(percent_match)
        
    print("\n")
    print(f"{names[0]} label match percentage with 26D: {channels_match_rate[0]}")
    print(f"{names[1]} label match percentage with 26D: {channels_match_rate[1]}")
    print(f"{names[2]} label match percentage with 26D: {channels_match_rate[2]}")
    print(f"{names[3]} label match percentage with 26D: {channels_match_rate[3]}")

    return channels_match_rate   

def plot_fft(
        ax,
        w,
        z,
        freq_bounds,
        low_pass,
        high_pass,
        fft_units,
        label_bins=True
        ):  
    """
    
    Plot normalized fft spectrum. Bins are placed according to feature vector 
    size. 
    
    """
    z = z / (max(z))  # normalize by maximum value (different for each event)
    ax.plot(w,z)    
    ax.set_ylim([0, 1.0])
    ax.set_ylabel('Normalized Amplitude')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_xlim([low_pass/fft_units,high_pass/fft_units])
    
    # Plot frequency intervals for partial power bins
    # The feature vector is calculated by taking the area under the fft curve
    # for each of these bins.
    ax.vlines(freq_bounds,0,1,color = 'purple',linestyles='--',
                linewidth = 1)
    
    # Write bin labels 
    if label_bins: # false when doing multiple axes to keep plot less busy
        spacing = freq_bounds[1]-freq_bounds[0] # interval size
        for idx, left_freq_bound in enumerate(freq_bounds):
            if np.mod(idx,2) == 1: # prevents labeling EACH bin, only every 2
                bin_number = str(idx+1)
                # annotate puts the text with an 'arrow' pointing to bin
                ax.annotate(bin_number,xy=(left_freq_bound+spacing/2,1),
                            xytext=(left_freq_bound+spacing/2,1.1),
                            arrowprops=dict(arrowstyle='-',
                            connectionstyle="arc3", color='purple'),
                            horizontalalignment="center", color='purple') 
        # Axes title
        ax.text(freq_bounds[0]-spacing*2,1.1,'Bin#',color='purple')
    
    return ax

def plot_stress_and_cumul_norm_energy(
        ax,
        time,
        stress,
        channel_labels,
        cumul_norm_energy=None,
        ax2=None,
        ev_idx=None
        ):    
    """
    
    Plot the overall stress vs time. If a second axes is passed with the cumul
    norm energy, then this will be plotted on the same x axis with a secondary
    axis.
    
    """
    
    # If the normalized energy is a list, then we're plotting all 4 channels
    # For 4 channels, don't distinguish between matrix and fiber since differs.
    # For single channel, we can differ between matrix or fiber.
    if type(cumul_norm_energy) is not list: 
        # Separate points into fiber and matrix based on cluster label
        stress_m = stress[np.where(channel_labels==0)] #
        time_m= time[np.where(channel_labels==0)]
        stress_f = stress[np.where(channel_labels==1)]
        time_f= time[np.where(channel_labels==1)]
    
        # Stress vs time plots on primary axis, distinguishing between f and m
        ax.scatter(time_m,stress_m,color='blue')   # plot matrix events
        ax.scatter(time_f,stress_f,color='red')    # plot fiber events
    else:
        ax.scatter(time,stress,color='black')
    
    ax.set_ylim([0,max(stress)+20])
    ax.set_ylabel('Stress')
    ax.set_xlabel('Time (sec)')
    
    # If an event index was passed an arg -> plot a line to indicate 
    # event of interest, and distinguish by color whether matrix or fiber
    if ev_idx is not None:
        if type(cumul_norm_energy) is not list:
            if channel_labels[ev_idx] == 0:
                ax.vlines(time[ev_idx],0,max(stress)+20,color='blue')
            if channel_labels[ev_idx] == 1:
                ax.vlines(time[ev_idx],0,max(stress)+20,color='red')
        else: # if its 4 channel we're not differentiating
            ax.vlines(time[ev_idx],0,max(stress)+20,color='black')

    # If a second axes was passed, we're also plotting cumul_norm_energy
    if ax2 is not None:
        # Cumulative Normalized AE Energy vs time plots on secondary axis
        names = ['S9225_1','S9225_2','B1025_1','B1025_2']
        if type(cumul_norm_energy) is not list:
            ax2.plot(time,cumul_norm_energy) 
            ax2.set_ylim([0,max(cumul_norm_energy)+0.01])
        else:
            ax2.plot(time,cumul_norm_energy[0],label=names[0])
            ax2.plot(time,cumul_norm_energy[1],label=names[1])
            ax2.plot(time,cumul_norm_energy[2],label=names[2])
            ax2.plot(time,cumul_norm_energy[3],label=names[3])
            ax2.set_ylim([0,np.amax(cumul_norm_energy)+0.01])
            ax2.legend()
            
        ax2.set_ylabel('Cumulative Norm AE Energy')
        ax2.set_xlabel('Time (sec)')
        
        # If an event index was passed an arg -> plot a line to indicate 
        # event of interest, and distinguish by color whether matrix or fiber
        if ev_idx is not None:
            if type(cumul_norm_energy) is not list:     
                if channel_labels[ev_idx] == 0:
                    ax2.vlines(time[ev_idx],0,500,color='blue')
                if channel_labels[ev_idx] == 1:
                    ax2.vlines(time[ev_idx],0,500,color='red')
            else: # if its 4 channel we're not differentiating
                ax.vlines(time[ev_idx],0,500,color='black')

        return ax, ax2
    
    else: # If only one axes was passed
        
        return ax


def split_array(
    array,
    labels
    ):
    """
    
    Split an array into two separate arrays based on fiber and matrix
    clustering results. The labels and array must have same size.
    
    """
    array_1 = []
    array_2 = []
    for idx, ev in enumerate(array):
        if labels[idx] == 0: # matrix
            array_1.append(array[idx,:])
        if labels[idx] == 1: # fiber
            array_2.append(array[idx,:])
            
    array_1 = np.array(array_1)
    array_2 = np.array(array_2)
    
    return array_1, array_2

    
def plot_feature_vector_in_time(
        ax,
        channel_feature_vector,
        fft_units,
        high_pass,
        low_pass,
        ev,
        channel_labels,
        v_max=0.25,
        v_min=0,
        ev_idx=None,
        annotate=True,
        ):
    """
    
    Plot feature vector, which is calculated by binning the frequency spectra 
    of a signal into intervals computing area with respect to total area, as it
    changes through time (plot each events feature vector). The plotting 
    function in use is matplotlib 'pcolormesh'. 
    
    From the pcolormesh doc:
        
    (X[i+1, j], Y[i+1, j])       (X[i+1, j+1], Y[i+1, j+1])
                          +-----+
                          |     |
                          +-----+
        (X[i, j], Y[i, j])       (X[i, j+1], Y[i, j+1])
    
    So for instance:
        x = np.array([[1,2,3],[4,5,6],[7,8,9]])
        [1 2 3]               7     8     9
        [4 5 6]               4     5     6 
        [7 8 9]               1     2     3
        
    The right arrangement of numbers is how pcolormesh arranges by default.

    """
    y_bins = channel_feature_vector.shape[0] # frequency / partial power bins
    x_bins = channel_feature_vector.shape[1] # time bins
    
    # Plot 2D array into heat map
    im=ax.pcolormesh(channel_feature_vector,vmin=v_min,vmax=v_max) 
    # vmax would be the numerical value associated with the brightest color
    # the max value is calculated in the max_feature_value function
    
    plt.colorbar(im, label = 'Partial Power Values')
    
    # Y labeling to label the bin rather than the line between
    ax.set_ylabel('Bin #',color = 'purple')
    ax.tick_params(axis='y',colors='purple')
    start, end = ax.get_ylim()
    ax.set_yticks(np.arange(start+1,end,2)+0.5) # place ticks [1.5,3.5...25.5]
    bin_labels = np.arange(start+1,end+1,dtype=int) # [1,2,3,4.....26]
    bin_labels = bin_labels[1::2] # [2,4,6,....26]
    ax.set_yticklabels(bin_labels) 

    # Set tick labels to be every four events 
    x_ticks = np.arange(0,len(ev))+0.5 # 0.5 shifts tick to center of pixel
    x_ticks = x_ticks[1::4] # every four starting from index 1 (2nd ev)
    ax.set_xticks(x_ticks)  # set tick locations with respect to bin #
    ax.set_xticklabels(ev[1::4], rotation=45, ha='right') # set tick labels
    ax.set_xlabel('Event #')
    
    # Mark event of interest if ev_idx was passed as arg
    if ev_idx is not None: 
        # Draw a line indicating the point of interest, most recently plotted
        # Distinguish whether clustered as matrix or fiber
        if channel_labels[ev_idx] == 0 and ev_idx is not None: # matrix
            ax.vlines(ev_idx+0.5,0,y_bins,color='blue',linewidth=1)
        if channel_labels[ev_idx] == 1 and ev_idx is not None: # fiber
            ax.vlines(ev_idx+0.5,0,y_bins,color='red',linewidth=1)
                       
    # Annotate where the first fiber event occured
    # This places a red line to the exact left of the bin / the start of bin
    first_fiber_ev = 0    # Determine first fiber failure event
    while(channel_labels[first_fiber_ev] != 1): # will stop at FIRST 1
        first_fiber_ev = first_fiber_ev + 1 
    # Label this first fiber crack    
    ax.vlines(first_fiber_ev,0,y_bins,color='red',linewidth=0.5,linestyle='--')
    if annotate: 
        ax.annotate('First fiber crack',xy=(first_fiber_ev,y_bins-5),
                xytext=(first_fiber_ev+3 ,y_bins),
                arrowprops=dict(arrowstyle='->',connectionstyle="arc3",\
                                color='red'),color='red')   
            
    # Saving in case I need in the future, used for frequency version
    # if spectrogram_type == 'frequency':
    #    tick_y_position = [0,y_bins/3,2*y_bins/3,y_bins]
    #    tick_y_label =  [int(low_pass/fft_units),
    #                     int((low_pass+(high_pass-low_pass)/3)/fft_units),
    #                     int((low_pass+(high_pass-low_pass)/3*2)/fft_units),
    #                     int(high_pass/fft_units)] 
    #    ytitle = 'Frequency(kHz)'
    #    ax.set_yticks(tick_y_position)
    #    ax.set_yticklabels(tick_y_label)   
    #    plt.colorbar(im, label='10*log10(A^2) ; Power dB')
      
    return ax

# def compute_cumul_featur_vector(
#         channel_feature_vector,
#         )
    
    
    
def plot_damage_location(
        ax,
        location,
        channel_labels,
        ev_idx,
        x_lim=[-7.5,7.5],
        y_lim=[-0.1,0.1]
        ):
    """
    
    Plot the location of events. Damage location calculated using time of 
    arrival method, taken from 'filter' .csv file.
    
    """
    # Check if event index argument was passed
    if ev_idx is not None:
        location = location[0:ev_idx+1] # only use locations up to event
    
    # Separate stress points into fiber and matrix based on cluster label
    location_m = location[np.where(channel_labels[0:ev_idx+1]==0)]
    location_f = location[np.where(channel_labels[0:ev_idx+1]==1)]
    
    # Plot location, distinguishing between fiber and matrix
    ax.scatter(location_m,np.zeros((len(location_m),1)),color='blue')
    ax.scatter(location_f,np.zeros((len(location_f),1)),color='red')
    
    # Draw a line indicating the point of interest, most recently plotted
    if channel_labels[ev_idx] == 0 and ev_idx is not None: # matrix
        ax.vlines(location[ev_idx],y_lim[0]-5,y_lim[1]+5,color='blue')
    if channel_labels[ev_idx] == 1 and ev_idx is not None: # fiber
        ax.vlines(location[ev_idx],y_lim[0]-5,y_lim[1]+5,color='red')
        
    # Set limits
    ax.set_xlabel('Location (mm)')
    ax.set_xlim(x_lim)
    ax.set_yticks([])
    ax.set_ylim(y_lim)
    
    return ax

def plot_stress_vs_norm_ae(
        ax,
        stress,
        channel_cumul_norm_energy,
        channel_labels,
        ev_idx=None
        ):
    """
    NOTE TO SELF: CURRENTLY THIS APPEARS TO BE COMPUTING THE WRONG CUMULATIVE ENERGY COMPARED TO PAPER
    THE CORRECT SHAPE IS THERE BUT NOT CORRECT MAGNITUDES, INVALIDATES THESE FIGURES FOR NOW
    
    Plot the cumulative normalize ae energy vs stress(x-axis). Plot the fiber
    vs matrix crack energies separately.
    
    """    
    channel_norm_energy_by_time = np.zeros(channel_cumul_norm_energy.shape[0])
    total_energy = sum(channel_cumul_norm_energy)
    norm_energy = channel_cumul_norm_energy/total_energy
    
    ev_m = np.argwhere(channel_labels==0)
    ev_f = np.argwhere(channel_labels==1)
    
    norm_energy_m = np.zeros(len(ev_m))
    norm_energy_f = np.zeros(len(ev_f))
    
    norm_energy_m[0] = norm_energy[ev_m[0]]
    norm_energy_f[0] = norm_energy[ev_f[0]]
    
    for i in range(1,len(ev_m)):
        norm_energy_m[i] = norm_energy_m[i-1]+\
            norm_energy[ev_m[i]]
            
    for i in range(1,len(ev_f)):
        norm_energy_f[i] = norm_energy_f[i-1]+\
            norm_energy[ev_f[i]]     
    
    stress_m = stress[np.where(channel_labels==0)]
    stress_f = stress[np.where(channel_labels==1)]

    ax.scatter(stress_m, norm_energy_m, color = 'blue')
    ax.scatter(stress_f, norm_energy_f, color = 'red')
    ax.set_ylim([0,0.95])
    ax.set_ylabel('Normalized cumulative AE energy')
    ax.set_xlabel('Stress (MPa)')
 
    if ev_idx is not None:
        if channel_labels[ev_idx] == 0:
            ax.vlines(stress[ev_idx],0,500,color='blue')
        if channel_labels[ev_idx] == 1:
            ax.vlines(stress[ev_idx],0,500,color='red')

    return ax

def compute_feature_vector_in_time(
        channel,
        dt,
        low_pass,
        high_pass,
        num_bins,
        fft_units
        ):
    """
    
    Computes the 2D array corresponding to the feature vector changing in time,
    similar to STFT. Given a channel, or an array containing the events from
    an individual transducer, function will compute the feature vector for each
    event, and compile into single array that can be visualized. The freq bound
    are also returned in case user needs to see how freq band is divided up.

    """
    # Create the 2D array, shape depends on number of bins and number of events
    channel_feature_vector = np.zeros((num_bins,channel.shape[0]))

    # Loop through all the AE events recorded for transducer during test
    for ev_idx, waveform in enumerate(channel):
        # 'signal' same as 'waveform'
        # Calculate feature vector (partial power)
        # The freq_bounds is used for plotting the bin lines in plot_fft
        feature_vector, freq_bounds, spacing = wave2vec(dt, waveform, low_pass,
                                                        high_pass, num_bins,
                                                        fft_units)
        
        # Append feature vector for each event
        channel_feature_vector[:,ev_idx] = feature_vector
         
    return channel_feature_vector, freq_bounds

def compute_cumul_norm_energy(
        energy
        ):
    """ 
    NOTE TO SELF: CURRENTLY THIS APPEARS TO BE COMPUTING THE WRONG CUMULATIVE ENERGY COMPARED TO PAPER
    THE CORRECT SHAPE IS THERE BUT NOT CORRECT MAGNITUDES, INVALIDATES THESE FIGURES FOR NOW
    
    Given a channel's energies, compute the normalize AE in time. In other 
    words, will compute the contribution of each AE event in terms of total
    energy.
    
    """       
    # If energy is a list, then we're looking at all 4 channels,
    # otherwise just for single channel.
    if type(energy) is list:
        
        channels_cumul_norm_energy = []
        for channel_energy in energy:
            total_energy = sum(channel_energy)
            norm_energy = channel_energy/total_energy
            channel_cumul_norm_energy = np.zeros(channel_energy.shape[0])
    
            for ev_idx in range(len(norm_energy)):
                channel_cumul_norm_energy[ev_idx] = \
                                                sum(norm_energy[0:ev_idx+1])
            
            channels_cumul_norm_energy.append(channel_cumul_norm_energy)
            
        return channels_cumul_norm_energy
        
    else: # single channel
            
        total_energy = sum(energy)
        norm_energy = energy/total_energy
        channel_cumul_norm_energy = np.zeros(energy.shape[0])
    
        for ev_idx in range(len(norm_energy)):
            channel_cumul_norm_energy[ev_idx] = sum(norm_energy[0:ev_idx+1])
            
        return channel_cumul_norm_energy

def compute_stft(
        dt,
        signal,
        low_pass,
        high_pass,
        n_fft=256,
        hop_length=128
        ):
    """
    
    For a single acoustic emissions signal, generally 1024 samples, compute the 
    STFT using librosa. Need to specify the window length and the filter
    thresholds. Uses hann windowing for each frame to prevent spectral leakage.
    
    """
    stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
    # uses hann windowing
 
    # Remove frequencies according to filtering done for partial power
    freq = librosa.fft_frequencies(sr=1/dt, n_fft=n_fft)
    if low_pass is not None:
        stft = stft[np.where(freq > low_pass)]
        freq = freq[np.where(freq > low_pass)]

    if high_pass is not None:
        stft = stft[np.where(freq < high_pass)]
        freq = freq[np.where(freq < high_pass)]

    return stft

def plot_spectrogram(
        ax,
        stft,
        dt,
        hop_length,
        sig_len,
        fft_units,
        high_pass,
        low_pass,
        amplitude_lim=0,
        ):
    """
    
    Plot spectrogram. Time-frequency representation.
    
    """
    
    im=ax.pcolormesh(stft,cmap='inferno')
    #plt.colorbar(im)

    num_bins = stft.shape[0] 
    num_slices = stft.shape[1]
    tick_y_position = \
        [0,num_bins/3,2*num_bins/3,num_bins]
    
    duration = sig_len*dt*10**6 # convert to us

    xticks_loc = ax.get_xticks() - 0.5
    ax.set_xticks(xticks_loc[1::])
    frame_times = np.linspace(0,duration,num_slices)
    xtick_label = [int(t) for t in frame_times]
    ax.set_xticklabels(xtick_label)
    ax.set_xlabel('Time (us)')
    tick_y_label =  [
        int(low_pass/fft_units),
        int((low_pass+(high_pass-low_pass)/3)/fft_units),
        int((low_pass+(high_pass-low_pass)/3*2)/fft_units),
        int(high_pass/fft_units)]
    ax.set_yticks(tick_y_position)
    ax.set_yticklabels(tick_y_label)
    ax.set_ylabel('Frequency(kHz)')
    
    return ax

def compute_max_feature_value(
        channels,
        dt,
        low_pass,
        high_pass,
        num_bins,
        fft_units
        ):
    """
    
    Calculate the feature vector in time for all 4 channels and returns the max
    value seen, for the purpose of consistent plotting for vmax in function
    plot_feature_vector_in_time.
    
    """
    max_feature_value = 0
    max_cumul_feature_value = 0
    
    for channel in channels:
        # Compute feature vector in time, this is what is visualized as a
        # spectrogram. Number of bins used for partial power method dictate 
        # shape, [num_bins, # of events]
        channel_feature_vector, freq_bounds = compute_feature_vector_in_time(
                                    channel, dt, low_pass, high_pass, num_bins, 
                                    fft_units)
        
        if np.amax(channel_feature_vector) > max_feature_value:
            max_feature_value = np.amax(channel_feature_vector)
            
        # sum across rows to get the max
        channel_cumul_feature_vec = np.sum(channel_feature_vector,axis=1)
        
        if np.amax(channel_cumul_feature_vec) > max_cumul_feature_value:
            max_cumul_feature_value = np.amax(channel_cumul_feature_vec)
    
    return max_feature_value, max_cumul_feature_value
    
def plot_cumul_feature_vec(
        ax,
        channel_cumul_feature_vec,
        max_cumul_feature_value
        ):
    
       y=np.arange(1,len(channel_cumul_feature_vec)+1)
       
       plt.barh(y,channel_cumul_feature_vec[:,0],color='blue')
       plt.barh(y,channel_cumul_feature_vec[:,1],color='red')

       #plt.xlim([0,max_cumul_feature_value])
       # Y labeling to label the bin rather than the line between
       ax.set_ylabel('Bin #',color = 'purple')
       ax.tick_params(axis='y',colors='purple')
       ax.set_yticks(y[1::2])
       ax.set_yticklabels(y[1::2])
       ax.set_ylim([0.5,26.5])
       ax.set_xlabel('Cumulative Spectrum Area')
       
       return ax

def plot_feature_vec(
    ax,
    channel_feature_vec,
    max_feature_value,
    color='blue'
    ):
    """
    
    Plot partial power feature vector in bar form, single event. The length of
    the feature vector MUST be 26 dimension. If this is not true, function must
    be modified for xlimit.

    Parameters
    ----------
    ax : axes
        Matplotlib class for figures.
    channel_feature_vec : array-like
        The partial power feature vector for a single event, 26 dimensions.
    max_feature_value : float
        Used to set axes.
    color : string, optional
        Color of bar plot. The default is 'blue'.

    Returns
    -------
    ax : axes
        Figure returned with feature vector plotted.

    """
    x=np.arange(1,len(channel_feature_vec)+1)
    plt.bar(x,channel_feature_vec,color=color)
    plt.ylim([0,max_feature_value])
    ax.set_xlabel('Bin #',color = 'purple')
    ax.tick_params(axis='x',colors='purple')
    ax.set_xticks(x[1::4])
    ax.set_xticklabels(x[1::4])
    ax.set_xlim([0.5,26.5])
    ax.set_ylabel('Spectrum Area')
    
    return ax

def create_4_channel_mp3(
        channels,
        labels,
        names,
        ev,
        stress,
        time,
        location,
        energy,
        fname,
        sig_len,
        dt,
        low_pass,
        high_pass,
        num_bins,
        fft_units,
        n_fft,  
        hop_length
        ):
    """
    
    Create mp3 for figures based off recording for 4 channels.
    
    """
    " ----- Directory Setup --------------------------------------------------"
    # Folder for images 
    sample_folder_name = fname+'_visuals'

    # Make folder for images if it doesn't exist
    folder_exist = os.path.exists(sample_folder_name) 
    if not folder_exist:
        os.mkdir(sample_folder_name)
    
    # Change directory to image folder
    os.chdir(sample_folder_name) 
    
    # Make folder for images 
    image_folder = '4_channels'
    folder_exist = os.path.exists(image_folder)
    if not folder_exist:
        os.mkdir(image_folder)
    os.chdir(image_folder)
   
    " ----- Additional Setup -------------------------------------------------"
    # Get max feature vector value for consistent plotting scale
    max_feature_value, max_cumul_feature_value = \
        compute_max_feature_value(channels, dt, low_pass,
                                                  high_pass, num_bins,
                                                  fft_units)
    
    # Compute cumulative normalize energy for each channel
    channels_cumul_norm_energy = [compute_cumul_norm_energy(energy[0]),
                           compute_cumul_norm_energy(energy[1]),
                           compute_cumul_norm_energy(energy[2]),
                           compute_cumul_norm_energy(energy[3])]
    
    # Cumulative feature vector
    channels_cumul_feature_vec = [np.zeros((num_bins,2)),
                                  np.zeros((num_bins,2)),
                                  np.zeros((num_bins,2)),
                                  np.zeros((num_bins,2))]
    
    " ------ Loop through all recorded events --------------------------------"
    num_events = channels[0].shape[0] 
    for ev_idx in range(num_events):   
        
        event_num = ev[ev_idx]         # Event number (some filtered)
        event_time = time[ev_idx]      # Time at which event occured
   
        " ----- Event Image Creation -------------------------------------"   
        # Create Figure
        columns = 16
        rows= 4
        suptitle = 'Sample: ' + fname + ' Event: ' + str(event_num)
        fig,spec2 = create_figure(suptitle, columns, rows)
        
        " ----- Loop through all 4 channels for the given event --------------"
        for idx, channel in enumerate(channels):
            
            # channel 2D array, [# of events, # of samples (1024 typical)] 
            signal = channel[ev_idx]      # waveform            
            channel_energy = energy[idx]  # energy per hit for channel
            channel_labels = labels[idx]  # cluster results (0 matrix, 1 fiber)
            channel_name = names[idx]     # channel name 
            
            # Compute feature vector in time, this is what is visualized as a
            # spectrogram. Number of bins used for partial power method dictate 
            # shape, [num_bins, # of events]
            channel_feature_vector, freq_bounds = \
                compute_feature_vector_in_time(channel, dt, low_pass,
                                               high_pass, num_bins, fft_units)
            
            # FOR THIS CHANNEL, Determine what this event was clustered as
            crack_type = ''      
            if channel_labels[ev_idx] == 1:
                crack_type = 'Fiber'
            else: # == 0
                crack_type = 'Matrix'
                
            # Add feature vector for event to appropriate col 
            event_feature_vector, freq_bounds, spacing = \
                wave2vec(dt, signal, low_pass,high_pass, num_bins, fft_units)
            
            if crack_type == 'Matrix':
                channels_cumul_feature_vec[idx][:,0] = \
                    channels_cumul_feature_vec[idx][:,0] \
                        + event_feature_vector
            else: # Fiber
                channels_cumul_feature_vec[idx][:,1] = \
                    channels_cumul_feature_vec[idx][:,1] + event_feature_vector
                
            " ----- Computations on signal -----------------------------------"
            # Compute fft for event signal (unscaled amplitude)
            w, z = fft(dt, signal, low_pass, high_pass)
            w = w/fft_units; # convert to kHZ
        
            " ----- Event Image Creation -------------------------------------"   
            row = idx # row of figure where plots go
            
            # Unimportant but necessary code just to prevent busy plots 
            bin_labels = True
            annotate = True
            if row != 0: 
                bin_labels = False 
                annotate = False

            # FFT
            fft_ax = fig.add_subplot(spec2[row,0:4])
            fft_ax = plot_fft(fft_ax, w, z, freq_bounds, low_pass, high_pass,
                              fft_units,bin_labels)
            fft_ax.text(1.01,.8,channel_name,horizontalalignment='left',
                            verticalalignment='bottom',fontsize=13,
                            transform = fft_ax.transAxes, weight='bold')
            if crack_type == 'Fiber': # fiber
                fft_ax.text(1.01,0.7,crack_type,horizontalalignment='left',
                            verticalalignment='bottom',fontsize=13,
                            transform = fft_ax.transAxes, weight='bold',
                            color='red')
            else: # matrix   
                fft_ax.text(1.01,0.7,crack_type,horizontalalignment='left',
                            verticalalignment='bottom',fontsize=13,
                            transform = fft_ax.transAxes, weight='bold',
                            color='blue')
                
            # Feature Vector Spectrogram
            feature_vec_ax = fig.add_subplot(spec2[row,5:12])
            feature_vec_ax = plot_feature_vector_in_time(feature_vec_ax, 
                                              channel_feature_vector,
                                              fft_units, high_pass, low_pass,
                                              ev, channel_labels,
                                              ev_idx=ev_idx,
                                              v_max=max_feature_value,
                                              annotate=annotate)
            plt.xticks(fontsize=8)
            #fontsize of the x tick labels

            # Cumulate Feature Vector
            # cumul_feature_vec_ax = fig.add_subplot(spec2[row,12:16])
            # cumul_feature_vec_ax = plot_cumul_feature_vec(
            #     cumul_feature_vec_ax,channels_cumul_feature_vec[idx],
            #     max_cumul_feature_value)
            
            # Remove extra x labels unless bottom row
            if row != rows-1: 
                fft_ax.set_xlabel('')
                feature_vec_ax.set_xlabel('')
                #cumul_feature_vec_ax.set_xlabel('')
        
        # Stress and Cumulative Norm Energy vs. Time
        stress_ax = fig.add_subplot(spec2[0:4,12:16])
        energy_ax = stress_ax.twinx()  
        stress_ax, energy_ax = plot_stress_and_cumul_norm_energy(stress_ax, 
                                            time, stress,channel_labels,
                                cumul_norm_energy=channels_cumul_norm_energy,
                                            ax2=energy_ax, ev_idx=ev_idx)
        
        # Let the tired programmer know that things are working
        print(f"Four channel - Event #: {ev[ev_idx]}")        
    
        # Image Save
        image_name = str(ev[ev_idx])
        image_name = image_name.zfill(4) # pad with zeros
        plt.savefig(image_name)   
        plt.close()
        
    # Create mp3 file with all event images in folder that were just created
    create_mp3_file(fname) 
    
    os.chdir('..') # Get out of the directory
    os.chdir('..') # GET OUT of the directory

    return
    
def create_individual_channel_mp3(
        channels,
        labels,
        names,
        ev,
        stress,
        time,
        location,
        energy,
        fname,
        sig_len,
        dt,
        low_pass,
        high_pass,
        num_bins,
        fft_units,
        n_fft,  
        hop_length
        ):
    """
    
    Create mp3 for figures based off recording by individual channel.
    
    """
    " ----- Directory Setup --------------------------------------------------"
    # Folder for images 
    sample_folder_name = fname+'_visuals'

    # Make folder for images if one does not exist
    folder_exist = os.path.exists(sample_folder_name) 
    if not folder_exist:
        os.mkdir(sample_folder_name)
    
    # Change directory to image folder
    os.chdir(sample_folder_name) 
    
    # Make folders for individual channels
    # Each folder will have the corresponding event figures saved into it
    for channel in names:
        folder_exist = os.path.exists(channel)
        if not folder_exist:
            os.mkdir(channel)
    
    " ------ Additional Setup ------------------------------------------------"
    # Get max feature vector value for consistent plotting scale
    max_feature_value,_ = compute_max_feature_value(channels, dt, low_pass,
                                                  high_pass, num_bins,
                                                  fft_units)
        
    " ------ Loop through all 4 transducer channels --------------------------"
    for idx, channel in enumerate(channels): # v0,v1,v2,v3
        
        # channel 2D array, [# of events, # of samples (1024 typical)]       
        channel_folder = names[idx]   # folder name
        channel_energy = energy[idx]  # energy per hit
        channel_labels = labels[idx]  # 1 fiber, 0 matrix
        
        # Change to directory of channel
        os.chdir(channel_folder)
        
        # Compute feature vector in time, this is what is visualized as a
        # spectrogram. Number of bins used for partial power method dictate 
        # shape, [num_bins, # of events]
        channel_feature_vector, freq_bounds = compute_feature_vector_in_time(
                                    channel, dt, low_pass, high_pass, num_bins,
                                    fft_units)
        
        # Compute cumulative normalized energy for channel
        channel_cumul_norm_ae = compute_cumul_norm_energy(channel_energy)
        
        " ------ Loop through all recorded event in channel ------------------"
        for ev_idx, signal in enumerate(channel):
            
            # signal is the raw waveform recorded by the transducer
            event_num = ev[ev_idx]         # Event number (some filtered)
            event_time = time[ev_idx]      # Time at which event occured
            
            " ----- Computations on signal -----------------------------------"
            # Compute fft for event signal (unscaled amplitude)
            w, z = fft(dt, signal, low_pass, high_pass)
            w = w/fft_units; # convert to kHZ
                        
            # Compute stft for event signal
            stft = compute_stft(dt, signal, low_pass, high_pass, n_fft=n_fft, 
                         hop_length=hop_length)
            
            " ----- Event Image Creation -------------------------------------"   
            # Create Figure
            columns = 3
            rows= 3
            crack_type = '' # Determine from labels the type of crack event 
            if channel_labels[ev_idx] == 1:
                crack_type = 'Fiber'
            else: # == 0
                crack_type = 'Matrix'
            suptitle = 'SAMPLE: '+ fname +'  EVENT: '+ str(event_num) \
                + '  CLUSTER: ' + crack_type # master title of figure
            fig,spec2 = create_figure(suptitle, columns, rows)
            
            # Stress and Cumulative Norm Energy vs. Time
            stress_ax = fig.add_subplot(spec2[0,2])
            energy_ax = stress_ax.twinx() 
            stress_ax, energy_ax = plot_stress_and_cumul_norm_energy(stress_ax, 
                                               time, stress,channel_labels,
                                     cumul_norm_energy=channel_cumul_norm_ae,
                                               ax2=energy_ax, ev_idx=ev_idx)

            # FFT
            fft_ax = fig.add_subplot(spec2[0,0])
            fft_ax = plot_fft(fft_ax, w, z, freq_bounds, low_pass, high_pass,
                              fft_units)
            
            # Spectrogram
            spectrogram_ax = fig.add_subplot(spec2[1,1])
            spectrogram_ax = plot_spectrogram(spectrogram_ax, stft, dt,
                                              hop_length, sig_len,
                                              fft_units,
                                              high_pass, low_pass)
            
            # Cumulative Norm Energy vs. Stress
            energy_stress_ax = fig.add_subplot(spec2[1,2])
            energy_stress_ax = plot_stress_vs_norm_ae(energy_stress_ax, stress,
                                                      channel_cumul_norm_ae,
                                                      channel_labels, 
                                                      ev_idx=ev_idx)
            
            # Event Signal 
            signal_ax = fig.add_subplot(spec2[0,1])
            signal_ax = plot_signal(signal_ax, signal, dt, sig_len)
            
            # Damage Location 
            location_ax = fig.add_subplot(spec2[1,0:1])
            location_ax = plot_damage_location(location_ax, location,
                                               channel_labels, ev_idx)
            
            # Feature Vector Spectrogram
            feature_vec_ax = fig.add_subplot(spec2[2,0:3])
            feature_vec_ax = plot_feature_vector_in_time(feature_vec_ax, 
                                              channel_feature_vector,
                                              fft_units, high_pass, low_pass,
                                              ev, channel_labels,
                                              ev_idx=ev_idx,
                                              v_max=max_feature_value)
                
            # Image Save
            image_name = str(ev[ev_idx])
            image_name = image_name.zfill(4) # pad with zeros
            plt.savefig(image_name)   
            plt.close()
            
            # Is the code working?
            print(f"{channel_folder} - Event #: {ev[ev_idx]}")
        
        create_mp3_file(fname + '_' + channel_folder) # create mp3 file 
        
        os.chdir('..') # Get out of directory
        
        # Create single large image of Feature Spectrogram in Time per channel
        suptitle = 'SAMPLE: '+ fname + ' CHANNEL: ' + channel_folder
        fig, spec2 = create_figure(suptitle, 1, 1)
        feature_vec_ax = fig.add_subplot(spec2[0,0])
        feature_vec_ax = plot_feature_vector_in_time(feature_vec_ax, 
                                  channel_feature_vector,
                                  fft_units, high_pass, low_pass,
                                  ev, channel_labels,
                                  ev_idx=ev_idx,
                                  v_max=max_feature_value)
        plt.savefig(channel_folder+'_feature_vector_spectrogram')
    
    os.chdir('..') # Go up a directory, your work here is done
    
    return

def create_mp3_file(
        video_name
        ):
    """
    
    Takes all the image files in the working directory and converts to mp3.
    
    """
    print("Creating mp3 file ... ")
    
    image_folder = '.'
    video_name = str(video_name) + '.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 5, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    
    print(f"Mp3 file saved as {video_name} \n\n")  
    
    return

