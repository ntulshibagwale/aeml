"""

Dedicated module containing functions for creating visuals.

Nick Tulshibagwale

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

def plot_signal(ax, signal, dt, sig_length):
    """
    
    Plot raw event signal waveform with appropriate time scaling.
    
    Parameters
    ----------
    ax : matplotlib object
        Axes for plotting.
    signal : array-like
        AE event / hit, waveform voltage values.
    dt : float
        Time between samples.
    sig_length : int
        Number of samples in waveform event / hit. 

    Returns
    -------
    ax : Matplotlib object
        Axes for plotting with signal plotted.

    """
    duration = sig_length*dt*10**6 # convert to us
    time = np.linspace(0,duration,sig_length) # discretization of signal time
    if type(signal) is list:
        for idx,sig in enumerate(signal):
            ax.plot(time,sig)
    else:
        ax.plot(time,signal)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (us)')
    ax.set_xlim([0,duration])

    return ax

def plot_fft(ax, w, z, freq_bounds, low_pass, high_pass, fft_units, 
             label_bins=True):  
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