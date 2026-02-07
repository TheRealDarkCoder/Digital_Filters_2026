"""
MATLAB-inspired convenience functions for the digital filters course
in the University of Oulu.

Routines in this module:

impz(b, a=1, N=512)
zplane(b, a, margin=0.2, size=6)
fvtool(b, a, fp, fs, Ap, As, xlim=(0, 1), ylim=(-100, 5))
zerofill(x_in, L)

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def impz(b, a=1, N=512):
    """
    Compute the impulse response of a digital filter.
    
    Given the numerator b and denominator a of a digital filter,
    compute the first N coefficients of its impulse response.
    
    Parameters
    ----------
    b : array_like
        Numerator of a linear filter.
    a : array_like
        Denominator of a linear filter.
    N : int, optional
        Number of coefficients to compute.
    
    Returns
    -------
    h : ndarray
        The first N coefficients of the impulse response.
    
    """
    x = np.zeros(N)
    x[0] = 1
    return signal.lfilter(b, a, x)


def zplane(b, a, margin=0.2, size=6):
    """
    Plot the zeros and poles of a digital filter.
    
    Given the numerator b and denominator a of a digital filter,
    plot the zeros and poles of the filter.
    
    Parameters
    ----------
    b : array_like
        Numerator of a linear filter.
    a : array_like
        Denominator of a linear filter.
    margin : float, optional
        The minimum amount of empty space at the borders of the plot.
    size : int, optional
        The size of the figure.
    
    """
    z, p, k = signal.tf2zpk(b, a)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.scatter(z.real, z.imag, label='zeros')
    ax.scatter(p.real, p.imag, label='poles', marker='x')
    ax.legend()
    ax.axhline(0, color='k', alpha=0.5, lw=0.5, ls='--')
    ax.axvline(0, color='k', alpha=0.5, lw=0.5, ls='--')
    ax.add_artist(plt.Circle((0, 0), 1, color='k', alpha=0.5, ls='--', fill=False))
    ax.set_xlim(np.min(np.concatenate(([-1-margin], z.real-margin, p.real-margin))),
                np.max(np.concatenate(([1+margin], z.real+margin, p.real+margin))))
    ax.set_ylim(np.min(np.concatenate(([-1-margin], z.imag-margin, p.imag-margin))),
                np.max(np.concatenate(([1+margin], z.imag+margin, p.imag+margin))))
    ax.set_aspect('equal')
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part');

	
def fvtool(b, a, fp, fs, Ap, As, xlim=(0, 1), ylim=(-100, 5)):
    """
    Plot the magnitude response of a digital filter.
    
    Given the numerator b and denominator a and design parameters
    fp (passband edge frequency), fs (stopband edge frequency),
    Ap (passband ripple) and As (stopband attenuation),
    plot the magnitude response of the filter along with line markers
    that demonstrate how the filter fulfills the given design criteria.
    
    Parameters
    ----------
    b : array_like
        Numerator of a linear filter.
    a : array_like
        Denominator of a linear filter.
    fp : float
        The normalized passband edge frequency of the filter
    fs : float
        The normalized stopband edge frequency of the filter
    Ap : scalar
        The passband ripple of the filter in decibels
    As : scalar
        The stopband attenuation of the filter in decibels
    xlim : (scalar, scalar), optional
        The x-axis limits in data coordinates
    ylim : (scalar, scalar), optional
        The y-axis limits in data coordinates
    
    """
    w, h = signal.freqz(b, a)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(w / np.pi, 20 * np.log10(np.abs(h)))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.hlines(0, xmin=xlim[0], xmax=fp, color='r', alpha=0.7, linestyles='dashed')
    ax.hlines(-Ap, xmin=xlim[0], xmax=fs, color='r', alpha=0.7, linestyles='dashed')
    ax.hlines(-As, xmin=fp, xmax=xlim[1], color='r', alpha=0.7, linestyles='dashed')
    ax.vlines(fp, ymin=-As, ymax=0, color='r', alpha=0.7, linestyles='dashed')
    ax.vlines(fs, ymin=ylim[0], ymax=-Ap, color='r', alpha=0.7, linestyles='dashed')
    ax.set_xlabel("Normalized frequency (* pi rad/sample)")
    ax.set_ylabel("Magnitude (dB)")
    fig.tight_layout()


def lim_cycles(a, A, B, N):
    """
    Calculate and plot two outputs of a simple IIR-filter.
	
	Parameters
    ----------
    a : scalar
        the only coefficient of the filter
    A : scalar
        amplitude of the impulse
    B : int
        number of bits used in quantizing
    N : int
        length of the arrays
    
	Returns
    -------
    y : ndarray
	    filtered output without quantization
    yq : ndarray
        filtered output with quantization
    """
    Q = 2 ** (1-B)  # quantizing step
    
    # initialize arrays
    y = np.zeros(N+1) 
    yq = np.zeros(N+1)
    x = np.zeros(N+1)
    
    x[1] = A  # input impulse
    for i in range(1, N+1):  # 1 is first index, otherwise i-1 < 0
        y[i] = a*y[i-1] + x[i]  # output
        yq[i] = Q*np.floor(a*yq[i-1]/Q + 0.5) + x[i] # quantized output
    
    # plot the outputs
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(y, picker=2)
    ax[0].set_title("non-quantized output y")
    ax[1].plot(yq, picker=2)
    ax[1].set_title("quantized output yq")
    
    # create a text object for each axes object
    texts = []
    for a in ax:
        texts.append(a.text(0, 0, "", va="bottom", ha="left"))
    
    fig.tight_layout()
    
    def on_pick(event):
        line = event.artist
        axes = line.axes
        xdata, ydata = line.get_data()
        ind = event.ind
        
        # remove old square highlights
        for l in axes.get_lines():
            if l != line:
                l.remove()
        
        # draw a square to highlight the picked value
        axes.plot(xdata[ind][0], ydata[ind][0], 'sk')
        
        # display the picked value in the correct subplot
        for (i, a) in enumerate(ax):
            if a == axes:
                txt = texts[i]
        
        txt.set_x(xdata[ind][0])
        txt.set_y(ydata[ind][0])
        txt.set_text("X: {}\nY: {}".format(xdata[ind][0], ydata[ind][0]))
    
    fig.canvas.mpl_connect('pick_event', on_pick)
    return y, yq


def zerofill(x_in, L):
    """
    ZFILL  is the "expander" operation used in multi-rate filters
    -----
                           /  x(n/L),  for n = 0 modulo L
                   y(n) = < 
                           \  0,       otherwise

       Usage:   y = zerofill(x, L)

           x : input signal vector
           L : fill with L-1 zeros between each sample.
           y : output signal vector ==> Length(y) = L*Length(x)

    ---------------------------------------------------------------
     copyright 1994, by C.S. Burrus, J.H. McClellan, A.V. Oppenheim,
     T.W. Parks, R.W. Schafer, & H.W. Schussler.  For use with the book
     "Computer-Based Exercises for Signal Processing Using MATLAB"
     (Prentice-Hall, 1994).
    ---------------------------------------------------------------
    """
    N = len(x_in)
    y_out = np.zeros(N*L)
    i = (np.arange(0, N*L) % L) == 0
    y_out[i] = x_in
    return y_out
