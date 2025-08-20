import numpy as np
from scipy.fftpack import fft, ifft
from scipy.interpolate import interp1d

def EWT1D(f, N=3, log=0):
    """
    Empirical Wavelet Transform (1D) - simplified offline version.
    Based on original implementation by Fernando Verdu et al.
    """
    l = len(f)
    ff = fft(f)
    mag = abs(ff[0:l // 2])
    mag = mag / np.max(mag)

    # Detect N boundaries (uniform for offline use)
    boundaries = np.linspace(0, l // 2, N + 2, dtype=int)[1:-1]

    ewt = []
    mfb = []

    for i in range(N):
        low = 0 if i == 0 else boundaries[i - 1]
        high = boundaries[i]

        filter_mask = np.zeros(l)
        filter_mask[low:high] = 1
        filter_mask[-high:-low] = 1

        subband = np.real(ifft(ff * filter_mask))
        ewt.append(subband)
        mfb.append(filter_mask)

    return np.array(ewt), np.array(mfb), boundaries