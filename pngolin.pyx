################################################################################
# Polyspectra code. Should be able to compute power spectrum, bispectrum,
# trispectrum, as well as other measurements that are useful for our
# consistency relations projects.
################################################################################
import numpy as np
import pyfftw
import time 

cimport numpy as np
cimport cython

from cython.parallel cimport prange, parallel
from libc.math cimport sqrt, pow, sin, pi, abs,  lround


################################################################################
# Parallel testing routines
################################################################################
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def test_sum(int N, int num_threads):
    """
    Simple sum function
    """
    
    cdef int i, tot

    tot = 0
    start = time.time()
    for i in prange(N, nogil=True, num_threads=1):
        tot += i
    end = time.time()
    print(end-start)
    return tot


################################################################################
# Fourier transform routines
################################################################################
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def FFT3Dr_f(np.ndarray[np.float32_t, ndim=3] a, int nthreads):
    """
    Single precision forward Fourier transform. Based on implementation in
    Pylians.

    Parameters
    ----------
    a: (Ngrid, Ngrid, Ngrid) float array
        Array to compute the FFT on. Should be a single point precision float
        array with Ngrid^3 dimensions. Typically this is delta(x).
    nthreads: integer
        Number of threads for the FFT

    Returns
    -------
    a_out: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier transformed array.
    """

    cdef int dims

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims, dims, dims), dtype='float32')
    a_out = pyfftw.empty_aligned((dims, dims, dims//2+1), dtype='complex64')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0, 1, 2),
                            flags=('FFTW_ESTIMATE',), direction='FFTW_FORWARD',
                            threads=nthreads)
                         
    # put input array into delta_r and perform FFTW
    a_in[:] = a
    fftw_plan(a_in, a_out)

    return a_out


@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def IFFT3Dr_f(np.complex64_t[:,:,::1] a, int nthreads):
    """
    Single precision inverse Fourier transform. Based on implementation in 
    Pylians.


    Parameters
    ----------
    a: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Array to compute the IFFT on. Should be a single point precision float
        array with Ngrid^3 dimensions. Typically this is delta(k).
    nthreads: integer
        Number of threads for the FFT

    Returns
    -------
    a_out: (Ngrid, Ngrid, Ngrid) float array
        Forward Fourier transformed array.
    """

    cdef int dims

    # align arrays
    dims  = len(a)
    a_in  = pyfftw.empty_aligned((dims, dims, dims//2+1), dtype='complex64')
    a_out = pyfftw.empty_aligned((dims, dims, dims), dtype='float32')

    # plan FFTW
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=(0, 1, 2),
                            flags=('FFTW_ESTIMATE',), direction='FFTW_BACKWARD', 
                            threads=nthreads)
                            
    # put input array into delta_r and perform FFTW
    a_in [:] = a
    fftw_plan(a_in,a_out)

    return a_out


@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def get_RFFT_mask(int ng, int nthreads):
    """
    Function that computes a mask for RFFTs that only counts kz=0 modes
    once. This is useful for computing sums over elements in RFFTs.

    Parameters
    ----------
    ng: integer
        Number of grid points on real space grid
    nthreads: integer
        Number of threads for the outer loop over grid indices and for FFTs.

    Returns
    -------
    RFFT_mask: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field that is one for modes with kz=0 and kx,ky>0
        and zero otherwise.
    """

    cdef int i, j, k, ki, kj, kk, middle
    cdef np.uint8_t[:,:,::1] RFFT_mask

    middle = ng//2 
    
    # Initialize output arrays
    RFFT_mask     = np.zeros((ng, ng, middle+1), dtype=np.uint8)

    # Loop over field and pick values in [k_min, k_max)
    for i in prange(ng,  nogil=True, num_threads=1):
        ki = (i-ng if (i>middle) else i)
        for j in range(ng):
            kj = (j-ng if (j>middle) else j)
                
            for k in range(middle+1): 
                kk = (k-ng if (k>middle) else k)

                # kz=0 and kz=middle planes should only be counted once
                if kk==0 or (kk==middle and ng%2==0):
                    if ki<0: continue
                    elif ki==0 or (ki==middle and ng%2==0):
                        if kj<0.0: continue

                RFFT_mask[i,j,k] = 1
    
    return RFFT_mask


################################################################################
# Density Field Manipulations
################################################################################
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def initialize_delta(np.ndarray[np.float32_t,ndim=3] delta, MAS, int nthreads):
    """
    Function to initialize a real space density field for pngolin calculations.
    This Fourier transforms and deconvolves the field with the appropriate
    window function based on the mass assignment scheme (MAS)

    Parameters
    ----------
    delta: (Ngrid, Ngrid, Ngrid) float array
        Real space density field.
    MAS: str
        Mass assignment scheme applied to real space density field. Currently 
        supports NGP, CIC, and TSC.
    nthreads: integer
        Number of threads for the outer loop over grid indices and for FFTs.

    Returns
    -------
    delta_k: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier transformed and window deconvolved density field.

    TO DO: output doesn't need to be complex!
    """

    cdef double wx, wy, wz
    cdef int ng, i, j, k, middle, power

    cdef np.complex64_t[:,:,::1] delta_k
    cdef np.ndarray[np.float64_t, ndim=1] ind_1d, window

    # Get appropriate power for window correction based on assignment scheme
    assert MAS in ['NGP', 'CIC', 'TSC'], "MAS must be in ['NGP', 'CIC', 'TSC']"

    if MAS=='NGP':
        power = 1
    elif MAS=='CIC':
        power = 2
    else:
        power = 3

    # Get grid size properties
    ng =  delta.shape[0]
    middle = ng//2

    # Fourier transform
    delta_k = FFT3Dr_f(delta, nthreads)

    # Deconvolve assignment scheme
    ind_1d = np.arange(ng, dtype=np.float64)
    ind_1d[ind_1d>ng/2]=ind_1d[ind_1d>ng/2]-ng

    window = 1/np.sinc(ind_1d/ng)**(power) # OPTIMIZE?
#    window = np.ones(ng)

    # Apply to input array
    for i in prange(ng, nogil=True, num_threads=1):
        wx = window[i]
        for j in range(ng):
            wy = window[j]
            for k in range(middle+1):
                wz = window[k]
                delta_k[i][j][k] = wx*wy*wz*delta_k[i][j][k] 


    return delta_k


@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def pick_field(np.complex64_t[:,:,::1] delta_k, int k_min, int k_max, 
               int nthreads, double alpha=0, return_k_bin=False):
    """
    Function that picks modes within a Fourier space density field delta_k 
    in interval [k_min, k_max). Note that k_min and k_max are in units of the 
    fundamental frequency. 

    Parameters
    ----------
    delta_k: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field.
    k_min: int
        Minimum wave number in units of kf.
    k_max: int 
        Maximum wave number in units of kf.
    nthreads: integer
        Number of threads for the outer loop over grid indices and for FFTs.
    alpha: integer (optional)
        Optional power alpha to have momentum weighted density fields. This
        is used to compute the momentum weighted spectra such as 
        <q^(2alpha) deltaq delta_{-q}> in the theory model.
    return_k_bin: bool (optional)
        Optional argument to specify whether or not to return the average k_bin
        used in this picked field. This can be used to better estimate the bin
        centers when comparing with theory.

    Returns
    -------
    delta_picked: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field that includes only modes in [k_min, k_max).
    I_picked: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space field that has value 1 for modes in [k_min, k_max) and 
        0 otherwise. This is used for normalizing pngolin calculations.
    k_bin: optional float
        If return_k_bin is True, then return the average value of k used in this
        FFT. This NEEDS to be rescaled in units of kf and normalized by Nmodes
        for actual computations. See the power spectrum routine for how this is
        done. 
    """

    cdef double knorm, k_bin, tot
    cdef int ng, i, j, k, ki, kj, kk, middle
    cdef np.complex64_t[:,:,::1] delta_picked, I_picked

    ng     =  delta_k.shape[0]
    middle = ng//2 
    
    # Initialize output arrays
    k_bin, tot   = 0.0, 0.0
    delta_picked = np.zeros_like(delta_k)
    I_picked     = np.zeros((ng, ng, middle+1), dtype=np.complex64)

    # Loop over field and pick values in [k_min, k_max)
    for i in prange(ng,  nogil=True, num_threads=1):
        ki = (i-ng if (i>middle) else i)
        if(ki>k_max): continue
        
        for j in range(ng):
            kj = (j-ng if (j>middle) else j)
            if(kj>k_max): continue
                
            for k in range(middle+1): 
                kk = (k-ng if (k>middle) else k)
                if(kk>k_max): continue

                knorm = sqrt(ki*ki+kj*kj+kk*kk)
                if(knorm<k_min or knorm>=k_max): continue

                delta_picked[i,j,k] = delta_k[i,j,k]*pow(knorm, alpha)
                I_picked[i,j,k]     = 1

                # kz=0 and kz=middle planes should not be included in any
                # sums computed over real FFTs
                if kk==0 or (kk==middle and ng%2==0):
                    if ki<0: continue
                    elif ki==0 or (ki==middle and ng%2==0):
                        if kj<0.0: continue

                k_bin += knorm
                tot   += 1
    
    if return_k_bin:
        k_bin = k_bin/tot
        return delta_picked, I_picked, k_bin
    else:
        return delta_picked, I_picked


@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def construct_pi_giri(np.complex64_t[:,:,::1] delta_k, int k_min, int k_max, 
                      int nthreads):
    """
    Constructs pi fields from https://arxiv.org/abs/2305.03070. These fields are
    meant to encapsulate an auxiliary field weighted by the local small-scale
    power spectrum. The precise definition of the auxiliary field is:
        pi(x) = (int d^3k/(2pi)^3 W^i(k)rho(k) e^{ikx})^2

    Parameters
    ----------
    delta_k: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field.
    k_min: int
        Minimum wave number in units of kf.
    k_max: int 
        Maximum wave number in units of kf.
    nthreads: integer
        Number of threads for the outer loop over grid indices and for FFTs.

    Returns #UPDATE THIS
    -------
    delta_picked: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field that includes only modes in [k_min, k_max).
    """

    cdef int ng, i, j, k
    cdef np.complex64_t[:,:,::1] delta_picked
    cdef np.float32_t[:,:,::1] deltax_picked, pix_picked

    # Useful quantitites
    ng     =  delta_k.shape[0]

    # Get filtered density field W^{i}(k)delta(k)
    delta_picked = pick_field(delta_k, k_min, k_max, nthreads)[0]

    # Compute IFFT
    deltax_picked = IFFT3Dr_f(delta_picked, nthreads)
    pix_picked = np.zeros((ng, ng, ng), dtype=np.float32)

    for i in range(ng):
        for j in range(ng):
            for k in range(ng):
                pix_picked[i,j,k] = pow(deltax_picked[i,j,k], 2)

    return pix_picked

@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def construct_Pk_mesh(int ng, Pk_interp, double box_len, int nthreads):
    """
    Function that constructs a mesh of P(k) values on a particular Fourier
    grid. This function is used in the python function implementations, but
    in the cython version all of these preliminary calculations are done
    over a single loop at the beginning of the function call.

    Parameters
    ----------
    ng: int
        Real space grid size.
    Pk_interp: interpolator
        Interpolator for k->P(k). Needs to be in units of [k]=1/box_len and
        [P(k)]=box_len^6.
    box_len: double
        Size of the box. Sets the dimensions of the problem.
    nthreads: int 
        Number of threads to use. Currently not parallelized

    Returns
    -------
    Pk_mesh: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field that has P(k) value at every location on the
        mesh.
    """

    cdef double knorm, kf
    cdef int i, j, k, ki, kj, kk, middle
    cdef np.float32_t[:,:,::1] Pk_mesh

    middle = ng//2 
    kf     = 2*pi/box_len
    
    # Initialize output arrays
    Pk_mesh = np.zeros((ng, ng, middle+1), dtype=np.float32)

    # Loop over field and pick values in [k_min, k_max)
    for i in range(ng):
        ki = (i-ng if (i>middle) else i)
        
        for j in range(ng):
            kj = (j-ng if (j>middle) else j)
                
            for k in range(middle+1): 
                kk = (k-ng if (k>middle) else k)

                knorm = sqrt(ki*ki+kj*kj+kk*kk)*kf
                Pk_mesh[i,j,k] = Pk_interp(knorm)#+0j

    return Pk_mesh

################################################################################
# Power spectrum routines
################################################################################
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def compute_Pk3D_single(np.complex64_t[:,:,::1] delta_k, int k_min, int k_max,
                        double box_len, int nthreads,  double alpha=0):
    """
    Computes 3D auto power power spectrum of a density field in a single bin 
    between [k_min, k_max) where k_min and k_max are in units of kf.

    Parameters
    ----------
    delta_k: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field.
    k_min: int
        Minimum wave number in units of kf=2pi/box_len.
    k_max: int 
        Maximum wave number in units of kf=2pi/box_len.
    box_len: float 
        Length of the box (typically in units of Mpc/h or Mpc). This sets the 
        dimensions of the power spectrum as well as the fundamental frequency.
    alpha: float (optional; default 0)
        Optional float that allows for momentum weighted power spectra. Used 
        in squeezed bispectrum modelling.
    nthreads: integer
        Number of threads for the outer loop over grid indices and for FFTs.

    Returns
    -------
    k_bin: float
        Average value of all Fourier modes used in this bin in units of 
        1/box_len~h/Mpc.
    Pk: float
        Power spectrum in units of [Vol]=[box_len]^3~(Mpc/h)^3.
    Nmodes: float
        Number of modes in the Fourier bin. Technically this should be an
        integer, but we keep it as a float.
    """

    cdef int ng, i, j, k
    cdef double signal, pairs, Pk, Nmodes, k_bin
    cdef np.complex64_t[:,:,::1] delta_k1, Ik1 # Fourier space fields
    cdef np.float32_t[:,:,::1] delta1, I1      # Real space fields

    ng   = delta_k.shape[0]
    delta_k1, Ik1, k_bin = pick_field(delta_k, k_min, k_max, nthreads, 
                                      alpha=alpha/2, return_k_bin=True)
    
    # Compute FFT
    delta1 = IFFT3Dr_f(delta_k1, nthreads)
    del delta_k1
    I1     = IFFT3Dr_f(Ik1, nthreads)
    del Ik1

    # Calculate sum and normalization
    signal, pairs = 0.0, 0.0

    for i in prange(ng, nogil=True, num_threads=1):
        for j in range(ng):
            for k in range(ng):
                signal += (delta1[i,j,k]*delta1[i,j,k])
                pairs  += (I1[i,j,k]*I1[i,j,k])
    
    # Normalize results 
    Pk     = (signal/pairs)*(box_len/ng**2)**3
    Nmodes = 0.5*pairs*ng**3 # Factor of 2 because real space double counts kz 
    k_bin  = k_bin*(2*pi/box_len)

    del delta1; del I1
    
    return k_bin, Pk, Nmodes


@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def compute_xPk3D_single(np.complex64_t[:,:,::1] delta_ka, 
                         np.complex64_t[:,:,::1] delta_kb, int k_min, int k_max,
                         double box_len, int nthreads,  double alpha=0):
    """
    Computes 3D auto power power spectrum of a density field in a single bin 
    between [k_min, k_max) where k_min and k_max are in units of kf.

    Parameters
    ----------
    delta_ka: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field.
    delta_kb: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field.
    k_min: int
        Minimum wave number in units of kf=2pi/box_len.
    k_max: int 
        Maximum wave number in units of kf=2pi/box_len.
    box_len: float 
        Length of the box (typically in units of Mpc/h or Mpc). This sets the 
        dimensions of the power spectrum as well as the fundamental frequency.
    alpha: float (optional; default 0)
        Optional float that allows for momentum weighted power spectra. Used 
        in squeezed bispectrum modelling.
    nthreads: integer
        Number of threads for the outer loop over grid indices and for FFTs.

    Returns
    -------
    k_bin: float
        Average value of all Fourier modes used in this bin in units of 
        1/box_len~h/Mpc.
    Pk: float
        Power spectrum in units of [Vol]=[box_len]^3~(Mpc/h)^3.
    Nmodes: float
        Number of modes in the Fourier bin. Technically this should be an
        integer, but we keep it as a float.
    """

    cdef int ng, i, j, k
    cdef double signal, pairs, Pk, Nmodes, k_bin
    cdef np.complex64_t[:,:,::1] delta_ka_k1, delta_kb_k1, Ik1 # Fourier space fields
    cdef np.float32_t[:,:,::1] delta_xa_k1, delta_xb_k1, Ix1      # Real space fields

    ng   = delta_ka.shape[0]
    delta_ka_k1, Ik1, k_bin = pick_field(delta_ka, k_min, k_max, nthreads, 
                                         alpha=alpha/2, return_k_bin=True)
    delta_kb_k1, _, _ = pick_field(delta_kb, k_min, k_max, nthreads, 
                                   alpha=alpha/2, return_k_bin=True)
    
    del delta_ka, delta_kb

    # Compute FFT
    delta_xa_k1 = IFFT3Dr_f(delta_ka_k1, nthreads)
    del delta_ka_k1
    delta_xb_k1 = IFFT3Dr_f(delta_kb_k1, nthreads)
    del delta_kb_k1
    Ix1     = IFFT3Dr_f(Ik1, nthreads)
    del Ik1

    # Calculate sum and normalization
    signal, pairs = 0.0, 0.0

    for i in prange(ng, nogil=True, num_threads=1):
        for j in range(ng):
            for k in range(ng):
                signal += (delta_xa_k1[i,j,k]*delta_xb_k1[i,j,k])
                pairs  += (Ix1[i,j,k]*Ix1[i,j,k])
    
    # Clean up
    del delta_xa_k1, delta_xa_k1, Ix1

    # Normalize results 
    Pk     = (signal/pairs)*(box_len/ng**2)**3
    norm   = pairs # Factor of 2 because real space double counts kz 
    k_bin  = k_bin*(2*pi/box_len)
    
    return k_bin, Pk, pairs


@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def compute_Pk3D_binned(np.complex64_t[:,:,::1] delta_k, int k_min, int k_max,
                        int dk, double box_len, int nthreads, double alpha=0,
                        verbose=False):
    """
    Computes 3D auto power power spectrum of a density field in bins between
    k_min and k_max with spacing dk. k_min, k_max, and dk are all in units of
    fundamental frequency.

    Parameters
    ----------
    delta_k: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field.
    k_min: int
        Minimum wave number in units of kf=2pi/box_len.
    k_max: int 
        Maximum wave number in units of kf=2pi/box_len.
    box_len: float 
        Length of the box (typically in units of Mpc/h or Mpc). This sets the 
        dimensions of the power spectrum as well as the fundamental frequency.
    nthreads: integer
        Number of threads for the outer loop over grid indices and for FFTs.
    alpha: float (optional; default 0)
        Optional float that allows for momentum weighted power spectra. Used 
        in squeezed bispectrum modelling.
    verbose: bool (optional; default False)
        Sets verbosity

    Returns
    -------
    k_cen_arr: float array
        Bin center of the power spectrum. This is simply the midpoint of the
        bin. More sophisticated treatment of binning could be necessary at 
        low k to mitigate binning effects. Units of 1/[box_len]~h/Mpc.
    Pk: float array
        Power spectrum in units of [Vol]=[box_len]^3~(Mpc/h)^3.
    Nmodes: float array
        Number of modes in the Fourier bin. Technically this should be an
        integer, but we keep it as a float.
    """

    cdef int Nk, ng, middle, i, j, k, k_ind, ki, kj, kk
    cdef double  kf, knorm, start, end
    cdef np.ndarray[np.int64_t, ndim=1] k_min_arr, k_max_arr 
    cdef np.ndarray[np.float64_t, ndim=1] k_arr, Pk_arr, Nmodes_arr

    # Initialize data products
    kf         = 2*pi/box_len
    k_min_arr  = np.arange(k_min, k_max, dk, dtype=np.int64) # Lower bin values (units of kf)
    k_max_arr  = k_min_arr+dk
    Nk         = len(k_min_arr)
    k_arr      = np.zeros(Nk, dtype=np.float64)
    Pk_arr     = np.zeros(Nk, dtype=np.float64)
    Nmodes_arr = np.zeros(Nk, dtype=np.float64)

    ng = delta_k.shape[0]
    middle = ng//2

    if verbose: 
        start = time.time()
        print("1. COMPUTING POWER SPECTRUM WITH {} THREADS".format(nthreads))

        if nthreads>1:
            print("-> WARNING: nthreads is greater than 1. This can bias the estimator")

        print("-> {} bins".format(Nk))

    # Loop over momentum bins
    for i in prange(ng,nogil=True, num_threads=1):
        ki = (i-ng if (i>middle) else i)
        
        for j in range(ng):
            kj = (j-ng if (j>middle) else j)
                
            for k in range(middle+1):
                kk = (k-ng if (k>middle) else k)

                if kk==0 or (kk==middle and ng%2==0):
                    if ki<0: continue
                    elif ki==0 or (ki==middle and ng%2==0):
                        if kj<0.0: continue

                knorm = sqrt(ki*ki+kj*kj+kk*kk)

                # Get index
                k_ind = lround((knorm-k_min)/dk-0.499999)

                if 0<=k_ind<Nk:
                    k_arr[k_ind] += knorm
                    Pk_arr[k_ind] += (pow(delta_k[i,j,k].real, 2)+
                                      pow(delta_k[i,j,k].imag, 2))*pow(knorm, alpha)
                    Nmodes_arr[k_ind] += 1

    k_arr  = kf*k_arr/Nmodes_arr
    Pk_arr = (Pk_arr/Nmodes_arr)*(box_len/ng**2)**3

    # Need to multiply by kf^alpha for momentum weighted spectra.
    Pk_arr = Pk_arr*pow(kf, alpha)

    if verbose:
        end = time.time()
        print("-> Time taken: {:.3f} seconds".format(end-start))

    return k_arr, Pk_arr, Nmodes_arr


@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def compute_xPk3D_binned(np.complex64_t[:,:,::1] delta_k_a,
                         np.complex64_t[:,:,::1] delta_k_b, int k_min, int k_max,
                        int dk, double box_len, int nthreads, double alpha=0,
                        verbose=False):
    """
    Computes 3D cross power spectrum of two density fields in bins between
    k_min and k_max with spacing dk. 
    Note: k_min, k_max, and dk are all in units of fundamental frequency.

    Parameters
    ----------
    delta_k_a: (Ngrid, Ngrid, Ngrid//2+1) complex array
        First Fourier space density field.
    delta_k_b: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Second Fourier space density field.
    k_min: int
        Minimum wave number in units of kf=2pi/box_len.
    k_max: int 
        Maximum wave number in units of kf=2pi/box_len.
    box_len: float 
        Length of the box (typically in units of Mpc/h or Mpc). This sets the 
        dimensions of the power spectrum as well as the fundamental frequency.
    nthreads: integer
        Number of threads for the outer loop over grid indices and for FFTs.
    alpha: float (optional; default 0)
        Optional float that allows for momentum weighted power spectra. Used 
        in squeezed bispectrum modelling.
    verbose: bool (optional; default False)
        Sets verbosity

    Returns
    -------
    k_cen_arr: float array
        Bin center of the power spectrum. This is simply the midpoint of the
        bin. More sophisticated treatment of binning could be necessary at 
        low k to mitigate binning effects. Units of 1/[box_len]~h/Mpc.
    Pk: float array
        Power spectrum in units of [Vol]=[box_len]^3~(Mpc/h)^3.
    Nmodes: float array
        Number of modes in the Fourier bin. Technically this should be an
        integer, but we keep it as a float.
    """

    cdef int Nk, ng, middle, i, j, k, k_ind, ki, kj, kk
    cdef double  kf, knorm, start, end
    cdef np.ndarray[np.int64_t, ndim=1] k_min_arr, k_max_arr 
    cdef np.ndarray[np.float64_t, ndim=1] k_arr, Pk_arr, Nmodes_arr

    # Initialize data products
    kf         = 2*pi/box_len
    k_min_arr  = np.arange(k_min, k_max, dk, dtype=np.int64) # Lower bin values (units of kf)
    k_max_arr  = k_min_arr+dk
    Nk         = len(k_min_arr)
    k_arr      = np.zeros(Nk, dtype=np.float64)
    Pk_arr     = np.zeros(Nk, dtype=np.float64)
    Nmodes_arr = np.zeros(Nk, dtype=np.float64)

    ng = delta_k_a.shape[0]
    middle = ng//2

    assert delta_k_a.shape != delta_k_b.shape, "Input fields must have same dimensions."

    if verbose: 
        start = time.time()
        print("1. COMPUTING CROSS POWER SPECTRUM WITH {} THREADS".format(nthreads))

        if nthreads>1:
            print("-> WARNING: nthreads is greater than 1. This can bias the estimator")

        print("-> {} bins".format(Nk))

    # Loop over momentum bins
    for i in prange(ng,nogil=True, num_threads=1):
        ki = (i-ng if (i>middle) else i)
        
        for j in range(ng):
            kj = (j-ng if (j>middle) else j)
                
            for k in range(middle+1):
                kk = (k-ng if (k>middle) else k)

                if kk==0 or (kk==middle and ng%2==0):
                    if ki<0: continue
                    elif ki==0 or (ki==middle and ng%2==0):
                        if kj<0.0: continue

                knorm = sqrt(ki*ki+kj*kj+kk*kk)

                # Get index
                k_ind = lround((knorm-k_min)/dk-0.499999)

                if 0<=k_ind<Nk:
                    k_arr[k_ind] += knorm
                    Pk_arr[k_ind] += ((delta_k_a[i,j,k].real*delta_k_b[i,j,k].real)+
                                      (delta_k_a[i,j,k].imag*delta_k_b[i,j,k].imag))*pow(knorm, alpha)
                    Nmodes_arr[k_ind] += 1

    k_arr  = kf*k_arr/Nmodes_arr
    Pk_arr = (Pk_arr/Nmodes_arr)*(box_len/ng**2)**3

    # Need to multiply by kf^alpha for momentum weighted spectra.
    Pk_arr = Pk_arr*pow(kf, alpha)

    if verbose:
        end = time.time()
        print("-> Time taken: {:.3f} seconds".format(end-start))

    return k_arr, Pk_arr, Nmodes_arr


################################################################################
# Bispectrum routines
################################################################################
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def compute_bk3d(np.complex64_t[:,:,::1] delta_k, int k1_min, int k1_max,
                 int k2_min, int k2_max, int k3_min, int k3_max, 
                 int dk1, int dk2, int dk3, double box_len, int nthreads,
                 verbose=False, np.ndarray[np.float32_t, ndim=1] Bqnorm_arr=None):
    """
    Computes 3D bispectrum for ki_min<ki<ki_max with i from 1 to 3. Note 
    that this assumes k1<k2<k3.

    Parameters (UPDATE)
    ----------
    delta_k: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field.
    ki_min: int
        Minimum wave number for the ith leg in units of kf=2pi/box_len.
    ki_max: int 
        Maximum wave number for the ith leg in units of kf=2pi/box_len.
    dki: int
        Bin spacing for the ith leg in units of kf.

    box_len: float 
        Length of the bo
        x (typically in units of Mpc/h or Mpc). This sets the 
        dimensions of the power spectrum as well as the fundamental frequency.
    nthreads: integer
        Number of threads for the outer loop over grid indices and for FFTs.

    Returns
    -------
    q_cen_arr: float array
        Bin center of the soft mode. This is computed as the average wavenumber
        of all Fourier modes in this bin.
    Bq_arr: float array
        Angle averaged bispectrum in units of [Vol]^2=[box_len]^6~(Mpc/h)^6.
    Bqnorm_arr: float array
        Number of triangles in the Fourier bin. This is currently wrong!
    """

    # Initialize variables
    cdef int ng, Nk1, Nk2, Nk3, k1_min_i, k2_min_i, k12_min_i, k3_min_i
    cdef int Nbin_tot, Nbin_uniq, i, j, k, k1_ind, k2_ind, k3_ind 
    cdef np.float32_t  kf, signal, Ntri,  k1, k2, k3
    cdef np.ndarray[np.int16_t, ndim=1] k1_min_arr, k2_min_arr, k3_min_arr    
    cdef np.ndarray[np.float32_t, ndim=1] Bk_arr
    cdef np.ndarray[np.float32_t, ndim=2] k_cen_arr

    # Define Fourier and real space fields
    cdef np.ndarray[np.float32_t, ndim=1] k_cen_uniq     
    cdef np.ndarray[np.complex64_t, ndim=4] Wk_uniq, deltak_uniq
    cdef np.ndarray[np.float32_t, ndim=4] Wr_uniq, deltax_uniq
    cdef np.float32_t[:,:,::1] deltax_k1, Wr_k1, deltax_k2, Wr_k2, deltax_k3, Wr_k3 

    # Compute constants and useful variables
    ng     = delta_k.shape[0]
    kf     = 2*pi/box_len

    # Bin initialization
    if verbose: 
        start = time.time()
        print("1. INITIALIZING BINS")
        
    k1_min_arr  = np.arange(k1_min, k1_max, dk1, dtype=np.int16)
    k2_min_arr  = np.arange(k2_min, k2_max, dk2, dtype=np.int16) 
    k3_min_arr  = np.arange(k3_min, k3_max, dk3, dtype=np.int16)  

    Nk1         = len(k1_min_arr)
    Nk2         = len(k2_min_arr)
    Nk3         = len(k3_min_arr)

    k_min_mat = []
    for i1 in range(Nk1):
        k1_min_i = k1_min_arr[i1]

        for i2 in range(Nk2):
            k2_min_i = k2_min_arr[i2]
            if k2_min_i < k1_min_i: continue # k1 ≤ k2
            for i3 in range(Nk3):
                k3_min_i = k3_min_arr[i3]
                if k3_min_i < k2_min_i: continue # k2 ≤ k3

                # Triangle inequality
                if not ((k1_min_i+dk1+k2_min_i+dk2>k3_min_i) &
                        (k2_min_i+dk2+k3_min_i+dk3>k1_min_i) &
                        (k3_min_i+dk3+k1_min_i+dk1>k2_min_i)): continue

                k_min_mat.append([k1_min_i, k2_min_i, k3_min_i])
        
    k_min_mat = np.asarray(k_min_mat)

    # The following line computes all pairs (k_min, k_max) that are used for the windows. 
    # This way we only have to compute them once and significantly reduces # of FFTs.
    # This also determines the indexing W_ind = k_bounds_uniq.index((kmin_i,kmax_i))
    k_bounds_uniq = list(set([(i, i+dk1) for i in k1_min_arr]+
                             [(i, i+dk2) for i in k2_min_arr]+
                             [(i, i+dk3) for i in k3_min_arr]))
    Nbin_tot = len(k_min_mat)
    N_uniq   = len(k_bounds_uniq)
    
    # Initialize data products
    if verbose:
        print("-> Total number of bins: {}".format(Nbin_tot))
        print("-> Number of unique k windows: {}".format(N_uniq))
        end = time.time()
        print("-> Time taken: {:.3f} seconds".format(end-start))

    if Bqnorm_arr is None:
        if verbose:
            start = time.time()
            print("2. COMPUTING UNIQUE WINDOWED FIELDS AND THEIR IFFTS")
            print("-> Normalization not specified. Computing this too!")

        k_cen_uniq  = np.empty(N_uniq, dtype=np.float32)
        deltak_uniq = np.zeros((N_uniq, ng, ng, ng//2+1), dtype=np.complex64)
        Wk_uniq     = np.zeros((N_uniq, ng, ng, ng//2+1), dtype=np.complex64)
        deltax_uniq = np.zeros((N_uniq, ng, ng, ng), dtype=np.float32)
        Wr_uniq     = np.zeros((N_uniq, ng, ng, ng), dtype=np.float32)

        # Compute all necessary real space fields
        for i in (range(N_uniq)):
            deltak_i, Wk_i, k_cen_uniq[i] = pick_field(delta_k, *k_bounds_uniq[i], nthreads, return_k_bin=True) 
            Wk_uniq[i]     = np.asarray(Wk_i)
            Wr_uniq[i]     = IFFT3Dr_f(Wk_uniq[i], nthreads)
            deltak_uniq[i] = np.asarray(deltak_i)
            deltax_uniq[i] = IFFT3Dr_f(deltak_uniq[i], nthreads)
            
        del Wk_uniq, deltak_uniq

        if verbose:
            end = time.time()
            print("-> Time taken: {:.3f} seconds".format(end-start))
            print("3. COMPUTING BINNED BISPECTRUM")
            start = time.time()

        # Initialize output data products
        k_cen_arr = np.zeros((Nbin_tot, 3), dtype=np.float32)
        Bk_arr    = np.zeros(Nbin_tot, dtype=np.float32)
        Bqnorm_arr  = np.zeros_like(Bk_arr)

        # Loop through bins and compute bispectrum 
        for ik, k_min in enumerate(k_min_mat):
            k1_min_i, k2_min_i, k3_min_i = k_min
            k1_ind = k_bounds_uniq.index((k1_min_i, k1_min_i+dk1))
            k2_ind = k_bounds_uniq.index((k2_min_i, k2_min_i+dk2))
            k3_ind = k_bounds_uniq.index((k3_min_i, k3_min_i+dk3))

            # Get bin center (this is the average value)
            k_cen_arr[ik] = np.asarray([k_cen_uniq[k1_ind], k_cen_uniq[k2_ind], 
                                        k_cen_uniq[k3_ind]])*kf
            
            # Get correct fields
            deltax_k1 = deltax_uniq[k1_ind]
            deltax_k2 = deltax_uniq[k2_ind]
            deltax_k3 = deltax_uniq[k3_ind] 
            Wr_k1 = Wr_uniq[k1_ind]
            Wr_k2 = Wr_uniq[k2_ind]
            Wr_k3 = Wr_uniq[k3_ind]

            # Compute sum in real space
            signal, Ntri = 0.0, 0.0
            
            # Can parallelize this with some work
            for i in prange(ng,nogil=True, num_threads=1):
                for j in range(ng):
                    for k in range(ng):

                        signal += (deltax_k1[i,j,k]*deltax_k2[i,j,k]*deltax_k3[i,j,k])         
                        Ntri += (Wr_k1[i,j,k]*Wr_k2[i,j,k]*Wr_k3[i,j,k])

            Bk_arr[ik]   = signal/Ntri*pow(box_len/ng, 6)/pow(ng, 3)
            Bqnorm_arr[ik] = Ntri

        if verbose:
            end = time.time()
            print("-> Time taken: {:.3f} seconds".format(end-start))

    else: 

        assert len(Bqnorm_arr)==len(k_min_mat), "Normalization has wrong length."

        if verbose:
            start = time.time()
            print("2. COMPUTING UNIQUE WINDOWED FIELDS AND THEIR IFFTS")
            print("-> Using input normalization.")

        k_cen_uniq  = np.empty(N_uniq, dtype=np.float32)
        deltak_uniq = np.zeros((N_uniq, ng, ng, ng//2+1), dtype=np.complex64)
        deltax_uniq = np.zeros((N_uniq, ng, ng, ng), dtype=np.float32)

        # Compute all necessary real space fields
        for i in (range(N_uniq)):
            deltak_i, _, k_cen_uniq[i] = pick_field(delta_k, *k_bounds_uniq[i], nthreads, return_k_bin=True) 
            
            deltak_uniq[i] = np.asarray(deltak_i)
            deltax_uniq[i] = IFFT3Dr_f(deltak_uniq[i], nthreads)
            
        del deltak_uniq

        if verbose:
            end = time.time()
            print("-> Time taken: {:.3f} seconds".format(end-start))
            print("3. COMPUTING BINNED BISPECTRUM")
            start = time.time()

        # Initialize output data products
        k_cen_arr = np.zeros((Nbin_tot, 3), dtype=np.float32)
        Bk_arr    = np.zeros(Nbin_tot, dtype=np.float32)

        # Loop through bins and compute bispectrum 
        for ik, k_min in enumerate(k_min_mat):
            k1_min_i, k2_min_i, k3_min_i = k_min
            k1_ind = k_bounds_uniq.index((k1_min_i, k1_min_i+dk1))
            k2_ind = k_bounds_uniq.index((k2_min_i, k2_min_i+dk2))
            k3_ind = k_bounds_uniq.index((k3_min_i, k3_min_i+dk3))

            # Get bin center (this is the average value)
            k_cen_arr[ik] = np.asarray([k_cen_uniq[k1_ind], k_cen_uniq[k2_ind], 
                                        k_cen_uniq[k3_ind]])*kf
            
            # Get correct fields
            deltax_k1 = deltax_uniq[k1_ind]
            deltax_k2 = deltax_uniq[k2_ind]
            deltax_k3 = deltax_uniq[k3_ind] 

            # Compute sum in real space
            signal = 0.0
            
            for i in prange(ng,nogil=True, num_threads=1): 
                for j in range(ng):
                    for k in range(ng):
                        signal += (deltax_k1[i,j,k]*deltax_k2[i,j,k]*deltax_k3[i,j,k])         

            Bk_arr[ik]   = signal*pow(box_len/ng, 6)/pow(ng, 3)

        # Normalize result
        Bk_arr = Bk_arr/Bqnorm_arr

        if verbose:
            end = time.time()
            print("-> Time taken: {:.3f} seconds".format(end-start))
    
    return k_cen_arr, Bk_arr, Bqnorm_arr


@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def compute_bk3d_ang_avg(np.complex64_t[:,:,::1] delta_k, int q_min, int q_max,
                         int dq, int k_min, int k_max, double box_len, 
                         int nthreads, int k3_min=0, int k3_max=1000000,
                         np.ndarray[np.float32_t, ndim=1] Bqnorm_arr=None):
    """
    Computes angular averaged bispectrum estimator used in 2209.06228. This is
    a squeezed bispectrum binned in the soft mode k1=q and integrated over hard
    modes k2 in [k_min, k_max) and 0<k3<infty subject to triangle inequality.

    Parameters (UPDATE)
    ----------
    delta_k: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field.
    q_min: int
        Minimum soft wave number in units of kf=2pi/box_len.
    q_max: int 
        Maximum soft wave number in units of kf.
    dq: int
        Bin spacing in units of kf.
    k_min: int
        Minimum hard mode wave number in units of kf.
    k_min: int
        Maximum hard mode wave number in units of kf.
    box_len: float 
        Length of the box (typically in units of Mpc/h or Mpc). This sets the 
        dimensions of the power spectrum as well as the fundamental frequency.
    nthreads: integer
        Number of threads for the outer loop over grid indices and for FFTs.

    Returns
    -------
    q_cen_arr: float array
        Bin center of the soft mode. This is computed as the average wavenumber
        of all Fourier modes in this bin.
    Bq_arr: float array
        Angle averaged bispectrum in units of [Vol]^2=[box_len]^6~(Mpc/h)^6.
    Bqnorm_arr: float array
        Number of triangles in the Fourier bin. This is currently wrong!
    """

    # Initialize variables
    cdef int Nq, ng, q_ind, i, j, k
    cdef double  kf, signal, Ntriangles
    cdef np.ndarray[np.int64_t, ndim=1] q_min_arr      
    cdef np.ndarray[np.float32_t, ndim=1] q_cen_arr, Bq_arr#, Bqnorm_arr

    # Define Fourier and real space fields
    cdef np.complex64_t[:,:,::1] delta_k1, I_k1, delta_k2, I_k2, delta_k3, I_k3 
    cdef np.float32_t[:,:,::1] deltax_k1, Ix_k1, deltax_k2, Ix_k2, deltax_k3, Ix_k3 

    # Initialize data products
    ng         = delta_k.shape[0]
    kf         = 2*pi/box_len
    q_min_arr  = np.arange(q_min, q_max, dq, dtype=np.int64) # Lower bin values (units of kf)
    Nq         = len(q_min_arr)

    # Compute normalization
    if Bqnorm_arr is None:

        # Output arrays
        q_cen_arr  = np.zeros(Nq, dtype=np.float32)
        Bq_arr     = np.zeros(Nq, dtype=np.float32)
        Bqnorm_arr = np.zeros(Nq, dtype=np.float32)

        # k_min<k2<k_max
        delta_k2, I_k2 = pick_field(delta_k, k_min, k_max, nthreads)
        deltax_k2 = IFFT3Dr_f(delta_k2, nthreads); del delta_k2
        Ix_k2     = IFFT3Dr_f(I_k2, nthreads); del I_k2

        # 0<k3<inf
        delta_k3, I_k3 = pick_field(delta_k, k3_min, k3_max, nthreads)
        deltax_k3 = IFFT3Dr_f(delta_k3, nthreads); del delta_k3
        Ix_k3     = IFFT3Dr_f(I_k3, nthreads); del I_k3

        # Loop over soft modes
        for q_ind in (range(Nq)):

            # q_i<k1<qi+dq
            delta_k1, I_k1, k1 = pick_field(delta_k, q_min_arr[q_ind], 
                                            q_min_arr[q_ind]+dq, nthreads,
                                            return_k_bin=True)
            deltax_k1 = IFFT3Dr_f(delta_k1, nthreads); del delta_k1
            Ix_k1     = IFFT3Dr_f(I_k1, nthreads); del I_k1

            # Calculate sum and normalization
            signal, Ntriangles = 0.0, 0.0     
            for i in prange(ng, nogil=True, num_threads=1):
                for j in range(ng):
                    for k in range(ng):
                        signal      += (deltax_k1[i,j,k]*deltax_k2[i,j,k]*deltax_k3[i,j,k])
                        Ntriangles  += (Ix_k1[i,j,k]*Ix_k2[i,j,k]*Ix_k3[i,j,k])
            
            q_cen_arr[q_ind] = k1*kf
            Bq_arr[q_ind]    = signal/Ntriangles*pow(box_len, 6)/pow(ng, 9)
            Bqnorm_arr[q_ind]  = Ntriangles#/pow(ng, 3)

        return q_cen_arr, Bq_arr, Bqnorm_arr
    
    # Use pre-computed normalization
    else:
        assert len(Bqnorm_arr)==len(q_min_arr), "Normalization has wrong length."

        # Output arrays
        q_cen_arr  = np.zeros(Nq, dtype=np.float32)
        Bq_arr     = np.zeros(Nq, dtype=np.float32)

        # k_min<k2<k_max
        delta_k2, _ = pick_field(delta_k, k_min, k_max, nthreads)
        deltax_k2 = IFFT3Dr_f(delta_k2, nthreads); del delta_k2

        # 0<k3<inf
        delta_k3, _ = pick_field(delta_k, k3_min, k3_max, nthreads)
        deltax_k3 = IFFT3Dr_f(delta_k3, nthreads); del delta_k3

        # Loop over soft modes
        for q_ind in (range(Nq)):

            # q_i<k1<qi+dq
            delta_k1, _, k1 = pick_field(delta_k, q_min_arr[q_ind], 
                                            q_min_arr[q_ind]+dq, nthreads,
                                            return_k_bin=True)
            deltax_k1 = IFFT3Dr_f(delta_k1, nthreads); del delta_k1

            # Calculate sum and normalization
            signal = 0.0     
            for i in prange(ng, nogil=True, num_threads=1):
                for j in range(ng):
                    for k in range(ng):
                        signal      += (deltax_k1[i,j,k]*deltax_k2[i,j,k]*deltax_k3[i,j,k])
            
            q_cen_arr[q_ind] = k1*kf
            Bq_arr[q_ind]    = signal*pow(box_len, 6)/pow(ng, 9)

        return q_cen_arr, Bq_arr/Bqnorm_arr, Bqnorm_arr


################################################################################
# Trispectrum routines
# -> Includes routine to compute trispectrum binned in k12 and integrated over
#    k12.
################################################################################
"""
Trispectrum estimator binned in k12
"""
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def compute_tk3d(np.complex64_t[:,:,::1] delta_k, int k1_min, int k1_max, 
                 int k2_min, int k2_max, int k3_min, int k3_max, 
                 int k4_min, int k4_max, int k12_min, int k12_max, 
                 int dk1, int dk2, int dk3, int dk4, int dk12, 
                 double box_len, int nthreads, verbose=False, 
                 Pk_interp=None, np.float32_t[:,:,::1] Pk_mesh=None,
                 np.ndarray[np.float32_t, ndim=1] Tk_disc_PP_arr=None,
                 np.ndarray[np.float32_t, ndim=1] Ntet_arr=None):
    """
    Cython implementation of the binned trispectrum estimator. For an input of 
    minimum and maximum wavenumbers as well as bin spacing, the code computes
    the trispectrum in all possible bins subject to:
    * k1 ≤ k2
    * k2 ≤ k3, (Will's paper uses k1≤k3)
    * k3 ≤ k4,
    * |k1-k2| ≤ k1+k2,
    * |k3-k4| ≤ k3+k4.
    N.B. the inequalities are checked at the bin minima, but probably should be
    imposed at the bin centers.

    Parameters 
    ----------
    delta_k: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field.
    ki_min: int
        Minimum wave number for the ith leg in units of kf=2pi/box_len.
    ki_max: int 
        Maximum wave number for the ith leg in units of kf=2pi/box_len.
    dki: int
        Bin spacing for the ith leg in units of kf.
    box_len: float 
        Length of the box (typically in units of Mpc/h or Mpc). This sets the 
        dimensions of the power spectrum as well as the fundamental frequency.
    nthreads: integer
        Number of threads for the outer loop over grid indices and for FFTs.
    verbose: bool
        Verbosity. Set to true to print progress.
    Pk_interp: function (optional; default=None)
        Function that computes the expected 3D power spectrum of the field.
        This is used to compute the disconnected term in the estimator. This
        must be passed if Pk_mesh is not passed. Function f(x) units should be
        [x]=1/box_len and [f(x)]=box_len^3
    Pk_mesh: (Ngrid, Ngrid, Ngrid//2+1) float array (optional; default=None)
        Array containing the value of the fiducial power spectrum at each kbin
        in Fourier space. This is used to compute the disconnected term of the 
        estimator and is much faster to use than Pk_interp.
    Tk_disc_PP_arr: float array (optional; default=None)
        Array containing the realization-independent contribution to the 
        disconnected estimator. Since this only needs to be computed once,
        it should be passed as input to significantly speed up the code. This 
        is also used to identify the bins where disconnected terms need to be
        computed.
    N_tet_arr: float array (optional; defaut=None)
        Array containing the normalization for the trispectrum. This also needs
        to only be computed once so passing it in as input signifcantly speeds
        up the code.

    Returns
    ----------
    """

    cdef int ng, middle, Nk1, Nk2, Nk3, Nk4, Nk12, i1, i2, i12, i3, i4, i, j, k, ik
    cdef int k1_min_i, k2_min_i, k12_min_i, k3_min_i, k4_min_i, k12_max_i, ki, kj, kk, 
    cdef double kf, start, end,  k12norm
    cdef np.ndarray[np.int16_t, ndim=1] k1_min_arr, k2_min_arr, k3_min_arr
    cdef np.ndarray[np.int16_t, ndim=1] k4_min_arr, k12_min_arr
    cdef np.ndarray[np.float32_t, ndim=1] Tk_tot_arr, Tk_disc_PD_arr
    cdef double Tk_tot, Ntet_tot, Tk_disc_PP_tot, Tk_disc_PD_tot
    cdef np.complex64_t[:,:,::1] Fk_2perm, Fk_4perm, Wk_12, Wk_34, Dk_12, Dk_34
    cdef np.ndarray[np.float32_t, ndim=1] k_cen_uniq     
    cdef np.ndarray[np.complex64_t, ndim=4] Wk_uniq, deltak_uniq
    cdef np.ndarray[np.float32_t, ndim=4] Wr_uniq, deltax_uniq
    cdef np.ndarray[np.complex64_t, ndim=5] Wk_filt_pairs_uniq, deltak_filt_pairs_uniq
    cdef np.ndarray[np.float32_t, ndim=5] Fpr_filt_pairs_uniq, Fdr_filt_pairs_uniq
    cdef np.float32_t[:,:,::1] deltak_sq

    # Compute some constants and useful variables
    ng     = delta_k.shape[0]
    middle = ng//2
    kf     = 2*pi/box_len

    # Initializaiton
    if verbose:
        print("0. INITIALIAZING TRISPECTRUM CALCULATION")
        start = time.time()

    if Pk_mesh is None:
        assert Pk_interp is not None, "Pk_interp or Pk_mesh must be specified"
        if verbose: 
            print("-> No Pk_mesh inputed. Computing from interpolator.")
        Pk_mesh = construct_Pk_mesh(ng, Pk_interp, box_len, nthreads)
    
    if Ntet_arr is None:
        assert Tk_disc_PP_arr is None, "If Ntet_arr is None, then T_disk_PP_arr must also be None"
        need_norm=True
        if verbose: print("-> No normalization specified. Routine will compute normalization and realization independent disconnected term.")
    else:
        assert Tk_disc_PP_arr is not None, "If Ntet_arr is not None, then T_disk_PP_arr must also be None"
        assert len(Tk_disc_PP_arr)==len(Ntet_arr), "Ntet_arr and T_disk_PP_arr must be the same length."
        need_norm = False
        if verbose: print("-> Normalization specified and realization independent disconnected term inputted. Only computing realization dependent terms.")

    if verbose:
        end = time.time()
        print("-> Time taken: {:.3f} seconds".format(end-start))

    # Bin initialization
    if verbose: 
        start = time.time()
        print("1. INITIALIZING BINS")
        

    k1_min_arr  = np.arange(k1_min, k1_max, dk1, dtype=np.int16)
    k2_min_arr  = np.arange(k2_min, k2_max, dk2, dtype=np.int16) 
    k3_min_arr  = np.arange(k3_min, k3_max, dk3, dtype=np.int16)  
    k4_min_arr  = np.arange(k4_min, k4_max, dk4, dtype=np.int16) 
    k12_min_arr = np.arange(k12_min, k12_max, dk12, dtype=np.int16) 

    Nk1  = len(k1_min_arr)
    Nk2  = len(k2_min_arr)
    Nk3  = len(k3_min_arr)
    Nk4  = len(k4_min_arr)
    Nk12 = len(k12_min_arr)

    k_min_mat = []
    for i1 in range(Nk1):
        k1_min_i = k1_min_arr[i1]

        for i2 in range(Nk2):
            k2_min_i = k2_min_arr[i2]
            if k2_min_i < k1_min_i: continue # k1 ≤ k2
            
            for i12 in range(Nk12):
                k12_min_i = k12_min_arr[i12]
                if k12_min_i<abs(k1_min_i-k2_min_i) or k12_min_i>k1_min_i+k2_min_i: continue # |k1-k2| ≤ k1+k2
                
                for i3 in range(Nk3):
                    k3_min_i = k3_min_arr[i3]
                    if k3_min_i < k2_min_i: continue # k2 ≤ k3
                    
                    for i4 in range(Nk4):
                        k4_min_i = k4_min_arr[i4]
                        if k4_min_i < k3_min_i: continue # k3 ≤ k4
                        if k1_min_i==k3_min_i and k2_min_i>k4_min_i: continue # Ordering
                        
                        if k12_min_i<abs(k3_min_i-k4_min_i) or k12_min_i>k3_min_i+k4_min_i: continue # |k3-k4| ≤ k3+k4
                        k_min_mat.append([k1_min_i, k2_min_i, k3_min_i, k4_min_i, k12_min_i])
        
    k_min_mat = np.asarray(k_min_mat)

    # The following line computes all pairs (k_min, k_max) that are used for the windows. 
    # This way we only have to compute them once and significantly reduces # of FFTs.
    # This also determines the indexing W_ind = k_bounds_uniq.index((kmin_i,kmax_i))
    k_bounds_uniq = list(set([(i, i+dk1) for i in k1_min_arr]+
                             [(i, i+dk2) for i in k2_min_arr]+
                             [(i, i+dk3) for i in k3_min_arr]+
                             [(i, i+dk4) for i in k4_min_arr]+
                             [(i, i+dk12) for i in k12_min_arr]))
    Nbin_tot = len(k_min_mat)
    N_uniq   = len(k_bounds_uniq)

    if verbose:
        print("-> Total number of bins: {}".format(Nbin_tot))
        print("-> Number of unique k windows: {}".format(N_uniq))
        end = time.time()
        print("-> Time taken: {:.3f} seconds".format(end-start))

    # No normalization or disconnected term specified. Compute everything!    
    if need_norm:
        # Compute all unique windows for the k bounds
        if verbose: 
            start = time.time()
            print("2. COMPUTING UNIQUE WINDOWED FIELDS AND THEIR IFFTS")

        
        k_cen_uniq  = np.empty(N_uniq, dtype=np.float32)
        k_cen_arr   = np.zeros((Nbin_tot, 5), dtype=np.float32)
        deltak_uniq = np.zeros((N_uniq, ng, ng, ng//2+1), dtype=np.complex64)
        Wk_uniq     = np.zeros((N_uniq, ng, ng, ng//2+1), dtype=np.complex64)
        deltax_uniq = np.zeros((N_uniq, ng, ng, ng), dtype=np.float32)
        Wr_uniq     = np.zeros((N_uniq, ng, ng, ng), dtype=np.float32)
        
        # Define Fourier and real space fields
        for i in (range(N_uniq)):
            deltak_i, Wk_i, k_cen_uniq[i] = pick_field(delta_k, *k_bounds_uniq[i], nthreads, return_k_bin=True) 
            Wk_uniq[i]     = np.asarray(Wk_i)
            Wr_uniq[i]     = IFFT3Dr_f(Wk_uniq[i], nthreads)
            deltak_uniq[i] = np.asarray(deltak_i)
            deltax_uniq[i] = IFFT3Dr_f(deltak_uniq[i], nthreads)
        
        # Compute products of windows 
        if verbose:
            end = time.time()
            print("-> Time taken: {:.3f} seconds".format(end-start))
            print("3. COMPUTING UNIQUE PRODUCTS OF WINDOWS AND THEIR FFTS")
            start = time.time()

        # Terms for total estimator
        Wk_filt_pairs_uniq     = np.zeros((N_uniq, N_uniq, ng, ng, ng//2+1), 
                                        dtype=np.complex64)
        deltak_filt_pairs_uniq = np.zeros((N_uniq, N_uniq, ng, ng, ng//2+1), 
                                        dtype=np.complex64)

        # Terms for disconnected estimator
        Fpr_filt_pairs_uniq = np.zeros((N_uniq, N_uniq, ng, ng, ng), 
                                    dtype=np.float32)
        Fdr_filt_pairs_uniq = np.zeros((N_uniq, N_uniq, ng, ng, ng), 
                                    dtype=np.float32)
        deltak_sq = np.zeros((ng,ng,ng//2+1), dtype=np.float32)

        for i in prange(ng, nogil=True, num_threads=1):
            for j in range(ng):
                for k in range(ng//2+1):
                    deltak_sq[i,j,k] = pow(delta_k[i,j,k].real,2)+pow(delta_k[i,j,k].imag, 2)

        for i in range(N_uniq):
            for j in range(i, N_uniq):
                Wk_filt_pairs_uniq[i,j] = FFT3Dr_f(Wr_uniq[i]*Wr_uniq[j], nthreads)
                deltak_filt_pairs_uniq[i,j] = FFT3Dr_f(deltax_uniq[i]*deltax_uniq[j], nthreads)
                Fpr_filt_pairs_uniq[i,j] = IFFT3Dr_f(np.asarray(Wk_uniq[i])*
                                                    np.asarray(Wk_uniq[j])*
                                                    Pk_mesh, nthreads)
                Fdr_filt_pairs_uniq[i,j] = IFFT3Dr_f(np.asarray(Wk_uniq[i])*
                                                    np.asarray(Wk_uniq[j])*
                                                    deltak_sq, nthreads)
        
        del deltak_sq, deltax_uniq, Wr_uniq, Pk_mesh 
                
        # Compute trispectrum for every bin
        if verbose:
            end = time.time()
            print("-> Time taken: {:.3f} seconds".format(end-start)) 
            print("4. COMPUTING BINNED TRISPECTRUM")
            start = time.time()

        

        # Realization dependent terms
        Tk_tot_arr = np.zeros(Nbin_tot, dtype=np.float32)
        Tk_disc_PD_arr = np.zeros_like(Tk_tot_arr)

        # Realization independent terms (only compute once)
        Ntet_arr = np.zeros_like(Tk_tot_arr)
        Tk_disc_PP_arr = np.zeros_like(Tk_tot_arr)


        for ik, k_min in enumerate(k_min_mat):
            # Get bin indices
            k1_min_i, k2_min_i, k3_min_i, k4_min_i, k12_min_i = k_min
            k1_ind = k_bounds_uniq.index((k1_min_i, k1_min_i+dk1))
            k2_ind = k_bounds_uniq.index((k2_min_i, k2_min_i+dk2))
            k3_ind = k_bounds_uniq.index((k3_min_i, k3_min_i+dk3))
            k4_ind = k_bounds_uniq.index((k4_min_i, k4_min_i+dk4))
            k12_ind = k_bounds_uniq.index((k12_min_i, k12_min_i+dk12))
            k12_max_i = k12_min_i+dk12

            k_cen_arr[ik] = np.asarray([k_cen_uniq[k1_ind], k_cen_uniq[k2_ind], 
                                        k_cen_uniq[k3_ind], k_cen_uniq[k4_ind],
                                        k_cen_uniq[k12_ind]])*kf
        
            # Compute product fields for total estimator.
            # -> Note that min and max are used to ensure *_uniq[i,j] has i<j
            Wk_12 = Wk_filt_pairs_uniq[min(k1_ind,k2_ind), max(k1_ind, k2_ind)]
            Wk_34 = Wk_filt_pairs_uniq[min(k3_ind,k4_ind), max(k3_ind, k4_ind)]
            Dk_12 = deltak_filt_pairs_uniq[min(k1_ind,k2_ind), max(k1_ind, k2_ind)]
            Dk_34 = deltak_filt_pairs_uniq[min(k3_ind,k4_ind), max(k3_ind, k4_ind)]

            # Compute FFTs for this bin used in disconnected estimator
            # -> There should be more efficient way to do this than having to FFT in loop.
        
            # Realization independent disconnected term
            Fk_2perm = FFT3Dr_f(Fpr_filt_pairs_uniq[min(k1_ind, k3_ind), max(k1_ind, k3_ind)]*
                                Fpr_filt_pairs_uniq[min(k2_ind, k4_ind), max(k2_ind, k4_ind)]+
                                Fpr_filt_pairs_uniq[min(k1_ind, k4_ind), max(k1_ind, k4_ind)]*
                                Fpr_filt_pairs_uniq[min(k2_ind, k3_ind), max(k2_ind, k3_ind)], nthreads)

            
            # Realization dependent disconnected term
            Fk_4perm = FFT3Dr_f(Fpr_filt_pairs_uniq[min(k1_ind, k3_ind), max(k1_ind, k3_ind)]*
                                Fdr_filt_pairs_uniq[min(k2_ind, k4_ind), max(k2_ind, k4_ind)]+
                                Fdr_filt_pairs_uniq[min(k1_ind, k3_ind), max(k1_ind, k3_ind)]*
                                Fpr_filt_pairs_uniq[min(k2_ind, k4_ind), max(k2_ind, k4_ind)]+
                                Fpr_filt_pairs_uniq[min(k1_ind, k4_ind), max(k1_ind, k4_ind)]*
                                Fdr_filt_pairs_uniq[min(k2_ind, k3_ind), max(k2_ind, k3_ind)]+
                                Fdr_filt_pairs_uniq[min(k1_ind, k4_ind), max(k1_ind, k4_ind)]*
                                Fpr_filt_pairs_uniq[min(k2_ind, k3_ind), max(k2_ind, k3_ind)], nthreads)

            # Compute Fourier space sum.
            Tk_tot = 0.0; Ntet_tot = 0.0; Tk_disc_PP_tot = 0.0; Tk_disc_PD_tot = 0.0
            for i in range(ng):
                ki = (i-ng if (i>middle) else i)
                if(ki>k12_max_i): continue
                
                for j in range(ng):
                    kj = (j-ng if (j>middle) else j)
                    
                    if(kj>k12_max_i): continue
                    for k in range(middle+1): 
                            kk = (k-ng if (k>middle) else k)
                            if(kk>k12_max_i): continue

                            # kz=0 and kz=middle planes should only be counted once
                            if kk==0 or (kk==middle and ng%2==0):
                                if ki<0: continue
                                elif ki==0 or (ki==middle and ng%2==0):
                                    if kj<0.0: continue
                            
                            # Select k bin
                            k12norm = sqrt(ki*ki+kj*kj+kk*kk)
                            if(k12norm<k12_min_i or k12norm>=k12_max_i): continue

                            Tk_tot   += np.real(Dk_12[i,j,k]*np.conj(Dk_34[i,j,k]))
                            Ntet_tot += np.real(Wk_12[i,j,k]*np.conj(Wk_34[i,j,k]))
                            Tk_disc_PP_tot += np.real(Fk_2perm[i,j,k])
                            Tk_disc_PD_tot += np.real(Fk_4perm[i,j,k])      

            Tk_tot_arr[ik]     = Tk_tot
            Ntet_arr[ik]       = Ntet_tot
            Tk_disc_PP_arr[ik] = Tk_disc_PP_tot
            Tk_disc_PD_arr[ik] = Tk_disc_PD_tot
        
        # Normalize output
        Tk_tot_arr     = Tk_tot_arr/Ntet_arr*pow(box_len/ng, 9)/pow(ng, 3)
        Tk_disc_PP_arr = Tk_disc_PP_arr/Ntet_arr*pow(box_len/ng, 3)
        Tk_disc_PD_arr = Tk_disc_PD_arr/Ntet_arr*pow(box_len/ng,6)/(2*pow(ng,3))
        
        if verbose:
            end = time.time()
            print("-> Time taken: {:.3f} seconds".format(end-start)) 

        return k_cen_arr, Tk_tot_arr, Tk_disc_PP_arr, Tk_disc_PD_arr, Ntet_arr

    else:
        assert len(Ntet_arr)==len(k_min_mat), "Normalization has wrong length."
        
        # Compute all unique windows for the k bounds
        if verbose: 
            start = time.time()
            print("2. COMPUTING UNIQUE WINDOWED FIELDS AND THEIR IFFTS")

        k_cen_uniq  = np.empty(N_uniq, dtype=np.float32)
        k_cen_arr   = np.zeros((Nbin_tot, 5), dtype=np.float32)
        deltak_uniq = np.zeros((N_uniq, ng, ng, ng//2+1), dtype=np.complex64)
        Wk_uniq     = np.zeros((N_uniq, ng, ng, ng//2+1), dtype=np.complex64)
        deltax_uniq = np.zeros((N_uniq, ng, ng, ng), dtype=np.float32)

        # Define Fourier and real space fields
        for i in (range(N_uniq)):
            deltak_i, Wk_i, k_cen_uniq[i] = pick_field(delta_k, *k_bounds_uniq[i], nthreads, return_k_bin=True) 
            Wk_uniq[i]     = np.asarray(Wk_i)
            deltak_uniq[i] = np.asarray(deltak_i)
            deltax_uniq[i] = IFFT3Dr_f(deltak_uniq[i], nthreads)
        
        # Compute products of windows 
        if verbose:
            end = time.time()
            print("-> Time taken: {:.3f} seconds".format(end-start))
            print("3. COMPUTING UNIQUE PRODUCTS OF WINDOWS AND THEIR FFTS")
            start = time.time()

        # Terms for total estimator
        Wk_filt_pairs_uniq     = np.zeros((N_uniq, N_uniq, ng, ng, ng//2+1), 
                                        dtype=np.complex64)
        deltak_filt_pairs_uniq = np.zeros((N_uniq, N_uniq, ng, ng, ng//2+1), 
                                        dtype=np.complex64)

        # Terms for disconnected estimator
        Fpr_filt_pairs_uniq = np.zeros((N_uniq, N_uniq, ng, ng, ng), 
                                    dtype=np.float32)
        Fdr_filt_pairs_uniq = np.zeros((N_uniq, N_uniq, ng, ng, ng), 
                                    dtype=np.float32)
        deltak_sq = np.zeros((ng,ng,ng//2+1), dtype=np.float32)

        for i in prange(ng, nogil=True, num_threads=1):
            for j in range(ng):
                for k in range(ng//2+1):
                    deltak_sq[i,j,k] = pow(delta_k[i,j,k].real,2)+pow(delta_k[i,j,k].imag, 2)

        for i in range(N_uniq):
            for j in range(i, N_uniq):
                deltak_filt_pairs_uniq[i,j] = FFT3Dr_f(deltax_uniq[i]*deltax_uniq[j], nthreads)
                Fpr_filt_pairs_uniq[i,j] = IFFT3Dr_f(np.asarray(Wk_uniq[i])*
                                                    np.asarray(Wk_uniq[j])*
                                                    Pk_mesh, nthreads)
                Fdr_filt_pairs_uniq[i,j] = IFFT3Dr_f(np.asarray(Wk_uniq[i])*
                                                    np.asarray(Wk_uniq[j])*
                                                    deltak_sq, nthreads)
        
        del deltak_sq, deltax_uniq, Pk_mesh 
                
        # Compute trispectrum for every bin
        if verbose:
            end = time.time()
            print("-> Time taken: {:.3f} seconds".format(end-start)) 
            print("4. COMPUTING BINNED TRISPECTRUM")
            start = time.time()

        # Realization dependent terms
        Tk_tot_arr = np.zeros(Nbin_tot, dtype=np.float32)
        Tk_disc_PD_arr = np.zeros_like(Tk_tot_arr)

        for ik, k_min in enumerate(k_min_mat):
            # Get bin indices
            k1_min_i, k2_min_i, k3_min_i, k4_min_i, k12_min_i = k_min
            k1_ind = k_bounds_uniq.index((k1_min_i, k1_min_i+dk1))
            k2_ind = k_bounds_uniq.index((k2_min_i, k2_min_i+dk2))
            k3_ind = k_bounds_uniq.index((k3_min_i, k3_min_i+dk3))
            k4_ind = k_bounds_uniq.index((k4_min_i, k4_min_i+dk4))
            k12_ind = k_bounds_uniq.index((k12_min_i, k12_min_i+dk4))
            k12_max_i = k12_min_i+dk12
            
            k_cen_arr[ik] = np.asarray([k_cen_uniq[k1_ind], k_cen_uniq[k2_ind], 
                                        k_cen_uniq[k3_ind], k_cen_uniq[k4_ind],
                                        k_cen_uniq[k12_ind]])*kf

            # Compute product fields for total estimator.
            # -> Note that min and max are used to ensure *_uniq[i,j] has i<j
            Dk_12 = deltak_filt_pairs_uniq[min(k1_ind,k2_ind), max(k1_ind, k2_ind)]
            Dk_34 = deltak_filt_pairs_uniq[min(k3_ind,k4_ind), max(k3_ind, k4_ind)]

            # Totals from Fourier space sum
            Tk_tot = 0.0; Tk_disc_PD_tot = 0.0

            # Compute realization-dependent disconnected term if non-zero
            if Tk_disc_PP_arr[ik]!=0:
                Fk_4perm = FFT3Dr_f(Fpr_filt_pairs_uniq[min(k1_ind, k3_ind), max(k1_ind, k3_ind)]*
                                    Fdr_filt_pairs_uniq[min(k2_ind, k4_ind), max(k2_ind, k4_ind)]+
                                    Fdr_filt_pairs_uniq[min(k1_ind, k3_ind), max(k1_ind, k3_ind)]*
                                    Fpr_filt_pairs_uniq[min(k2_ind, k4_ind), max(k2_ind, k4_ind)]+
                                    Fpr_filt_pairs_uniq[min(k1_ind, k4_ind), max(k1_ind, k4_ind)]*
                                    Fdr_filt_pairs_uniq[min(k2_ind, k3_ind), max(k2_ind, k3_ind)]+
                                    Fdr_filt_pairs_uniq[min(k1_ind, k4_ind), max(k1_ind, k4_ind)]*
                                    Fpr_filt_pairs_uniq[min(k2_ind, k3_ind), max(k2_ind, k3_ind)], nthreads)
                
                # Compute sum in Fourier space
                for i in range(ng): 
                    ki = (i-ng if (i>middle) else i)
                    if(ki>k12_max_i): continue
                    
                    for j in range(ng):
                        kj = (j-ng if (j>middle) else j)
                        
                        if(kj>k12_max_i): continue
                        for k in range(middle+1): 
                                kk = (k-ng if (k>middle) else k)
                                if(kk>k12_max_i): continue

                                # kz=0 and kz=middle planes should only be counted once
                                if kk==0 or (kk==middle and ng%2==0):
                                    if ki<0: continue
                                    elif ki==0 or (ki==middle and ng%2==0):
                                        if kj<0.0: continue
                                
                                # Select k bin
                                k12norm = sqrt(ki*ki+kj*kj+kk*kk)
                                if(k12norm<k12_min_i or k12norm>=k12_max_i): continue

                                Tk_tot   += np.real(Dk_12[i,j,k]*np.conj(Dk_34[i,j,k]))
                                Tk_disc_PD_tot += np.real(Fk_4perm[i,j,k])

            else:
                # Compute sum in Fourier space
                for i in range(ng): 
                    ki = (i-ng if (i>middle) else i)
                    if(ki>k12_max_i): continue
                    
                    for j in range(ng):
                        kj = (j-ng if (j>middle) else j)
                        
                        if(kj>k12_max_i): continue
                        for k in range(middle+1): 
                                kk = (k-ng if (k>middle) else k)
                                if(kk>k12_max_i): continue

                                # kz=0 and kz=middle planes should only be counted once
                                if kk==0 or (kk==middle and ng%2==0):
                                    if ki<0: continue
                                    elif ki==0 or (ki==middle and ng%2==0):
                                        if kj<0.0: continue
                                
                                # Select k bin
                                k12norm = sqrt(ki*ki+kj*kj+kk*kk)
                                if(k12norm<k12_min_i or k12norm>=k12_max_i): continue

                                Tk_tot   += np.real(Dk_12[i,j,k]*np.conj(Dk_34[i,j,k]))


            Tk_tot_arr[ik]     = Tk_tot
            Tk_disc_PD_arr[ik] = Tk_disc_PD_tot
        
        # Normalize output
        Tk_tot_arr     = Tk_tot_arr/Ntet_arr*pow(box_len/ng, 9)/pow(ng, 3)
        Tk_disc_PD_arr = Tk_disc_PD_arr/Ntet_arr*pow(box_len/ng,6)/(2*pow(ng,3))
        
        if verbose:
            end = time.time()
            print("-> Time taken: {:.3f} seconds".format(end-start)) 

        return k_cen_arr, Tk_tot_arr, Tk_disc_PP_arr, Tk_disc_PD_arr, Ntet_arr


"""
Trispectrum estimator binned in k12 and averaged over k1min<k1<k1max,
0<k2<inf, k3min<k3<k3max, 0<k4<inf.
"""
@cython.boundscheck(False)
@cython.cdivision(False)
@cython.wraparound(False)
def compute_tk3d_ang_avg(np.complex64_t[:,:,::1] delta_k, int k1_min, int k1_max, 
                         int k3_min, int k3_max,  int k12_min, int k12_max,
                         int dk12, double box_len, int nthreads, verbose=False, 
                         Pk_interp=None, np.float32_t[:,:,::1] Pk_mesh=None,
                         np.ndarray[np.float32_t, ndim=1] Tk_disc_PP_arr=None,
                         np.ndarray[np.float32_t, ndim=1] Ntet_arr=None):
    """
    Cython implementation of the binned trispectrum estimator. For an input of 
    minimum and maximum wavenumbers as well as bin spacing, the code computes
    the trispectrum in all possible bins subject to:
    * k1 ≤ k2
    * k2 ≤ k3, (Will's paper uses k1≤k3)
    * k3 ≤ k4,
    * |k1-k2| ≤ k1+k2,
    * |k3-k4| ≤ k3+k4.
    N.B. the inequalities are checked at the bin minima, but probably should be
    imposed at the bin centers.
    UPDATE

    Parameters 
    ----------
    delta_k: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field.
    ki_min: int
        Minimum wave number for the ith leg in units of kf=2pi/box_len.
    ki_max: int 
        Maximum wave number for the ith leg in units of kf=2pi/box_len.
    dki: int
        Bin spacing for the ith leg in units of kf.
    box_len: float 
        Length of the box (typically in units of Mpc/h or Mpc). This sets the 
        dimensions of the power spectrum as well as the fundamental frequency.
    nthreads: integer
        Number of threads for the outer loop over grid indices and for FFTs.
    verbose: bool
        Verbosity. Set to true to print progress.
    Pk_interp: function (optional; default=None)
        Function that computes the expected 3D power spectrum of the field.
        This is used to compute the disconnected term in the estimator. This
        must be passed if Pk_mesh is not passed. Function f(x) units should be
        [x]=1/box_len and [f(x)]=box_len^3
    Pk_mesh: (Ngrid, Ngrid, Ngrid//2+1) float array (optional; default=None)
        Array containing the value of the fiducial power spectrum at each kbin
        in Fourier space. This is used to compute the disconnected term of the 
        estimator and is much faster to use than Pk_interp.
    Tk_disc_PP_arr: float array (optional; default=None)
        Array containing the realization-independent contribution to the 
        disconnected estimator. Since this only needs to be computed once,
        it should be passed as input to significantly speed up the code. This 
        is also used to identify the bins where disconnected terms need to be
        computed.
    N_tet_arr: float array (optional; defaut=None)
        Array containing the normalization for the trispectrum. This also needs
        to only be computed once so passing it in as input signifcantly speeds
        up the code.

    Returns
    ----------
    """

    cdef int ng, middle, Nk12, i1, i2, i12, i3, i4, i, j, k, ik
    cdef int k12_min_i, k12_max_i, ki, kj, kk, 
    cdef double kf, start, end,  k12norm, k12tot, N12
    cdef np.ndarray[np.int16_t, ndim=1] k12_min_arr
    cdef np.ndarray[np.float32_t, ndim=1] Tk_tot_arr, Tk_disc_PD_arr
    cdef double Tk_tot, Ntet_tot, Tk_disc_PP_tot, Tk_disc_PD_tot
    cdef np.complex64_t[:,:,::1] Fk_2perm, Fk_4perm, Wk_12, Wk_34, Dk_12, Dk_34
    cdef np.ndarray[np.float32_t, ndim=1] k_cen_uniq     
    cdef np.float32_t[:,:,::1] deltak_sq

    # Compute some constants and useful variables
    ng     = delta_k.shape[0]
    middle = ng//2
    kf     = 2*pi/box_len

    # Initializaiton
    if verbose:
        print("0. INITIALIAZING TRISPECTRUM CALCULATION")
        start = time.time()

    if Pk_mesh is None:
        assert Pk_interp is not None, "Pk_interp or Pk_mesh must be specified"
        if verbose: 
            print("-> No Pk_mesh inputed. Computing from interpolator.")
        Pk_mesh = construct_Pk_mesh(ng, Pk_interp, box_len, nthreads)
    
    if Ntet_arr is None:
        assert Tk_disc_PP_arr is None, "If Ntet_arr is None, then T_disk_PP_arr must also be None"
        need_norm=True
        if verbose: print("-> No normalization specified. Routine will compute normalization and realization independent disconnected term.")
    else:
        assert Tk_disc_PP_arr is not None, "If Ntet_arr is not None, then T_disk_PP_arr must also be None"
        assert len(Tk_disc_PP_arr)==len(Ntet_arr), "Ntet_arr and T_disk_PP_arr must be the same length."
        need_norm = False
        if verbose: print("-> Normalization specified and realization independent disconnected term inputted. Only computing realization dependent terms.")

    if verbose:
        end = time.time()
        print("-> Time taken: {:.3f} seconds".format(end-start))

    # Bin initialization
    if verbose: 
        start = time.time()
        print("1. INITIALIZING BINS")
        
    k12_min_arr = np.arange(k12_min, k12_max, dk12, dtype=np.int16) 
    Nk12 = len(k12_min_arr)     
    k12_cen_arr = np.zeros(Nk12, dtype=np.float32)   

    Nbin_tot = Nk12

    # No normalization or disconnected term specified. Compute everything!    
    if need_norm:
        # Compute all unique windows for the k bounds
        deltak_1, Wk_1, k1_cen = pick_field(delta_k, k1_min, k1_max, nthreads,
                                            return_k_bin=True) 
        deltak_3, Wk_3, k3_cen = pick_field(delta_k, k3_min, k3_max, nthreads,
                                            return_k_bin=True)
        Wk = np.ones_like(Wk_1)

        deltak_sq = np.zeros((ng,ng,ng//2+1), dtype=np.float32)

        for i in prange(ng, nogil=True, num_threads=1):
            for j in range(ng):
                for k in range(ng//2+1):
                    deltak_sq[i,j,k] = pow(delta_k[i,j,k].real,2)+pow(delta_k[i,j,k].imag, 2)

        # Compute real space fields
        Wr_1 = IFFT3Dr_f(Wk_1, nthreads)
        Wr_3 = IFFT3Dr_f(Wk_3, nthreads)
        Wr   = IFFT3Dr_f(Wk, nthreads)

        Wk = np.asarray(Wk)
        Wk_1 = np.asarray(Wk_1)
        Wk_3 = np.asarray(Wk_3)

        deltax_k1 = IFFT3Dr_f(deltak_1, nthreads)
        deltax_k3 = IFFT3Dr_f(deltak_3, nthreads)
        deltar    = IFFT3Dr_f(delta_k, nthreads)

        Wk_12 = FFT3Dr_f(Wr_1*Wr, nthreads)
        Wk_34 = FFT3Dr_f(Wr_3*Wr, nthreads)
        Dk_12 = FFT3Dr_f(deltax_k1*deltar, nthreads)
        Dk_34 = FFT3Dr_f(deltax_k3*deltar, nthreads)

        del Wr_1, Wr_3, Wr, deltax_k1, deltax_k3, deltar

        # Realization independent disconnected term. Note that Wk_2=Wk_4=I
        Fpr_13 = IFFT3Dr_f(Wk_1*Wk_3, nthreads)
        Fpr_24 = IFFT3Dr_f(Wk*Pk_mesh, nthreads) # Multiply by Wk=Ik to type cast
        Fpr_14 = IFFT3Dr_f(Wk_1*Pk_mesh, nthreads)
        Fpr_23 = IFFT3Dr_f(Wk_3*Pk_mesh, nthreads)
        Fk_2perm = FFT3Dr_f(Fpr_13*Fpr_24+Fpr_14*Fpr_23, nthreads)

        del Pk_mesh,

        # Realization dependent disconnected term. Note that Wk_2=Wk_4=I
        Fdr_13   = IFFT3Dr_f(Wk_1*Wk_3*deltak_sq, nthreads)
        Fdr_24   = IFFT3Dr_f(Wk*deltak_sq, nthreads)
        Fdr_14   = IFFT3Dr_f(Wk_1*deltak_sq, nthreads)
        Fdr_23   = IFFT3Dr_f(Wk_3*deltak_sq, nthreads)
        Fk_4perm = FFT3Dr_f(Fpr_13*Fdr_24+Fdr_13*Fpr_24+
                            Fpr_14*Fdr_23+Fdr_14*Fpr_23, nthreads)

        del deltak_sq, Fdr_13, Fdr_24, Fdr_14, Fdr_23, Wk_1, Wk_3

        # Realization dependent terms
        Tk_tot_arr = np.zeros(Nbin_tot, dtype=np.float32)
        Tk_disc_PD_arr = np.zeros_like(Tk_tot_arr)

        # Realization independent terms (only compute once)
        Ntet_arr = np.zeros_like(Tk_tot_arr)
        Tk_disc_PP_arr = np.zeros_like(Tk_tot_arr)

        for ik, k12_min_i in enumerate(k12_min_arr): 
            k12_max_i = k12_min_i+dk12

            # Compute Fourier space sum.
            Tk_tot = 0.0; Ntet_tot = 0.0; Tk_disc_PP_tot = 0.0; Tk_disc_PD_tot = 0.0
            k12tot = 0.0; Ntot=0.0
            for i in range(ng): 
                ki = (i-ng if (i>middle) else i)
                if(ki>k12_max_i): continue
                
                for j in range(ng):
                    kj = (j-ng if (j>middle) else j)
                    
                    if(kj>k12_max_i): continue
                    for k in range(middle+1): 
                            kk = (k-ng if (k>middle) else k)
                            if(kk>k12_max_i): continue

                            # kz=0 and kz=middle planes should only be counted once
                            if kk==0 or (kk==middle and ng%2==0):
                                if ki<0: continue
                                elif ki==0 or (ki==middle and ng%2==0):
                                    if kj<0.0: continue
                            
                            # Select k bin
                            k12norm = sqrt(ki*ki+kj*kj+kk*kk)
                            if(k12norm<k12_min_i or k12norm>=k12_max_i): continue

                            Tk_tot   += np.real(Dk_12[i,j,k]*np.conj(Dk_34[i,j,k]))
                            Ntet_tot += np.real(Wk_12[i,j,k]*np.conj(Wk_34[i,j,k]))
                            Tk_disc_PP_tot += np.real(Fk_2perm[i,j,k])
                            Tk_disc_PD_tot += np.real(Fk_4perm[i,j,k])

                            k12tot += k12norm
                            Ntot   += 1      

            k12_cen_arr[ik]    = kf*k12tot/Ntot
            Tk_tot_arr[ik]     = Tk_tot
            Ntet_arr[ik]       = Ntet_tot
            Tk_disc_PP_arr[ik] = Tk_disc_PP_tot
            Tk_disc_PD_arr[ik] = Tk_disc_PD_tot
        
        # Normalize output
        Tk_tot_arr     = Tk_tot_arr/Ntet_arr*pow(box_len/ng, 9)/pow(ng, 3)
        Tk_disc_PP_arr = Tk_disc_PP_arr/Ntet_arr*pow(box_len/ng, 3)
        Tk_disc_PD_arr = Tk_disc_PD_arr/Ntet_arr*pow(box_len/ng,6)/(2*pow(ng,3))

        return k12_cen_arr, Tk_tot_arr, Tk_disc_PP_arr, Tk_disc_PD_arr, Ntet_arr



"""
Integrated trispectrum estimator binned integrated over k12
"""
def compute_Itk3d_py(delta_k, k1_min, k1_max, k2_min, k2_max, k3_min,
                     k3_max, k4_min, k4_max, dk1, dk2, dk3, dk4, box_len, 
                     nthreads, verbose=False, Pk_interp=None, Pk_mesh=None):
    """
    Computes 3D integrated trispectrum for ki_min<ki<ki_max with i from 1 to 4.
    Only returns non-trivial components which are specified by
        * k1 ≤ k2
        * k2 ≤ k3, (Will's paper uses k1≤k3)
        * k3 ≤ k4,
        * |k1-k2| ≤ k1+k2,
        * |k3-k4| ≤ k3+k4.
    N.B.: these are imposed at the minimum values of the bins, but should
          should be imposed for the bin centers.

    Parameters 
    ----------
    delta_k: (Ngrid, Ngrid, Ngrid//2+1) complex array
        Fourier space density field.
    ki_min: int
        Minimum wave number for the ith leg in units of kf=2pi/box_len.
    ki_max: int 
        Maximum wave number for the ith leg in units of kf=2pi/box_len.
    dki: int
        Bin spacing for the ith leg in units of kf.

    box_len: float 
        Length of the box (typically in units of Mpc/h or Mpc). This sets the 
        dimensions of the power spectrum as well as the fundamental frequency.
    nthreads: integer
        Number of threads for the outer loop over grid indices and for FFTs.

    Returns
    -------
    ki_arr: float array
        Bin center for each mode. This is computed as the average wavenumber
        of all Fourier modes in this bin.
    Tk_arr: float array
        3D trispectrum in units of [Vol]^3~(Mpc/h)^9
    Ntet_arr: float array
        Number of tetrahedra in the Fourier bin. This is currently wrong!
    """
    # Compute some constants and useful variables
    ng = delta_k.shape[0]
    middle = ng//2
    kf = 2*pi/box_len
    RFFT_mask = np.asarray(get_RFFT_mask(ng, nthreads))
    deltak_sq = np.asarray((np.abs(delta_k).real)**2)

    # Compute power spectrum mesh if not computed
    if Pk_mesh is None:
        assert Pk_interp is not None, "Pk_interp or Pk_mesh must be specified"

        if verbose: 
            print("0. No Pk_mesh inputed. Computing from interpolator.")
            start = time.time()

        Pk_mesh = construct_Pk_mesh(ng, Pk_interp, box_len, nthreads)

        if verbose:
            end = time.time()
            print("-> Time taken: {:.3f} seconds".format(end-start))

    # Bin initialization
    if verbose: 
        print("1. INITIALIZING BINS")
        start = time.time()

    k1_min_arr  = np.arange(k1_min, k1_max, dk1, dtype=np.int16)
    k2_min_arr  = np.arange(k2_min, k2_max, dk2, dtype=np.int16) 
    k3_min_arr  = np.arange(k3_min, k3_max, dk3, dtype=np.int16)  
    k4_min_arr  = np.arange(k4_min, k4_max, dk4, dtype=np.int16) 

    Nk1         = len(k1_min_arr)
    Nk2         = len(k2_min_arr)
    Nk3         = len(k3_min_arr)
    Nk4         = len(k4_min_arr)

    k_min_mat = []
    for i1 in range(Nk1):
        k1_min_i = k1_min_arr[i1]

        for i2 in range(Nk2):
            k2_min_i = k2_min_arr[i2]
            if k2_min_i < k1_min_i: continue # k1 ≤ k2
            if k1_min_i+k2_min_i<abs(k1_min_i-k2_min_i): continue #|k1-k2| ≤ k1+k2,
                
            for i3 in range(Nk3):
                k3_min_i = k3_min_arr[i3]
                if k3_min_i < k2_min_i: continue # k2 ≤ k3
                
                for i4 in range(Nk4):
                    k4_min_i = k4_min_arr[i4]
                    if k4_min_i < k3_min_i: continue # k3 ≤ k4
                    if k3_min_i+k4_min_i<abs(k3_min_i-k4_min_i): continue #|k3-k4| ≤ k3+k4, 
                    if k1_min_i==k3_min_i and k2_min_i>k4_min_i: continue # Ordering

                    k_min_mat.append([k1_min_i, k2_min_i, k3_min_i, k4_min_i])
    
    k_min_mat = np.asarray(k_min_mat)

    # The following line computes all pairs (k_min, k_max) that are used for the windows. 
    # This way we only have to compute them once and significantly reduces # of FFTs.
    # This also determines the indexing W_ind = k_bounds_uniq.index((kmin_i,kmax_i))
    k_bounds_uniq = list(set([(i, i+dk1) for i in k1_min_arr]+
                            [(i, i+dk2) for i in k2_min_arr]+
                            [(i, i+dk3) for i in k3_min_arr]+
                            [(i, i+dk4) for i in k4_min_arr]))
    Nbin_tot = len(k_min_mat)
    N_uniq   = len(k_bounds_uniq)

    if verbose:
        print("-> Total number of bins: {}".format(Nbin_tot))
        print("-> Number of unique k windows: {}".format(N_uniq))
        end = time.time()
        print("-> Time taken: {:.3f} seconds".format(end-start))

    # Compute all unique windows for the k bounds
    if verbose: 
        start = time.time()
        print("2. COMPUTING UNIQUE WINDOWED FIELDS AND THEIR IFFTS")

    deltak_uniq = np.empty(N_uniq, dtype=object) 
    Wk_uniq     = np.empty(N_uniq, dtype=object)
    k_cen_uniq  = np.empty(N_uniq, dtype=float)

    for i in (range(N_uniq)):
        deltak_uniq[i], Wk_uniq[i], k_cen_uniq[i] = pick_field(delta_k, *k_bounds_uniq[i], nthreads, return_k_bin=True) 

    Wr_uniq     = [IFFT3Dr_f(np.asarray(Wk_uniq[i])+0.0j, nthreads) for i in range(N_uniq)]
    deltax_uniq = [IFFT3Dr_f(np.asarray(deltak_uniq[i])+0.0j, nthreads) for i in range(N_uniq)]

    # Comptue trispectrum looping over bins
    k_cen_arr      = np.zeros((Nbin_tot, 4), dtype=float)
    Ntet_arr       = np.zeros(Nbin_tot, dtype=float)
    Tk_tot_arr     = np.zeros_like(Ntet_arr)
    Tk_disc_DD_arr = np.zeros_like(Ntet_arr)
    Tk_disc_PD_arr = np.zeros_like(Ntet_arr)
    Tk_disc_PP_arr = np.zeros_like(Ntet_arr)
    Ntet_PD_arr    = np.zeros_like(Ntet_arr)

    for i, k_min in enumerate(k_min_mat):
        # Get bin indices
        k1_min_i, k2_min_i, k3_min_i, k4_min_i = k_min
        k1_ind = k_bounds_uniq.index((k1_min_i, k1_min_i+dk1))
        k2_ind = k_bounds_uniq.index((k2_min_i, k2_min_i+dk2))
        k3_ind = k_bounds_uniq.index((k3_min_i, k3_min_i+dk3))
        k4_ind = k_bounds_uniq.index((k4_min_i, k4_min_i+dk4))

        k_cen_arr[i] = [k_cen_uniq[k1_ind], k_cen_uniq[k2_ind], 
                        k_cen_uniq[k3_ind], k_cen_uniq[k4_ind]]

        # Compute total and normalization: connected+disconnected
        Tk_tot_arr[i] = np.sum(deltax_uniq[k1_ind]*deltax_uniq[k2_ind]*
                               deltax_uniq[k3_ind]*deltax_uniq[k4_ind])
        Ntet_arr[i] = np.sum(Wr_uniq[k1_ind]*Wr_uniq[k2_ind]*
                             Wr_uniq[k3_ind]*Wr_uniq[k4_ind])

        # Compute disconnected (delta^2)^2 term
        Tk_disc_DD_arr[i] = ((np.sum(deltax_uniq[k1_ind]*deltax_uniq[k2_ind])*
                              np.sum(deltax_uniq[k3_ind]*deltax_uniq[k4_ind]))+
                             (np.sum(deltax_uniq[k1_ind]*deltax_uniq[k3_ind])*
                              np.sum(deltax_uniq[k2_ind]*deltax_uniq[k4_ind]))+
                             (np.sum(deltax_uniq[k1_ind]*deltax_uniq[k4_ind])*
                              np.sum(deltax_uniq[k2_ind]*deltax_uniq[k3_ind])))

        # Compute disconnected P*delta^2 term and normalization
        Wk1 = np.asarray(Wk_uniq[k1_ind])
        Wk2 = np.asarray(Wk_uniq[k2_ind])
        Wk3 = np.asarray(Wk_uniq[k3_ind])
        Wk4 = np.asarray(Wk_uniq[k4_ind])

        DK12 = np.sum(RFFT_mask*Wk1*Wk2*deltak_sq).real
        DK34 = np.sum(RFFT_mask*Wk3*Wk4*deltak_sq).real
        DK13 = np.sum(RFFT_mask*Wk1*Wk3*deltak_sq).real
        DK24 = np.sum(RFFT_mask*Wk2*Wk4*deltak_sq).real
        DK14 = np.sum(RFFT_mask*Wk1*Wk4*deltak_sq).real
        DK23 = np.sum(RFFT_mask*Wk2*Wk3*deltak_sq).real

        PK12 = np.sum(RFFT_mask*Wk1*Wk2*Pk_mesh).real
        PK34 = np.sum(RFFT_mask*Wk3*Wk4*Pk_mesh).real
        PK13 = np.sum(RFFT_mask*Wk1*Wk3*Pk_mesh).real
        PK24 = np.sum(RFFT_mask*Wk2*Wk4*Pk_mesh).real
        PK14 = np.sum(RFFT_mask*Wk1*Wk4*Pk_mesh).real
        PK23 = np.sum(RFFT_mask*Wk2*Wk3*Pk_mesh).real

        Tk_disc_PD_arr[i] = (PK12*DK34+PK34*DK12+
                             PK13*DK24+PK24*DK13+
                             PK14*DK23+PK23*DK14)
        Tk_disc_PP_arr[i] = (PK12*PK34+PK13*PK24+PK23*PK14)

    # Normalize output
    Tk_tot_arr     = Tk_tot_arr/Ntet_arr*box_len**9/ng**12
    Tk_disc_DD_arr = Tk_disc_DD_arr/Ntet_arr*box_len**9/ng**15
    Tk_disc_PD_arr = 2*Tk_disc_PD_arr/Ntet_arr*(box_len/ng)**6/(ng**9)
    Tk_disc_PP_arr = 4*Tk_disc_PP_arr/Ntet_arr*(box_len/ng)**6/ng**3/box_len**3#/(ng**9)/(box_len**3)*ng**6

    if verbose:
        end = time.time()
        print("-> Time taken: {:.3f} seconds".format(end-start)) 

    return kf*k_cen_arr, Tk_tot_arr, Tk_disc_DD_arr, Tk_disc_PD_arr, Tk_disc_PP_arr, Ntet_arr, 