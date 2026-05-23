import numpy as np

def extend_series(x: np.ndarray,
                  extension_type: str,
                  left_ext: int,
                  right_ext: int,
                  extend_left: bool = True,
                  extend_right: bool = True) -> np.ndarray:
    '''
    Extends time series to perform Singular Spectrum Analysis.  https://doi.org/10.1016/j.sigpro.2020.107824
    This function extends the time series at the beginning and end.
    Based on https://github.com/jbogalo/CiSSA
    -------------------------------------------------------------------------
    References:
    [1] Bógalo, J., Poncela, P., & Senra, E. (2021).
        "Circulant singular spectrum analysis: a new automated procedure for signal extraction".
          Signal Processing, 179, 107824.
        https://doi.org/10.1016/j.sigpro.2020.107824.
    -------------------------------------------------------------------------

    Parameters
    ----------
    x : np.ndarray
        DESCRIPTION: Column vector with the original time series. Must be size (N,1) where N is the length of the vector.
    extension_type : str
        DESCRIPTION: extension type for left and right ends of the time series
                    Options are:
                        'Mirror' - data at the start and end of the series is mirrored, x_{L+1} = X_{L-1},
                        'NoExt' - no extension applied (Not recommended...),
                        'AR_LR' - autoregressive extension applied to start (L) and end (R) of x,
                        'AR_L' - autoregressive extension applied to start (L) of x, ,
                        'AR_R - autoregressive extension applied to end (R) of x'
    left_ext: int
        DESCRIPTION: length of left time series extension
    right_ext: int
        DESCRIPTION: length of right time series extension
    extend_left: bool
        DESCRIPTION: Whether to actually apply the left extension logic
    extend_right: bool
        DESCRIPTION: Whether to actually apply the right extension logic

    Returns
    -------
    xe : np.ndarray
        DESCRIPTION: Extended inpit series

    '''

    import numpy as np
    import statsmodels.api as sm
    from pycissa.utilities.autoregression_functions import aryule
    from scipy.signal import lfilter


    ###########################################################################
    # 0) x size checking, extension_type type checking
    ###########################################################################
    #check extension_type is an integer
    if not type(extension_type) == str:
        raise('Input parameter "extension_type" should be a string')

    rows, cols = x.shape
    if cols != 1:
        raise ValueError(f'Input "x" should be a column vector (i.e. only contain a single column). The size of x is ({rows},{cols})')
    ###########################################################################


    ###########################################################################
    # 1) Dimensions
    ###########################################################################
    T = len(x)
    ###########################################################################


    ###########################################################################
    # 2) Extend
    ###########################################################################
    if extension_type == 'NoExt':    #No extension
        xe = x.copy()

    elif extension_type == 'Mirror':  #Mirroring
        left_part = np.flipud(x) if extend_left else np.empty((0, 1))
        right_part = np.flipud(x) if extend_right else np.empty((0, 1))
        xe = np.append(np.append(left_part, x), right_part)
        xe = xe.reshape(len(xe),1)

    elif extension_type == 'AR_LR':         #Autoregressive extension
        # AR coefficients of the differentiated series
        p = np.fix(T/3)
        dx = np.diff(x, axis = 0)
        Aold, cccold = sm.regression.yule_walker(dx, order=int(p),method="adjusted")
        [A, var, reflec] = aryule(dx, int(p))
        # Right extension
        y = x.copy().reshape(-1)
        if extend_right:
            dy = np.diff(y, axis = 0)
            er = lfilter(np.append(1,A), 1, dy)

            dy = lfilter([1],np.append(1,A),np.append(er,np.zeros(right_ext)))
            y = y[0]+np.append(0,np.cumsum(dy))
        # Left extension
        y = np.flipud(y)
        if extend_left:
            dy = np.diff(y, axis = 0)
            er = lfilter(np.append(1,A), 1, dy)
            dy = lfilter([1],np.append(1,A),np.append(er,np.zeros(left_ext)))
            y = y[0]+np.append(0,np.cumsum(dy))
        # Extended series
        xe = np.flipud(y)
        xe = xe.reshape(len(xe),1)

    elif extension_type == 'AR_L':  #Autoregressive extension  on left side only
        # AR coefficients of the differentiated series
        p = np.fix(T/3)
        dx = np.diff(x, axis = 0)
        Aold, cccold = sm.regression.yule_walker(dx, order=int(p),method="adjusted")
        [A, var, reflec] = aryule(dx, int(p))
        # Right extension (None basically here, just process the left extension step)
        y = x.copy().reshape(-1)
        dy = np.diff(y, axis = 0)
        er = lfilter(np.append(1,A), 1, dy)

        dy = lfilter([1],np.append(1,A),np.append(er,np.zeros(0)))
        y = y[0]+np.append(0,np.cumsum(dy))

        # Left extension
        y = np.flipud(y)
        if extend_left:
            dy = np.diff(y, axis = 0)
            er = lfilter(np.append(1,A), 1, dy)
            dy = lfilter([1],np.append(1,A),np.append(er,np.zeros(left_ext)))
            y = y[0]+np.append(0,np.cumsum(dy))
        # Extended series
        xe = np.flipud(y)
        xe = xe.reshape(len(xe),1)

    elif extension_type == 'AR_R':  #Autoregressive extension  on right side only
        # AR coefficients of the differentiated series
        p = np.fix(T/3)
        dx = np.diff(x, axis = 0)
        Aold, cccold = sm.regression.yule_walker(dx, order=int(p),method="adjusted")
        [A, var, reflec] = aryule(dx, int(p))
        # Right extension
        y = x.copy().reshape(-1)
        if extend_right:
            dy = np.diff(y, axis = 0)
            er = lfilter(np.append(1,A), 1, dy)

            dy = lfilter([1],np.append(1,A),np.append(er,np.zeros(right_ext)))
            y = y[0]+np.append(0,np.cumsum(dy))

        # # Extended series
        xe = y.copy()
        xe = xe.reshape(len(xe),1)

    ###########################################################################

    return xe