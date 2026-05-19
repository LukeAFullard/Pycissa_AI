import numpy as np
import warnings
import matplotlib.pyplot as plt

def initial_mdata_checks(t: np.ndarray, x: np.ndarray, use_32_bit: bool):
    """
    Data checks to ensure t,x are numpy arrays of the correct shape.
    Will try to convert to the correct shape if they are not
    x should be (T, M)
    """
    if not isinstance(x, np.ndarray):
        try:
            x = np.array(x)
        except Exception:
            raise ValueError('Input "x" is not a numpy array, nor can be converted to one.')

    myshape = x.shape
    if len(myshape) != 2:
        raise ValueError(f'Input "x" should be a 2D matrix (T, M). The size of x is ({myshape})')

    if use_32_bit:
        new_x = np.empty_like(x, dtype=object)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                try:
                    new_x[i, j] = np.float32(x[i, j])
                except (ValueError, OverflowError):
                    new_x[i, j] = x[i, j]
        x = new_x
        try:
            x = x.astype(np.float32)
        except ValueError:
            pass

    if not isinstance(t, np.ndarray):
        try:
            t = np.array(t)
            t = t.reshape(len(t),)
        except Exception:
            raise ValueError('Input "t" is not a numpy array, nor can be converted to one.')

    if len(t.shape) != 1:
        try:
            t = t.reshape(len(t),)
        except Exception:
            raise ValueError(f'Input "t" should be a 1D vector. The size of t is ({t.shape})')

    return t, x

class MCissa:
    """
    Multivariate Circulant Singular Spectrum Analysis
    """
    def __init__(self, t: np.ndarray, x: np.ndarray, use_32_bit: bool = False):
        self.use_32_bit = use_32_bit
        t, x = initial_mdata_checks(t, x, self.use_32_bit)

        self.x_raw = x
        self.t_raw = t

        self.t = t
        self.x = x

        self.information_text = ''
        if not hasattr(self, 'figures'):
            self.figures = {}
        self.figures.update({'mcissa': {}})

    def fit(self, L: int, extension_type: str = 'AR_LR', extend_left: bool = True, extend_right: bool = True):
        """
        Function to fit M-CiSSA to a multivariate timeseries.
        """
        target_dtype = np.float32 if self.use_32_bit else np.float64
        try:
            self.x = np.asarray(self.x, dtype=target_dtype)
        except ValueError:
            raise ValueError("All elements in the input array 'x' must be numeric or convertible to numeric type before fitting.")

        from pycissa.processing.matrix_operations.m_matrix_operations import run_mcissa

        self.Z_stacked, self.psd, self.Zs = run_mcissa(
            self.x, L,
            extension_type=extension_type,
            extend_left=extend_left,
            extend_right=extend_right
        )

        # We can store Z components like univariate CiSSA.
        # self.Z_stacked has shape (T, M, nft)

        self.L = L
        self.extension_type = extension_type

        # generate initial results dictionary
        # In multivariate, we have psd for each variable or cross-psd. Let's provide a structure.
        self.results = {'mcissa': {}}
        self.results['mcissa']['model parameters'] = {
            'extension_type': extension_type,
            'L': L
        }

        return self

    def plot_components(self, num_components: int = 3, variable_names: list = None, component_names: list = None):
        """
        Groups the components by total variance across all variables and plots the top `num_components`.

        Parameters:
        num_components: int - Number of top components to plot (ordered by variance)
        variable_names: list of str - Names for the variables
        component_names: list of str - Names for the extracted components
        """
        from pycissa.utilities.plotting import plot_m_components

        if not hasattr(self, 'Z_stacked'):
            raise ValueError("Model not fitted. Call fit() before plotting.")

        M = self.Z_stacked.shape[1]
        nft = self.Z_stacked.shape[2]

        # Calculate total variance across all variables for each frequency
        variances = [np.sum([np.var(self.Z_stacked[:, m, i]) for m in range(M)]) for i in range(nft)]

        # Sort indices by descending variance
        sorted_indices = np.argsort(variances)[::-1]

        # Extract the top components
        extracted_components = []
        for i in range(min(num_components, nft)):
            extracted_components.append(self.Z_stacked[:, :, sorted_indices[i]])

        fig = plot_m_components(self.t, self.x, extracted_components, variable_names, component_names)
        self.figures.get('mcissa').update({'figure_components': fig})

        return fig
