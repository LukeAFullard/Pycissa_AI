import numpy as np
from pycissa.processing.mcissa.mcissa import MCissa

class OverlapMCissa(MCissa):
    """
    Overlap-SSA (ov-SSA) decomposition methodology for multivariate data.
    Based on Leles et al. (2018), it uses an overlap-save technique
    to decompose long time series without boundary artifacts.
    """
    def __init__(self, t: np.ndarray, x: np.ndarray, Z: int, q: int, L: int, use_32_bit: bool = False, **mcissa_kwargs):
        super().__init__(t, x, use_32_bit)
        self.Z_len = Z
        self.q = q
        self.L = L

        # Calculate discarded boundary length (must satisfy Z = q + 2 * L_bar)
        self.L_bar = (self.Z_len - self.q) // 2

        if self.Z_len != self.q + 2 * self.L_bar:
            raise ValueError(f"Z ({self.Z_len}) must equal q ({self.q}) + 2 * L_bar. Ensure (Z - q) is even.")

        self.mcissa_kwargs = mcissa_kwargs

    def _generate_segments(self):
        """
        Iterates through the full time series from i=0 to N in steps of q.
        Yields (segment_t, segment_data, needs_ar_left, needs_ar_right, pad_left).
        """
        N = len(self.x)
        for i in range(0, N, self.q):
            start_idx = max(0, i - self.L_bar)
            end_idx = min(N, i + self.q + self.L_bar)

            # If the segment is too small to compute L window, extend it backwards.
            # We also compute how much extra we padded so we discard it later.
            pad_left = 0
            if end_idx - start_idx <= 2 * self.L:
                new_start = max(0, end_idx - 2 * self.L - 1)
                pad_left = start_idx - new_start
                start_idx = new_start

            segment_t = self.t[start_idx:end_idx]
            # Since self.x is a 2D array, we slice the rows and take all columns
            segment_data = self.x[start_idx:end_idx, :]

            needs_ar_left = (i == 0)
            # The right extension is needed only if this block's true data
            # slice reaches the end of the array.
            needs_ar_right = (end_idx == N)

            yield segment_t, segment_data, needs_ar_left, needs_ar_right, pad_left

    def fit(self, extension_type: str = 'AR_LR', extend_left: bool = True, extend_right: bool = True):
        """
        Global Decomposition/Reconstruction over blocks for multivariate data.
        """
        reconstructed_blocks_stacked = []
        global_psd = []

        M = self.x.shape[1]
        reconstructed_blocks_Zs = [[] for _ in range(M)]

        # Iterate over segments
        for segment_t, segment_data, needs_ar_left, needs_ar_right, pad_left in self._generate_segments():
            local_mcissa = MCissa(t=segment_t, x=segment_data, use_32_bit=self.use_32_bit)

            # Use original boolean switches for extensions but only apply when at true edges
            do_extend_left = needs_ar_left and extend_left
            do_extend_right = needs_ar_right and extend_right

            local_mcissa.fit(
                L=self.L,
                extension_type=extension_type,
                extend_left=do_extend_left,
                extend_right=do_extend_right
            )

            Z_stacked = local_mcissa.Z_stacked
            Zs = local_mcissa.Zs

            start_slice = 0 if needs_ar_left else self.L_bar
            start_slice += pad_left
            end_slice = len(Z_stacked) if needs_ar_right else start_slice + self.q

            if needs_ar_left and not needs_ar_right:
                end_slice = self.q

            reconstructed_blocks_stacked.append(Z_stacked[start_slice:end_slice, :, :])
            global_psd.append(local_mcissa.psd)

            for m in range(M):
                reconstructed_blocks_Zs[m].append(Zs[m][start_slice:end_slice, :, :])

        # Concatenate reconstructed blocks
        self.Z_stacked = np.vstack(reconstructed_blocks_stacked)
        self.Zs = [np.vstack(reconstructed_blocks_Zs[m]) for m in range(M)]

        # Average the PSDs as an approximation for the global PSD
        self.psd = np.mean(np.array(global_psd), axis=0)

        # Generate standard results dictionary
        from pycissa.utilities.generate_cissa_result_dictionary import generate_m_results_dictionary
        self.results = generate_m_results_dictionary(self.Z_stacked, self.psd, self.L, cissa_type='mcissa')

        from pycissa.postprocessing.grouping.grouping_functions import generate_grouping
        self.frequencies = generate_grouping(np.zeros(self.L), self.L, trend=True)

        results = self.results
        results.get('mcissa').setdefault('model parameters', {})
        results.get('mcissa').setdefault('noise component tests', {})
        results.get('mcissa').setdefault('fractal scaling results', {})
        results.get('mcissa').get('model parameters').update({
            'extension_type': extension_type,
            'L': self.L,
            'Z': self.Z_stacked.shape[0],
        })
        self.results = results
        self.extension_type = extension_type

        return self
