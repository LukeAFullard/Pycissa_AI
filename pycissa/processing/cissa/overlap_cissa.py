import numpy as np
from pycissa.processing.cissa.cissa import Cissa, initial_data_checks

class OverlapCissa(Cissa):
    """
    Overlap-SSA (ov-SSA) decomposition methodology.
    Based on Leles et al. (2018), it uses an overlap-save technique
    to decompose long time series without boundary artifacts.
    """
    def __init__(self, t: np.ndarray, x: np.ndarray, Z: int, q: int, L: int, use_32_bit: bool = False, **cissa_kwargs):
        super().__init__(t, x, use_32_bit)
        self.Z_len = Z
        self.q = q
        self.L = L

        # Calculate discarded boundary length (must satisfy Z = q + 2 * L_bar)
        self.L_bar = (self.Z_len - self.q) // 2

        if self.Z_len != self.q + 2 * self.L_bar:
            raise ValueError(f"Z ({self.Z_len}) must equal q ({self.q}) + 2 * L_bar. Ensure (Z - q) is even.")

        self.cissa_kwargs = cissa_kwargs

    def _generate_segments(self):
        """
        Iterates through the full time series from i=0 to N in steps of q.
        Yields (segment_t, segment_data, needs_ar_left, needs_ar_right).
        """
        N = len(self.x)
        for i in range(0, N, self.q):
            start_idx = max(0, i - self.L_bar)
            end_idx = min(N, i + self.q + self.L_bar)

            segment_t = self.t[start_idx:end_idx]
            segment_data = self.x[start_idx:end_idx]

            needs_ar_left = (i == 0)
            # The right extension is needed only if this block's true data
            # slice reaches the end of the array.
            needs_ar_right = (end_idx == N)

            yield segment_t, segment_data, needs_ar_left, needs_ar_right

    def fit(self, extension_type: str = 'AR_LR', multi_thread_run: bool = True, num_workers: int = 2):
        """
        Global Decomposition/Reconstruction over blocks.
        """
        reconstructed_blocks = []
        global_psd = []

        # Iterate over segments
        for segment_t, segment_data, needs_ar_left, needs_ar_right in self._generate_segments():
            local_cissa = Cissa(t=segment_t, x=segment_data, use_32_bit=self.use_32_bit)
            local_cissa.fit(
                L=self.L,
                extension_type=extension_type,
                multi_thread_run=multi_thread_run,
                num_workers=num_workers,
                extend_left=needs_ar_left,
                extend_right=needs_ar_right
            )

            # The result shape is (segment_length + right_ext + left_ext, num_components). Wait.
            # run_cissa groups paired frequencies and its return Z shape is
            # Z = Z[left_ext:lcol-right_ext,:]
            # So Z's length matches exactly the length of the input data (segment_data), which is min(N, i + q + L_bar) - max(0, i - L_bar).

            Z = local_cissa.Z

            # Now we need to discard the L_bar boundary extensions that overlap with other blocks
            # Wait, if we are at the very beginning (needs_ar_left == True), start_idx was 0, so there are no real data boundary on the left.
            # However, the length of the segment is still q + L_bar (or q + 2*L_bar if not on boundaries).
            # Leles et al. Overlap-save: the internal block of size Z = 2*L_bar + q is decomposed.
            # We save the middle q samples.
            # Let's verify the exact truncation logic.

            # If needs_ar_left is True, the segment length is typically q + L_bar. We should save the FIRST q samples.
            # If needs_ar_right is True, we save the LAST remaining samples (usually q).
            # Otherwise, we save the middle q samples, which means we slice [L_bar : L_bar + q].

            start_slice = 0 if needs_ar_left else self.L_bar
            end_slice = len(Z) if needs_ar_right else start_slice + self.q

            # Wait, for the first block (needs_ar_left=True), the start_idx is 0, end_idx is q + L_bar.
            # The length is q + L_bar. We only want to save the FIRST q samples!
            # So start_slice = 0, end_slice = q
            if needs_ar_left and not needs_ar_right:
                end_slice = self.q

            reconstructed_blocks.append(Z[start_slice:end_slice, :])
            global_psd.append(local_cissa.psd)

        # Concatenate reconstructed blocks
        self.Z = np.vstack(reconstructed_blocks)

        # We can also average the PSDs as an approximation for the global PSD
        self.psd = np.mean(np.array(global_psd), axis=0)

        # Generate standard results dictionary
        from pycissa.utilities.generate_cissa_result_dictionary import generate_results_dictionary
        self.results = generate_results_dictionary(self.Z, self.psd, self.L)

        from pycissa.postprocessing.grouping.grouping_functions import generate_grouping
        self.frequencies = generate_grouping(self.psd, self.L, trend=True)

        results = self.results
        results.get('cissa').setdefault('model parameters', {})
        results.get('cissa').setdefault('noise component tests', {})
        results.get('cissa').setdefault('fractal scaling results', {})
        results.get('cissa').get('model parameters').update({
            'extension_type': extension_type,
            'L': self.L,
            'Z': self.Z.shape[0], # Not strictly the parameter Z, but we can store it
            'multi_thread_run': multi_thread_run,
        })
        self.results = results

        return self
