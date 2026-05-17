import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from pycissa.processing.cissa.cissa import Cissa
from pycissa.processing.cissa.overlap_cissa import OverlapCissa

def reconstruct_top_k(cissa_obj, k=1):
    """
    Sort components by PSD share and sum the top k components.
    k=1 corresponds to the trend component (highest variance).
    """
    components = cissa_obj.results['cissa']['components']
    comp_shares = []
    for key, data in components.items():
        comp_shares.append((data['array_position'], data['percentage_share_of_psd']))
    comp_shares.sort(key=lambda x: x[1], reverse=True)
    top_indices = [x[0] for x in comp_shares[:k]]
    return np.sum(cissa_obj.Z[:, top_indices], axis=1)

if __name__ == '__main__':
    # Generate a synthetic time series of moderate length
    N = 3_000
    t = np.arange(N)
    # Signal with a slow trend, a periodic component, and some noise
    x = 0.005 * t + np.sin(2 * np.pi * t / 100) + 0.3 * np.random.randn(N)

    print(f"--- Profiling Performance on Series of Length {N} ---")
    L = 50

    # 1. Standard CiSSA
    print("\n[1] Running standard CiSSA...")
    tracemalloc.start()
    start_time = time.time()
    cissa_std = Cissa(t, x)
    cissa_std.fit(L=L, multi_thread_run=False)
    x_rec_std = reconstruct_top_k(cissa_std, k=1)
    std_time = time.time() - start_time
    _, std_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Standard CiSSA Time: {std_time:.2f} s")
    print(f"Standard CiSSA Peak Memory: {std_peak_mem / 1024 / 1024:.2f} MB")

    # 2. Overlap CiSSA - Small blocks
    print("\n[2] Running Overlap CiSSA (Small Blocks)...")
    tracemalloc.start()
    start_time = time.time()
    oc_many = OverlapCissa(t, x, Z=200, q=100, L=L)
    oc_many.fit(multi_thread_run=False)
    x_rec_many = reconstruct_top_k(oc_many, k=1)
    many_time = time.time() - start_time
    _, many_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Overlap CiSSA (Small Blocks) Time: {many_time:.2f} s")
    print(f"Overlap CiSSA (Small Blocks) Peak Memory: {many_peak_mem / 1024 / 1024:.2f} MB")

    # 3. Overlap CiSSA - Large blocks
    print("\n[3] Running Overlap CiSSA (Large Blocks)...")
    tracemalloc.start()
    start_time = time.time()
    oc_few = OverlapCissa(t, x, Z=1000, q=900, L=L)
    oc_few.fit(multi_thread_run=False)
    x_rec_few = reconstruct_top_k(oc_few, k=1)
    few_time = time.time() - start_time
    _, few_peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"Overlap CiSSA (Large Blocks) Time: {few_time:.2f} s")
    print(f"Overlap CiSSA (Large Blocks) Peak Memory: {few_peak_mem / 1024 / 1024:.2f} MB")

    subset = min(1000, N)
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(t[:subset], x[:subset], label='Original Data', color='black', alpha=0.5)
    axes[0].set_title('Original Time Series (First 1000 pts)')
    axes[0].legend()
    axes[1].plot(t[:subset], x[:subset], label='Original Data', color='black', alpha=0.3)
    axes[1].plot(t[:subset], x_rec_std[:subset], label='Standard CiSSA Trend', color='blue')
    axes[1].set_title(f'Standard CiSSA (L={L}) | Time: {std_time:.2f}s | Mem: {std_peak_mem / 1024/1024:.2f}MB')
    axes[1].legend()
    axes[2].plot(t[:subset], x[:subset], label='Original Data', color='black', alpha=0.3)
    axes[2].plot(t[:subset], x_rec_many[:subset], label='Overlap CiSSA Trend (Z=200)', color='red')
    axes[2].set_title(f'Overlap CiSSA - Small Blocks | Time: {many_time:.2f}s | Mem: {many_peak_mem / 1024/1024:.2f}MB')
    axes[2].legend()
    axes[3].plot(t[:subset], x[:subset], label='Original Data', color='black', alpha=0.3)
    axes[3].plot(t[:subset], x_rec_few[:subset], label='Overlap CiSSA Trend (Z=1000)', color='green')
    axes[3].set_title(f'Overlap CiSSA - Large Blocks | Time: {few_time:.2f}s | Mem: {few_peak_mem / 1024/1024:.2f}MB')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig('overlap_cissa_performance.png')
    print("\nPlot saved to overlap_cissa_performance.png")
