import numpy as np
import matplotlib.pyplot as plt
from pycissa.processing.mcissa.mcissa import MCissa

def run_scenario(scenario_name, t_uneven, t_target, x_full, true_x, missing_idx, plot_filename):
    print(f"--- Running {scenario_name} ---")

    t_missing = np.delete(t_uneven, missing_idx)
    x_missing = np.delete(x_full, missing_idx, axis=0)

    print("Running Multivariate Uneven Gap Filling...")
    model_multi = MCissa(t_missing.copy(), x_missing.copy())
    model_multi.pre_fill_uneven_timeseries(
        L_values=[12],
        dt=30.0,
        gap_threshold=20.0,
        center_data=True,
        multivariate=True,
        plot=False,
        estimate_error=False,
        verbose=False,
        component_selection_method='drop_smallest_proportion',
        eigenvalue_proportion=0.99
    )
    x_multi_filled = model_multi.x
    t_multi_even = model_multi.t

    print("Running Univariate Uneven Gap Filling...")
    model_uni = MCissa(t_missing.copy(), x_missing.copy())
    model_uni.pre_fill_uneven_timeseries(
        L_values=[12],
        dt=30.0,
        gap_threshold=20.0,
        center_data=True,
        multivariate=False,
        plot=False,
        estimate_error=False,
        verbose=False,
        component_selection_method='drop_smallest_proportion',
        eigenvalue_proportion=0.99
    )
    x_uni_filled = model_uni.x
    t_uni_even = model_uni.t

    # Calculate RMSE just for Channel 1 at the gap locations
    rmse_multi_true = np.sqrt(np.mean((x_multi_filled[missing_idx, 0] - true_x[missing_idx, 0])**2))
    rmse_uni_true = np.sqrt(np.mean((x_uni_filled[missing_idx, 0] - true_x[missing_idx, 0])**2))

    print(f"Channel 1 Gap Recovery - Univariate RMSE: {rmse_uni_true:.4f}, Multivariate RMSE: {rmse_multi_true:.4f}\n")

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for m in range(3):
        axes[m].plot(t_uneven, x_full[:, m], 'ko-', alpha=0.2, label='Original Uneven Data + Noise')
        axes[m].plot(t_missing, x_missing[:, m], 'ks', markersize=4, label='Input Data')

        axes[m].plot(t_uni_even, x_uni_filled[:, m], 'r--', label='Univariate Imputed')
        axes[m].plot(t_multi_even, x_multi_filled[:, m], 'g-', alpha=0.8, label='Multivariate Imputed')

        gap_t = t_target[missing_idx]
        axes[m].plot(gap_t, x_uni_filled[missing_idx, m], 'rX', markersize=8)
        axes[m].plot(gap_t, x_multi_filled[missing_idx, m], 'gX', markersize=8)

        axes[m].set_ylabel(f'Channel {m+1}')
        if m == 0:
            axes[m].legend(loc='upper left', fontsize=8)

    axes[2].set_xlabel('Time (Days)')
    title_str = (
        f'{scenario_name}: Univariate vs Multivariate\n'
        f'Channel 1 Gap Recovery RMSE - Univariate: {rmse_uni_true:.2f}, Multivariate: {rmse_multi_true:.2f}'
    )
    plt.suptitle(title_str, y=0.98)
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close('all')
    print(f"Saved plot to {plot_filename}\n")

if __name__ == "__main__":
    np.random.seed(42)

    days_in_month = 30
    months = 60
    t_target = np.arange(15, (months + 1) * days_in_month, days_in_month)[:months]
    t_uneven = t_target + np.random.uniform(-5, 5, size=months)
    missing_idx = [20, 21, 22, 40, 41]

    # --- SCENARIO 1: Basic Correlated Seasonality ---
    base_signal = 10 * np.sin(2 * np.pi * t_uneven / 360)
    c1 = base_signal + np.random.normal(0, 0.5, size=months)
    c2 = 1.2 * base_signal + np.random.normal(0, 0.2, size=months)
    c3 = 0.8 * base_signal + np.random.normal(0, 0.2, size=months)
    x_full_1 = np.column_stack([c1, c2, c3])

    true_c1 = 10 * np.sin(2 * np.pi * t_target / 360)
    true_x_1 = np.column_stack([true_c1, true_c1*1.2, true_c1*0.8])

    run_scenario(
        "Scenario 1: Highly Correlated Seasonality",
        t_uneven, t_target, x_full_1, true_x_1, missing_idx,
        'examples/mcissa/monthly_centering_scenario_1.png'
    )

    # --- SCENARIO 2: Mixed Frequencies and Trends ---
    # Mixed frequencies: 360 days (annual) and 180 days (semi-annual) with different trends
    s2_annual = 10 * np.sin(2 * np.pi * t_uneven / 360)
    s2_semi_annual = 5 * np.sin(2 * np.pi * t_uneven / 180)

    # Channel 1: Just annual + small trend
    c1_2 = s2_annual + 0.01 * t_uneven + np.random.normal(0, 0.5, size=months)
    # Channel 2: Annual + semi-annual + strong trend
    c2_2 = s2_annual + s2_semi_annual + 0.05 * t_uneven + np.random.normal(0, 0.2, size=months)
    # Channel 3: Semi-annual inverted + small trend
    c3_2 = -s2_semi_annual + 0.01 * t_uneven + np.random.normal(0, 0.2, size=months)

    x_full_2 = np.column_stack([c1_2, c2_2, c3_2])

    true_c1_2 = 10 * np.sin(2 * np.pi * t_target / 360) + 0.01 * t_target
    true_x_2 = np.column_stack([true_c1_2, true_c1_2, true_c1_2]) # Only checking C1 anyway

    run_scenario(
        "Scenario 2: Mixed Frequencies and Trends",
        t_uneven, t_target, x_full_2, true_x_2, missing_idx,
        'examples/mcissa/monthly_centering_scenario_2.png'
    )

    # --- SCENARIO 3: Phase-Shifted Signals ---
    # Phase shifts are captured efficiently by M-CiSSA spatial eigenvectors
    s3_base = 10 * np.sin(2 * np.pi * t_uneven / 360)

    # C1 is standard
    c1_3 = s3_base + np.random.normal(0, 0.5, size=months)

    # C2 is delayed by 30 days (1 month phase shift)
    c2_3 = 10 * np.sin(2 * np.pi * (t_uneven - 30) / 360) + np.random.normal(0, 0.2, size=months)

    # C3 is advanced by 60 days
    c3_3 = 10 * np.sin(2 * np.pi * (t_uneven + 60) / 360) + np.random.normal(0, 0.2, size=months)

    x_full_3 = np.column_stack([c1_3, c2_3, c3_3])

    true_c1_3 = 10 * np.sin(2 * np.pi * t_target / 360)
    true_x_3 = np.column_stack([true_c1_3, true_c1_3, true_c1_3])

    run_scenario(
        "Scenario 3: Phase-Shifted Signals",
        t_uneven, t_target, x_full_3, true_x_3, missing_idx,
        'examples/mcissa/monthly_centering_scenario_3.png'
    )

    # --- SCENARIO 4: Gap Size Sensitivity ---
    # We will loop through varying gap sizes to see how performance degrades.
    print("\n--- Running Scenario 4: Gap Size Sensitivity ---")
    gap_sizes_list = [2, 3, 4, 6]
    uni_rmses = []
    multi_rmses = []

    for g_size in gap_sizes_list:
        g_idx = list(range(24, 24 + g_size))

        t_m = t_uneven.copy()
        x_m = x_full_1.copy()

        # We simulate a sensor failure where ONLY Channel 1 stops recording for g_size months.
        # Channels 2 and 3 continue recording, allowing multivariate to use their information!
        x_m[g_idx, 0] = np.nan

        # Multi
        mod_m = MCissa(t_m.copy(), x_m.copy())
        mod_m.pre_fill_uneven_timeseries(
            L_values=[12], dt=30.0, gap_threshold=20.0,
            center_data=True, multivariate=True, plot=False,
            estimate_error=False, verbose=False,
            component_selection_method='drop_smallest_proportion', eigenvalue_proportion=0.99
        )

        # Uni
        mod_u = MCissa(t_m.copy(), x_m.copy())
        mod_u.pre_fill_uneven_timeseries(
            L_values=[12], dt=30.0, gap_threshold=20.0,
            center_data=True, multivariate=False, plot=False,
            estimate_error=False, verbose=False,
            component_selection_method='drop_smallest_proportion', eigenvalue_proportion=0.99
        )

        err_m = np.sqrt(np.mean((mod_m.x[g_idx, 0] - true_x_1[g_idx, 0])**2))
        err_u = np.sqrt(np.mean((mod_u.x[g_idx, 0] - true_x_1[g_idx, 0])**2))

        multi_rmses.append(err_m)
        uni_rmses.append(err_u)

        print(f"Gap Size {g_size} -> Univariate RMSE: {err_u:.4f} | Multivariate RMSE: {err_m:.4f}")

    # Plot sensitivity
    plt.figure(figsize=(8, 5))
    plt.plot(gap_sizes_list, uni_rmses, 'ro--', label='Univariate Imputation')
    plt.plot(gap_sizes_list, multi_rmses, 'go-', label='Multivariate Imputation')
    plt.xlabel('Gap Size (Consecutive Missing Months in Channel 1)')
    plt.ylabel('Recovery RMSE (Channel 1)')
    plt.title('Gap Recovery Performance vs Gap Size\n(Multivariate utilizes intact C2 & C3 across the gap)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('examples/mcissa/monthly_centering_scenario_4.png')
    plt.close('all')
    print("Saved plot to examples/mcissa/monthly_centering_scenario_4.png\n")

    # --- SCENARIO 5: Pure Centering Test (0 missing months) ---
    print("\n--- Running Scenario 5: Pure Centering Test (No Missing Months) ---")

    # We use x_full_1 which has NO deleted elements, just the uneven sampling jitter.

    # Multi
    mod_m_full = MCissa(t_uneven.copy(), x_full_1.copy())
    mod_m_full.pre_fill_uneven_timeseries(
        L_values=[12], dt=30.0, gap_threshold=20.0,
        center_data=True, multivariate=True, plot=False,
        estimate_error=False, verbose=False,
        component_selection_method='drop_smallest_proportion', eigenvalue_proportion=0.99
    )

    # Uni
    mod_u_full = MCissa(t_uneven.copy(), x_full_1.copy())
    mod_u_full.pre_fill_uneven_timeseries(
        L_values=[12], dt=30.0, gap_threshold=20.0,
        center_data=True, multivariate=False, plot=False,
        estimate_error=False, verbose=False,
        component_selection_method='drop_smallest_proportion', eigenvalue_proportion=0.99
    )

    # Measure against true clean underlying signal across all evenly spaced points
    err_m_full = np.sqrt(np.mean((mod_m_full.x[:, 0] - true_x_1[:, 0])**2))
    err_u_full = np.sqrt(np.mean((mod_u_full.x[:, 0] - true_x_1[:, 0])**2))

    print(f"Overall Series Reconstruction - Univariate RMSE: {err_u_full:.4f} | Multivariate RMSE: {err_m_full:.4f}")
