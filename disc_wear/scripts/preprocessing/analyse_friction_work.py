import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
from sklearn.preprocessing import MinMaxScaler

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

def analyse_friction():
    """
    Analyzes the friction energy data, plots growth curves, and calculates statistics.
    """
    # --- 1. Parameters ---
    parser = argparse.ArgumentParser(description="Analyze friction energy data.")
    parser.add_argument(
        '--input_csv',
        type=str,
        default=str(project_root / 'data/processed/tbm_data_with_friction_energy.csv'),
        help='Path to the input CSV file with friction energy data.'
    )
    parser.add_argument(
        '--output_cumulative_plot',
        type=str,
        default=str(project_root / 'results/cumulative_friction_growth_curve.png'),
        help='Path to save the cumulative friction growth plot file.'
    )
    parser.add_argument(
        '--output_ratio_plot',
        type=str,
        default=str(project_root / 'results/negative_friction_ratio.png'),
        help='Path to save the negative friction ratio plot file.'
    )
    parser.add_argument(
        '--output_corr_plot',
        type=str,
        default=str(project_root / 'results/friction_vs_torque_correlation.png'),
        help='Path to save the correlation scatter plot file.'
    )
    parser.add_argument(
        '--output_mean_trend_plot',
        type=str,
        default=str(project_root / 'results/mean_derivative_trend.png'),
        help='Path to save the mean derivative trend across rings plot file.'
    )
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    Path(args.output_cumulative_plot).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_ratio_plot).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_corr_plot).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_mean_trend_plot).parent.mkdir(parents=True, exist_ok=True)

    # --- 2. Load Data ---
    print(f"Loading data from {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input_csv}")
        return

    # --- 3. Calculate Total and Cumulative Friction per Ring ---
    print("Calculating total and cumulative friction per ring...")
    
    ring_summary = df.groupby('ring_number').agg(
        total_friction_in_ring=('window_cumulative_friction_energy', 'last'),
        total_friction_from_zeroed_in_ring=('window_cumulative_friction_from_zeroed', 'last')
    ).reset_index()

    ring_summary['cumulative_friction_across_rings'] = ring_summary['total_friction_in_ring'].cumsum()
    ring_summary['cumulative_friction_zeroed_across_rings'] = ring_summary['total_friction_from_zeroed_in_ring'].cumsum()

    # --- 4. Plot Cumulative Friction Growth Curve ---
    print(f"Plotting cumulative friction growth curve and saving to {args.output_cumulative_plot}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(ring_summary['ring_number'], ring_summary['cumulative_friction_across_rings'], marker='o', linestyle='-', label='Cumulative Original Friction')
    ax1.plot(ring_summary['ring_number'], ring_summary['cumulative_friction_zeroed_across_rings'], marker='x', linestyle='--', label='Cumulative Friction (Negatives Zeroed)')

    ax1.set_title('Cumulative Friction Energy Across Rings', fontsize=16)
    ax1.set_xlabel('Ring Number', fontsize=12)
    ax1.set_ylabel('Cumulative Friction Energy', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig(args.output_cumulative_plot, dpi=300)
    plt.close(fig1)
    print("Cumulative plot saved successfully.")

    # --- 5. Calculate and Plot Negative Friction Ratio ---
    print("\nCalculating proportion of negative friction timesteps per ring...")

    def negative_ratio(group):
        negative_count = (group['friction_energy_timestep'] < 0).sum()
        total_count = len(group)
        return negative_count / total_count if total_count > 0 else 0

    rings_with_friction = df.dropna(subset=['friction_energy_total_in_window'])['ring_number'].unique()
    analysis_df = df[df['ring_number'].isin(rings_with_friction)]

    neg_friction_ratio = analysis_df.groupby('ring_number').apply(negative_ratio)
    neg_friction_ratio = neg_friction_ratio.sort_index() # Sort by ring number for correct plot order
    
    print("--- Negative Friction Timestep Ratio per Ring ---")
    with pd.option_context('display.max_rows', None, 'display.float_format', '{:.2%}'.format):
        print(neg_friction_ratio)
    print("-------------------------------------------------")

    print(f"Plotting negative friction ratio and saving to {args.output_ratio_plot}...")
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    neg_friction_ratio.plot(kind='bar', ax=ax2, color='skyblue', edgecolor='black')

    ax2.set_title('Proportion of Negative Friction Timesteps per Ring', fontsize=16)
    ax2.set_xlabel('Ring Number', fontsize=12)
    ax2.set_ylabel('Ratio of Negative Friction Timesteps', fontsize=12)
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax2.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.savefig(args.output_ratio_plot, dpi=300)
    plt.close(fig2)
    print("Ratio plot saved successfully.")

    # --- 6. Correlation Analysis ---
    print("\nAnalyzing correlation between cumulative friction and cumulative torque...")

    friction_col = 'global_cumulative_friction_energy'
    torque_col = 'torque_work_cumulative_total'

    if friction_col in df.columns and torque_col in df.columns:
        correlation = df[friction_col].corr(df[torque_col])
        print(f"Pearson Correlation between '{friction_col}' and '{torque_col}': {correlation:.4f}")

        print(f"Plotting correlation scatter plot and saving to {args.output_corr_plot}...")
        fig3, ax3 = plt.subplots(figsize=(10, 10))
        
        # Using a hexbin plot for better visualization of dense data
        hb = ax3.hexbin(df[torque_col], df[friction_col], gridsize=50, cmap='inferno', bins='log')
        
        ax3.set_title('Cumulative Friction vs. Cumulative Torque', fontsize=16)
        ax3.set_xlabel('Global Cumulative Torque Work', fontsize=12)
        ax3.set_ylabel('Global Cumulative Friction Energy', fontsize=12)
        
        cb = fig3.colorbar(hb, ax=ax3)
        cb.set_label('Log-scaled point density')

        plt.tight_layout()
        plt.savefig(args.output_corr_plot, dpi=300)
        plt.close(fig3)
        print("Correlation plot saved successfully.")
    else:
        print(f"Warning: Could not find required columns for correlation analysis ('{friction_col}' and/or '{torque_col}').")

    # --- 7. Per-Ring Derivative Trend Analysis ---
    print("\nAnalyzing derivative trends across all rings...")

    friction_derivative_col = 'friction_energy_timestep'
    torque_derivative_col = 'torque_work_incremental'
    ring_col = 'ring_number'

    if friction_derivative_col in df.columns and torque_derivative_col in df.columns:
        # Calculate the mean of the derivatives for each ring
        mean_summary = df.groupby(ring_col).agg(
            mean_friction_timestep=(friction_derivative_col, 'mean'),
            mean_torque_incremental=(torque_derivative_col, 'mean')
        ).reset_index()

        # Normalize both series to a 0-1 range for trend comparison
        scaler = MinMaxScaler()
        
        # The scaler expects a 2D array
        mean_summary_scaled = scaler.fit_transform(mean_summary[['mean_friction_timestep', 'mean_torque_incremental']])
        mean_summary['mean_friction_scaled'] = mean_summary_scaled[:, 0]
        mean_summary['mean_torque_scaled'] = mean_summary_scaled[:, 1]

        # Create and save the plot
        print(f"Plotting mean trend comparison and saving to {args.output_mean_trend_plot}...")
        fig4, ax4 = plt.subplots(figsize=(15, 7))
        
        ax4.plot(mean_summary[ring_col], mean_summary['mean_friction_scaled'], label='Mean Friction Energy Timestep', color='r', marker='.')
        ax4.plot(mean_summary[ring_col], mean_summary['mean_torque_scaled'], label='Mean Torque Work Timestep', color='b', marker='.')
        
        ax4.set_title('Trend of Mean Per-Timestep Derivatives Across Rings (Normalized)', fontsize=16)
        ax4.set_xlabel('Ring Number', fontsize=12)
        ax4.set_ylabel('Normalized Mean Value', fontsize=12)
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(args.output_mean_trend_plot, dpi=300)
        plt.close(fig4)
        print("Mean trend comparison plot saved successfully.")
    else:
        print(f"Warning: Could not find required columns for trend analysis ('{friction_derivative_col}' and/or '{torque_derivative_col}').")


if __name__ == '__main__':
    analyse_friction()