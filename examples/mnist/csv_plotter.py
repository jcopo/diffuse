import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt


def process_csv_files(root_dir):
    # Store all dataframes
    dfs = []

    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "scores.csv":
                filepath = os.path.join(dirpath, filename)
                df = pd.read_csv(filepath)
                dfs.append(df)

    # Concatenate all dataframes
    if not dfs:
        raise ValueError("No scores.csv files found")

    combined_df = pd.concat(dfs, ignore_index=True)

    # Group by Method and calculate mean and std for each metric
    summary = combined_df.groupby("Method").agg({"PSNR": ["median", "std"], "SSIM": ["median", "std"]})

    return summary


def plot_measurement_curves(root_dir):
    results = {}

    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "scores_during_opt.csv":
                # Extract method name from the path
                method_dir = os.path.basename(dirpath)
                if method_dir in ["theta", "random"]:
                    filepath = os.path.join(dirpath, filename)
                    df = pd.read_csv(filepath)

                    # Initialize method's dataframe list if not exists
                    if method_dir not in results:
                        results[method_dir] = []

                    # Append dataframe to appropriate list
                    results[method_dir].append(df)

    # After collecting all dataframes, concatenate and calculate stats
    for method in results:
        # Concatenate all dataframes for this method
        combined_df = pd.concat(results[method], ignore_index=True)

        # Calculate statistics on combined data
        stats = combined_df.groupby("Measurement").agg({"PSNR": ["median", "std"], "SSIM": ["median", "std"]})

        # Store final stats
        results[method] = stats

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    colors = {"theta": "blue", "random": "red"}

    for method, stats in results.items():
        measurements = np.array(stats.index.astype(float))

        if method == "theta":
            label = "CoDiff"
        else:
            label = "Random"

        # Plot PSNR
        psnr_median = stats["PSNR"]["median"].to_numpy()
        psnr_std = stats["PSNR"]["std"].to_numpy()
        # print median
        # print(psnr_median)
        ax1.plot(measurements, psnr_median, label=label, color=colors[method])
        ax1.fill_between(
            measurements,
            psnr_median - psnr_std,
            psnr_median + psnr_std,
            alpha=0.2,
            color=colors[method],
        )

        # Plot SSIM
        ssim_median = stats["SSIM"]["median"].to_numpy()
        print(ssim_median)
        ssim_std = stats["SSIM"]["std"].to_numpy()
        ax2.plot(measurements, ssim_median, label=label, color=colors[method])
        ax2.fill_between(
            measurements,
            ssim_median - ssim_std,
            ssim_median + ssim_std,
            alpha=0.2,
            color=colors[method],
        )

    ax1.set_title("PSNR over Measurements")
    ax1.set_xlabel("Measurement")
    ax1.set_ylabel("PSNR")
    ax1.legend()
    ax1.grid(True)

    ax2.set_title("SSIM over Measurements")
    ax2.set_xlabel("Measurement")
    ax2.set_ylabel("SSIM")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing subdirectories with scores.csv files",
    )
    args = parser.parse_args()

    summary = process_csv_files(args.root_dir)
    print("\nSummary Statistics:")
    print("==================")
    print(summary)

    plot_measurement_curves(args.root_dir)
