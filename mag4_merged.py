#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Venus Magnetic Field Analysis with Dual Modes (Per-File & Merged)
================================================================

Processes Venus magnetic field CSV files named like:
MAG_YYYY-MM-ddT00-00-00_DOY_XXX_S004_3_VSO_SAPV.csv

Modes:
1. Per-File Analysis:
   - Runs plots 01 → 15 for each CSV separately
   - Generates plot 16 as multi-day magnitude overlay
   - Saves stats for each file

2. Merged Dataset Analysis:
   - Merges all CSV files into one DataFrame
   - Runs plots 01 → 16 for entire dataset
   - Saves one combined stats CSV

@author: mp10
@coding assistant: TGC-01082025
# pip install pandas numpy matplotlib seaborn scikit-learn pywt scipy
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pywt
from scipy.signal import spectrogram

sns.set_theme(style="whitegrid", context="talk")

# ======== USER CONFIG ========
RUN_PER_FILE = False   # Run per-file analysis
RUN_MERGED = True     # Run merged dataset analysis
INPUT_DIR = Path(".")  # Change to your CSV folder
OUTPUT_DIR = Path("./output_merged")    # Change to your output folder
# =============================


@dataclass
class MagneticFieldDataset:
    file_path: Path
    df: Optional[pd.DataFrame] = None


class VenusMagneticFieldAnalyzer:
    def __init__(self, input_dir: Path, output_dir: Path) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.csv_files = sorted(self.input_dir.glob("MAG_*.csv"))

        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "stats").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

    def load_dataset(self, file_path: Path) -> MagneticFieldDataset:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        self._add_derived_columns(df)
        return MagneticFieldDataset(file_path=file_path, df=df)

    def load_all_datasets(self) -> MagneticFieldDataset:
        dfs = []
        for file_path in self.csv_files:
            df = pd.read_csv(file_path, parse_dates=["timestamp"])
            self._add_derived_columns(df)
            dfs.append(df)
        merged_df = pd.concat(dfs, ignore_index=True).sort_values("timestamp")
        return MagneticFieldDataset(file_path=Path("merged_dataset"), df=merged_df)

    def _add_derived_columns(self, df: pd.DataFrame) -> None:
        df["B_SC_mag"] = np.sqrt(df[["b_sc_x", "b_sc_y", "b_sc_z"]].pow(2).sum(axis=1))
        df["B_VSO_mag"] = np.sqrt(df[["b_vso_x", "b_vso_y", "b_vso_z"]].pow(2).sum(axis=1))

    def save_statistics(self, dataset: MagneticFieldDataset, prefix: str) -> None:
        stats = dataset.df.describe()
        stats_path = self.output_dir / "stats" / f"{prefix}_stats.csv"
        stats.to_csv(stats_path)
        print(f"📊 Saved stats: {stats_path}")

    # === PLOTS 01 → 16 ===
    def plot_time_series(self, ds, num, prefix):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        ds.df.plot(x="timestamp", y=["b_sc_x", "b_sc_y", "b_sc_z"], ax=axes[0],
                   title="Spacecraft Frame Magnetic Field (nT)")
        ds.df.plot(x="timestamp", y=["b_vso_x", "b_vso_y", "b_vso_z"], ax=axes[1],
                   title="VSO Frame Magnetic Field (nT)")
        ds.df.plot(x="timestamp", y=["B_SC_mag", "B_VSO_mag"], ax=axes[2],
                   title="Magnetic Field Magnitudes (nT)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_timeseries.png", dpi=300)
        plt.close()

    def plot_histogram(self, ds, num, prefix):
        plt.figure(figsize=(10, 6))
        sns.histplot(ds.df["B_VSO_mag"], kde=True, color="royalblue", bins=50)
        plt.title("Distribution of VSO Magnetic Field Magnitude")
        plt.xlabel("B_VSO_mag (nT)")
        plt.ylabel("Frequency")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_histogram.png", dpi=300)
        plt.close()

    def plot_trajectory(self, ds, num, prefix):
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(ds.df["pos_vso_x"], ds.df["pos_vso_y"], ds.df["pos_vso_z"], color="blue", lw=1.5)
        ax.set_xlabel("VSO X (km)")
        ax.set_ylabel("VSO Y (km)")
        ax.set_zlabel("VSO Z (km)")
        ax.set_title("Trajectory in VSO Frame")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_trajectory.png", dpi=300)
        plt.close()

    def plot_fft(self, ds, num, prefix):
        b = ds.df["B_VSO_mag"].values
        dt = np.median(np.diff(ds.df["timestamp"].values).astype("timedelta64[ms]").astype(float)) / 1000.0
        freqs = np.fft.rfftfreq(len(b), dt)
        spec = np.abs(np.fft.rfft(b - np.mean(b)))**2
        plt.figure(figsize=(10, 6))
        plt.loglog(freqs[1:], spec[1:], color="purple")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title("FFT Power Spectrum (B_VSO_mag)")
        plt.grid(True, which="both", ls="--")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_fft.png", dpi=300)
        plt.close()

    def plot_wavelet(self, ds, num, prefix):
        b = ds.df["B_VSO_mag"].values
        scales = np.arange(1, 128)
        coeffs, _ = pywt.cwt(b, scales, "cmor")
        plt.figure(figsize=(12, 6))
        plt.imshow(np.abs(coeffs), extent=[0, len(b), scales.min(), scales.max()],
                   cmap="viridis", aspect="auto", origin="lower")
        plt.colorbar(label="Wavelet Power")
        plt.xlabel("Sample index (time)")
        plt.ylabel("Scale")
        plt.title("Wavelet Scalogram (B_VSO_mag)")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_wavelet.png", dpi=300)
        plt.close()

    def detect_bow_shock(self, ds, num, prefix):
        Rv = 6052.0
        r = np.sqrt(ds.df["pos_vso_x"]**2 + ds.df["pos_vso_y"]**2 + ds.df["pos_vso_z"]**2) / Rv
        shocks = ds.df.index[(r < 3) & (ds.df["B_VSO_mag"].diff() > 5)]
        plt.figure(figsize=(14, 6))
        plt.plot(ds.df["timestamp"], ds.df["B_VSO_mag"], label="B_VSO_mag", color="blue")
        plt.scatter(ds.df.loc[shocks, "timestamp"], ds.df.loc[shocks, "B_VSO_mag"],
                    color="red", label="Probable Bow Shock", zorder=5)
        plt.legend()
        plt.title("Bow Shock Detection")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_bowshock.png", dpi=300)
        plt.close()

    def plot_vector_direction(self, ds, num, prefix):
        bx, by, bz = ds.df["b_vso_x"], ds.df["b_vso_y"], ds.df["b_vso_z"]
        phi = np.degrees(np.arctan2(by, bx))
        theta = np.degrees(np.arctan2(bz, np.sqrt(bx**2 + by**2)))
        plt.figure(figsize=(14, 6))
        plt.plot(ds.df["timestamp"], phi, label="Azimuth (deg)")
        plt.plot(ds.df["timestamp"], theta, label="Elevation (deg)")
        plt.legend()
        plt.title("Magnetic Field Direction (Spherical Coordinates)")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_vector_direction.png", dpi=300)
        plt.close()

    def plot_hodogram(self, ds, num, prefix):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(ds.df["b_vso_x"], ds.df["b_vso_y"], ".-")
        plt.xlabel("B_x")
        plt.ylabel("B_y")
        plt.title("Hodogram: By vs. Bx")
        plt.subplot(1, 2, 2)
        plt.plot(ds.df["b_vso_x"], ds.df["b_vso_z"], ".-")
        plt.xlabel("B_x")
        plt.ylabel("B_z")
        plt.title("Hodogram: Bz vs. Bx")
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_hodogram.png", dpi=300)
        plt.close()

    def plot_mag_vs_radial(self, ds, num, prefix):
        Rv = 6052.0
        r = np.sqrt(ds.df["pos_vso_x"]**2 + ds.df["pos_vso_y"]**2 + ds.df["pos_vso_z"]**2) / Rv
        plt.figure(figsize=(8, 6))
        plt.scatter(r, ds.df["B_VSO_mag"], alpha=0.5)
        plt.xlabel("Radial Distance (R_V)")
        plt.ylabel("B_VSO_mag (nT)")
        plt.title("Magnetic Field Magnitude vs. Radial Distance")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_mag_vs_radial.png", dpi=300)
        plt.close()

    def plot_mag_vs_localtime(self, ds, num, prefix):
        lst = (np.degrees(np.arctan2(ds.df["pos_vso_y"], ds.df["pos_vso_x"])) / 15.0 + 12) % 24
        plt.figure(figsize=(8, 6))
        plt.scatter(lst, ds.df["B_VSO_mag"], alpha=0.5)
        plt.xlabel("Local Solar Time (hours)")
        plt.ylabel("B_VSO_mag (nT)")
        plt.title("Magnetic Field Magnitude vs. Local Time")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_mag_vs_lst.png", dpi=300)
        plt.close()

    def plot_field_aligned(self, ds, num, prefix):
        B = ds.df[["b_vso_x", "b_vso_y", "b_vso_z"]].values
        B0 = np.mean(B, axis=0)
        B0_unit = B0 / np.linalg.norm(B0)
        B_parallel = np.dot(B, B0_unit)
        B_perp = np.linalg.norm(B - np.outer(B_parallel, B0_unit), axis=1)
        plt.figure(figsize=(14, 6))
        plt.plot(ds.df["timestamp"], B_parallel, label="Parallel to B0")
        plt.plot(ds.df["timestamp"], B_perp, label="Perpendicular to B0")
        plt.legend()
        plt.title("Field-Aligned Coordinates")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_field_aligned.png", dpi=300)
        plt.close()

    def plot_correlation_matrix(self, ds, num, prefix):
        corr = ds.df[["b_vso_x", "b_vso_y", "b_vso_z", "B_SC_mag", "B_VSO_mag",
                      "pos_vso_x", "pos_vso_y", "pos_vso_z"]].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Matrix")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_correlation.png", dpi=300)
        plt.close()

    def plot_running_rms(self, ds, num, prefix, window=60):
        rms = ds.df["B_VSO_mag"].rolling(window).apply(lambda x: np.sqrt(np.mean(x**2)))
        plt.figure(figsize=(14, 6))
        plt.plot(ds.df["timestamp"], rms, label=f"{window}-point RMS")
        plt.legend()
        plt.title("Running RMS of B_VSO_mag")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_running_rms.png", dpi=300)
        plt.close()

    def plot_spectrogram(self, ds, num, prefix):
        b = ds.df["B_VSO_mag"].values
        dt = np.median(np.diff(ds.df["timestamp"].values).astype("timedelta64[ms]").astype(float)) / 1000.0
        f, t, Sxx = spectrogram(b, fs=1/dt, nperseg=256)
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, 10*np.log10(Sxx), shading="gouraud", cmap="viridis")
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.title("Spectrogram (STFT)")
        plt.colorbar(label="Power (dB)")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_spectrogram.png", dpi=300)
        plt.close()

    def plot_pca(self, ds, num, prefix):
        B = ds.df[["b_vso_x", "b_vso_y", "b_vso_z"]]
        pca = PCA(n_components=3)
        pcs = pca.fit_transform(B)
        plt.figure(figsize=(14, 6))
        plt.plot(ds.df["timestamp"], pcs[:, 0], label="PC1")
        plt.plot(ds.df["timestamp"], pcs[:, 1], label="PC2")
        plt.plot(ds.df["timestamp"], pcs[:, 2], label="PC3")
        plt.legend()
        plt.title("PCA Components of Magnetic Field")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_pca.png", dpi=300)
        plt.close()

    def plot_multiday_overview(self, datasets, num, prefix):
        plt.figure(figsize=(14, 6))
        for ds in datasets:
            plt.plot(ds.df["timestamp"], ds.df["B_VSO_mag"], alpha=0.3)
        plt.title("Multi-Day Magnetic Field Overview")
        plt.xlabel("Time")
        plt.ylabel("B_VSO_mag (nT)")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_multiday_overview.png", dpi=300)
        plt.close()

    def plot_orbital_field_map(self, ds, num, prefix):
        """
        Plot magnetic field magnitude in Venus orbital plane using VSO coordinates.
        X-axis: Sun-Venus line, Y-axis: orbital plane.
        Color shows magnetic field magnitude.
        """
        plt.figure(figsize=(8, 8))
        sc = plt.scatter(
            ds.df["pos_vso_x"] / 6052,  # Venus radii
            ds.df["pos_vso_y"] / 6052,
            c=ds.df["B_VSO_mag"],
            cmap="plasma",
            s=10,
            alpha=0.7
        )
        plt.colorbar(sc, label="B_VSO_mag (nT)")
        plt.xlabel("VSO X (R_V)")
        plt.ylabel("VSO Y (R_V)")
        plt.axhline(0, color="gray", lw=0.5)
        plt.axvline(0, color="gray", lw=0.5)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Magnetic Field Variation Around Venus Orbit (VSO Coordinates)")
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_orbital_field_map.png", dpi=300)
        plt.close()


    def plot_orbital_field_polar(self, ds, num, prefix):
        """
        Polar plot of spacecraft position in Venus orbital plane with magnetic field strength as color.
        Useful for visualizing bow shock location and asymmetry.
        """
        # Convert to polar coordinates
        r = np.sqrt(ds.df["pos_vso_x"]**2 + ds.df["pos_vso_y"]**2) / 6052  # Venus radii
        theta = np.arctan2(ds.df["pos_vso_y"], ds.df["pos_vso_x"])

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="polar")
        sc = ax.scatter(theta, r, c=ds.df["B_VSO_mag"], cmap="plasma", s=10, alpha=0.7)
        plt.colorbar(sc, ax=ax, pad=0.1, label="B_VSO_mag (nT)")
        ax.set_theta_zero_location("E")  # 0° at +X (Sun direction)
        ax.set_theta_direction(-1)       # Clockwise from Sun
        ax.set_rlabel_position(135)
        ax.set_title("Magnetic Field Variation in Polar Orbit View (VSO Coordinates)", pad=20)
        plt.savefig(self.output_dir / "plots" / f"{num:02d}_{prefix}_orbital_field_polar.png", dpi=300)
        plt.close()


    # === Main Processing ===
    def process_per_file(self):
        print("🔍 Running Per-File Analysis...")
        datasets = []
        log_file = self.output_dir / "logs" / "processing_log_per_file.txt"
        with open(log_file, "w") as log:
            for csv_file in self.csv_files:
                ds = self.load_dataset(csv_file)
                datasets.append(ds)
                prefix = csv_file.stem
                self.save_statistics(ds, prefix)
                self._run_all_plots(ds, prefix)
                log.write(f"Processed: {csv_file.name}\n")
            self.plot_multiday_overview(datasets, 16, "per_file")
        print(f"✅ Per-File analysis done. Log: {log_file}")

    def process_merged(self):
        print("🔍 Running Merged Dataset Analysis...")
        merged_ds = self.load_all_datasets()
        prefix = "merged"
        self.save_statistics(merged_ds, prefix)
        self._run_all_plots(merged_ds, prefix)
        print("✅ Merged dataset analysis done.")

    def _run_all_plots(self, ds, prefix):
        self.plot_time_series(ds, 1, prefix)
        self.plot_histogram(ds, 2, prefix)
        self.plot_trajectory(ds, 3, prefix)
        self.plot_fft(ds, 4, prefix)
        self.plot_wavelet(ds, 5, prefix)
        self.detect_bow_shock(ds, 6, prefix)
        self.plot_vector_direction(ds, 7, prefix)
        self.plot_hodogram(ds, 8, prefix)
        self.plot_mag_vs_radial(ds, 9, prefix)
        self.plot_mag_vs_localtime(ds, 10, prefix)
        self.plot_field_aligned(ds, 11, prefix)
        self.plot_correlation_matrix(ds, 12, prefix)
        self.plot_running_rms(ds, 13, prefix)
        self.plot_spectrogram(ds, 14, prefix)
        self.plot_pca(ds, 15, prefix)
        self.plot_orbital_field_map(ds, 17, prefix)
        self.plot_orbital_field_polar(ds, 18, prefix)


    def run(self):
        if RUN_PER_FILE:
            self.process_per_file()
        if RUN_MERGED:
            self.process_merged()


def main():
    analyzer = VenusMagneticFieldAnalyzer(INPUT_DIR, OUTPUT_DIR)
    analyzer.run()


if __name__ == "__main__":
    main()
