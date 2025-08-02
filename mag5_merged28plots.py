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
   - Runs plots 01 → 28 for entire dataset
   - Saves one combined stats CSV

@author: mp10
@coding assistant: TGC-01082025
# pip install pandas numpy matplotlib seaborn scikit-learn pywt scipy plotly scikit-learn
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
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D  # noqa
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import gc  # Added for garbage collection

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
        Rv = 6052.0
        df["R_V"] = np.sqrt(df["pos_vso_x"]**2 + df["pos_vso_y"]**2 + df["pos_vso_z"]**2) / Rv

    def save_statistics(self, dataset: MagneticFieldDataset, prefix: str) -> None:
        stats = dataset.df.describe()
        stats_path = self.output_dir / "stats" / f"{prefix}_stats.csv"
        stats.to_csv(stats_path)
        print(f"📊 Saved stats: {stats_path}")

    # === PLOTS 01 → 15 ===
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
        shocks = ds.df.index[(ds.df["R_V"] < 3) & (ds.df["B_VSO_mag"].diff() > 5)]
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
        plt.figure(figsize=(8, 6))
        plt.scatter(ds.df["R_V"], ds.df["B_VSO_mag"], alpha=0.5)
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
        rms = ds.df["B_VSO_mag"].rolling(window).apply(lambda x: np.sqrt(np.mean(x**2)), raw=False)
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
        # Add a small number to Sxx to avoid log(0)
        plt.pcolormesh(t, f, 10*np.log10(Sxx + 1e-10), shading="gouraud", cmap="viridis")
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

    # ================================================================
    # NEW PLOTS from mag5_merged28plots.py
    # ================================================================

    # PLOT 16: Daily Magnetic Field Profiles
    def plot_daily_profiles(self, ds, num, prefix):
        """
        Purpose:
        - Compare daily variation patterns in B magnitude.
        """
        df = ds.df.copy()
        df["date"] = df["timestamp"].dt.date
        grouped = df.groupby("date")

        fig, axs = plt.subplots(len(grouped), 1, figsize=(12, 3 * len(grouped)), sharex=True)
        if len(grouped) == 1:
            axs = [axs]  # Ensure iterable

        for ax, (day, subdf) in zip(axs, grouped):
            ax.plot(subdf["timestamp"], subdf["B_VSO_mag"], lw=0.8)
            ax.set_ylabel("B (nT)")
            ax.set_title(f"{day}")
            ax.grid(True)
        axs[-1].set_xlabel("Time (UTC)")

        plt.tight_layout()
        out_path = self.output_dir / "plots" / f"{num:02d}_{prefix}_daily_profiles.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # PLOT 17: Spacecraft Orbit in VSO Coordinates (2D XY)
    def plot_orbit_xy(self, ds, num, prefix):
        """
        Purpose:
        - Plot spacecraft track in Venus Solar Orbital (VSO) XY-plane.
        - Color by B magnitude to highlight bow shock crossings.
        """
        plt.figure(figsize=(8, 8))
        sc = plt.scatter(ds.df["pos_vso_x"], ds.df["pos_vso_y"],
                         c=ds.df["B_VSO_mag"], cmap="plasma", s=5, alpha=0.8)
        plt.colorbar(sc, label="B Magnitude (nT)")
        plt.xlabel("X_VSO (km)")
        plt.ylabel("Y_VSO (km)")
        plt.grid(True)
        plt.axis("equal")
        plt.title("Spacecraft Orbit (XY) Colored by B Magnitude")
        plt.tight_layout()
        out_path = self.output_dir / "plots" / f"{num:02d}_{prefix}_orbit_XY.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # PLOT 18: Orbit in VSO XZ-plane
    def plot_orbit_xz(self, ds, num, prefix):
        """
        Purpose:
        - See vertical excursions relative to Venus-Sun line.
        """
        plt.figure(figsize=(8, 8))
        sc = plt.scatter(ds.df["pos_vso_x"], ds.df["pos_vso_z"],
                         c=ds.df["B_VSO_mag"], cmap="viridis", s=5, alpha=0.8)
        plt.colorbar(sc, label="B Magnitude (nT)")
        plt.xlabel("X_VSO (km)")
        plt.ylabel("Z_VSO (km)")
        plt.grid(True)
        plt.axis("equal")
        plt.title("Spacecraft Orbit (XZ) Colored by B Magnitude")
        plt.tight_layout()
        out_path = self.output_dir / "plots" / f"{num:02d}_{prefix}_orbit_XZ.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # PLOT 19: Orbit in VSO YZ-plane
    def plot_orbit_yz(self, ds, num, prefix):
        """
        Purpose:
        - Polar view of spacecraft trajectory relative to Venus's equator.
        """
        plt.figure(figsize=(8, 8))
        sc = plt.scatter(ds.df["pos_vso_y"], ds.df["pos_vso_z"],
                         c=ds.df["B_VSO_mag"], cmap="coolwarm", s=5, alpha=0.8)
        plt.colorbar(sc, label="B Magnitude (nT)")
        plt.xlabel("Y_VSO (km)")
        plt.ylabel("Z_VSO (km)")
        plt.grid(True)
        plt.axis("equal")
        plt.title("Spacecraft Orbit (YZ) Colored by B Magnitude")
        plt.tight_layout()
        out_path = self.output_dir / "plots" / f"{num:02d}_{prefix}_orbit_YZ.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # PLOT 20: 3D Orbit in VSO Coordinates (Matplotlib)
    def plot_orbit_3d(self, ds, num, prefix):
        """
        Purpose:
        - Full 3D trajectory around Venus colored by B magnitude.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(ds.df["pos_vso_x"], ds.df["pos_vso_y"], ds.df["pos_vso_z"],
                        c=ds.df["B_VSO_mag"], cmap="plasma", s=5)
        ax.set_xlabel("X_VSO (km)")
        ax.set_ylabel("Y_VSO (km)")
        ax.set_zlabel("Z_VSO (km)")
        ax.set_title("3D Orbit (Matplotlib)")
        fig.colorbar(sc, ax=ax, label="B Magnitude (nT)")
        plt.tight_layout()
        out_path = self.output_dir / "plots" / f"{num:02d}_{prefix}_orbit_3D_matplotlib.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # PLOT 21: 3D Interactive Orbit (Plotly)
    def plot_orbit_3d_interactive(self, ds, num, prefix):
        """
        Purpose:
        - Interactive rotation, zoom, hover for B magnitude exploration.
        """
        fig = go.Figure(data=[go.Scatter3d(
            x=ds.df["pos_vso_x"], y=ds.df["pos_vso_y"], z=ds.df["pos_vso_z"],
            mode='markers',
            marker=dict(size=2, color=ds.df["B_VSO_mag"], colorscale='Plasma', colorbar=dict(title="B (nT)")),
            text=ds.df["timestamp"].astype(str)
        )])
        fig.update_layout(scene=dict(
            xaxis_title="X_VSO (km)", yaxis_title="Y_VSO (km)", zaxis_title="Z_VSO (km)"
        ), title="3D Orbit (Interactive)")
        out_path = self.output_dir / "plots" / f"{num:02d}_{prefix}_orbit_3D_plotly.html"
        fig.write_html(str(out_path))

    # PLOT 22: 3D Magnetic Field Vector Arrows
    def plot_3d_field_vectors(self, ds, num, prefix):
        """
        Purpose:
        - Show actual B direction vectors along spacecraft path.
        - Downsample for readability.
        """
        step = max(1, len(ds.df) // 500)
        x, y, z = ds.df["pos_vso_x"].values[::step], ds.df["pos_vso_y"].values[::step], ds.df["pos_vso_z"].values[::step]
        u, v, w = ds.df["b_vso_x"].values[::step], ds.df["b_vso_y"].values[::step], ds.df["b_vso_z"].values[::step]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(x, y, z, u, v, w, length=500, normalize=True, color='blue', alpha=0.5)
        ax.set_xlabel("X_VSO (km)")
        ax.set_ylabel("Y_VSO (km)")
        ax.set_zlabel("Z_VSO (km)")
        ax.set_title("3D Magnetic Field Vectors")
        plt.tight_layout()
        out_path = self.output_dir / "plots" / f"{num:02d}_{prefix}_3d_field_vectors.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # PLOT 23: Magnetic Field Magnitude along Orbit Path (Color Line)
    def plot_colorline_orbit(self, ds, num, prefix):
        """
        Purpose:
        - Color-code orbit line by B magnitude.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        points = np.array([ds.df["pos_vso_x"], ds.df["pos_vso_y"], ds.df["pos_vso_z"]]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, cmap="plasma", norm=plt.Normalize(ds.df["B_VSO_mag"].min(), ds.df["B_VSO_mag"].max()))
        lc.set_array(ds.df["B_VSO_mag"])
        lc.set_linewidth(2)
        ax.add_collection3d(lc)
        ax.set_xlim(ds.df["pos_vso_x"].min(), ds.df["pos_vso_x"].max())
        ax.set_ylim(ds.df["pos_vso_y"].min(), ds.df["pos_vso_y"].max())
        ax.set_zlim(ds.df["pos_vso_z"].min(), ds.df["pos_vso_z"].max())
        ax.set_xlabel("X_VSO (km)")
        ax.set_ylabel("Y_VSO (km)")
        ax.set_zlabel("Z_VSO (km)")
        plt.tight_layout()
        out_path = self.output_dir / "plots" / f"{num:02d}_{prefix}_colorline_orbit.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # PLOT 24: Radial Distance vs Z Position Colored by B Magnitude
    def plot_radius_vs_z(self, ds, num, prefix):
        """
        Purpose:
        - Show how B magnitude varies with height above/below Venus equator.
        """
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(ds.df["R_V"], ds.df["pos_vso_z"], c=ds.df["B_VSO_mag"], cmap="inferno", s=5, alpha=0.8)
        plt.colorbar(sc, label="B Magnitude (nT)")
        plt.xlabel("Radial Distance (R_V)")
        plt.ylabel("Z_VSO (km)")
        plt.grid(True)
        plt.title("B vs Z position")
        plt.tight_layout()
        out_path = self.output_dir / "plots" / f"{num:02d}_{prefix}_radius_vs_z.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # PLOT 25: 3D Bow Shock Surface Fit (Approximation)
    def plot_bowshock_surface(self, ds, num, prefix):
        """
        Purpose:
        - Approximate bow shock location by fitting 3D surface to high-B points.
        """
        highB = ds.df[ds.df["B_VSO_mag"] > ds.df["B_VSO_mag"].quantile(0.9)]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(highB["pos_vso_x"], highB["pos_vso_y"], highB["pos_vso_z"],
                   c=highB["B_VSO_mag"], cmap="plasma", s=5)
        ax.set_xlabel("X_VSO (km)")
        ax.set_ylabel("Y_VSO (km)")
        ax.set_zlabel("Z_VSO (km)")
        ax.set_title("Approx. Bow Shock Surface Points")
        plt.tight_layout()
        out_path = self.output_dir / "plots" / f"{num:02d}_{prefix}_bowshock_surface.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # PLOT 26: 3D Position Density Map (Kernel Density Estimate)
    def plot_position_density(self, ds, num, prefix, sample_size=50000):
        """
        Purpose:
        - Show where spacecraft spends most time in orbit.
        - Downsampled for quicker computation on large datasets.
        """
        df = ds.df
        if len(df) > sample_size:
            print(f"Downsampling data for Plot {num} from {len(df)} to {sample_size} points for performance.")
            df_sampled = df.sample(n=sample_size, random_state=42)
        else:
            df_sampled = df
        
        xy = np.vstack([df_sampled["pos_vso_x"], df_sampled["pos_vso_y"]])
        kde = gaussian_kde(xy)(xy)
        plt.figure(figsize=(8, 8))
        sc = plt.scatter(df_sampled["pos_vso_x"], df_sampled["pos_vso_y"], c=kde, s=5, cmap="viridis")
        plt.colorbar(sc, label="Density")
        plt.xlabel("X_VSO (km)")
        plt.ylabel("Y_VSO (km)")
        plt.grid(True)
        plt.axis("equal")
        plt.title("Spacecraft Position Density")
        plt.tight_layout()
        out_path = self.output_dir / "plots" / f"{num:02d}_{prefix}_position_density.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # PLOT 27: 3D Magnetic Topology Clustering
    def plot_field_clustering(self, ds, num, prefix):
        """
        Purpose:
        - Cluster B vector directions to find distinct regions (e.g., magnetosheath).
        """
        Bnorm = ds.df[["b_vso_x", "b_vso_y", "b_vso_z"]].values
        kmeans = KMeans(n_clusters=4, n_init='auto', random_state=0).fit(Bnorm)
        labels = kmeans.labels_

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(ds.df["pos_vso_x"], ds.df["pos_vso_y"], ds.df["pos_vso_z"],
                         c=labels, cmap="tab10", s=5)
        ax.set_xlabel("X_VSO (km)")
        ax.set_ylabel("Y_VSO (km)")
        ax.set_zlabel("Z_VSO (km)")
        ax.set_title("Magnetic Field Direction Clusters")
        plt.tight_layout()
        out_path = self.output_dir / "plots" / f"{num:02d}_{prefix}_field_clustering.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # PLOT 28: 3D Velocity vs Magnetic Field Magnitude
    def plot_velocity_vs_B(self, ds, num, prefix):
        """
        Purpose:
        - If velocity data available, compare it to B magnitude in 3D space.
        - Here, using position change as proxy velocity (approx).
        """
        dt = np.gradient(ds.df["timestamp"].astype(np.int64) / 1e9)  # seconds
        vx = np.gradient(ds.df["pos_vso_x"]) / dt
        vy = np.gradient(ds.df["pos_vso_y"]) / dt
        vz = np.gradient(ds.df["pos_vso_z"]) / dt
        speed = np.sqrt(vx**2 + vy**2 + vz**2)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(speed, ds.df["R_V"], ds.df["B_VSO_mag"], c=ds.df["B_VSO_mag"], cmap="plasma", s=5)
        ax.set_xlabel("Speed (km/s)")
        ax.set_ylabel("Radial Distance (R_V)")
        ax.set_zlabel("B Magnitude (nT)")
        ax.set_title("Velocity vs B Magnitude vs Radius")
        fig.colorbar(sc, ax=ax, label="B Magnitude (nT)")
        plt.tight_layout()
        out_path = self.output_dir / "plots" / f"{num:02d}_{prefix}_velocity_vs_B.png"
        plt.savefig(out_path, dpi=150)
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
                self._run_all_plots(ds, prefix, extended_plots=True)
                log.write(f"Processed: {csv_file.name}\n")
            self.plot_multiday_overview(datasets, 16, "per_file")
        print(f"✅ Per-File analysis done. Log: {log_file}")
        gc.collect()

    def process_merged(self):
        print("🔍 Running Merged Dataset Analysis...")
        merged_ds = self.load_all_datasets()
        prefix = "merged"
        self.save_statistics(merged_ds, prefix)
        self._run_all_plots(merged_ds, prefix, extended_plots=True)
        print("✅ Merged dataset analysis done.")
        gc.collect()

    def _run_all_plots(self, ds, prefix, extended_plots=True):
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
        if extended_plots:
            self.plot_daily_profiles(ds, 16, prefix)
            self.plot_orbit_xy(ds, 17, prefix)
            self.plot_orbit_xz(ds, 18, prefix)
            self.plot_orbit_yz(ds, 19, prefix)
            self.plot_orbit_3d(ds, 20, prefix)
            self.plot_orbit_3d_interactive(ds, 21, prefix)
            self.plot_3d_field_vectors(ds, 22, prefix)
            self.plot_colorline_orbit(ds, 23, prefix)
            self.plot_radius_vs_z(ds, 24, prefix)
            print("Starting plot 25...")
            self.plot_bowshock_surface(ds, 25, prefix)
            print("Starting plot 26...")
            self.plot_position_density(ds, 26, prefix)
            print("Starting plot 27...")
            self.plot_field_clustering(ds, 27, prefix)
            print("Starting plot 28...")
            self.plot_velocity_vs_B(ds, 28, prefix)
        else:
            self.plot_multiday_overview([ds], 16, prefix) # The `plot_multiday_overview` logic is for a list of datasets, so I will pass a list with a single dataset in per-file mode.
        gc.collect()

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