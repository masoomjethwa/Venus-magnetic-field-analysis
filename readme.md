
# Venus Magnetic Field Analysis

This repository hosts a Python script for the comprehensive analysis and visualization of Venus magnetic field data. Designed to process data from CSV files, it provides deep insights into the planet's induced magnetosphere and its intricate interactions with the solar wind.

---

### Features 🌟

* **Dual-Mode Operation**: The script offers two flexible processing modes:
    * **Per-File Analysis**: Generates 16 distinct plots and statistical summaries for each individual CSV data file.
    * **Merged Dataset Analysis**: Combines all available CSV files into a single, cohesive DataFrame for a holistic analysis, producing 28 detailed plots and a unified statistical report.
* **Advanced Visualization**: From fundamental time series and histograms to complex 3D orbital trajectories, magnetic field hodograms, and interactive Plotly visualizations, the script provides a rich array of graphical outputs.
* **Sophisticated Analysis**: It incorporates advanced techniques such as **Fast Fourier Transform (FFT)** for frequency analysis, **wavelet transforms** for time-frequency insights, **spectrograms**, **Principal Component Analysis (PCA)** for dimensionality reduction, and **K-Means clustering** to identify distinct magnetic field topologies.
* **Optimized Data Handling**: The script automates data loading, computes essential derived physical quantities (e.g., magnetic field magnitude, radial distance), and integrates **garbage collection** to enhance memory efficiency, crucial when handling large scientific datasets.

---

### Files for a GitHub Repository 📁

To successfully set up and run this project, your GitHub repository should include the following:

1.  **`mag5_merged28plots.py`**: This is the **core Python script**. It encapsulates all the logic for data ingestion, processing, analytical computations, and the generation of all plots.
2.  **`README.md`**: This Markdown file (the one you are currently reading) serves as the primary documentation for the project. It outlines the project's purpose, features, and provides instructions for setup and execution.
3.  **`data/` Directory**: Create a dedicated folder named `data/` at the root of your repository. All your input CSV files, which should follow the `MAG_*.csv` naming convention, must be placed within this directory. The script is configured to automatically locate and process these files.
4.  **`requirements.txt`**: This file lists all the Python libraries and their specific versions required for the script to function correctly. You can generate this file easily from your active Python environment using the command:
    ```bash
    pip freeze > requirements.txt
    ```

A typical repository structure would be:

````

venus-magnetic-field-analysis/
├── data/
│   ├── MAG\_YYYY-MM-ddT00-00-00\_DOY\_XXX\_S004\_3\_VSO\_SAPV.csv
│   └── ... (additional CSV data files)
├── output\_merged/
│   └── (This directory will be automatically generated and populated with plots and statistics)
├── mag5\_merged28plots.py
├── README.md
└── requirements.txt

````

---

### Getting Started 🚀

Follow these steps to get the analysis running on your local machine.

#### Prerequisites
Ensure you have **Python 3.8 or newer** installed on your system.

#### Installation
1.  **Clone the repository**: Begin by cloning this repository to your local machine:
    ```bash
    git clone [https://github.com/your-username/venus-magnetic-field-analysis.git](https://github.com/your-username/venus-magnetic-field-analysis.git)
    cd venus-magnetic-field-analysis
    ```
2.  **Place your data**: Deposit all your `MAG_*.csv` data files into the newly created `data/` directory within the cloned repository.
3.  **Install dependencies**: Install all required Python libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

#### Configuration
Open the `mag5_merged28plots.py` file in a text editor. Navigate to the `======== USER CONFIG ========` section near the top of the script and adjust the following variables as needed:

* `RUN_PER_FILE`: Set to `True` to enable analysis of each individual CSV file, or `False` to skip it.
* `RUN_MERGED`: Set to `True` to enable comprehensive analysis of all merged CSV files, or `False` to skip it.
* `INPUT_DIR`: Specify the path to your data folder. For the recommended structure, this would be `Path("./data")`.
* `OUTPUT_DIR`: Define the directory where all generated plots and statistics will be saved. For example, `Path("./output_merged")`.

#### Running the Script
After configuring the script and placing your data, execute the script from your terminal:

```bash
python mag5_merged28plots.py
````

The script will print progress messages to the console. Upon completion, the `output_merged/` directory (or your specified `OUTPUT_DIR`) will contain all the generated plots (PNG and HTML for interactive plots) and statistical CSV files.

-----

# Workflow for Magnetic Field Analysis Using Bx, By, Bz Data

This section outlines a general workflow for analyzing magnetic field data, specifically using components like $B\_x$, $B\_y$, and $B\_z$.

-----

## 1\. Data Preparation

### a. **Load Data**

  * Read your CSV/XML file, ensuring it contains columns for `timestamp`, `b_sc_x`, `b_sc_y`, `b_sc_z`, `b_vso_x`, `b_vso_y`, `b_vso_z`, `pos_vso_x`, `pos_vso_y`, and `pos_vso_z`.
  * Parse timestamps as datetime objects for proper temporal analysis.

### b. **Clean Data**

  * Check for and handle missing or NaN values, typically via interpolation or removal.
  * Filter outliers or erroneous values (e.g., magnetic field magnitudes exceeding physically plausible limits).

-----

## 2\. Coordinate and Frame Considerations

  * Understand the coordinate system of your data (e.g., VSO - Venus Solar Orbital, SC - Spacecraft).
  * Convert between coordinate systems if necessary for specific analyses.
  * Use position vectors to spatially relate magnetic field measurements.

-----

## 3\. Basic Statistical Analysis

  * Compute **magnetic field magnitude** ($B$):
    $$
    $$$$B = \\sqrt{B\_x^2 + B\_y^2 + B\_z^2}
    $$
    $$$$
    $$
  * Calculate **mean, median, and standard deviation** for $B\_x$, $B\_y$, $B\_z$, and $B$.
  * Identify temporal trends or periodicities using:
      * Rolling averages
      * Fourier Transform / Power Spectral Density analysis

-----

## 4\. Visualization

### a. Time Series Plots

  * Plot $B\_x$, $B\_y$, $B\_z$, and $|B|$ versus time to visualize magnetic field variations.

### b. Vector Plots / Hodograms

  * Plot $B\_x$ vs. $B\_y$ or $B\_y$ vs. $B\_z$ to study vector orientation and rotations.

### c. 3D Scatter Plot

  * Plot spacecraft position (e.g., `pos_vso_x`, `pos_vso_y`, `pos_vso_z`) colored by $|B|$ or individual components to visualize spatial magnetic field distribution.

### d. Magnetic Field Lines (Advanced)

  * Interpolate the vector field onto a grid around Venus and plot field lines (streamlines) for a more advanced visualization.

-----

## 5\. Event Identification & Analysis

  * Detect magnetic field features such as:
      * Sudden increases/decreases (e.g., shocks, discontinuities)
      * Magnetic reconnection signatures (e.g., rotations, reversals)
      * Boundary crossings (e.g., bow shock, magnetopause)
  * Use thresholds or machine learning clustering techniques for automated event detection.

-----

## 6\. Correlation with External Data

  * Compare magnetic field variations with:
      * Solar wind parameters (e.g., Interplanetary Magnetic Field (IMF), velocity, density)
      * Plasma measurements (if available)
      * Spacecraft trajectory and position relative to Venus

-----

## 7\. Modeling and Interpretation

  * Utilize empirical or theoretical models of Venus's induced magnetosphere to interpret observed magnetic structures.
  * Compare your data with simulation results if available.

-----

## 8\. Reporting & Documentation

  * Summarize findings using clear figures, tables, and descriptive text.
  * Document all data processing steps thoroughly to ensure reproducibility.

-----

# Example Python Outline

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("venus_magnetic_field.csv", parse_dates=["timestamp"])

# 2. Clean data
df = df.dropna(subset=["b_vso_x", "b_vso_y", "b_vso_z"])

# 3. Calculate magnitude
df['B_magnitude'] = np.sqrt(df['b_vso_x']**2 + df['b_vso_y']**2 + df['b_vso_z']**2)

# 4a. Plot time series
plt.figure(figsize=(10,6))
plt.plot(df['timestamp'], df['b_vso_x'], label='Bx')
plt.plot(df['timestamp'], df['b_vso_y'], label='By')
plt.plot(df['timestamp'], df['b_vso_z'], label='Bz')
plt.plot(df['timestamp'], df['B_magnitude'], label='|B|', linestyle='--', color='k')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Magnetic Field (nT)')
plt.title('Venus Magnetic Field Components Over Time')
plt.show()

# 4b. Hodogram example Bx vs By
plt.figure(figsize=(6,6))
plt.scatter(df['b_vso_x'], df['b_vso_y'], s=1)
plt.xlabel('Bx (nT)')
plt.ylabel('By (nT)')
plt.title('Hodogram of Bx vs By')
plt.grid(True)
plt.show()
```

-----

