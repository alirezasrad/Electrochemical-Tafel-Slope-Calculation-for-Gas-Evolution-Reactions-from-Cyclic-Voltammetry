#!/usr/bin/env python3
"""
Tafel analysis (forward scan) from Excel CV data — floating window method.

- Reads Excel file named: "tafel.xlsx"
- Keeps the forward scan in a broad user-defined potential range.
- Tests many small moving potential windows inside that broad range.
- Finds the window where E vs log10(|i|) has the strongest linear fit.
- Reports the Tafel slope for the best window.
- Saves:
    - tafel_floating_results.csv
    - best_floating_window_data.csv
    - tafel_floating_plot.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# USER SETTINGS
# ==========================================================

# Excel file name
EXCEL_NAME = "tafel.xlsx"

# Broad acceptable potential range, in V vs RHE
# Change these based on the reaction you are studying.
E_MAX = -0.10   # less negative limit
E_MIN = -0.40   # more negative limit

# Floating window settings
WINDOW_SIZE = 0.05    # V; for example 0.05 V = 50 mV window
STEP_SIZE = 0.005     # V; for example 0.005 V = move by 5 mV each time
MIN_POINTS = 10       # minimum number of points needed to accept a fit


# ==========================================================
# LOAD AND CLEAN DATA
# ==========================================================

def load_excel_forward_only(path: str) -> pd.DataFrame:
    """
    Load Excel data and isolate the forward cathodic scan.

    Returns a table with:
        E     = potential, V
        i     = current density, mA/cm^2
        log_i = log10(|i|)
    """

    df = pd.read_excel(path, sheet_name=0)

    # Try to automatically detect potential and current columns
    col_E = None
    col_i = None

    for name in df.columns:
        low = name.lower()

        # Detect potential column
        if ("v" in low and "(" in low) or low.strip() == "e":
            col_E = name if col_E is None else col_E

        # Detect current column
        if ("ma" in low and "cm" in low) or low.strip() == "i":
            col_i = name if col_i is None else col_i

    if col_E is None or col_i is None:
        raise ValueError(
            "Could not detect potential/current columns. "
            "Expected something like 'V (mv)', 'E', 'I (mA/cm2)', or 'i'."
        )

    # Rename columns to simple names
    df = df.rename(columns={col_E: "E", col_i: "i"})

    # Keep only the broad potential region and negative current
    df = df[(df["E"] <= E_MAX) & (df["E"] >= E_MIN) & (df["i"] < 0)].copy()

    if df.empty:
        raise ValueError(
            "No data found in the selected potential range with negative current. "
            "Check E_MAX, E_MIN, and your Excel columns."
        )

    # Use first occurrence at each 1 mV potential bin as a simple forward-scan proxy
    df["E_bin_mV"] = (df["E"] * 1000).round(0)
    df = df.drop_duplicates(subset="E_bin_mV", keep="first").copy()

    # Calculate log10(|i|)
    df["log_i"] = np.log10(-df["i"].values)

    # Sort from high potential to low potential
    df = df.sort_values("E", ascending=False).reset_index(drop=True)

    return df[["E", "i", "log_i"]]


# ==========================================================
# LINEAR REGRESSION
# ==========================================================

def linreg_np(x: np.ndarray, y: np.ndarray):
    """
    Fit a straight line:
        y = a*x + b

    Returns:
        a  = slope
        b  = intercept
        r2 = R squared
    """

    a, b = np.polyfit(x, y, 1)
    y_pred = a * x + b

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return a, b, r2


# ==========================================================
# FIT ONE WINDOW
# ==========================================================

def fit_range(df_src: pd.DataFrame, E_hi: float, E_lo: float):
    """
    Fit E vs log10(|i|) inside one potential window.

    Returns:
        slope_mVdec = Tafel slope in mV/dec
        r2          = R squared
        n           = number of points
        intercept   = intercept in V
    """

    sub = df_src[(df_src["E"] <= E_hi) & (df_src["E"] >= E_lo)].copy()
    n = len(sub)

    if n < MIN_POINTS:
        return np.nan, np.nan, n, np.nan

    x = sub["log_i"].values
    y = sub["E"].values

    slope_Vdec, intercept, r2 = linreg_np(x, y)

    slope_mVdec = slope_Vdec * 1000.0

    return slope_mVdec, r2, n, intercept


# ==========================================================
# GENERATE FLOATING WINDOWS
# ==========================================================

def generate_floating_ranges(E_max: float, E_min: float, window_size: float, step_size: float):
    """
    Generate many moving potential windows inside the broad range.

    Example:
        E_max = -0.10
        E_min = -0.40
        window_size = 0.05
        step_size = 0.005

    Windows:
        -0.100 to -0.150
        -0.105 to -0.155
        -0.110 to -0.160
        ...
    """

    ranges = []
    E_hi = E_max

    while E_hi - window_size >= E_min:
        E_lo = E_hi - window_size
        ranges.append((round(E_hi, 4), round(E_lo, 4)))
        E_hi -= step_size

    return ranges


# ==========================================================
# MAIN ANALYSIS
# ==========================================================

def main():

    if not os.path.exists(EXCEL_NAME):
        raise FileNotFoundError(f"Could not find '{EXCEL_NAME}' in the current folder.")

    print(f"Analyzing: {EXCEL_NAME}")

    df = load_excel_forward_only(EXCEL_NAME)

    floating_ranges = generate_floating_ranges(
        E_MAX,
        E_MIN,
        WINDOW_SIZE,
        STEP_SIZE
    )

    rows = []
    per_range_subsets = {}

    # Fit every floating window
    for E_hi, E_lo in floating_ranges:

        sub = df[(df["E"] <= E_hi) & (df["E"] >= E_lo)].copy()
        slope_mVdec, r2, npts, intercept = fit_range(df, E_hi, E_lo)

        rows.append({
            "Range (V)": f"{E_hi:.4f} to {E_lo:.4f}",
            "E high (V)": E_hi,
            "E low (V)": E_lo,
            "Window size (V)": WINDOW_SIZE,
            "Points": npts,
            "Tafel slope (mV/dec)": slope_mVdec,
            "Intercept (V)": intercept,
            "R²": r2
        })

        per_range_subsets[(E_hi, E_lo)] = sub

    results = pd.DataFrame(rows)

    if results.empty:
        raise ValueError("No floating windows were generated. Check E_MAX, E_MIN, and WINDOW_SIZE.")

    # Save all floating-window results
    results.to_csv("tafel_floating_results.csv", index=False)
    print("Saved: tafel_floating_results.csv")

    # Find best R² window
    results_nonan = results.copy()
    results_nonan["R²"] = results_nonan["R²"].fillna(-np.inf)

    idx_best = results_nonan["R²"].idxmax()
    best_row = results.iloc[idx_best]

    best_E_hi = float(best_row["E high (V)"])
    best_E_lo = float(best_row["E low (V)"])

    best_sub = per_range_subsets[(best_E_hi, best_E_lo)].copy()

    best_tab = best_sub[["E", "i", "log_i"]].rename(
        columns={
            "E": "E (V)",
            "i": "i (mA/cm^2)",
            "log_i": "log10(|i|)"
        }
    )

    best_tab.to_csv("best_floating_window_data.csv", index=False)
    print("Saved: best_floating_window_data.csv")

    # Print summary
    print("\nBest floating Tafel window:")
    print(f"Potential range: {best_E_hi:.4f} to {best_E_lo:.4f} V")
    print(f"Points: {int(best_row['Points'])}")
    print(f"Tafel slope: {best_row['Tafel slope (mV/dec)']:.2f} mV/dec")
    print(f"R²: {best_row['R²']:.6f}")

    print("\nBest-window data for copy-paste into Excel:")
    print(best_tab.to_csv(sep="\t", index=False))

    # ======================================================
    # PLOT
    # ======================================================

    plt.figure(figsize=(7, 6))

    # Plot all accepted data
    plt.scatter(
        df["log_i"],
        df["E"],
        s=8,
        alpha=0.5,
        label="Forward scan data"
    )

    # Plot best window points
    plt.scatter(
        best_sub["log_i"],
        best_sub["E"],
        s=25,
        label="Best linear window"
    )

    # Plot best fit line
    a, b, _ = linreg_np(best_sub["log_i"].values, best_sub["E"].values)
    xfit = np.linspace(best_sub["log_i"].min(), best_sub["log_i"].max(), 100)
    yfit = a * xfit + b

    plt.plot(
        xfit,
        yfit,
        linewidth=2,
        label=f"Best fit: {best_E_hi:.4f} to {best_E_lo:.4f} V"
    )

    plt.xlabel("log10(|i|) (mA/cm²)")
    plt.ylabel("E (V vs RHE)")
    plt.title("Floating-window Tafel analysis")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tafel_floating_plot.png", dpi=200)

    print("Saved: tafel_floating_plot.png")


if __name__ == "__main__":
    main()
