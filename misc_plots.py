import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    return mo, np, pd, plt


@app.cell
def _(pd):
    df = pd.read_csv('./FPbase_Spectra.csv')
    df
    return (df,)


@app.cell
def _(df, mo, np, plt):
    import cramerif
    import matplotlib.colors as mplc

    # --- 1. Style Customization ---
    plt.rcParams['font.family'] = 'sans'
    plt.rcParams['text.color'] = '#282828'

    # --- 2. Create Figure and Axes ---
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # --- 3. Data & Calculations ---
    cramerif.use('batlow_categorical')
    cmap = plt.get_cmap('batlow_categorical')
    norm = mplc.Normalize(vmin=0, vmax=1)
    colors = cmap(norm([0, 1]))

    peak_mkate = df['Wavelength'][df['mKate EM'].argmax()]
    peak_mcherry = df['Wavelength'][df['mCherry EM'].argmax()]

    # Calculate Overlap Percentage
    x_vals = df['Wavelength'].to_numpy()
    y_mkate = np.nan_to_num(df['mKate EM'].to_numpy())
    y_mcherry = np.nan_to_num(df['mCherry EM'].to_numpy())
    overlap_curve = np.minimum(y_mkate, y_mcherry)
    overlap_area = np.trapz(overlap_curve, x_vals)
    total_area = np.trapz(y_mkate, x_vals)
    overlap_percentage = (overlap_area / total_area)
    # **FIX**: Use Mathtext (dollar signs) to correctly render the Greek letter Chi
    overlap_text = fr'$\chi = {overlap_percentage:.3f}$'

    # --- 4. Plotting ---
    ax.plot(df['Wavelength'], df['mKate EM'], c=colors[1], label='mKate', linewidth=2)
    ax.plot(df['Wavelength'], df['mCherry EM'], c=colors[0], label='mCherry', linewidth=2)
    ax.scatter([peak_mkate, peak_mcherry], [1, 1], c=[colors[1], colors[0]], s=40, zorder=5)

    # --- 5. Annotations ---
    # **NEW**: Add a formal whisker line for peak separation
    whisker_y = 1.05  # Vertical position of the whisker line
    whisker_height = 0.015 # Height of the ticks at each end

    # Draw the horizontal line and the two vertical end ticks
    ax.plot([peak_mkate, peak_mcherry], [whisker_y, whisker_y], color='#555555', linewidth=1)
    ax.plot([peak_mkate, peak_mkate], [whisker_y - whisker_height, whisker_y + whisker_height], color='#555555', linewidth=1)
    ax.plot([peak_mcherry, peak_mcherry], [whisker_y - whisker_height, whisker_y + whisker_height], color='#555555', linewidth=1)

    # Add the separation text label above the whisker line
    separation = abs(peak_mcherry - peak_mkate)
    separation_text = f'{separation:.1f} nm'
    midpoint_x = (peak_mkate + peak_mcherry) / 2
    ax.text(midpoint_x, whisker_y + 0.02, separation_text, ha='center', va='bottom', fontsize=10)

    # --- 6. Professional Styling & Final Touches ---
    ax.set_title('Fluorescent Protein Emission Spectra', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(r'$\lambda$ (nm)', fontsize=14)
    ax.set_ylabel('Normalized Intensity', fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#AAAAAA')
    ax.spines['bottom'].set_color('#AAAAAA')
    ax.tick_params(axis='x', colors='#555555')
    ax.tick_params(axis='y', colors='#555555')

    ax.grid(True, which='major', linestyle='--', color='#CCCCCC', alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_yticks([0, 1])

    ax.text(0.95, 0.65, overlap_text, transform=ax.transAxes,
            ha='right', va='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#AAAAAA', lw=1, alpha=0.9))

    ax.legend(frameon=False, fontsize=11, loc='upper left')

    # Adjust y-limit to make room for the new whisker annotation
    ax.set_ylim(0, 1.2)

    fig.tight_layout()

    mo.mpl.interactive(fig)
    return


if __name__ == "__main__":
    app.run()
