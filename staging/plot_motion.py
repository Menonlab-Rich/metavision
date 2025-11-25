import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt

    import matplotlib.animation as animation

    import pandas as pd

    import polars as pl

    import numpy as np

    import marimo as mo

    from dataclasses import dataclass, InitVar, field

    from typing import Union

    from pathlib import Path



    @dataclass

    class Parquet():

        phenotype: str

        path: InitVar[Union[str,Path]]

        _path: Path = field(init=False, repr=False)



        def __post_init__(self, path: Union[str,Path]):

            self._path = Path(path)



        @property

        def path(self):

            return self._path



        @property

        def df(self) -> pl.DataFrame:

            with open(self.path, mode='rb') as f:

                return pl.read_parquet(f)







    parquets = [

        Parquet(path='recording_2025-08-18_12-46-51.parquet', phenotype='17618_HIC'),

        Parquet(path='recording_2025-08-18_12-23-27.parquet', phenotype='6188_DR1'),

        Parquet(path='recording_2025-08-18_12-12-09.parquet', phenotype='N2'),

        Parquet(path='recording_2025-08-18_12-14-55_cd.parquet', phenotype='UG1180_LITE1'),

    ]
    return animation, mo, np, parquets, pl, plt


@app.cell
def _(parquets):
    parquets[1].df.height
    return


@app.cell
def _(parquets, pl):
    def resample_df(df: pl.DataFrame) -> pl.DataFrame:

        # Define all columns to be aggregated and then exploded

        value_cols = [

            "track_id",

            "x",

            "y",

            "p",

            "t",

            "new_frame"

        ]



        return (

            df.with_columns(

                [

                    pl.col("track_id").cast(pl.Int64),

                    pl.col("x").cast(pl.Float32),

                    pl.col("y").cast(pl.Float32),

                    # 1. Create a datetime column from the integer 't'

                    pl.from_epoch(pl.col("t"), time_unit="us").alias("t_datetime"),

                ]

            )

            .unique(subset=["frame", "track_id"], keep="first")

            .sort("t_datetime")

            .group_by_dynamic(

                # 2. Use the new datetime column as the index

                index_column="t_datetime",

                every="100us",

            )

            .agg(

                # 3. Exclude the new datetime index from the aggregation

                pl.all().exclude("t_datetime")

            )

            .with_row_count("new_frame")

            .explode(pl.exclude("t_datetime", "new_frame"))

            .select(value_cols)

            .rename({"new_frame": "frame"})

        )



    parquet = parquets[0]

    df_resampled = parquet.df.sort('t').pipe(resample_df)



    df_final = (

        df_resampled

        # Crucially, sort by track_id and the frame number

        .sort("track_id", "frame")

        .upsample(

            time_column="frame", # The column that defines the timeline

            every="1i", # "1i" means every 1 integer step

            group_by="track_id",

            maintain_order=True,

        )

        .with_columns(

            # Interpolate the columns that should change smoothly

            pl.col("track_id").forward_fill(),

            pl.col("x").interpolate().over("track_id"),

            pl.col("y").interpolate().over("track_id"),

            pl.col("t").interpolate().over("track_id").cast(pl.Int64),

            # Forward fill columns that should be constant for the track

            pl.col("p").forward_fill().over("track_id"),

        )

        .with_columns(

            # Calculate distance

            distance=(

                (pl.col("x") - pl.col("x").shift(1).over("track_id"))**2 +

                (pl.col("y") - pl.col("y").shift(1).over("track_id"))**2

            ).sqrt()

        )

        .filter(

            # Keep rows that move <= 5 pixels or are the first point of a track

            ((pl.col("distance") <= 5) | (pl.col("distance").is_null())) & (pl.col("track_id").count().over("track_id") > 3)

        )

        .drop('distance')

        .drop_nulls(subset=pl.col('t'))

        .sort('frame', 'track_id')

    )



    df_final
    return df_final, parquet


@app.cell
def _(df_final, pl):
    sample_duration = df_final['t'].max() - df_final['t'].min()

    event_duration = 1_000

    padding_duration = 100_000

    center_of_sample = df_final['t'].min() + (sample_duration / 2)

    half_event_window = (event_duration / 2) + padding_duration



    start_bound = center_of_sample - half_event_window

    end_bound = center_of_sample + half_event_window



    print(f"Filtering between {start_bound / 1e6:.3f}s and {end_bound / 1e6:.3f}s...")



    # --- 3. Apply the Filter ---

    df_event_window = df_final.filter(pl.col("t").is_between(start_bound, end_bound))



    df_event_window
    return (df_event_window,)


@app.cell
def _(df_event_window, pl):
    avg_displacement_per_frame = (

        df_event_window

        .sort("track_id", "frame")

        .with_columns(

            # Calculate distance

            distance=(

                (pl.col("x") - pl.col("x").shift(1).over(["track_id"]))**2 +

                (pl.col("y") - pl.col("y").shift(1).over("track_id"))**2

            ).sqrt(),

            frame=(

                pl.col('frame') - pl.col('frame').min()

            )

        )

        .with_columns(

            displacement_um=(pl.col('distance') * 15/10) # convert pixels to um: px * pitch/magnification

        )

        # Now group by frame and calculate the mean

        .group_by("frame", maintain_order=True)

        .agg(

            pl.col("distance").mean().alias("avg_displacement_px"),

            pl.col('displacement_um').mean().alias('avg_displacement_um'),

            pl.col('t').max().alias('t')

        )

        .sort("t")

    )



    avg_displacement_per_frame
    return (avg_displacement_per_frame,)


@app.cell
def _(avg_displacement_per_frame, mo, parquet, pl, plt):
    def _():

        # --- NEW: Calculate the rolling average ---

        window_size = 10  # Adjust this value to increase or decrease smoothing

        df_to_plot = avg_displacement_per_frame.with_columns(

            rolling_avg_um = pl.col("avg_displacement_um")

                .rolling_mean(window_size=window_size, min_samples=1, center=True)

        )



        # --- PLOTTING ---

        fig, ax = plt.subplots(figsize=(12, 7))



        # 1. Plot the raw, noisy per-frame average as a thin, semi-transparent line

        ax.plot(

            df_to_plot["frame"],

            df_to_plot["avg_displacement_um"],

            label="Per-Frame Average",

            color="lightblue",

            linewidth=1.5,

            alpha=0.8

        )



        # 2. Plot the smoothed rolling average as a solid, prominent line

        ax.plot(

            df_to_plot["frame"],

            df_to_plot["rolling_avg_um"],

            label=f"{window_size}-Frame Rolling Average",

            color="navy",

            linewidth=2.5

        )



        # 3. Add labels, title, grid, and legend for clarity

        ax.set_xlabel("Accumulation Period (100 µs)")

        ax.set_ylabel("Average Displacement (µm)")

        ax.set_title(f"Average Event Displacement Over Time\n{parquet.phenotype}")

        ax.legend()

        ax.grid(True, linestyle='--', alpha=0.6)



        # Display the plot

        return mo.mpl.interactive(plt.gcf())





    _()
    return


@app.cell
def _(animation, df_event_window, mo, parquets, pl, plt):
    def create_movie():

        fig, ax = plt.subplots(figsize=(8, 6))





        # Set plot limits based on the event window's data range to zoom in.

        x_min, x_max = df_event_window["x"].min(), df_event_window["x"].max()

        y_min, y_max = df_event_window["y"].min(), df_event_window["y"].max()



        # Add a 10% margin for better visualization

        margin_x = (x_max - x_min) * 0.1

        margin_y = (y_max - y_min) * 0.1

        ax.set_xlim(x_min - margin_x, x_max + margin_x)

        ax.set_ylim(y_min - margin_y, y_max + margin_y)



        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel("X Coordinate")

        ax.set_ylabel("Y Coordinate")



        # Initialize plot elements

        scatter = ax.scatter([], [], s=10) # s is marker size

        title = ax.set_title("")





        # Get the min and max frame numbers

        min_anim_frame = df_event_window["frame"].min()

        max_anim_frame = df_event_window["frame"].max()



        def update(frame):

            # Map animation index (0 to N-1) to the actual frame numbers

            current_frame = min_anim_frame + frame



            # Filter the event window DataFrame for the current frame

            data_for_frame = df_event_window.filter(pl.col("frame") == current_frame)



            # Update the scatter plot data

            scatter.set_offsets(data_for_frame[["x", "y"]].to_numpy())



            # Update the title

            title.set_text(f"Frame: {current_frame}")



            return scatter, title



        # Calculate the number of frames based on the event window's range

        num_frames = max_anim_frame - min_anim_frame + 1



        # Create the animation object

        anim = animation.FuncAnimation(fig, update, frames=num_frames, blit=True)



        # Save the animation as an MP4 (assuming ffmpeg is installed)

        output_filename = f"{parquets[0].phenotype}_event_window_50ms_pol_0.mp4"

        anim.save(writer='ffmpeg', filename=output_filename, fps=12)



        plt.close(fig) # Prevents the static plot from displaying in your notebook

        print(f"Animation saved to {output_filename}")



    mo.ui.button(on_click=create_movie, label="Press to create movie")
    return


@app.cell
def _(df_event_window, pl):
    # --- 2. Select Tracks and Generate Line Segments ---



    # Find the 5 longest tracks to plot

    top_5_tracks = (

        df_event_window.group_by("track_id")

        .len()

        .sort("len", descending=True)

        .head(15)["track_id"]

    )



    # --- 1. Prepare data with per-step calculations ---

    df_with_steps = (

        df_event_window

        .filter(pl.col("track_id").is_in(top_5_tracks))

        .with_columns(

            # Calculate per-step distance first

            step_distance=(

                (pl.col("x") - pl.col("x").shift(1).over("track_id"))**2 +

                (pl.col("y") - pl.col("y").shift(1).over("track_id"))**2

            ).sqrt()

        )

    )



    # --- 2. Calculate and filter track-level metrics ---

    good_track_ids = (

        df_with_steps

        .group_by("track_id")

        .agg(

            # Calculate metrics for each track

            net_displacement=(

                (pl.col("x").last() - pl.col("x").first())**2 +

                (pl.col("y").last() - pl.col("y").first())**2

            ).sqrt(),

            path_length=pl.col("step_distance").sum()

        )

        .with_columns(

            # Calculate straightness from the metrics

            straightness=(pl.col("net_displacement") / pl.col("path_length"))

        )

        # .filter(

        #     # Keep tracks that are NOT long AND straight

        #     (pl.col("path_length") < 180)

        # )

        .select("track_id") # Select only the IDs of the good tracks

    )



    # --- 3. Join back to get full data for good tracks and apply final transforms ---

    df_for_plot = (

        df_event_window

        # A semi join is a filtering join. It keeps all rows from df_event_window

        # where the track_id exists in the good_track_ids DataFrame.

        .join(good_track_ids, on="track_id", how="semi")

        .with_columns(

            # Now apply the final transformations for plotting

            t_relative=pl.col("t") - pl.col("t").min(),

            x_um=pl.col("x") * 15/10,

            y_um=pl.col("y") * 15/10,

        )
         .filter(
                pl.col('x_um').is_between(400, 800) &
                pl.col('y_um').is_between(0, 300)
          )

    )

    df_for_plot
    return df_for_plot, top_5_tracks


@app.cell
def _(df_for_plot, mo, np, parquet, pl, plt, top_5_tracks):
    def _():

        import matplotlib as mpl

        from matplotlib.collections import LineCollection

        import cramerif

        cramerif.use('batlow')



        # Assume df_event_window is your fully processed DataFrame



        # --- 1. Prepare the Plot ---

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.set_xlabel("X Coordinate (µm)")

        ax.set_ylabel("Y Coordinate (µm)")

        ax.set_title(f"Track Positions Over Time\n{parquet.phenotype}")

        ax.set_aspect('equal', adjustable='box')






        vmin, vmax = (0, 230e3)

        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        cmap = plt.get_cmap('batlow')



        # Loop through each of the selected track IDs

        for tid in top_5_tracks:

            track_df = df_for_plot.filter(pl.col("track_id") == tid)



            # Get x, y, and t data as numpy arrays

            x = track_df["x_um"].to_numpy()

            y = track_df["y_um"].to_numpy()

            t = track_df["t_relative"].to_numpy()



            # Create an array of points, shape: (N, 1, 2)

            points = np.array([x, y]).T.reshape(-1, 1, 2)



            # Create an array of segments, shape: (N-1, 2, 2)

            segments = np.concatenate([points[:-1], points[1:]], axis=1)



            # --- 3. Create and Add the LineCollection ---



            # Create the LineCollection object

            lc = LineCollection(segments, cmap=cmap, linewidth=2, norm=norm)



            # Set the values used for color mapping (the timestamp at the start of each segment)

            lc.set_array(t[:-1])



            # Add the collection to the plot

            ax.add_collection(lc)

            #ax.set_xlim(left = 0, right = 960)

            #ax.set_ylim(bottom=0, top=720)



        # --- 4. Finalize Plot ---

        ax.autoscale_view() # Adjust plot limits to fit the data



        # Add a color bar to show the time mapping

        cbar = fig.colorbar(lc, ax=ax, norm=norm)

        cbar.set_label("Time (µs)")



        ax.grid(True, linestyle='--', alpha=0.6)

        return mo.mpl.interactive(plt.gcf())



    _()
    return


@app.cell
def _(df_for_plot, mo, top_5_tracks):
    def _():

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        import cramerif
        import polars as pl
        import numpy as np

        # Assuming df_event_window and parquet are defined in your scope
        # For example:
        # import polars.testing
        # df_event_window = polars.testing.make_dataframe() # Placeholder
        # class Parquet:
        #    phenotype = "Example Phenotype"
        # parquet = Parquet() # Placeholder

        cramerif.use('batlow')

        # --- 1. Prepare the Plot ---
        # Create 3 subplots in one row, sharing X and Y axes
        rows = 5
        cols = 5
        step = 230_000//(rows * cols)
        times = np.arange(0,230_000 + step, step)
        fig, axes = plt.subplots(cols, rows, figsize=(10, 40), sharex=True, sharey=True,)
        axes = np.reshape(axes, -1)

        # --- 4. Setup Colormap and Normalization ---
        # This normalization is shared across all subplots
        vmin, vmax = (0, 230e3) # Using your new vmax
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap('batlow')

        # --- 5. Loop Through Subplots and Plot Data ---

        # Define the time bins and titles for each subplot
        # subplots_info = [
        #     (axes[0], (0, 15000), "0 - 15,000 µs"),
        #     (axes[1], (15000, 18000), "15,000 - 18,000 µs"),
        #     (axes[2], (16000, 30000), "18,000 - 30,000 µs"),
        #     (axes[3], (30000, 230000), "30,000 - 230,000 µs")
        # ]

        subplots_info = []
        print(times)
        for i, ax in enumerate(axes):
                from_time = times[i]
                to_time = times[i + 1]
                subplots_info.append((ax, (from_time, to_time), f'{from_time} μs - {to_time} μs'))

        # Loop through each subplot (ax), its time bin (t_min, t_max), and title
        # We use enumerate to get the index `i` to identify the last bin
        for i, (ax, (t_min, t_max), title) in enumerate(subplots_info):

            # Set properties for this specific subplot
            ax.set_xlabel("X Coordinate (µm)")
            ax.set_title(title)
            ax.set_aspect('auto', adjustable='box')
            ax.grid(True, linestyle='--', alpha=0.6)

            # Only set Y label for the first plot
            if not (i % rows):
                ax.set_ylabel("Y Coordinate (µm)")

            # Loop through each of the selected track IDs
            for tid in top_5_tracks:
                # Get the full data for this track
                track_df = df_for_plot.filter(pl.col("track_id") == tid)

                # Get x, y, and t data as numpy arrays
                x = track_df["x_um"].to_numpy()
                y = track_df["y_um"].to_numpy()
                t = track_df["t_relative"].to_numpy()

                # Need at least 2 points to make a segment
                if len(t) < 2:
                    continue

                # Create an array of points, shape: (N, 1, 2)
                points = np.array([x, y]).T.reshape(-1, 1, 2)

                # Create an array of segments, shape: (N-1, 2, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Get the time for the *start* of each segment
                segment_times = t[:-1]

                # --- Filter segments based on the time bin ---
                # Check if this is the last bin in the list
                is_last_bin = (i == len(subplots_info) - 1)

                if is_last_bin: # Make the last bin inclusive of the endpoint
                     mask = (segment_times >= t_min) & (segment_times <= t_max)
                else: # Other bins are [min, max)
                     mask = (segment_times >= t_min) & (segment_times < t_max)

                # Apply the mask to get only segments and times in this bin
                segments_in_bin = segments[mask]
                times_in_bin = segment_times[mask]

                # If no segments from this track are in this time bin, skip
                if len(segments_in_bin) == 0:
                    continue

                # --- Create and Add the LineCollection ---
                lc = LineCollection(segments_in_bin, cmap=cmap, linewidth=2, norm=norm)
                lc.set_array(times_in_bin)
                ax.add_collection(lc)

            # Adjust plot limits to fit the data *for this subplot*
            # Since axes are shared, this will expand the limits to fit all data
            ax.autoscale_view()

        # --- 6. Finalize Plot ---

        # Create a "dummy" mappable object for the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) # We just need the norm and cmap

        # --- MODIFIED SECTION ---

        # Add a single color bar to the figure, spanning all subplots
        cbar = fig.colorbar(
            sm,
            ax=axes,                     # Apply to all axes
            orientation='horizontal',    # Place it horizontally
            pad=0.3,                     # Padding below the plots
            shrink=0.8,                  # Make it 80% of the total width
            aspect=40                    # Control the width (higher is thinner)
        )
        cbar.set_label("Time (µs)")

        # Adjust layout to prevent titles from overlapping and reduce spacing
        # We use this instead of tight_layout for more control
        fig.subplots_adjust(
            left=0.05,    # Left margin
            right=0.95,   # Right margin
            bottom=0.25,   # Bottom margin (make space for colorbar)
            top=0.9,      # Top margin (make space for suptitle)
            wspace=0.05   # Width space between plots (very close)
        )

        return mo.mpl.interactive(plt.gcf())
        #plt.show() # Use plt.show() for a standard environment


    # Call the function
    _()
    return


@app.cell
def _(df_for_plot, mo, pl, plt):
    def plot_flow_components():
        import cramerif
        import matplotlib.colors as mcolors
        import matplotlib.ticker as ticker

        # Setup colormap (using diverging 'vik' or 'RdBu_r')
        try:
            cramerif.use('vik')
            cmap_name = 'vik'
        except:
            cmap_name = 'RdBu_r'

        px_to_um = 1.5

        # --- 1. Calculate Raw Velocities for Both Axes ---
        df_temp = df_for_plot.sort("track_id", "t").with_columns(
            raw_dx = pl.col("x").diff().over("track_id") * px_to_um,
            raw_dy = pl.col("y").diff().over("track_id") * px_to_um
        ).filter(
            pl.col("raw_dx").is_not_null() & pl.col("raw_dy").is_not_null()
        )

        # --- 2. Define Helper to Process and Plot a Single Component ---
        def create_hovmoller(component_col, title_prefix, noise_threshold=0.2):
            # A. Calculate Bias (Median)
            bias = df_temp.select(pl.col(component_col).median()).item()
            print(f"{title_prefix} Bias Correction: {bias:.4f} µm/step")

            # B. Apply Correction and Noise Gate
            df_clean = df_temp.with_columns(
                corrected = pl.col(component_col) - bias
            ).with_columns(
                clean_val = pl.when(pl.col("corrected").abs() < noise_threshold)
                           .then(0.0)
                           .otherwise(pl.col("corrected"))
            )

            # C. Binning (Always using X-Position for the spatial axis)
            x_bins = 50
            t_bins = 100000

            heatmap_data = (
                df_clean
                .with_columns(
                    x_bin = ((pl.col("x") - pl.col("x").min()) / (pl.col("x").max() - pl.col("x").min()) * (x_bins-1)).round(),
                    t_bin = ((pl.col("t") - pl.col("t").min()) / (pl.col("t").max() - pl.col("t").min()) * (t_bins-1)).round()
                )
                .group_by(["x_bin", "t_bin"])
                .agg(
                    mean_velocity = (pl.col("clean_val").mean() / 100)
                )
                .sort("x_bin", "t_bin")
            )

            # D. Pivot to Matrix
            matrix_df = (
                heatmap_data
                .pivot(values="mean_velocity", index="t_bin", columns="x_bin")
                .fill_null(0)
            )
            data_matrix = matrix_df.select(pl.all().exclude("t_bin")).to_numpy().T
            # E. Plotting
            fig, ax = plt.subplots(figsize=(10, 6))

            x_min_um = df_clean["x"].min() * px_to_um
            x_max_um = df_clean["x"].max() * px_to_um
            t_extent = (df_clean["t"].max() - df_clean["t"].min()) / 1000

            extent = [
                0,  t_extent,
                x_min_um, x_max_um,
            ]

            # Center colorbar at 0
            norm = mcolors.CenteredNorm(vcenter=0)

            im = ax.imshow(data_matrix, aspect='auto', origin='lower',
                           cmap=cmap_name, norm=norm, extent=extent, interpolation='nearest')

            ax.set_title(f"{title_prefix} Flow Density")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("X Position (μm)")
            ax.xaxis.set_major_locator(ticker.MultipleLocator(base=20))

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(f"Mean {title_prefix} Velocity (µm/μs)")
            ax.axvline((t_extent // 2), ls='--', color='r' )
            return mo.mpl.interactive(fig)

        # --- 3. Generate the Two Plots ---
        # Plot 1: Longitudinal Flow (X-Velocity)
        plot_x = create_hovmoller("raw_dx", "Longitudinal (X)")

        # Plot 2: Transverse Flow (Y-Velocity)
        plot_y = create_hovmoller("raw_dy", "Transverse (Y)")

        return plot_x, plot_y

    plot_flow_components()
    return


if __name__ == "__main__":
    app.run()
