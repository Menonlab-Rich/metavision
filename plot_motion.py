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
    from typing import Union, Tuple
    from pathlib import Path

    @dataclass
    class Parquet():
        phenotype: str
        crop: Tuple[int]
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

    def um_to_pixel(*args):
        return tuple(int(arg * 10/15) for arg in args)



    parquets = [
        # Parquet(path='combo_nomod_no_filter.parquet', phenotype='COMBO', crop=(0,0,-1,-1))#(280,150, 390, 300)),
        Parquet(path='recording_2025-08-18_12-23-27.parquet', phenotype='6188_DR1', crop=(350, 160, -1, -1)),
        Parquet(path='recording_2025-08-18_12-12-09.parquet', phenotype='N2', crop=um_to_pixel(0, 0, -1, -1)),
        Parquet(path='recording_2025-08-18_12-14-55.parquet', phenotype='UG1180_LITE1', crop=um_to_pixel(0, 200, 200,300)),
    ]
    return animation, mo, np, parquets, pl, plt


@app.cell
def _(mo):
    picker = mo.ui.number(debounce=True, label="Parquet Number", start=0, step=1, stop=3)
    picker
    return (picker,)


@app.cell
def _(parquets, picker):
    parquet = parquets[picker.value]
    parquet.df
    return (parquet,)


@app.cell
def _(parquet, pl):
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
                    pl.from_epoch(pl.col("start_time"), time_unit="us").alias("t"),
                ]
            )
            .unique(subset=["frame", "track_id"], keep="first")
            .sort("t")
            .group_by_dynamic(
                # 2. Use the new datetime column as the index
                index_column="t",
                every="100us",
            )
            .agg(
                # 3. Exclude the new datetime index from the aggregation
                pl.all().exclude("t")
            )
            .with_row_count("new_frame")
            .explode(pl.exclude("t", "new_frame"))
            .select(value_cols)
            .rename({"new_frame": "frame"})
        )


    df_resampled = parquet.df.sort('start_time').pipe(resample_df)

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
            ((pl.col("distance") <= 25) | (pl.col("distance").is_null())) & (pl.col("track_id").count().over("track_id") > 0)
            # & pl.col('p').eq(0)
        )
        .drop('distance')
        .drop_nulls(subset=pl.col('t'))
        .sort('frame', 'track_id')
    )

    df_final
    return (df_final,)


@app.cell
def _(df_final, parquet, pl):
    sample_duration = df_final['t'].max() - df_final['t'].min()
    event_duration = 10_000
    padding_duration = 10_000
    center_of_sample = df_final['t'].min() + (sample_duration / 2)
    half_event_window = (event_duration / 2) + padding_duration

    start_bound = center_of_sample - half_event_window
    end_bound = center_of_sample + half_event_window

    print(f"Filtering between {start_bound / 1e6:.3f}s and {end_bound / 1e6:.3f}s...")
    df_event_window = df_final.filter(
        pl.col('t').is_between(start_bound, end_bound)
    )
    # --- 3. Apply the Filter ---
    if parquet.crop is not None:
        x1, y1, x2, y2 = parquet.crop
        x1 = max(0, x1)
        y1 = max(0, y1)
        if x2 < 1:
            x2 = df_event_window.select(pl.max("x")).item()
            print(x2)
        if y2 < 1:
            y2 = df_event_window.select(pl.max("y")).item()

        print(x1, y1, x2, y2)
        df_event_window = df_event_window.filter(
            pl.col("t").is_between(start_bound, end_bound) &
            pl.col("x").is_between(x1, x2) &
            pl.col("y").is_between(y1, y2)
        )

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
def _(avg_displacement_per_frame, mo, parquet, pl):
    def _():

        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "figure.figsize": (6, 4),
            "figure.dpi": 300,
            "font.family": "sans-serif",
            "font.serif": ["DejaVu sans"],
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "axes.labelpad": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "axes.grid": True,
            "grid.color": "0.85",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,

        })

        # --- NEW: Calculate the rolling average ---
        window_size = 10  # Adjust this value to increase or decrease smoothing
        df_to_plot = avg_displacement_per_frame.with_columns(
            rolling_avg_um = pl.col("avg_displacement_um")
                .rolling_mean(window_size=window_size, min_samples=1, center=True)
        )
        # --- NEW: Define event boundaries based on time ('t') ---
        # The event is the central 30ms of the 50ms window
        center_t = df_to_plot['t'].min() + (df_to_plot['t'].max() - df_to_plot['t'].min()) / 2
        event_duration_t = 30_000  # 30ms in microseconds

        event_start_t = center_t - (event_duration_t / 2)
        event_end_t = center_t + (event_duration_t / 2)

        # Filter the DataFrame to get just the event segment
        df_event_segment = df_to_plot.filter(
            pl.col("t").is_between(event_start_t, event_end_t)
        )

        # --- PLOTTING ---
        fig, ax = plt.subplots(figsize=(12, 7))

        # 1. Plot the raw, noisy per-frame average
        ax.plot(
            df_to_plot["frame"],
            df_to_plot["avg_displacement_um"],
            label="Per-Frame Average",
            color="lightblue",
            linewidth=1.5,
            alpha=0.8
        )

        # 2. Plot the full smoothed rolling average in navy
        ax.plot(
            df_to_plot["frame"],
            df_to_plot["rolling_avg_um"],
            label=f"{window_size}-Frame Rolling Average",
            color="navy",
            linewidth=2.5
        )


        # 2. Check if the event segment is empty and plot accordingly
        if df_event_segment.height < 3:
            print("No events found in the 30ms window, plotting zero-line.")

            # Find the frame closest to the event start time
            start_frame = df_to_plot.sort(
                (pl.col("t") - event_start_t).abs()
            ).row(0, named=True)['frame']

            # Find the frame closest to the event end time
            end_frame = df_to_plot.sort(
                (pl.col("t") - event_end_t).abs()
            ).row(0, named=True)['frame']

            # Plot a horizontal line at y=0 between these frames
            ax.plot(
                [start_frame, end_frame],
                [0, 0],
                color="green",
                linewidth=2.5,
                label="Event (30ms, no data)"
            )
        else:
            # If data exists, overplot the event segment in green
            ax.plot(
                df_event_segment["frame"],
                df_event_segment["rolling_avg_um"],
                color="green",
                linewidth=2.5,
                label="Event (30ms)"
            )

        # 4. Add labels, title, grid, and legend for clarity
        ax.set_xlabel("Accumulation Period (100 µs)")
        ax.set_ylabel("Average Displacement (µm)")
        ax.set_title(f"Average Event Displacement Over Time\n{parquet.phenotype} - Single Worm")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # Display the plot
        return mo.mpl.interactive(plt.gcf())


    _()
    return


@app.cell
def _(df_event_window, mo, np, parquet, pl, plt):
    global df_for_plot
    df_for_plot = None

    def _():
        import matplotlib as mpl
        from matplotlib.collections import LineCollection
        import cramerif
        global df_for_plot
        cramerif.use('batlow')

        # Assume df_event_window is your fully processed DataFrame

        # --- 1. Prepare the Plot ---
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel("X Coordinate (µm)")
        ax.set_ylabel("Y Coordinate (µm)")
        ax.set_title(f"Track Positions Over Time\n{parquet.phenotype} - Single Worm")
        ax.set_aspect('equal', adjustable='box')


        # --- 2. Select Tracks and Generate Line Segments ---

        # Find the 5 longest tracks to plot
        top_5_tracks = (
            df_event_window.group_by("track_id")
            .len()
            .sort("len", descending=True)
            .head(100)["track_id"]
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
        )


        vmin, vmax = (0, 30e3)
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
            ax.set_xlim(left = 0, right = 960)
            ax.set_ylim(bottom=0, top=720)

        # --- 4. Finalize Plot ---
        ax.autoscale_view() # Adjust plot limits to fit the data

        # Add a color bar to show the time mapping
        cbar = fig.colorbar(lc, ax=ax, norm=norm)
        cbar.set_label("Time (µs)")

        ax.grid(True, linestyle='--', alpha=0.6)

        # --- 1. Data Preparation (Your existing code is good) ---
        # Assume df_for_plot and parquet are defined from your previous cells.
        # df_for_plot should have columns: track_id, x_um, y_um, t_relative
        # cramerif.use('batlow')

        # --- 2. Define Plot Windows and Global Settings ---

        # Define the parameters for each of the three subplots
        # Times are in microseconds (µs)
        plot_windows = [
            {"start": 20000, "end": 30000, "title": "20-30 ms"},
            {"start": 30000, "end": 60000, "title": "30-60 ms"},
            {"start": 60000, "end": 70000, "title": "60-70 ms"},
        ]

        # Determine the overall time range for a consistent color bar
        global_vmin = 20000
        global_vmax = 70000
        norm = mpl.colors.Normalize(vmin=global_vmin, vmax=global_vmax)
        cmap = plt.get_cmap('batlow')

        # --- 3. Create the Figure and Subplots ---
        # Create 1 row, 3 columns of subplots that share their x and y axes
        fig1, axes = plt.subplots(
            1, 3,
            figsize=(21, 7),
            sharex=True,
            sharey=True
        )
        fig1.suptitle(f"Track Positions Over Time\n{parquet.phenotype} - Single Worm", fontsize=16)

        # --- 4. Loop Through and Create Each Plot ---
        for i, window in enumerate(plot_windows):
            ax = axes[i]
            ax.set_title(window["title"])
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, linestyle='--', alpha=0.6)

            # Filter the main DataFrame for the current time window
            df_subplot_data = df_for_plot.filter(
                pl.col("t_relative").is_between(window["start"], window["end"])
            )

            # Get the unique track IDs present in this time slice
            track_ids_in_window = df_subplot_data["track_id"].unique()

            # Loop through each track present in this window
            for tid in track_ids_in_window:
                track_df = df_subplot_data.filter(pl.col("track_id") == tid)

                # Skip tracks with fewer than 2 points
                if track_df.height < 2:
                    continue

                x = track_df["x_um"].to_numpy()
                y = track_df["y_um"].to_numpy()
                t = track_df["t_relative"].to_numpy()

                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Create the LineCollection, passing the GLOBAL norm object
                lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
                lc.set_array(t[:-1])
                ax.add_collection(lc)

        # --- 5. Finalize and Add Shared Elements ---
        # Find the overall data range to set a consistent x/y scale
        x_min, x_max = df_for_plot["x_um"].min(), df_for_plot["x_um"].max()
        y_min, y_max = df_for_plot["y_um"].min(), df_for_plot["y_um"].max()
        axes[0].set_xlim(left = 0, right = 960)
        axes[0].set_ylim(bottom=0, top=720)

        # Add labels to the outer plots
        axes[0].set_ylabel("Y Coordinate (µm)")
        fig1.supxlabel("X Coordinate (µm)", y=0.3)


        # Add a single, shared color bar for the entire figure
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig1.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        location='bottom',
        orientation='horizontal',
        shrink=0.5,
        pad=0.4  # Adjust padding between plots and bar
    )
        cbar.ax.in_layout = False
        cbar.set_label("Relative Time (µs)")
        fig1.tight_layout(rect=[.15, 0.3, .85, 0.95]) # Adjust for suptitle and supxlabel
        # return mo.mpl.interactive(plt.gcf())

        return mo.vstack([
            mo.mpl.interactive(fig),
            mo.mpl.interactive(fig1)
        ])

    _()
    return (df_for_plot,)


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
        ax.set_xlim(0, 960)
        ax.set_ylim(0, 720)

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
def _(df_for_plot, pl):
    # 1. Identify the first point for each track to determine its channel.
    # We sort by time ('t_relative') and then find the unique first entry for each track.
    track_initial_points = (
        df_for_plot
        .sort("t_relative")
        .unique(subset=["track_id"], keep="first")
    )

    track_channels = track_initial_points.with_columns(
        pl.when(pl.col("y") >= 160)
        .then(pl.lit("mkate"))
        .otherwise(pl.lit("mcherry"))
        .alias("channel")
    ).select(["track_id", "channel"]) # We only need the track ID and its new channel label.

    # 3. Propagate these labels to all points in the original DataFrame.
    # This joins the channel label back to the main DataFrame, so every point for
    # a given track_id now has the correct 'mcherry' or 'mkate' label.
    df_labeled = df_for_plot.join(track_channels, on="track_id", how="left").with_columns(
        channel_code = pl.col('channel').cast(pl.Categorical).to_physical()
    )
    channel_mapping = (
        df_labeled
        .select('channel', 'channel_code')
        .unique()
        .sort('channel_code')
    )

    # --- 1. Define Linking Thresholds ---
    # Max time gap (in µs) to consider a link
    TIME_THRESHOLD = 5000  # 5 ms
    # Max spatial distance (in µm) to consider a link
    DISTANCE_THRESHOLD = 300 # ~20 pixels

    # --- 2. Find the Start and End Point of Every Track ---
    # Ensure data is sorted by time before finding the first/last points
    track_endpoints = (
        df_labeled.sort("t_relative")
        .group_by("track_id")
        .agg(
            # Get the first (start) and last (end) value for each column
            pl.first("x_um").alias("x_start"),
            pl.first("y_um").alias("y_start"),
            pl.first("t_relative").alias("t_start"),
            pl.last("x_um").alias("x_end"),
            pl.last("y_um").alias("y_end"),
            pl.last("t_relative").alias("t_end"),
            pl.first("channel").alias("channel"),
            pl.first("channel_code").alias("channel_code"),
        )
    )

    # --- 3. Find Plausible Links Using a Cross Join ---
    # Prepare two copies of the endpoints: one for "ends" and one for "starts"
    df_ends = track_endpoints.select(
        pl.col("track_id").alias("end_track_id"),
        pl.col("x_end"),
        pl.col("y_end"),
        pl.col("t_end"),
        pl.col("channel"),
    )
    df_starts = track_endpoints.select(
        pl.col("track_id").alias("start_track_id"),
        pl.col("x_start"),
        pl.col("y_start"),
        pl.col("t_start"),
        pl.col("channel"),
    )

    # Find all possible end-to-start pairs and filter them by our criteria
    # --- 1. Count the number of data points in each track ---
    df_with_counts = df_labeled.with_columns(
        track_length = pl.col("track_id").count().over("track_id")
    )

    # --- 2. Split the DataFrame based on the track length ---
    # Keep tracks with more than one point for line plotting
    df_tracks = df_with_counts.filter(pl.col("track_length") > 3)

    # Isolate tracks that are only a single point
    df_lone_points = df_with_counts.filter(pl.col("track_length")  < 3)

    df_tracks
    return channel_mapping, df_lone_points, df_tracks


@app.cell
def _(channel_mapping, df_tracks, mo, parquet, plt):
    # Import necessary modules
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import cramerif
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.lines import Line2D # Needed for the new legend entry

    # Use a categorical colormap for the channels
    cramerif.use('batlow_categorical')
    cmap = plt.get_cmap('batlow_categorical')

    # --- 1. Setup ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # --- 2. Normalize colors for channels ---
    if df_tracks.height > 0:
        c = df_tracks['channel_code'].to_numpy()
        norm = mcolors.Normalize(vmin=c.min(), vmax=c.max())
    else:
        norm = mcolors.Normalize(vmin=0, vmax=1) # Fallback

    # --- 3. Plot Multi-Point Tracks ---
    legend_handles = [
        # Time arrow handles
        Line2D([0], [0], marker='o', color='#00FF00', label='Track Start',
               linestyle='None', markersize=6, mec='black'),
        Line2D([0], [0], marker='X', color='#FF0000', label='Track End',
               linestyle='None', markersize=7)
    ]

    if df_tracks.height > 0:
        # Add channel handles to legend
        for row in channel_mapping.iter_rows(named=True):
            patch = mpatches.Patch(color=cmap(norm(row['channel_code'])),
                                   label=f"Channel: {row['channel']}")
            legend_handles.append(patch)

        for track_id, track_df in df_tracks.group_by("track_id"):
            track_df = track_df.sort("t_relative")
            x, y, t = track_df["x_um"], track_df["y_um"], track_df["t_relative"]

            # Get the channel color
            track_color = cmap(norm(track_df["channel_code"][0]))

            # --- Plot with new axis mapping: (X, Time, Y) ---
            ax.plot(x, t, y,
                    color=track_color,
                    alpha=0.7,
                    linewidth=1.5)

            # --- Add Start/End Markers ---
            # Add a 'start' marker (green circle)
            ax.scatter(x[0], t[0], y[0], color='#00FF00', marker='o', s=20, ec='black', zorder=10)
            # Add an 'end' marker (red 'X')
            ax.scatter(x[-1], t[-1], y[-1], color='#FF0000', marker='X', s=25, zorder=10)

    ax.legend(handles=legend_handles)

    # --- 4. Final Touches ---
    # ax.set_xlim(500, 800) # Spatial X
    # ax.set_zlim(0, 300) # Spatial Y
    # ax.set_ylim(5000, 20000)

    # Set new labels based on your request
    ax.set_xlabel("X Coordinate (µm)")
    ax.set_ylabel("Relative Time (µs)") # <--- TIME is Y-axis
    ax.set_zlabel("Y Coordinate (µm)") # <--- Spatial Y is Z-axis

    ax.set_title(f"Observed Motion Tracks (Length > 1)\n{parquet.phenotype}")

    # Adjust view: elev=elevation, azim=azimuth. This view looks at the
    # XZ plane from above and to the side, showing time moving "up".
    ax.view_init(elev=20, azim=-120)

    # Invert the Z-axis (which is now Spatial Y) for imaging convention
    #ax.invert_zaxis()
    ax.invert_xaxis()
    ax.invert_yaxis()

    mo.mpl.interactive(fig)
    return Line2D, mcolors, mpatches


@app.cell
def _(df_lone_points, df_tracks, pl):
    # --- 1. Define Linking Parameters ---
    MAX_SEARCH_DIST_UM = 500
    MAX_SEARCH_TIME_US = 50000
    SPACE_SCALE = 50.0
    TIME_SCALE = 20000.0

    # --- 2. Prepare DataFrames for Joining (now including channel_code for both) ---
    lone_points_subset = df_lone_points.select(
        pl.col("track_id").alias("lone_track_id"),
        "x_um", "y_um", "t_relative", "channel_code"
    )

    tracks_subset = df_tracks.select(
        pl.col("track_id").alias("neighbor_track_id"),
        pl.col("x_um").alias("x_neighbor"),
        pl.col("y_um").alias("y_neighbor"),
        pl.col("t_relative").alias("t_neighbor"),
        pl.col("channel_code").alias("neighbor_channel_code"), # Rename for clarity
    )

    # --- 3. Find the Best Neighbor for Each Lone Point ---
    lone_point_links = pl.DataFrame()
    if lone_points_subset.height > 0 and tracks_subset.height > 0:
        lone_point_links = (
            lone_points_subset.join(tracks_subset, how="cross")

            # ** THE FIX: Ensure channels match before doing anything else **
            .filter(pl.col("channel_code") == pl.col("neighbor_channel_code"))

            # Calculate spatial and temporal distances
            .with_columns(
                dist_um=((pl.col("x_um") - pl.col("x_neighbor"))**2 + (pl.col("y_um") - pl.col("y_neighbor"))**2).sqrt(),
                time_diff_us=(pl.col("t_relative") - pl.col("t_neighbor")).abs(),
            )
            # Filter to a reasonable search radius
            .filter(
                (pl.col("dist_um") < MAX_SEARCH_DIST_UM) &
                (pl.col("time_diff_us") < MAX_SEARCH_TIME_US)
            )
            # Calculate a unified score
            .with_columns(
                score=(pl.col("dist_um") / SPACE_SCALE)**2 + (pl.col("time_diff_us") / TIME_SCALE)**2
            )
            # For each lone point, find the single point on another track with the best score
            .sort("score")
            .group_by("lone_track_id")
            .first()
        )
    return (lone_point_links,)


@app.cell
def _(
    Line2D,
    channel_mapping,
    df_lone_points,
    df_tracks,
    lone_point_links,
    mcolors,
    mo,
    mpatches,
    parquet,
    plt,
):
    def _():
            # --- 1. Setup ---
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('batlow_categorical')

        # --- 2. Plot Multi-Point Tracks (Solid Lines) ---
        if df_tracks.height > 0:
            c = df_tracks['channel_code'].to_numpy()
            norm = mcolors.Normalize(vmin=c.min(), vmax=c.max())
            for track_id, track_df in df_tracks.group_by("track_id"):
                track_df = track_df.sort("t_relative")
                x, y, t = track_df["x_um"], track_df["y_um"], track_df["t_relative"]
                track_color = cmap(norm(track_df["channel_code"][0]))
                ax.plot(x, y, t, color=track_color, alpha=0.7, linewidth=1.5, marker='o', markersize=2)

        # --- 3. Plot Lone Points (Dark Gray Markers) ---
        if df_lone_points.height > 0:
            ax.scatter(
                df_lone_points["x_um"], df_lone_points["y_um"], df_lone_points["t_relative"],
                color='#555555', s=20, marker='x', alpha=0.6
            )

        # --- 4. Plot Lone Point -> Neighbor Links (Dotted Lines) ---
        if lone_point_links.height > 0:
            for link in lone_point_links.iter_rows(named=True):
                # The link should have the color of the track it's connecting to
                link_color = cmap(norm(link["channel_code"]))

                # Define the start (lone point) and end (neighbor) of the line
                link_x = [link["x_um"], link["x_neighbor"]]
                link_y = [link["y_um"], link["y_neighbor"]]
                link_t = [link["t_relative"], link["t_neighbor"]]

                ax.plot(link_x, link_y, link_t, color=link_color, linestyle=':', linewidth=1.2, alpha=0.9)

        # --- 5. Create Legend ---
        legend_handles = []
        if df_tracks.height > 0:
            for row in channel_mapping.iter_rows(named=True):
                patch = mpatches.Patch(color=cmap(norm(row['channel_code'])), label=f"Channel: {row['channel']}")
                legend_handles.append(patch)

        if df_lone_points.height > 0:
            lone_handle = Line2D([0], [0], marker='x', color='#555555', label='Lone Point', linestyle='None', markersize=6)
            legend_handles.append(lone_handle)

        if lone_point_links.height > 0:
            link_handle = Line2D([0], [0], linestyle=':', color='gray', label='Neighbor Link')
            legend_handles.append(link_handle)

        if legend_handles:
            ax.legend(handles=legend_handles)

        # --- 6. Final Touches ---
        ax.set_xlim(0, 960)
        ax.set_ylim(0, 720)
        ax.set_xlabel("X Coordinate (µm)")
        ax.set_ylabel("Y Coordinate (µm)")
        ax.set_zlabel("Relative Time (µs)")
        ax.set_title(f"Tracks with Lone Point Linking\n{parquet.phenotype}")
        ax.view_init(elev=25., azim=-75)
        ax.invert_yaxis()

        return mo.mpl.interactive(fig)

    _()
    return


if __name__ == "__main__":
    app.run()
