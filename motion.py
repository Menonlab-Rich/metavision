import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import logging

    # Configure the logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(__name__)
    os.environ['JAX_LOGGING_LEVEL'] = 'ERROR'
    return (logger,)


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from metavision_core.event_io import EventsIterator
    from metavision_sdk_cv import TrailFilterAlgorithm
    from metavision_sdk_core import RoiFilterAlgorithm, PolarityFilterAlgorithm
    import marimo as mo
    import tempfile
    from pathlib import Path
    from functools import partial
    import multiprocessing
    import gc
    import jax
    import jax.numpy as jnp
    import threading
    from tqdm import tqdm
    from dataclasses import dataclass, InitVar, field
    from numbers import Number
    from scipy.spatial import KDTree
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
    from datetime import timedelta
    from typing import Optional
    return (
        EventsIterator,
        InitVar,
        KDTree,
        KalmanFilter,
        Optional,
        Path,
        PolarityFilterAlgorithm,
        Q_discrete_white_noise,
        dataclass,
        field,
        jax,
        jnp,
        mo,
        np,
        partial,
        pd,
        timedelta,
        tqdm,
    )


@app.cell
def _(KDTree, KalmanFilter, Q_discrete_white_noise, dataclass, np):
    @dataclass
    class Track:
        """A simple data class to hold the state for a single tracked point."""
        track_id: int
        kf: KalmanFilter
        age: int = 1
        hits: int = 1
        invisible_count: int = 0

    class Tracker:
        """Manages all active tracks using Kalman Filters."""
        def __init__(self, dist_thresh, max_age, min_hits):
            self.dist_thresh = dist_thresh
            self.max_age = max_age
            self.min_hits = min_hits
            self.tracks = {}
            self.next_id = 0

        def _create_kalman_filter(self, measurement):
            """Initializes a new Kalman Filter for a point."""
            # State is [x, y, vx, vy]
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.x = np.array([measurement[0], measurement[1], 0., 0.])

            # State Transition Matrix (constant velocity model)
            # Adjust dt later if needed, but 1.0 is a good start
            dt = 100
            kf.F = np.array([[1, 0, dt, 0],
                               [0, 1, 0, dt],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

            # Measurement Function
            kf.H = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]])

            # Measurement Noise Covariance
            kf.R = np.eye(2) * 10

            # Process Noise Covariance
            kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.1)

            # State Covariance Matrix
            kf.P *= 500.
            return kf

        def update(self, detections):
            """
            Updates the tracker with a new set of detections from a frame.

            Args:
                detections (np.ndarray): An array of shape (N, 2) of [x, y] coordinates.

            Returns:
                np.ndarray: An array containing the IDs of the matched detections.
            """
            if not self.tracks:
                # First frame, initialize all detections as new tracks
                for det in detections:
                    kf = self._create_kalman_filter(det)
                    self.tracks[self.next_id] = Track(track_id=self.next_id, kf=kf)
                    self.next_id += 1
                # Return an array of IDs corresponding to the initial detections
                return np.arange(len(detections))

            # 1. Predict the next state for all existing tracks
            predicted_positions = []
            track_ids = []
            for track_id, track in self.tracks.items():
                track.kf.predict()
                predicted_positions.append(track.kf.x[:2])
                track_ids.append(track_id)

            if not predicted_positions: # No active tracks, treat all as new
                 return self.update([]) # Recurse with empty tracks to init new ones

            predicted_positions = np.array(predicted_positions)

            # 2. Match detections to predictions
            # Use KDTree for efficient nearest neighbor search
            tree = KDTree(predicted_positions)
            dist, indices = tree.query(detections)

            matched_detections_idx = set()
            matched_tracks_idx = set()

            matches = []
            for det_idx, (d, track_idx) in enumerate(zip(dist, indices)):
                if d < self.dist_thresh:
                    matches.append((det_idx, track_idx))
                    matched_detections_idx.add(det_idx)
                    matched_tracks_idx.add(track_idx)

            # 3. Update matched tracks
            for det_idx, track_idx in matches:
                track_id = track_ids[track_idx]
                track = self.tracks[track_id]
                track.kf.update(detections[det_idx])
                track.hits += 1
                track.invisible_count = 0

            # 4. Manage unmatched tracks and new tracks
            # Unmatched tracks are aged
            for i, track_id in enumerate(track_ids):
                if i not in matched_tracks_idx:
                    self.tracks[track_id].invisible_count += 1

            # Unmatched detections become new tracks
            unmatched_detections = [d for i, d in enumerate(detections) if i not in matched_detections_idx]
            for det in unmatched_detections:
                kf = self._create_kalman_filter(det)
                self.tracks[self.next_id] = Track(track_id=self.next_id, kf=kf)
                self.next_id += 1

            # 5. Prune old tracks
            dead_tracks = [tid for tid, t in self.tracks.items() if t.invisible_count > self.max_age]
            for tid in dead_tracks:
                del self.tracks[tid]

            # 6. Re-build the ID list for the current detections
            # This is the most complex part: we need to map the original detection index to a track ID
            output_ids = np.full(len(detections), -1, dtype=int)

            # Go through successful matches again to assign IDs
            for det_idx, track_idx in matches:
                 output_ids[det_idx] = track_ids[track_idx]

            # Find the IDs for the newly created tracks
            new_track_ids = sorted([t.track_id for t in self.tracks.values() if t.hits == 1 and t.age == 1])

            unmatched_det_indices = [i for i, _ in enumerate(detections) if i not in matched_detections_idx]

            # Assign new IDs to the unmatched detections
            for i, det_idx in enumerate(unmatched_det_indices):
                if i < len(new_track_ids):
                    output_ids[det_idx] = new_track_ids[i]

            return output_ids
    return (Tracker,)


@app.cell
def _(
    EventsIterator,
    InitVar,
    Optional,
    Path,
    PolarityFilterAlgorithm,
    Tracker,
    dataclass,
    field,
    jax,
    jnp,
    logger,
    mo,
    np,
    partial,
    pd,
    timedelta,
    tqdm,
):
    jax.config.update("jax_debug_nans", True)

    # --- WARMUP STEP ---

    # Create some dummy data on the GPU
    _key = jax.random.PRNGKey(0)
    _dummy_a = jax.random.normal(_key, (10, 10))
    _dummy_b = jax.random.normal(_key, (10, 10))


    # Perform a small, representative calculation to "warm up" the GPU.
    # .block_until_ready() ensures the operation completes before moving on.
    print("Warming up the GPU...")
    (_dummy_a @ _dummy_b).block_until_ready() 
    print("Warmup complete.")

    @dataclass
    class Crop():
        x0: int
        y0: int
        width: int
        height: int

        @property
        def x1(self):
            return self.x0 + self.width

        @property
        def y1(self):
            return self.y0 + self.height


    @dataclass
    class File():
        path: Path
        event_start: timedelta
        crop: Optional[Crop] = None
        duration: InitVar[timedelta]
        _duration: timedelta = field(init=False, repr=False)
        event_duration: timedelta = timedelta(milliseconds=30)
        dt: int = 100

        def __post_init__(self, duration):
            self._duration = duration

        @property
        def duration(self) -> int:
            return int(self._duration.total_seconds() * 1e6)

        @property
        def start_time(self) -> int:
            """The start of the centered window in µs, aligned to the dt grid."""
            # 1. Calculate the ideal, unaligned start time
            window_start = self.event_start - (self._duration - self.event_duration) / 2
            ideal_start_us = window_start.total_seconds() * 1e6

            # 2. Round the time down to the nearest multiple of self.dt
            aligned_start_us = (ideal_start_us // self.dt) * self.dt

            return max(0, int(aligned_start_us))



    def progress_monitor(queue, file_stems):

        """Listens to the queue and updates tqdm bars."""

        pbars = {stem: tqdm(desc=stem, leave=False) for stem in file_stems}



        while True:

            msg = queue.get()

            if msg == 'shutdown':

                break



            stem = msg.get('id')

            pbar = pbars.get(stem)

            if pbar is not None:

                if 'total' in msg:

                    pbar.total = msg['total']

                    pbar.refresh()

                if 'progress' in msg:

                    pbar.update(msg['progress'])



        for pbar in pbars.values():

            pbar.close()



    class XY_Pairs_Generator:

        def __init__(self, file_path, dt, start_time=0, duration=None, max_points=1024):

            self.evts_iter = EventsIterator(str(file_path), delta_t=dt, start_ts=int(start_time), max_duration=int(duration))

            self.height, self.width = self.evts_iter.get_size()

            #self.roi_filter = RoiFilterAlgorithm(100, 100, 500, 480, False)
            self.polarity_filter = PolarityFilterAlgorithm(polarity=0) # keep only off events
            self.max_points=max_points



        def __iter__(self):

            #roi_buffer = RoiFilterAlgorithm.get_empty_output_buffer()
            polarity_buffer = PolarityFilterAlgorithm.get_empty_output_buffer()

            prev_idx, prev_evts_structured = None, None



            # Helper function for efficient conversion

            def structured_to_jnp(structured_array):

                arr = np.stack(

                    [structured_array['x'], structured_array['y'], structured_array['p'], structured_array['t']], 

                    axis=-1

                )

                return jnp.asarray(arr)



            for idx, evts in enumerate(self.evts_iter):

                #self.polarity_filter.process_events(evts, polarity_buffer)

                #current_evts_structured = polarity_buffer.numpy(copy=True)
                current_evts_structured = evts


                # min number of pts = 10
                if len(current_evts_structured) > 0:
                    if prev_evts_structured is not None:

                        # Convert to JAX arrays
                        # breakpoint()
                        source_points = structured_to_jnp(prev_evts_structured)
                        target_points = structured_to_jnp(current_evts_structured)
                        time_gap = idx - prev_idx



                        if len(source_points) < 10:

                            yield None
                            continue


                        cov_source = jnp.cov(source_points[:, :2], rowvar=False)
                        cov_target = jnp.cov(target_points[:, :2], rowvar=False)

                        det_source = jnp.linalg.det(cov_source)
                        det_target = jnp.linalg.det(cov_target)

                        degeneracy_threshold = 1e-9

                        # If either determinant is near zero, the geometry is bad.
                        if det_source < degeneracy_threshold or det_target < degeneracy_threshold:
                            print('Degenerate!')
                            # This frame is degenerate (likely collinear), so we skip it.
                            continue # Skips to the next iteration of the for loop



                        # 1. Sample down if too many points

                        if source_points.shape[0] > self.max_points:

                            key = jax.random.PRNGKey(idx)

                            perm = jax.random.permutation(key, source_points.shape[0])

                            source_points = source_points[perm[:self.max_points], :]



                        if target_points.shape[0] > self.max_points:

                            key = jax.random.PRNGKey(idx + 1)

                            perm = jax.random.permutation(key, target_points.shape[0])

                            target_points = target_points[perm[:self.max_points], :]



                        # Store the number of real points BEFORE padding

                        num_source_points = source_points.shape[0]

                        num_target_points = target_points.shape[0]



                        # 2. Pad up if too few points

                        pad_source_count = self.max_points - num_source_points

                        if pad_source_count > 0:

                            # Pad with (x=0, y=0, p=3, t=0) p=3 ensures that padding doesn't affect the prob map

                            padding = jnp.zeros((pad_source_count, 4), dtype=jnp.float32).at[:, -2].set(3)

                            source_points = jnp.vstack([source_points, padding])



                        pad_target_count = self.max_points - num_target_points

                        if pad_target_count > 0:

                            padding = jnp.zeros((pad_target_count, 4), dtype=jnp.float32).at[:, -2].set(3)

                            target_points = jnp.vstack([target_points, padding])



                        # Yield the padded arrays AND the original counts

                        yield (prev_idx, source_points, target_points, time_gap, 

                               num_source_points, num_target_points)

                    else:

                        yield None

                    prev_idx, prev_evts_structured = idx, current_evts_structured









    @partial(jax.jit, static_argnames=('max_iterations', 'tolerance'))

    def calc_cpd_values(Y, X, w, lamda, beta, max_iterations=100, tolerance=1e-5):

        jax.debug.print("Initializing calculations...")

        eps = jnp.finfo(float).eps

        M, D = Y.shape

        N = X.shape[0]



        # Initialization (pre-computed outside the loop)

        G = jnp.exp(-jnp.sum((Y[:, None, :] - Y[None, :, :])**2, axis=-1) / (2 * beta**2))



        sq_dist_xy = jnp.sum((X[:, None, :] - Y[None, :, :])**2, axis=-1)

        sigma2_init = jnp.sum(sq_dist_xy) / (D * N * M)



        # The state of our loop must be in a tuple: (iteration, W, sigma2, sigma2_old)

        initial_state = (0, jnp.zeros((M, D)), sigma2_init, jnp.inf)



        def loop_body(state):

            jax.debug.print("Looping")

            i, W, sigma2, sigma2_old = state



            # --- E-Step ---

            jax.debug.print("E Step")

            T = Y + G @ W

            _sq_dist = jnp.sum((X[:, None, :] - T[None, :, :])**2, axis=-1)

            prob_mn = jnp.exp(-_sq_dist / (2 * sigma2))



            c = (2 * jnp.pi * sigma2)**(D / 2) * (w / (1 - w)) * (M / N)

            denominators = jnp.sum(prob_mn, axis=1) + c

            P = (prob_mn / denominators[:, None]).T



            # --- M-Step (Optimized) ---

            jax.debug.print("M Step")

            P1 = jnp.sum(P, axis=1)

            PX = P @ X



            diag_P1 = jnp.diag(P1)

            A_new = diag_P1 @ G + lamda * sigma2 * jnp.identity(M)

            B_new = PX - diag_P1 @ Y

            W_new = jnp.linalg.solve(A_new, B_new)



            # --- Update sigma2 ---

            jax.debug.print("Update Sigma")

            Np = jnp.sum(P1)

            T_new = Y + G @ W_new

            _sq_dist_new = jnp.sum((X[:, None, :] - T_new[None, :, :])**2, axis=-1)

            sigma2_new = jnp.sum(P.T * _sq_dist_new) / (Np * D + eps)

            sigma2_new = jnp.maximum(sigma2_new, 1e-10)



            return (i + 1, W_new, sigma2_new, sigma2)



        def loop_cond(state):

            i, W, sigma2, sigma2_old = state

            err = jnp.abs(sigma2 - sigma2_old)

            return jnp.logical_and(i < max_iterations, err > tolerance)



        # Run the JAX-native while loop

        final_state = jax.lax.while_loop(loop_cond, loop_body, initial_state)



        # Run the E-step one final time with the converged parameters
        # to get the final P and T
        _, W_final, sigma2_final, _ = final_state

        T_final = Y + G @ W_final
        _sq_dist_final = jnp.sum((X[:, None, :] - T_final[None, :, :])**2, axis=-1)
        prob_mn_final = jnp.exp(-_sq_dist_final / (2 * sigma2_final))

        c_final = (2 * jnp.pi * sigma2_final)**(D / 2) * (w / (1 - w)) * (M / N)
        denominators_final = jnp.sum(prob_mn_final, axis=1) + c_final
        P_final = (prob_mn_final / denominators_final[:, None]).T

        # Return the transformed points AND the probability matrix
        return T_final, P_final



    def process_file(file_info, alpha, beta, lamda, w, dt, max_points=500, progress_queue=None):
        """
        This function processes a single file, applies CPD, and performs stateful tracking.
        """
        # Unpack file info
        fp = file_info.path
        st = file_info.start_time
        dur = file_info.duration

        # --- TRACKER INITIALIZATION ---
        # These parameters may need tuning
        tracker = Tracker(dist_thresh=25, max_age=25, min_hits=3)

        xy_gen = XY_Pairs_Generator(file_path=fp, dt=dt, start_time=st, duration=dur, max_points=max_points)

        all_results = []

        for frame_data in mo.status.progress_bar(xy_gen, total=dur/dt, title=fp.stem):
            if frame_data is None:
                continue

            try:
                i, padded_source_pts, padded_target_pts, time_gap, num_source, num_target = frame_data

                # --- Perform CPD as before ---
                inverted_p = (1.0 - padded_target_pts[:, 2]) * alpha
                Y = padded_source_pts.at[:, -2].multiply(alpha)
                X = padded_target_pts.at[:, -2].set(inverted_p)
                T_f, P_f = calc_cpd_values(Y[:, :-1], X[:, :-1], w, lamda=lamda, beta=beta, max_iterations=100, tolerance=1e-5)

                target_match_probs = jnp.sum(P_f, axis=0)


               # 1. Find all matched indices from the padded array (0 to max_points-1)
                matched_indices_all = jnp.where(target_match_probs > 0.3)[0]

                # 2. Filter these indices to only keep "real" points (0 to num_target-1)
                valid_real_indices = np.array(matched_indices_all)
                valid_real_indices = valid_real_indices[valid_real_indices < num_target]

                # 3. Get the FULL data (x, y, p, t) for these valid points
                #    You only need to index the padded array ONCE using the valid indices.
                matched_full_data = np.array(padded_target_pts)[valid_real_indices]

                # 4. Now, create your tracker detections from this filtered data
                detections_np = matched_full_data[:, :2] # Just the [x, y]
                if detections_np.shape[0] > 0:
                    track_ids = tracker.update(detections_np)

                    target_df = pd.DataFrame({
                        "frame": i,
                        "track_id": track_ids,
                        "x": matched_full_data[:, 0],
                        "y": matched_full_data[:, 1],
                        "p": matched_full_data[:, 2],
                        "t": matched_full_data[:, 3],
                        "time_gap": np.float32(time_gap),
                    })
                    all_results.append(target_df)

                if progress_queue:
                    progress_queue.put({'id': fp.stem, 'progress': 1})

            except Exception as e:
                logger.error(f"[{fp.stem}] Error on frame index {frame_data[0]}: {e}. Saving problematic data.")
                # Your existing error handling for saving data
                problem_source = np.array(frame_data[1])
                problem_target = np.array(frame_data[2])
                np.savez(f'problem_frame_{fp.stem}_{frame_data[0]}.npz', source=problem_source, target=problem_target)
                continue

        if not all_results:
            return None

        final_df = pd.concat(all_results, ignore_index=True)
        return final_df
    return File, process_file


@app.cell
def _(File, Path, process_file, timedelta):
    files = [
        # File(path = Path("/code/metavision/combo_nomod.raw"), 
        #       event_start = timedelta(minutes=0, seconds=11, microseconds=766549), duration=timedelta(seconds=30), dt=1e3),
        File(path = Path("/code/metavision/recording_2025-08-18_12-14-55_cd.dat"), 
             event_start= timedelta(minutes=3, seconds=37, microseconds=995800), duration=timedelta(seconds=2), dt=1e3),
        File(path = Path("/code/metavision/recording_2025-08-18_12-23-27.raw"), 
             event_start = timedelta(seconds=41, microseconds=25500), duration=timedelta(seconds=2), dt=1e3),
        File(path = Path("/code/metavision/recording_2025-08-18_12-46-51.raw"), 
             event_start = timedelta(seconds=34, microseconds=737900), duration=timedelta(seconds=2), dt=1e3),
    ]


    alpha = 100
    beta = 2
    lamda = 2
    w = 0.1
    dt = 100

    meta = {
        "frame": "int64",
        "type": "object",
        "x": "float32",
        "y": "float32",
        "p": "float32",
        "dx": "float32",
        "dy": "float32",
        "dp": "float32",
        "time_gap": "float32",
    }

    def save_df(df, path):
        if df is not None and not df.empty:
            df.to_parquet(path)
            return f"Saved {path}"
        return f"Skipped empty file for {path.stem}"

    # 1. Create a delayed object for each file that will result in a pandas DataFrame
    for file in files:
        # Get the file's unique name stem for the output path
        stem = file.path.stem
        output_path = Path("./") / f"{stem}_no_filter.parquet"

        # Create the task that processes the file
        df = process_file(file, alpha, beta, lamda, w, file.dt)
        print(save_df(df, output_path))



    print("All processing complete.")
    return (files,)


@app.cell
def _(files):
    files[0].duration
    return


if __name__ == "__main__":
    app.run()
