import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from metavision_core.event_io import EventsIterator
    from metavision_ml.preprocessing import histo
    import numpy as np
    return EventsIterator, histo, mo, np


@app.cell
def _(mo):
    fb = mo.ui.file_browser(initial_path='data')
    fb
    return (fb,)


@app.cell
def _(EventsIterator, fb, histo, np):
    raw = fb.value[0].path
    iterator = EventsIterator(str(raw), delta_t=1e6)
    volume = np.zeros((2, 1, *(iterator.get_size())))
    histo(next(iter(iterator)), volume, 1e6)

    import matplotlib.pyplot as plt
    print(volume.max())
    plt.imshow(np.squeeze(volume.mean(axis=0)))
    return


if __name__ == "__main__":
    app.run()
