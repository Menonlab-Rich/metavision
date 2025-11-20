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
    return parquets, pl


@app.cell
def _(parquets, pl):
    parquets[0].df.filter(pl.col('p') == 0).head()
    return


if __name__ == "__main__":
    app.run()
