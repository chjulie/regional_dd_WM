import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import datetime
    import numpy as np
    import torch
    from collections import defaultdict

    import earthkit.data as ekd
    import earthkit.regrid as ekr

    from anemoi.inference.runners.simple import SimpleRunner
    from anemoi.inference.outputs.printer import print_state

    from ecmwf.opendata import Client as OpendataClient
    return


if __name__ == "__main__":
    app.run()
