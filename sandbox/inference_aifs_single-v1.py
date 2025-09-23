import datetime
import numpy as np
import torch
from collections import defaultdict

import earthkit.data as ekd
import earthkit.regrid as ekr

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state

from ecmwf.opendata import Client as OpendataClient

PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
# msl: Mean sea level pressure
# skt: Skin temperature
# sp: Surface pressure
# tcw: Total column vertically-integrated water vapour
# lsm: Land Sea Mask
# z: Geopotential
# slor: Slope of sub-gridscale orography (step 0)
# sdor: Standard deviation of sub-gridscale orography (step 0)
PARAM_SOIL =["vsw","sot"]
# vsw: Volumetric soil water (layers 1-4)
# sot: Soil temperature (layers 1-4)
PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
# q: Specific humidity
# w: vertical velocity
LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1,2]

## Date of initial conditions
DATE = OpendataClient().latest()

def get_open_data(param, levelist=[]):
    fields = defaultdict(list)
    # Get data at time t and t-1:
    for date in [DATE - datetime.timedelta(hours=6), DATE]:
        data = ekd.from_source("ecmwf-open-data", date=date, param=param, levelist=levelist) # <class 'earthkit.data.readers.grib.file.GRIBReader'>
        for f in data:  # <class 'earthkit.data.readers.grib.codes.GribField'>
            assert f.to_numpy().shape == (721,1440)
            values = np.roll(f.to_numpy(), -f.shape[1] // 2, axis=1)
            # Interpolate the data to from 0.25°x0.25° (regular lat-lon grid, 2D) to N320 (reduced gaussian grid, 1D, see definition here: https://www.ecmwf.int/en/forecasts/documentation-and-support/gaussian_n320) 
            values = ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"})
            # Add the values to the list
            name = f"{f.metadata('param')}_{f.metadata('levelist')}" if levelist else f.metadata("param")
            fields[name].append(values)

    # Create a single matrix for each parameter
    for param, values in fields.items():
        fields[param] = np.stack(values)

    return fields


if __name__ == "__main__":

    ## Import initial conditions from ECMWF Open Data
    fields = {}
    fields.update(get_open_data(param=PARAM_SFC))

    fields.update(get_open_data(param=PARAM_PL, levelist=LEVELS))
    # Convert geopotential height into geopotential (transform GH to Z)
    for level in LEVELS:
        gh = fields.pop(f"gh_{level}")
        fields[f"z_{level}"] = gh * 9.80665
        
    soil=get_open_data(param=PARAM_SOIL,levelist=SOIL_LEVELS)

    # soil parameters need to be renamed to be consistent with training
    mapping = {'sot_1': 'stl1', 'sot_2': 'stl2',
            'vsw_1': 'swvl1','vsw_2': 'swvl2'}
    for k,v in soil.items():
        fields[mapping[k]]=v

    print(" > data downloaded! ")

    # Create initial state
    input_state = dict(date=DATE, fields=fields)

    ## Load the model and run the forecast
    checkpoint = {"huggingface":"ecmwf/aifs-single-1.0"}
    print(' > CUDA availability: ', torch.cuda.is_available())

    runner = SimpleRunner(checkpoint, device="cuda")

    # Run the forecast
    for state in runner.run(input_state=input_state, lead_time=12):
        print_state(state)

    print(" > Program finished successfully!")

