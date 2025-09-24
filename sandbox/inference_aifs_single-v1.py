import datetime
import os
import numpy as np
import torch
from collections import defaultdict

import earthkit.data as ekd
import earthkit.regrid as ekr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.tri as tri

from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state
from anemoi.models.layers.processor import TransformerProcessor
from ecmwf.opendata import Client as OpendataClient

# Create dummy flash_attn package and submodule
import sys
import types

# --- Create dummy flash_attn package ---
flash_attn = types.ModuleType('flash_attn')
flash_attn_interface = types.ModuleType('flash_attn_interface')

# Dummy function to satisfy checkpoint
def flash_attn_func(*args, **kwargs):
    raise RuntimeError("This is a dummy flash_attn_func. Should not be called during inference with replaced processor.")

flash_attn_interface.flash_attn_func = flash_attn_func

# Register modules
flash_attn.flash_attn_interface = flash_attn_interface
sys.modules['flash_attn'] = flash_attn
sys.modules['flash_attn.flash_attn_interface'] = flash_attn_interface

# --- Definition of constants ---
# SCRIPT CONSTANT
RESULTS_FOLDER = "../reports/figures"
EXPERIENCE = "inference_aifs_single-v1"

# INPUT VARIABLE
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

def fix(lons):
    # Shift the longitudes from 0-360 to -180-180
    return np.where(lons > 180, lons - 360, lons)

if __name__ == "__main__":
    # Create necessary dir
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

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

    # Modify model to NOT use flash-attn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("../data/aifs-single-mse-1.0.ckpt", map_location=device, weights_only=False).to(device)
    model.model.processor = TransformerProcessor(
        num_layers=16,
        window_size=1024,
        num_channels=1024,
        num_chunks=2,
        activation='GELU',
        num_heads=16,
        mlp_hidden_ratio=4,
        dropout_p=0.0,
        attention_implementation="scaled_dot_product_attention").to(device)

    print(" > Model modified to use 'scaled_dot_product_attention'.")
    runner = SimpleRunner(checkpoint, device="cuda")
    runner.model = model

    # Run the forecast
    for state in runner.run(input_state=input_state, lead_time=12):
        print_state(state)

    ## Plot generation
    DISP_VAR = "100u"
    latitudes = state["latitudes"]
    longitudes = state["longitudes"]
    values = state["fields"][DISP_VAR]

    fixed_longitudes = fix(longitudes)
    print(' -- longitudes --')
    print('- len: ', len(fixed_longitudes))
    print('- [0]: ', fixed_longitudes[0])
    print('- [-1]: ', fixed_longitudes[-1])
    print('- res: ', fixed_longitudes[1] - fixed_longitudes[0])
    print('\n')

    print(' -- latitudes --')
    print('- len: ', len(latitudes))
    print('- [0]: ', latitudes[0])
    print('- [-1]: ', latitudes[-1])
    print('- res: ', latitudes[1] - latitudes[0])
    print('\n')

    fig, ax = plt.subplots(figsize=(11, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")

    triangulation = tri.Triangulation(fix(longitudes), latitudes)

    contour=ax.tricontourf(triangulation, values, levels=20, transform=ccrs.PlateCarree(), cmap="RdBu")
    cbar = fig.colorbar(contour, ax=ax, orientation="vertical", shrink=0.7, label="100u")

    plt.title("100m winds (100u) at {}".format(state["date"]))
    plt.savefig(os.path.join(RESULTS_FOLDER, f"{EXPERIENCE}_{DISP_VAR}_{DATE}"), )

    print(" > Program finished successfully!")

