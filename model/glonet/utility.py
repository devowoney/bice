import gc
import xesmf as xe
import copernicusmarine as cmc
from datetime import datetime, timedelta
import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from netCDF4 import Dataset as ncDataset
import os
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import numpy as np


def get_normalizer1(model_dir: str):
    level = "L0"
    gmean = np.concatenate(
        [
            [np.load(model_dir + "/" + level + "/zos_mean.npy")],
            np.load(model_dir + "/" + level + "/thetao_mean.npy"),
            np.load(model_dir + "/" + level + "/so_mean.npy"),
            np.load(model_dir + "/" + level + "/uo_mean.npy"),
            np.load(model_dir + "/" + level + "/vo_mean.npy"),
        ]
    )

    gstd = np.concatenate(
        [
            [np.load(model_dir + "/" + level + "/zos_std.npy")],
            np.load(model_dir + "/" + level + "/thetao_std.npy"),
            np.load(model_dir + "/" + level + "/so_std.npy"),
            np.load(model_dir + "/" + level + "/uo_std.npy"),
            np.load(model_dir + "/" + level + "/vo_std.npy"),
        ]
    )

    transform = transforms.Normalize(mean=gmean, std=gstd)

    return transform


def get_denormalizer1(model_dir: str):
    level = "L0"
    gmean = np.concatenate(
        [
            [np.load(model_dir + "/" + level + "/zos_mean.npy")],
            np.load(model_dir + "/" + level + "/thetao_mean.npy"),
            np.load(model_dir + "/" + level + "/so_mean.npy"),
            np.load(model_dir + "/" + level + "/uo_mean.npy"),
            np.load(model_dir + "/" + level + "/vo_mean.npy"),
        ]
    )

    gstd = np.concatenate(
        [
            [np.load(model_dir + "/" + level + "/zos_std.npy")],
            np.load(model_dir + "/" + level + "/thetao_std.npy"),
            np.load(model_dir + "/" + level + "/so_std.npy"),
            np.load(model_dir + "/" + level + "/uo_std.npy"),
            np.load(model_dir + "/" + level + "/vo_std.npy"),
        ]
    )

    denormalizer = transforms.Normalize(
        mean=[-m / s for m, s in zip(gmean, gstd)], std=[1 / s for s in gstd]
    )
    return denormalizer


####################################################################3
################################################################################


def get_normalizer2(model_dir: str):
    levels = {
        "L50",
        "L100",
        "L150",
        "L222",
        "L318",
        "L380",
        "L450",
        "L540",
        "L640",
        "L763",
    }
    mean_thetao = []
    mean_so = []
    mean_uo = []
    mean_vo = []
    std_thetao = []
    std_so = []
    std_uo = []
    std_vo = []
    for level in sorted(levels):
        mean_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_mean.npy")[0]
        )
        mean_so.append(np.load(model_dir + "/" + level + "/so_mean.npy")[0])
        mean_uo.append(np.load(model_dir + "/" + level + "/uo_mean.npy")[0])
        mean_vo.append(np.load(model_dir + "/" + level + "/vo_mean.npy")[0])

        std_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_std.npy")[0]
        )
        std_so.append(np.load(model_dir + "/" + level + "/so_std.npy")[0])
        std_uo.append(np.load(model_dir + "/" + level + "/uo_std.npy")[0])
        std_vo.append(np.load(model_dir + "/" + level + "/vo_std.npy")[0])

    gmean = np.concatenate([mean_thetao, mean_so, mean_uo, mean_vo])
    gstd = np.concatenate([std_thetao, std_so, std_uo, std_vo])

    transform = transforms.Normalize(mean=gmean, std=gstd)

    return transform


def get_denormalizer2(model_dir: str):
    levels = {
        "L50",
        "L100",
        "L150",
        "L222",
        "L318",
        "L380",
        "L450",
        "L540",
        "L640",
        "L763",
    }
    mean_thetao = []
    mean_so = []
    mean_uo = []
    mean_vo = []
    std_thetao = []
    std_so = []
    std_uo = []
    std_vo = []
    for level in sorted(levels):
        mean_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_mean.npy")[0]
        )
        mean_so.append(np.load(model_dir + "/" + level + "/so_mean.npy")[0])
        mean_uo.append(np.load(model_dir + "/" + level + "/uo_mean.npy")[0])
        mean_vo.append(np.load(model_dir + "/" + level + "/vo_mean.npy")[0])

        std_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_std.npy")[0]
        )
        std_so.append(np.load(model_dir + "/" + level + "/so_std.npy")[0])
        std_uo.append(np.load(model_dir + "/" + level + "/uo_std.npy")[0])
        std_vo.append(np.load(model_dir + "/" + level + "/vo_std.npy")[0])

    gmean = np.concatenate([mean_thetao, mean_so, mean_uo, mean_vo])
    gstd = np.concatenate([std_thetao, std_so, std_uo, std_vo])

    denormalizer = transforms.Normalize(
        mean=[-m / s for m, s in zip(gmean, gstd)], std=[1 / s for s in gstd]
    )
    return denormalizer


####################################################################3
def get_normalizer3(model_dir: str):
    levels = {
        "L902",
        "L1245",
        "L1684",
        "L2225",
        "L3220",
        "L3597",
        "L3992",
        "L4405",
        "L4833",
        "L5274",
    }
    mean_thetao = []
    mean_so = []
    mean_uo = []
    mean_vo = []
    std_thetao = []
    std_so = []
    std_uo = []
    std_vo = []
    for level in sorted(levels):
        mean_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_mean.npy")[0]
        )
        mean_so.append(np.load(model_dir + "/" + level + "/so_mean.npy")[0])
        mean_uo.append(np.load(model_dir + "/" + level + "/uo_mean.npy")[0])
        mean_vo.append(np.load(model_dir + "/" + level + "/vo_mean.npy")[0])

        std_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_std.npy")[0]
        )
        std_so.append(np.load(model_dir + "/" + level + "/so_std.npy")[0])
        std_uo.append(np.load(model_dir + "/" + level + "/uo_std.npy")[0])
        std_vo.append(np.load(model_dir + "/" + level + "/vo_std.npy")[0])

    gmean = np.concatenate([mean_thetao, mean_so, mean_uo, mean_vo])
    gstd = np.concatenate([std_thetao, std_so, std_uo, std_vo])

    transform = transforms.Normalize(mean=gmean, std=gstd)

    return transform


def get_denormalizer3(model_dir: str):
    levels = {
        "L902",
        "L1245",
        "L1684",
        "L2225",
        "L3220",
        "L3597",
        "L3992",
        "L4405",
        "L4833",
        "L5274",
    }
    mean_thetao = []
    mean_so = []
    mean_uo = []
    mean_vo = []
    std_thetao = []
    std_so = []
    std_uo = []
    std_vo = []
    for level in sorted(levels):
        mean_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_mean.npy")[0]
        )
        mean_so.append(np.load(model_dir + "/" + level + "/so_mean.npy")[0])
        mean_uo.append(np.load(model_dir + "/" + level + "/uo_mean.npy")[0])
        mean_vo.append(np.load(model_dir + "/" + level + "/vo_mean.npy")[0])

        std_thetao.append(
            np.load(model_dir + "/" + level + "/thetao_std.npy")[0]
        )
        std_so.append(np.load(model_dir + "/" + level + "/so_std.npy")[0])
        std_uo.append(np.load(model_dir + "/" + level + "/uo_std.npy")[0])
        std_vo.append(np.load(model_dir + "/" + level + "/vo_std.npy")[0])

    gmean = np.concatenate([mean_thetao, mean_so, mean_uo, mean_vo])
    gstd = np.concatenate([std_thetao, std_so, std_uo, std_vo])

    denormalizer = transforms.Normalize(
        mean=[-m / s for m, s in zip(gmean, gstd)], std=[1 / s for s in gstd]
    )
    return denormalizer

####################################################################3
def get_data(date, depth, fn):
    start_datetime=str(date-timedelta(days=1))
    end_datetime=str(date)
    if (depth==0.):
        print("yes")
        mlist=['cmems_mod_glo_phy_anfc_0.083deg_P1D-m',
               'cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m',
               'cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m',
               'cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m']    
        mvar=[['zos'],['uo','vo'], ['so'], ['thetao']]
    else:
        print("no")
        mlist=['cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m',
               'cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m',
               'cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m']    
        mvar=[['uo','vo'], ['so'], ['thetao']]
    
    ds=[]
    for i in range(0, len(mlist)):
        print(i)
        data=cmc.open_dataset(
            dataset_id=mlist[i], 
            variables=mvar[i],
            minimum_longitude=-180,
            maximum_longitude=180,
            minimum_latitude=-90,
            maximum_latitude=90,
            minimum_depth=depth,
            maximum_depth=depth,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
        ds.append(data)
    print("merging..")
    ds=xr.merge(ds)
    print(ds)
    ds_out = xr.Dataset(
    {
        "lat": (["lat"], np.arange(data.latitude.min(), data.latitude.max(), 1/4)),
        "lon": (["lon"], np.arange(data.longitude.min(), data.longitude.max(), 1/4)),
    }
    )

    print("loading regridder")
    regridder = xe.Regridder(data, ds_out, 'bilinear', weights=fn, reuse_weights=True)
    print("regridder ready")
    ds_out = regridder(ds)
    print("done regridding")
    ds_out=ds_out.sel(lat=slice(ds_out.lat[8],ds_out.lat[-1]))
    #print(ds_out)
    del regridder, ds, data
    gc.collect()    
    return ds_out

def get_data_ref(date, depth, fn):
    start_datetime=str(date+timedelta(days=1))
    end_datetime=str(date+timedelta(days=10))
    if (depth==0.):
        print("yes")
        mlist=['cmems_mod_glo_phy_anfc_0.083deg_P1D-m',
               'cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m',
               'cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m',
               'cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m']    
        mvar=[['zos'],['uo','vo'], ['so'], ['thetao']]
    else:
        print("no")
        mlist=['cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m',
               'cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m',
               'cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m']    
        mvar=[['uo','vo'], ['so'], ['thetao']]
    
    ds=[]
    for i in range(0, len(mlist)):
        print(i)
        data=cmc.open_dataset(
            dataset_id=mlist[i], 
            variables=mvar[i],
            minimum_longitude=-180,
            maximum_longitude=180,
            minimum_latitude=-90,
            maximum_latitude=90,
            minimum_depth=depth,
            maximum_depth=depth,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
        ds.append(data)
    print("merging..")
    ds=xr.merge(ds)
    print(ds)
    ds_out = xr.Dataset(
    {
        "lat": (["lat"], np.arange(data.latitude.min(), data.latitude.max(), 1/4)),
        "lon": (["lon"], np.arange(data.longitude.min(), data.longitude.max(), 1/4)),
    }
    )

    print("loading regridder")
    regridder = xe.Regridder(data, ds_out, 'bilinear', weights=fn, reuse_weights=True)
    print("regridder ready")
    ds_out = regridder(ds)
    print("done regridding")
    ds_out=ds_out.sel(lat=slice(ds_out.lat[8],ds_out.lat[-1]))
    #print(ds_out)
    del regridder, ds, data
    gc.collect()    
    return ds_out

def get_data_refg(date, depth, fn):
    start_datetime=str(date+timedelta(days=1))
    end_datetime=str(date+timedelta(days=10))
    if (depth==0.):
        print("yes")
        mlist='cmems_mod_glo_phy_myint_0.083deg_P1D-m'    
        mvar=['zos','uo','vo', 'so', 'thetao']
    else:
        print("no")
        mlist='cmems_mod_glo_phy_myint_0.083deg_P1D-m'    
        mvar=['uo','vo', 'so', 'thetao']
    
    ds=[]
    data=cmc.open_dataset(
        dataset_id=mlist, 
        variables=mvar,
        minimum_longitude=-180,
        maximum_longitude=180,
        minimum_latitude=-90,
        maximum_latitude=90,
        minimum_depth=depth,
        maximum_depth=depth,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
    ds=data
    print("merging..")
    print(ds)
    ds_out = xr.Dataset(
    {
        "lat": (["lat"], np.arange(data.latitude.min(), data.latitude.max(), 1/4)),
        "lon": (["lon"], np.arange(data.longitude.min(), data.longitude.max(), 1/4)),
    }
    )

    print("loading regridder")
    regridder = xe.Regridder(data, ds_out, 'bilinear', weights=fn, reuse_weights=True)
    print("regridder ready")
    ds_out = regridder(ds)
    print("done regridding")
    ds_out=ds_out.sel(lat=slice(ds_out.lat[8],ds_out.lat[-1]))
    #print(ds_out)
    del regridder, ds, data
    gc.collect()    
    return ds_out



def get_data_refgr(date, depth):
    start_datetime=str(date+timedelta(days=1))
    end_datetime=str(date+timedelta(days=10))
    if (depth==0.):
        print("yes")
        mlist='cmems_mod_glo_phy_myint_0.083deg_P1D-m'    
        mvar=['zos','uo','vo', 'so', 'thetao']
    else:
        print("no")
        mlist='cmems_mod_glo_phy_myint_0.083deg_P1D-m'    
        mvar=['uo','vo', 'so', 'thetao']
    
    ds=[]
    data=cmc.open_dataset(
        dataset_id=mlist, 
        variables=mvar,
        minimum_longitude=-180,
        maximum_longitude=180,
        minimum_latitude=-90,
        maximum_latitude=90,
        minimum_depth=depth,
        maximum_depth=depth,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
    return data

def glo_in1(date):
    inp=get_data(date,0,"xe_weights14/L0.nc")
    return inp
def glo_in2(date):
    inp=[]
    inp.append(get_data(date,50,"xe_weights14/L50.nc"))
    inp.append(get_data(date,100,"xe_weights14/L100.nc"))
    inp.append(get_data(date,150,"xe_weights14/L150.nc"))
    inp.append(get_data(date,222,"xe_weights14/L222.nc"))
    inp.append(get_data(date,318,"xe_weights14/L318.nc"))
    inp.append(get_data(date,380,"xe_weights14/L380.nc"))
    inp.append(get_data(date,450,"xe_weights14/L450.nc"))
    inp.append(get_data(date,540,"xe_weights14/L540.nc"))
    inp.append(get_data(date,640,"xe_weights14/L640.nc"))
    inp.append(get_data(date,763,"xe_weights14/L763.nc"))
    inp=xr.concat(inp, dim="depth")
    return inp

def glo_in3(date):
    inp=[]
    inp.append(get_data(date,902,"xe_weights14/L902.nc"))
    inp.append(get_data(date,1245,"xe_weights14/L1245.nc"))
    inp.append(get_data(date,1684,"xe_weights14/L1684.nc"))
    inp.append(get_data(date,2225,"xe_weights14/L2225.nc"))
    inp.append(get_data(date,3220,"xe_weights14/L3220.nc"))
    inp.append(get_data(date,3597,"xe_weights14/L3597.nc"))
    inp.append(get_data(date,3992,"xe_weights14/L3992.nc"))
    inp.append(get_data(date,4405,"xe_weights14/L4405.nc"))
    inp.append(get_data(date,4405,"xe_weights14/L4405.nc"))
    inp.append(get_data(date,5274,"xe_weights14/L5274.nc"))
    inp=xr.concat(inp, dim="depth")
    return inp

def glo_inall(date):
    inp=[]
    inp.append(get_data_refg(date,0,"xe_weights14/L0.nc"))
    inp.append(get_data_refg(date,50,"xe_weights14/L50.nc"))
    inp.append(get_data_refg(date,100,"xe_weights14/L100.nc"))
    inp.append(get_data_refg(date,150,"xe_weights14/L150.nc"))
    inp.append(get_data_refg(date,222,"xe_weights14/L222.nc"))
    inp.append(get_data_refg(date,318,"xe_weights14/L318.nc"))
    inp.append(get_data_refg(date,380,"xe_weights14/L380.nc"))
    inp.append(get_data_refg(date,450,"xe_weights14/L450.nc"))
    inp.append(get_data_refg(date,540,"xe_weights14/L540.nc"))
    inp.append(get_data_refg(date,640,"xe_weights14/L640.nc"))
    inp.append(get_data_refg(date,763,"xe_weights14/L763.nc"))
    inp.append(get_data_refg(date,902,"xe_weights14/L902.nc"))
    inp.append(get_data_refg(date,1245,"xe_weights14/L1245.nc"))
    inp.append(get_data_refg(date,1684,"xe_weights14/L1684.nc"))
    inp.append(get_data_refg(date,2225,"xe_weights14/L2225.nc"))
    inp.append(get_data_refg(date,3220,"xe_weights14/L3220.nc"))
    inp.append(get_data_refg(date,3597,"xe_weights14/L3597.nc"))
    inp.append(get_data_refg(date,3992,"xe_weights14/L3992.nc"))
    inp.append(get_data_refg(date,4405,"xe_weights14/L4405.nc"))
    inp.append(get_data_refg(date,4405,"xe_weights14/L4405.nc"))
    inp.append(get_data_refg(date,5274,"xe_weights14/L5274.nc"))
    inp=xr.concat(inp, dim="depth")
    return inp

def glo_inallgr(date):
    inp=[]
    inp.append(get_data_refgr(date,0 )) 
    inp.append(get_data_refgr(date,2.6)) 
    inp.append(get_data_refgr(date,5.07)) 
    inp.append(get_data_refgr(date,7.9)) 
    inp.append(get_data_refgr(date,11.4)) 
    inp.append(get_data_refgr(date,15.8)) 
    inp.append(get_data_refgr(date,21.5)) 
    inp.append(get_data_refgr(date,29.4)) 
    inp.append(get_data_refgr(date,40.3)) 
    inp.append(get_data_refgr(date,55.7)) 
    inp.append(get_data_refgr(date,77.8)) 
    inp.append(get_data_refgr(date,92.32)) 
    inp.append(get_data_refgr(date,109.7)) 
    inp.append(get_data_refgr(date,130.6)) 
    inp.append(get_data_refgr(date,155.8)) 
    inp.append(get_data_refgr(date,186.1)) 
    inp.append(get_data_refgr(date,222)) 
    inp.append(get_data_refgr(date,266)) 
    inp.append(get_data_refgr(date,318)) 
    inp.append(get_data_refgr(date,380)) 
    inp.append(get_data_refgr(date,453)) 
    inp.append(get_data_refgr(date,541)) 
    inp.append(get_data_refgr(date,643)) 
    inp=xr.concat(inp, dim="depth")
    return inp

def create_data(ds_out, depth):
    thetao=ds_out['thetao'].data
    so=ds_out['so'].data
    uo=ds_out['uo'].data
    vo=ds_out['vo'].data
    if depth ==0:
        zos=np.expand_dims(ds_out['zos'].data, axis=1)
        tt=np.concatenate([zos,thetao,so,uo,vo], axis=1)
    else:
        tt=np.concatenate([thetao,so,uo,vo], axis=1)

    lat=ds_out.lat.data
    lon=ds_out.lon.data
    time=ds_out.time.data

    bb=xr.Dataset(
        {
            "data": (("time", "ch", "lat", "lon"), tt),
        },
        coords={
            "time": ("time", time),
            "ch": ("ch", range(0, tt.shape[1])),
            "lat": ("lat", lat),
            "lon": ("lon", lon),
        },
    )
    return bb
