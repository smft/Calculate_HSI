#!/usr/bin/env python

import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from ctypes import *
from netCDF4 import Dataset
from scipy import interpolate

""" make shared memory space """
def make_shared_array_3D(nz,ny,nx):
    shared_array_input_base=mp.Array(c_double,nz*ny*nx)
    data=np.ctypeslib.as_array(shared_array_input_base.get_obj())
    data=data.reshape(nz,ny,nx)
    return data

def make_shared_array_2D(ny,nx):
    shared_array_input_base=mp.Array(c_double,ny*nx)
    data=np.ctypeslib.as_array(shared_array_input_base.get_obj())
    data=data.reshape(ny,nx)
    return data

def make_shared_array_1D(nx):
    shared_array_input_base=mp.Array(c_double,nx)
    data=np.ctypeslib.as_array(shared_array_input_base.get_obj())
    data=data.reshape(nx)
    return data

""" select regional data from large area """
def chop_domain(obs_file,mod_file):
    fill_value=255
    obs_lon_2d,obs_lat_2d=np.meshgrid(obs_file.variables['lon'][:],obs_file.variables['lat'][:])
    mod_lon_2d=mod_file.variables['XLONG_M'][0,:,:]
    mod_lat_2d=mod_file.variables['XLAT_M'][0,:,:]
    mod_lon_ll=np.min(mod_lon_2d)
    mod_lon_ur=np.max(mod_lon_2d)
    mod_lat_ll=np.min(mod_lat_2d)
    mod_lat_ur=np.max(mod_lat_2d)
    dist=np.sqrt((obs_lat_2d-mod_lat_ll)**2+(obs_lon_2d-mod_lon_ll)**2)
    idx_y_ll,idx_x_ll=np.unravel_index(dist.argmin(),dist.shape)
    dist=np.sqrt((obs_lat_2d-mod_lat_ur)**2+(obs_lon_2d-mod_lon_ur)**2)
    idx_y_ur,idx_x_ur=np.unravel_index(dist.argmin(),dist.shape)
    return obs_file.variables[0,idx_y_ll-5:idx_y_ur+5,idx_x_ll-5:idx_x_ur+5].data,\
            obs_lat_2d[idx_y_ll-5:idx_y_ur+5,idx_x_ll-5:idx_x_ur+5],\
            obs_lon_2d[idx_y_ll-5:idx_y_ur+5,idx_x_ll-5:idx_x_ur+5],\
            mod_lat_2d,mod_lon_2d,fill_value

""" horizontal interp """
def horizontal_interp_module(mod_lat,mod_lon,interp_lat,interp_lon,data,fill_value,rslt,ny,idx_x_s,processlock):
    warnings.filterwarnings("ignore")
    for j in range(ny):
        for k in idx_x_s:
            dist=np.sqrt((mod_lat-interp_lat[j,k])**2+(mod_lon-interp_lon[j,k])**2)
            idx_y,idx_x=np.unravel_index(dist.argmin(),dist.shape)
            if (data[idx_y-2:idx_y+3,idx_x-2:idx_x+3]!=-1e30).all()==True:
                f=interpolate.interp2d(mod_lat[idx_y-2:idx_y+3,idx_x-2:idx_x+3],\
                                        mod_lon[idx_y-2:idx_y+3,idx_x-2:idx_x+3],\
                                        data[idx_y-2:idx_y+3,idx_x-2:idx_x+3],\
                                        kind='cubic',bounds_error=False,fill_value=(-1e30,-1e30))
                rslt[j,k]+=f(interp_lat[j,k],interp_lon[j,k])
            elif (data[idx_y-2:idx_y+3,idx_x-2:idx_x+3]!=fill_value).all()==False and (data[idx_y-2:idx_y+3,idx_x-2:idx_x+3]!=fill_value).any()==True:
                #trans_data=data[i,idx_y-2:idx_y+3,idx_x-2:idx_x+3][data[i,idx_y-2:idx_y+3,idx_x-2:idx_x+3]!=-1e30]
                #trans_dist=1/dist[idx_y-2:idx_y+3,idx_x-2:idx_x+3][data[i,idx_y-2:idx_y+3,idx_x-2:idx_x+3]!=-1e30]
                #rslt[i,j,k]+=np.sum(trans_data*trans_dist)/np.sum(trans_dist)
                #rslt[i,j,k]+=f(interp_lat[j,k],interp_lon[j,k])
                rslt[j,k]+=np.mean(data[idx_y-2:idx_y+3,idx_x-2:idx_x+3][data[idx_y-2:idx_y+3,idx_x-2:idx_x+3]!=fill_value])
            else:
                rslt[j,k]+=0
    processlock.release()

def horizontal_interp(mod_lat,mod_lon,interp_lat,interp_lon,data,fill_value,rslt):
    cpu_count=mp.cpu_count()
    ny,nx=np.shape(interp_lat)
    pids=[]
    processlock=mp.BoundedSemaphore(cpu_count)
    for cell in np.array_split(np.arange(0,nx,1),cpu_count,axis=0):
        processlock.acquire()
        p=mp.Process(target=horizontal_interp_module,args=(mod_lat,mod_lon,interp_lat,interp_lon,data,\
                                                            rslt,ny,cell,processlock))
        p.start()
        pids+=[p]
    for p in pids:
        p.join()

###################
""" test!!!test """
###################
# retrieve ingest data
flag_obs=Dataset(raw_input('OBS NDVI DIR : '))
flag_mod=Dataset(raw_input('MOD GRID DIR : '))
ndvi,obs_lat,obs_lon,mod_lat,mod_lon,fill_value=chop_domain(flag_obs,flag_mod)
ny_obs,nx_obs=np.shape(obs_lon)
ny_mod,nx_mod=np.shape(mod_lon)
# make shared mem
# obs
obs_data_shared=make_shared_array_2D(ny_obs,nx_obs)
obs_data_shared+=ndvi
obs_lat_shared=make_shared_array_2D(ny_obs,nx_obs)
obs_lat_shared+=obs_lat
obs_lon_shared=make_shared_array_2D(ny_obs,nx_obs)
obs_lon_shared+=obs_lon
# mod
mod_lat_shared=make_shared_array_2D(ny_mod,nx_mod)
mod_lat_shared+=mod_lat
mod_lon_shared=make_shared_array_2D(ny_mod,nx_mod)
mod_lon_shared+=mod_lon
rslt=make_shared_array_2D(ny_mod,nx_mod)
# horizontal interp
horizontal_interp(obs_lat_shared,obs_lon_shared,mod_lat_shared,mod_lon_shared,obs_data_shared,fill_value,rslt)
plt.imshow(rslt)
plt.colorbar()
plt.show()
