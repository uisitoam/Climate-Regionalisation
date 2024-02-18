from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta

# to read .nc files or any kinf of extension

def reader(filename, era5=False):
    """
    

    Parameters
    ----------
    filename : str
        Path of the file to be read. 
    era5 : bool, optional
        Make it True to read .nc files. The default is False.

    Returns
    -------
    np.ndarray
        The arrays containing the data of the file. 

    """
    
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")
    
    if era5: #.nc file
        rawData = Dataset(filename,'r')
        longitude = np.array(rawData.variables['longitude'][:])
        latitude = np.array(rawData.variables['latitude'][:])
        level = np.array(rawData.variables['level'][:]) # (500, 700, 850, 1000) [mba]
        timer = np.array(rawData.variables['time'][:]) # (2 per day)
        
        time = [0]*len(timer)
        
        for i in range(len(timer)):
            delta = timedelta(hours = int(timer[i]))
            start_date = datetime(1900, 1, 1, 0, 0, 0)
            result = start_date + delta
            
            if len(str(result.month)) == 1 and len(str(result.day)) == 1:
                time[i] = str(result.year) + str(0) + str(result.month) + str(0) + str(result.day)
                
            elif len(str(result.month)) == 1:
                time[i] = str(result.year) + str(0) + str(result.month) + str(result.day)
                
            elif len(str(result.day)) == 1:
                time[i] = str(result.year) + str(result.month) + str(0) + str(result.day)
                
            else:
                time[i] = str(result.year) + str(result.month) + str(result.day)
            

        #dimensions --> (time, level, latitude, longitude)
        z = np.array(rawData.variables['z'][:,:,:,:]) # geopotential height [m^2/s^2]
        q = np.array(rawData.variables['q'][:,:,:,:]) # specific humidity [kg/kg]
        t = np.array(rawData.variables['t'][:,:,:,:]) # temperature [K]
        u = np.array(rawData.variables['u'][:,:,:,:]) # u wind component (eastward) [m/s]
        v = np.array(rawData.variables['v'][:,:,:,:]) # v wind component (northward) [m/s]
        rawData.close()

        return time, longitude, latitude, level, z, q, t, u, v
    
    else: #another file type
        x1, x2, x3, x4, x5 = np.loadtxt(filename, skiprows=2, unpack=True, delimiter=',')
        
        return x1, x2, x3, x4, x5