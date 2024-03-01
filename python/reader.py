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
        
        
    def todate(times, date):
        """
        Convert time of era5 files to an actual date.
        
        """
        
        if date[0] > 1910:
            times = times / 60
            
        time = [0]*len(times)
        
        for i in range(len(times)):
            delta = timedelta(hours = int(times[i]))
            start_date = datetime(date[0], date[1], date[2], date[3], date[4], date[5])
            result = start_date + delta
            
            if len(str(result.month)) == 1 and len(str(result.day)) == 1:
                time[i] = str(result.year) + str(0) + str(result.month) + str(0) + str(result.day)
                
            elif len(str(result.month)) == 1:
                time[i] = str(result.year) + str(0) + str(result.month) + str(result.day)
                
            elif len(str(result.day)) == 1:
                time[i] = str(result.year) + str(result.month) + str(0) + str(result.day)
                
            else:
                time[i] = str(result.year) + str(result.month) + str(result.day)
        
        return time
    
    if era5: #.nc file
        rawData = Dataset(filename,'r')
        longitude = np.array(rawData.variables['longitude'][:])
        latitude = np.array(rawData.variables['latitude'][:])
        level = np.array(rawData.variables['level'][:]) # (500, 700, 850, 1000) [mba]
        timer = np.array(rawData.variables['time'][:]) # (2 per day)
        
        time = todate(timer, [1900, 1, 1, 0, 0, 0]) #año, mes, dia, hora, minuto, segundo
        
        #dimensions --> (time, level, latitude, longitude)
        z = np.array(rawData.variables['z'][:,:,:,:]) # geopotential height [m^2/s^2]
        q = np.array(rawData.variables['q'][:,:,:,:]) # specific humidity [kg/kg]
        t = np.array(rawData.variables['t'][:,:,:,:]) # temperature [K]
        u = np.array(rawData.variables['u'][:,:,:,:]) # u wind component (eastward) [m/s]
        v = np.array(rawData.variables['v'][:,:,:,:]) # v wind component (northward) [m/s]
        rawData.close()

        return time, longitude, latitude, level, z, q, t, u, v
    
    elif (filename[-3:] == '.nc'):
        rawData = Dataset(filename,'r')
        longitude = np.array(rawData.variables['lon'][:])
        latitude = np.array(rawData.variables['lat'][:])
        
        if 't2mean' in filename:
            timer = np.array(rawData.variables['Times'][:], dtype=int)
            time = np.array(timer, dtype=str)
            tmean = np.array(rawData.variables['T2MEAN'][:,:,:])
            rawData.close()
            
            return time, tmean, longitude, latitude
        
        elif 'pr' in filename:
            timer = np.array(rawData.variables['XTIME'][:])
            time = todate(timer, [1981, 9, 1, 0, 0, 0]) #año, mes, dia, hora, minuto, segundo
            precip = np.array(rawData.variables['pr'][:,:,:])
            rawData.close()
            
            return time, precip, longitude, latitude
        
    else: #another file type
        x1, x2, x3, x4, x5 = np.loadtxt(filename, skiprows=2, unpack=True, delimiter=',')
        
        return x1, x2, x3, x4, x5
