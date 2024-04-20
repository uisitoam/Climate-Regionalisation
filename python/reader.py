from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta
from calendar import leapdays



# to read .nc files or any kinf of extension

def reader(filename, era5=False):
    """
    Read .nc and txt files

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
            fecha_sb = start_date + delta
            result = fecha_sb + timedelta(days=leapdays(start_date.year, fecha_sb.year))
            
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
        rawData = Dataset(filename, 'r')
        longitude = np.array(rawData.variables['longitude'][:])
        latitude = np.array(rawData.variables['latitude'][:])
        level = np.array(rawData.variables['level'][:3]) # (500, 700, 850, 1000) [mba]
        timer = np.array(rawData.variables['time'][:]) # (2 per day)
        
        time = todate(timer, [1900, 1, 1, 0, 0, 0]) #año, mes, dia, hora, minuto, segundo
        
        #dimensions --> (time, level, latitude, longitude)
        z = np.array(rawData.variables['z'][:,:3,:,:]) # geopotential height [m^2/s^2]
        q = np.array(rawData.variables['q'][:,:3,:,:]) # specific humidity [kg/kg]
        t = np.array(rawData.variables['t'][:,:3,:,:]) # temperature [K]
        u = np.array(rawData.variables['u'][:,:3,:,:]) # u wind component (eastward) [m/s]
        v = np.array(rawData.variables['v'][:,:3,:,:]) # v wind component (northward) [m/s]
        rawData.close()

        return time, longitude, latitude, level, z, q, t, u, v
    
    elif 'wrf' in filename: # WRF files
        rawData = Dataset(filename, 'r')
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
    
    elif 'all' in filename:
        rawData = Dataset(filename, 'r')
        longitude = np.array(rawData.variables['longitude'][:])
        latitude = np.array(rawData.variables['latitude'][:])
        timer = np.array(rawData.variables['time'][:])*24

        if 'GFDL' in filename:
            time = todate(timer, [1861, 1, 1, 0, 0, 0]) #año, mes, dia, hora, minuto, segundo
        else:
            time = todate(timer, [1850, 1, 1, 0, 0, 0]) #año, mes, dia, hora, minuto, segundo


        z = np.array(rawData.variables['z'][:,:3,:,:]) # geopotential height [m^2/s^2]
        q = np.array(rawData.variables['q'][:,:3,:,:]) # specific humidity [kg/kg]
        t = np.array(rawData.variables['t'][:,:3,:,:]) # temperature [K]
        u = np.array(rawData.variables['u'][:,:3,:,:]) # u wind component (eastward) [m/s]
        v = np.array(rawData.variables['v'][:,:3,:,:]) # v wind component (northward) [m/s]

        return time, z, q, t, u, v
        
    else: #another file type
        x1, x2, x3, x4, x5 = np.loadtxt(filename, skiprows=2, unpack=True, delimiter=',')
        
        return x1, x2, x3, x4, x5
    



def maskreader(filename):
    
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
    
    rawData = Dataset(filename,'r')
    longitude = np.array(rawData.variables['lon'][:])
    latitude = np.array(rawData.variables['lat'][:])
    precip = np.where(type(rawData.variables['pr'][:]) == np.ma.core.MaskedConstant, 
                      -999., rawData.variables['pr'][:]) 
    
    timer = np.array(rawData.variables['XTIME'][:])
    time = todate(timer, [1981, 9, 1, 0, 0, 0]) #año, mes, dia, hora, minuto, segundo
    
    return time, precip, longitude, latitude
    


























