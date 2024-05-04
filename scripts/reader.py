from netCDF4 import Dataset
import numpy as np
from datetime import datetime, timedelta
from functools import reduce





def time_to_date(time, unit, baseDate):
    """
    Convert a time in minutes, hours, or days to a date string.

    Parameters
    ----------
    time : float
        The amount of time to convert. This can be in minutes, hours, or days, 
        depending on the 'unit' parameter.
    unit : str
        The unit of the 'time' parameter. This must be either 'minutes', 'hours', 
        or 'days'.
    baseDate : str
        The base date from which to calculate the new date. This should be a string 
        in the format 'YYYYMMDD'.
        
    Returns
    -------
    date_str : str
        The new date, calculated by adding 'time' (in the unit specified by 'unit') 
        to 'baseDate'. The date is returned as a string in the format 'YYYYMMDD'.
        
    Raises
    ------
    TypeError
        If 'time' is not a number, 'unit' is not a string, or 'baseDate' is not a string.
    ValueError
        If 'unit' is not 'minutes', 'hours', or 'days', or if 'baseDate' is not in the 
        format 'YYYYMMDD'.

    """

    # check input parameters
    assert isinstance(time, (int, float, np.float32, np.float64)), "time must be a number"
    assert unit in ['minutes', 'hours', 'days'], "unit must be 'minutes', 'hours', or 'days'"
    assert isinstance(baseDate, str) and len(baseDate) == 8, "baseDate must be a string in the format 'YYYYMMDD'"
    
    
    try:
        baseDate = datetime.strptime(baseDate, '%Y%m%d') # baseDate to datetime object

    except ValueError:
        raise ValueError("baseDate must be in the format 'YYYYMMDD'")
    
    try:
        # Calculate the number of days
        if unit == 'minutes':
            days = time / 1440

        elif unit == 'hours':
            days = time / 24

        else:
            days = time

        date = baseDate + timedelta(days=days)
        date_str = date.strftime('%Y%m%d') # format date as 'YYYYMMDD'

        return date_str

    except ValueError as ve:
        raise ValueError(f"ValueError: {ve}")

    except TypeError as te:
        raise TypeError(f"TypeError: {te}")

    except OverflowError as oe:
        raise OverflowError(f"OverflowError: {oe}")





def reader(filename, unit=None, baseDate=None, wrf=False, conversor=time_to_date):
    """
    Reads data from a file, giving the date in 'YYYYMMDD' format.

    Parameters
    ----------
    filename : str
        The name of the file to read.
    unit : str, optional
        The unit of the 'time' parameter. This must be either 'minutes', 'hours', or 'days'.
    baseDate : str, optional
        The base date from which to calculate the new date. This should be a string in the format 'YYYYMMDD'.
    wrf : bool, optional
        Whether the file is in WRF format. Default is False.
    conversor : function, optional
        The function to convert time. Default is time_to_date.

    Returns
    -------
    np.ndarray
        The time array.
    np.ndarray
        The variables arrays.

    Raises
    ------
    ValueError
        If time variable not found in file.
        If weather variables 'T2MEAN' or 'pr' not found in file.
    TypeError
        If 'filename' is not a string, 'unit' is not a string, 'baseDate' is not a string, 'wrf' is not a boolean, or 'conversor' is not a function.

    """

    # cheack input parameters
    assert isinstance(filename, str), "'filename' must be a string"
    assert unit is None or unit in ['minutes', 'hours', 'days'], "unit must be 'minutes', 'hours', or 'days'"
    assert baseDate is None or (isinstance(baseDate, str) and len(baseDate) == 8), "'baseDate' must be a string in the format 'YYYYMMDD'"
    assert isinstance(wrf, bool), "'wrf' must be a boolean"
    assert callable(conversor), "'conversor' must be a function"


    rawData = Dataset(filename, 'r')

    # for WRF files
    if wrf:
        # read the time variable, which can be named 'Times', 'XTIME', or 'time' 
        name = None
        for name in ['Times', 'XTIME', 'time']:
            try:
                fechas = np.array(rawData.variables[name][:])
                break
            except KeyError:
                continue
        
        if name is None:
            raise ValueError("Time variable not found in file")
        

        # convert the time to 'YYYYMMDD' format using the 'conversor' function
        # each type of time variable has a different format
        if name == 'Times':
            time = np.array([str(int(fecha)) for fecha in fechas])

        elif name == 'XTIME':
            time = np.array([conversor(fecha, unit, baseDate) for fecha in fechas])
    

        # read the longitude and latitude variables, which can be named 'longitude' or 'lon' and 'latitude' or 'lat'
        for lon, lat in zip(['longitude', 'lon'], ['latitude', 'lat']):
            try:
                longitude = np.array(rawData.variables[lon][:])
                latitude = np.array(rawData.variables[lat][:])
                break

            except KeyError:
                continue
        
        # read the weather variables, which can be named 'T2MEAN' or 'pr'
        var = None
        for var in ['T2MEAN', 'pr']:
            try:
                variable = np.array(rawData.variables[var][:,:,:])
                break

            except KeyError:
                continue

        if var is None:
            raise ValueError("Weather variables 'T2MEAN' or 'pr' not found in file")
            
        return time, variable
    
    # for GCM/Era5 files
    else:
        # read the longitude and latitude variables, which can be named 'longitude' or 'lon' and 'latitude' or 'lat'
        for lon, lat in zip(['longitude', 'lon'], ['latitude', 'lat']):
            try:
                longitude = np.array(rawData.variables[lon][:])
                latitude = np.array(rawData.variables[lat][:])
                break

            except KeyError:
                continue

        # read the time variable, which can be named 'Times', 'XTIME', or 'time'
        fechas = None
        for name in ['Times', 'XTIME', 'time']:
            try:
                fechas = np.array(rawData.variables[name][:])
                break

            except KeyError:
                continue
        
        if fechas is None:
            raise ValueError("Time variable not found in file")

        # convert the time to 'YYYYMMDD' format using the 'conversor' function 
        time = np.array([conversor(fecha, unit, baseDate) for fecha in fechas])

        # read mask variable
        try:
            tierra = np.array(rawData.variables['HGT'][:])
            boolTierra = tierra == 1 #mascara booleana

            boolFlat = tierra.flatten() == 1 #mascara booleana

            #maskGeo = np.reshape(maskMask, (np.shape(maskMask)[1] * np.shape(maskMask)[2])) # mascara 0, 1
            #maskGeo2 = np.reshape(maskGeo, (1, len(maskGeo)))
            #boolMask = maskGeo == 1 # mascara booleana
            rawData.close()

            return longitude, latitude, boolTierra, boolFlat

        # read predictors variables
        except KeyError:
            #dimensions --> (time, level (500, 700, 850, 1000) [mba], latitude, longitude)
            z = np.array(rawData.variables['z'][:,:3,:,:]) # geopotential height [m^2/s^2]
            q = np.array(rawData.variables['q'][:,:3,:,:]) # specific humidity [kg/kg]
            t = np.array(rawData.variables['t'][:,:3,:,:]) # temperature [K]
            u = np.array(rawData.variables['u'][:,:3,:,:]) # u wind component (eastward) [m/s]
            v = np.array(rawData.variables['v'][:,:3,:,:]) # v wind component (northward) [m/s]
            rawData.close()

            return time, longitude, latitude, z, q, t, u, v





def common_elements(arrs):
    """
    Finds the common elements in a list of arrays.

    Parameters
    ----------
    arrs : list of np.ndarray
        The list of arrays to find common elements in.

    Returns
    -------
    np.ndarray
        The array of common elements.

    """
    return reduce(np.intersect1d, arrs)





def aplicador(mask, time=None, timeWRF=None, data=None):
    """
    Applies a mask to the provided time and data arrays.

    Parameters
    ----------
    mask : np.ndarray
        The mask to apply.
    time : np.ndarray, optional
        The time array to reduce. Must be provided if timeWRF is not.
    timeWRF : np.ndarray, optional
        The WRF time array to reduce. Must be provided if time is not.
    data : list of np.ndarray / np.ndarray, optional
        The data arrays to reduce.

    Returns
    -------
    time, timeWRF : np.ndarray
        The reduced time array.
    np.ndarray
        The reduced data arrays.

    Raises
    ------
    ValueError
        If both time and timeWRF are provided or neither of them.

    """

    # check input parameters
    assert not (time is not None and timeWRF is not None), "Only one time variable is allowed"
    assert not (time is None and timeWRF is None), "One time variable is required"

    
    # apply mask to predictors variables (samples)
    if time is not None:
        mascara = np.isin(time, mask)
        time2 = time[mascara]
        z = data[0][mascara, :, :, :]
        q = data[1][mascara, :, :, :]
        t = data[2][mascara, :, :, :]
        u = data[3][mascara, :, :, :]
        v = data[4][mascara, :, :, :]

        return time2, z, q, t, u, v
    
    # apply mask to WRF variables (labels)
    elif timeWRF is not None:
        mascaraWRF = np.isin(timeWRF, mask)
        timeWRF2 = timeWRF[mascaraWRF]

        # avoid repeated values
        _, unique_indices = np.unique(timeWRF2, return_index=True) 
        mascaraFea = np.full(timeWRF2.shape, False)
        mascaraFea[unique_indices] = True

        varWRF = data[mascaraWRF, :, :]
        
        return timeWRF2[mascaraFea], varWRF[mascaraFea, :, :]





# READ DATA FILES 
# read mask
maskLon, maskLat, maskMap, maskBool = reader("Datos/mask.nc", 'minutes', '19810901') # (158,), (68,), (1, 68, 158), (10744,)


# read GCMs Data
timeGFDL1, lonGFDL1, latGFDL1, zGFDL1, qGFDL1, tGFDL1, uGFDL1, vGFDL1 = reader("Datos/GCMs/all_GFDL_crop_1980_2009.nc", 'days', '18610101') # (10918,), (17,), (10,), (10918, 3, 10, 17), ...
timeGFDL2, lonGFDL2, latGFDL2, zGFDL2, qGFDL2, tGFDL2, uGFDL2, vGFDL2 = reader("Datos/GCMs/all_GFDL_crop_2030_2059.nc", 'days', '18610101') # (10905,), (17,), (10,), (10905, 3, 10, 17), ...
timeGFDL3, lonGFDL3, latGFDL3, zGFDL3, qGFDL3, tGFDL3, uGFDL3, vGFDL3 = reader("Datos/GCMs/all_GFDL_crop_2070_2099.nc", 'days', '18610101') # (10895,), (17,), (10,), (10895, 3, 10, 17), ...

timeIPSL1, lonIPSL1, latIPSL1, zIPSL1, qIPSL1, tIPSL1, uIPSL1, vIPSL1 = reader("Datos/GCMs/all_IPSL_crop_1980_2009.nc", 'days', '18500101') # (10918,), (17,), (10,), (10918, 3, 10, 17), ...
timeIPSL2, lonIPSL2, latIPSL2, zIPSL2, qIPSL2, tIPSL2, uIPSL2, vIPSL2 = reader("Datos/GCMs/all_IPSL_crop_2030_2059.nc", 'days', '18500101') # (10905,), (17,), (10,), (10905, 3, 10, 17), ...
timeIPSL3, lonIPSL3, latIPSL3, zIPSL3, qIPSL3, tIPSL3, uIPSL3, vIPSL3 = reader("Datos/GCMs/all_IPSL_crop_2070_2099.nc", 'days', '18500101') # (10950,), (17,), (10,), (10950, 3, 10, 17), ...

timeMIROC1, lonMIROC1, latMIROC1, zMIROC1, qMIROC1, tMIROC1, uMIROC1, vMIROC1 = reader("Datos/GCMs/all_MIROC_crop_1980_2009.nc", 'days', '18500101') # (10918,), (17,), (10,), (10918, 3, 10, 17), ...
timeMIROC2, lonMIROC2, latMIROC2, zMIROC2, qMIROC2, tMIROC2, uMIROC2, vMIROC2 = reader("Datos/GCMs/all_MIROC_crop_2030_2059.nc", 'days', '18500101') # (10905,), (17,), (10,), (10905, 3, 10, 17), ...
timeMIROC3, lonMIROC3, latMIROC3, zMIROC3, qMIROC3, tMIROC3, uMIROC3, vMIROC3 = reader("Datos/GCMs/all_MIROC_crop_2070_2099.nc", 'days', '18500101') # (10895,), (17,), (10,), (10895, 3, 10, 17), ...


# read WRF Data (temperature and precipitation)
tempTimeWRF_GFDL1, tmeanWRF_GFDL1 = reader("Datos/WRFprojections/data_wrf_gfdl_t2mean_1980_2009_rcp85.nc", None, None, True) # (10572,), (10572, 68, 158)
tempTimeWRF_GFDL2, tmeanWRF_GFDL2 = reader("Datos/WRFprojections/data_wrf_gfdl_t2mean_2030_2059_rcp85.nc", None, None, True) # (10905,), (10905, 68, 158)
tempTimeWRF_GFDL3, tmeanWRF_GFDL3 = reader("Datos/WRFprojections/data_wrf_gfdl_t2mean_2070_2099_rcp85.nc", None, None, True) # (10885,), (10885, 68, 158)

tempTimeWRF_IPSL1, tmeanWRF_IPSL1 = reader("Datos/WRFprojections/data_wrf_ipsl_t2mean_1980_2009_rcp85.nc", None, None, True) # (10916,), (10916, 68, 158)
tempTimeWRF_IPSL2, tmeanWRF_IPSL2 = reader("Datos/WRFprojections/data_wrf_ipsl_t2mean_2030_2059_rcp85.nc", None, None, True) # (10905,), (10905, 68, 158)
tempTimeWRF_IPSL3, tmeanWRF_IPSL3 = reader("Datos/WRFprojections/data_wrf_ipsl_t2mean_2070_2099_rcp85.nc", None, None, True) # (10895,), (10895, 68, 158)

tempTimeWRF_MIROC1, tmeanWRF_MIROC1 = reader("Datos/WRFprojections/data_wrf_miroc_t2mean_1980_2009_rcp85.nc", None, None, True) # (10918,), (10918, 68, 158)
tempTimeWRF_MIROC2, tmeanWRF_MIROC2 = reader("Datos/WRFprojections/data_wrf_miroc_t2mean_2030_2059_rcp85.nc", None, None, True) # (10813,), (10813, 68, 158)
tempTimeWRF_MIROC3, tmeanWRF_MIROC3 = reader("Datos/WRFprojections/data_wrf_miroc_t2mean_2070_2099_rcp85.nc", None, None, True) # (10895,), (10895, 68, 158)

precipTimeWRF_GFDL1, precipWRF_GFDL1 = reader("Datos/WRFprojections/data_wrf_gfdl_pr_1980_2009_rcp85.nc", None, None, True) # (10595,), (10595, 68, 158)
precipTimeWRF_GFDL2, precipWRF_GFDL2 = reader("Datos/WRFprojections/data_wrf_gfdl_pr_2030_2059_rcp85.nc", None, None, True) # (9871,), (9871, 68, 158)
precipTimeWRF_GFDL3, precipWRF_GFDL3 = reader("Datos/WRFprojections/data_wrf_gfdl_pr_2070_2099_rcp85.nc", None, None, True) # (10956,), (10956, 68, 158)

precipTimeWRF_IPSL1, precipWRF_IPSL1 = reader("Datos/WRFprojections/data_wrf_ipsl_pr_1980_2009_rcp85.nc", None, None, True) # (10918,), (10918, 68, 158)
precipTimeWRF_IPSL2, precipWRF_IPSL2 = reader("Datos/WRFprojections/data_wrf_ipsl_pr_2030_2059_rcp85.nc", None, None, True) # (10905,), (10905, 68, 158)
precipTimeWRF_IPSL3, precipWRF_IPSL3 = reader("Datos/WRFprojections/data_wrf_ipsl_pr_2070_2099_rcp85.nc", None, None, True) # (10895,), (10895, 68, 158)

precipTimeWRF_MIROC1, precipWRF_MIROC1 = reader("Datos/WRFprojections/data_wrf_miroc_pr_1980_2009_rcp85.nc", None, None, True) # (10553,), (10553, 68, 158)
precipTimeWRF_MIROC2, precipWRF_MIROC2 = reader("Datos/WRFprojections/data_wrf_miroc_pr_2030_2059_rcp85.nc", None, None, True) # (10540,), (10540, 68, 158)
precipTimeWRF_MIROC3, precipWRF_MIROC3 = reader("Datos/WRFprojections/data_wrf_miroc_pr_2070_2099_rcp85.nc", None, None, True) # (10895,), (10895, 68, 158)





# find common times
temp1980 = common_elements([timeGFDL1, tempTimeWRF_GFDL1, timeIPSL1, tempTimeWRF_IPSL1, timeMIROC1, tempTimeWRF_MIROC1])
temp2030 = common_elements([timeGFDL2, tempTimeWRF_GFDL2, timeIPSL2, tempTimeWRF_IPSL2, timeMIROC2, tempTimeWRF_MIROC2])
temp2070 = common_elements([timeGFDL3, tempTimeWRF_GFDL3, timeIPSL3, tempTimeWRF_IPSL3, timeMIROC3, tempTimeWRF_MIROC3])
precip1980 = common_elements([timeGFDL1, precipTimeWRF_GFDL1, timeIPSL1, precipTimeWRF_IPSL1, timeMIROC1, precipTimeWRF_MIROC1])
precip2030 = common_elements([timeGFDL2, precipTimeWRF_GFDL2, timeIPSL2, precipTimeWRF_IPSL2, timeMIROC2, precipTimeWRF_MIROC2])
precip2070 = common_elements([timeGFDL3, precipTimeWRF_GFDL3, timeIPSL3, precipTimeWRF_IPSL3, timeMIROC3, precipTimeWRF_MIROC3])





#reduce time dimensions to common times and apply the mask to the predictors and labels (temperature and precipitation)
timeGFDL1_temp, zGFDL1_temp, qGFDL1_temp, tGFDL1_temp, uGFDL1_temp, vGFDL1_temp = aplicador(temp1980, time=timeGFDL1, data=[zGFDL1, qGFDL1, tGFDL1, uGFDL1, vGFDL1])
tempTimeWRF_GFDL1, tmeanWRF_GFDL1 = aplicador(temp1980, timeWRF=tempTimeWRF_GFDL1, data=tmeanWRF_GFDL1)
timeGFDL2_temp, zGFDL2_temp, qGFDL2_temp, tGFDL2_temp, uGFDL2_temp, vGFDL2_temp = aplicador(temp2030, time=timeGFDL2, data=[zGFDL2, qGFDL2, tGFDL2, uGFDL2, vGFDL2])
tempTimeWRF_GFDL2, tmeanWRF_GFDL2 = aplicador(temp2030, timeWRF=tempTimeWRF_GFDL2, data=tmeanWRF_GFDL2)
timeGFDL3_temp, zGFDL3_temp, qGFDL3_temp, tGFDL3_temp, uGFDL3_temp, vGFDL3_temp = aplicador(temp2070, time=timeGFDL3, data=[zGFDL3, qGFDL3, tGFDL3, uGFDL3, vGFDL3])
tempTimeWRF_GFDL3, tmeanWRF_GFDL3 = aplicador(temp2070, timeWRF=tempTimeWRF_GFDL3, data=tmeanWRF_GFDL3)

timeIPSL1_temp, zIPSL1_temp, qIPSL1_temp, tIPSL1_temp, uIPSL1_temp, vIPSL1_temp = aplicador(temp1980, time=timeIPSL1, data=[zIPSL1, qIPSL1, tIPSL1, uIPSL1, vIPSL1])
tempTimeWRF_IPSL1, tmeanWRF_IPSL1 = aplicador(temp1980, timeWRF=tempTimeWRF_IPSL1, data=tmeanWRF_IPSL1)
timeIPSL2_temp, zIPSL2_temp, qIPSL2_temp, tIPSL2_temp, uIPSL2_temp, vIPSL2_temp = aplicador(temp2030, time=timeIPSL2, data=[zIPSL2, qIPSL2, tIPSL2, uIPSL2, vIPSL2])
tempTimeWRF_IPSL2, tmeanWRF_IPSL2 = aplicador(temp2030, timeWRF=tempTimeWRF_IPSL2, data=tmeanWRF_IPSL2)
timeIPSL3_temp, zIPSL3_temp, qIPSL3_temp, tIPSL3_temp, uIPSL3_temp, vIPSL3_temp = aplicador(temp2070, time=timeIPSL3, data=[zIPSL3, qIPSL3, tIPSL3, uIPSL3, vIPSL3])
tempTimeWRF_IPSL3, tmeanWRF_IPSL3 = aplicador(temp2070, timeWRF=tempTimeWRF_IPSL3, data=tmeanWRF_IPSL3)

timeMIROC1_temp, zMIROC1_temp, qMIROC1_temp, tMIROC1_temp, uMIROC1_temp, vMIROC1_temp = aplicador(temp1980, time=timeMIROC1, data=[zMIROC1, qMIROC1, tMIROC1, uMIROC1, vMIROC1])
tempTimeWRF_MIROC1, tmeanWRF_MIROC1 = aplicador(temp1980, timeWRF=tempTimeWRF_MIROC1, data=tmeanWRF_MIROC1)
timeMIROC2_temp, zMIROC2_temp, qMIROC2_temp, tMIROC2_temp, uMIROC2_temp, vMIROC2_temp = aplicador(temp2030, time=timeMIROC2, data=[zMIROC2, qMIROC2, tMIROC2, uMIROC2, vMIROC2])
tempTimeWRF_MIROC2, tmeanWRF_MIROC2 = aplicador(temp2030, timeWRF=tempTimeWRF_MIROC2, data=tmeanWRF_MIROC2)
timeMIROC3_temp, zMIROC3_temp, qMIROC3_temp, tMIROC3_temp, uMIROC3_temp, vMIROC3_temp = aplicador(temp2070, time=timeMIROC3, data=[zMIROC3, qMIROC3, tMIROC3, uMIROC3, vMIROC3])
tempTimeWRF_MIROC3, tmeanWRF_MIROC3 = aplicador(temp2070, timeWRF=tempTimeWRF_MIROC3, data=tmeanWRF_MIROC3)

timeGFDL1_precip, zGFDL1_precip, qGFDL1_precip, tGFDL1_precip, uGFDL1_precip, vGFDL1_precip = aplicador(precip1980, time=timeGFDL1, data=[zGFDL1, qGFDL1, tGFDL1, uGFDL1, vGFDL1])
precipTimeWRF_GFDL1, precipWRF_GFDL1 = aplicador(precip1980, timeWRF=precipTimeWRF_GFDL1, data=precipWRF_GFDL1)
timeGFDL2_precip, zGFDL2_precip, qGFDL2_precip, tGFDL2_precip, uGFDL2_precip, vGFDL2_precip = aplicador(precip2030, time=timeGFDL2, data=[zGFDL2, qGFDL2, tGFDL2, uGFDL2, vGFDL2])
precipTimeWRF_GFDL2, precipWRF_GFDL2 = aplicador(precip2030, timeWRF=precipTimeWRF_GFDL2, data=precipWRF_GFDL2)
timeGFDL3_precip, zGFDL3_precip, qGFDL3_precip, tGFDL3_precip, uGFDL3_precip, vGFDL3_precip = aplicador(precip2070, time=timeGFDL3, data=[zGFDL3, qGFDL3, tGFDL3, uGFDL3, vGFDL3])
precipTimeWRF_GFDL3, precipWRF_GFDL3 = aplicador(precip2070, timeWRF=precipTimeWRF_GFDL3, data=precipWRF_GFDL3)

timeIPSL1_precip, zIPSL1_precip, qIPSL1_precip, tIPSL1_precip, uIPSL1_precip, vIPSL1_precip = aplicador(precip1980, time=timeIPSL1, data=[zIPSL1, qIPSL1, tIPSL1, uIPSL1, vIPSL1])
precipTimeWRF_IPSL1, precipWRF_IPSL1 = aplicador(precip1980, timeWRF=precipTimeWRF_IPSL1, data=precipWRF_IPSL1)
timeIPSL2_precip, zIPSL2_precip, qIPSL2_precip, tIPSL2_precip, uIPSL2_precip, vIPSL2_precip = aplicador(precip2030, time=timeIPSL2, data=[zIPSL2, qIPSL2, tIPSL2, uIPSL2, vIPSL2])
precipTimeWRF_IPSL2, precipWRF_IPSL2 = aplicador(precip2030, timeWRF=precipTimeWRF_IPSL2, data=precipWRF_IPSL2)
timeIPSL3_precip, zIPSL3_precip, qIPSL3_precip, tIPSL3_precip, uIPSL3_precip, vIPSL3_precip = aplicador(precip2070, time=timeIPSL3, data=[zIPSL3, qIPSL3, tIPSL3, uIPSL3, vIPSL3])
precipTimeWRF_IPSL3, precipWRF_IPSL3 = aplicador(precip2070, timeWRF=precipTimeWRF_IPSL3, data=precipWRF_IPSL3)

timeMIROC1_precip, zMIROC1_precip, qMIROC1_precip, tMIROC1_precip, uMIROC1_precip, vMIROC1_precip = aplicador(precip1980, time=timeMIROC1, data=[zMIROC1, qMIROC1, tMIROC1, uMIROC1, vMIROC1])
precipTimeWRF_MIROC1, precipWRF_MIROC1 = aplicador(precip1980, timeWRF=precipTimeWRF_MIROC1, data=precipWRF_MIROC1)
timeMIROC2_precip, zMIROC2_precip, qMIROC2_precip, tMIROC2_precip, uMIROC2_precip, vMIROC2_precip = aplicador(precip2030, time=timeMIROC2, data=[zMIROC2, qMIROC2, tMIROC2, uMIROC2, vMIROC2])
precipTimeWRF_MIROC2, precipWRF_MIROC2 = aplicador(precip2030, timeWRF=precipTimeWRF_MIROC2, data=precipWRF_MIROC2)
timeMIROC3_precip, zMIROC3_precip, qMIROC3_precip, tMIROC3_precip, uMIROC3_precip, vMIROC3_precip = aplicador(precip2070, time=timeMIROC3, data=[zMIROC3, qMIROC3, tMIROC3, uMIROC3, vMIROC3])
precipTimeWRF_MIROC3, precipWRF_MIROC3 = aplicador(precip2070, timeWRF=precipTimeWRF_MIROC3, data=precipWRF_MIROC3)