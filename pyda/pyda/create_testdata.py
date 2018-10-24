import wrf
from netCDF4 import Dataset, num2date

def obtain_background_data(nc_f, left_bottom, right_top, 
                           fields):
    background_data = {}
    
    nc_fid = Dataset(nc_f, 'r')
    lats_2d = nc_fid.variables['XLAT'][0, :, :]
    lons_2d = nc_fid.variables['XLONG'][0, :, :]

    left_bottom_xy = wrf.ll_to_xy(nc_fid, left_bottom[0], left_bottom[1])
    right_top_xy = wrf.ll_to_xy(nc_fid, right_top[0], right_top[1])
    
    
    if left_bottom_xy[1].values == right_top_xy[1].values:
        lat_start = left_bottom_xy[1].values
        lat_end = right_top_xy[1].values + 1
    else:
        lat_start = left_bottom_xy[1].values
        lat_end = right_top_xy[1].values

    if left_bottom_xy[0].values == right_top_xy[0].values:
        lon_start = int(left_bottom_xy[0].values)
        lon_end = int(right_top_xy[0].values + 1)
    else:
        lon_start = int(left_bottom_xy[0].values)
        lon_end = int(right_top_xy[0].values)

    background_data['lat'] = lats_2d[
            lat_start:lat_end, lon_start:lon_end]

    background_data['lon'] = lons_2d[
            lat_start:lat_end, lon_start:lon_end]

    for field in fields:
        if not field in ['dbz', 'T', 'U', 'V']:
            raise Exception('{} is not supported'.format(field))
        
        if field == 'dbz':
            pressure = wrf.g_pressure.get_pressure(nc_fid).values
            temperature = wrf.tk(pressure, wrf.g_temp.get_theta(nc_fid))
            water_vapor_mixing_ratio = nc_fid.variables['QVAPOR'][0, :, :, :]
            rain_mixig_ratio = nc_fid.variables['QRAIN'][0, :, :, :]
            snow_mixig_ratio = nc_fid.variables['QSNOW'][0, :, :, :]
            graupel_mixing_ratio = nc_fid.variables['QGRAUP'][0, :, :, :]
            cur_background_data = wrf.dbz(pressure, temperature,
                          water_vapor_mixing_ratio,
                          rain_mixig_ratio, snow_mixig_ratio,
                                graupel_mixing_ratio,
                                use_varint=True, meta=False)[0, lat_start:lat_end, lon_start:lon_end]
        else:
            if field == 'T':
                field_in = 'T2'
            elif field == 'U' or field == 'V':
                field_in = field + '10'
            else:
                field_in = field
                
            cur_background_data = nc_fid.variables[field_in][
                0, lat_start:lat_end, lon_start:lon_end]
        
        background_data[field] = cur_background_data

    return background_data



def obtain_obs_data(observation_data_dict):
    observation_data = {}
    
    for n, cur_obstype in enumerate(observation_data_dict):
        cur_data = observation_data_dict[cur_obstype]
        if cur_obstype not in observation_data:
            observation_data[cur_obstype] = {}
            observation_data[cur_obstype]['value'] = []
            observation_data[cur_obstype]['lat'] = []
            observation_data[cur_obstype]['lon'] = []
        observation_data[cur_obstype]['value'].append(cur_data['value'])
        observation_data[cur_obstype]['lat'].append(cur_data['lat'])
        observation_data[cur_obstype]['lon'].append(cur_data['lon'])
        
        # flat the list
        observation_data[cur_obstype]['value'] = [y for x in observation_data[cur_obstype]['value'] for y in x]
        observation_data[cur_obstype]['lat'] = [y for x in observation_data[cur_obstype]['lat'] for y in x]
        observation_data[cur_obstype]['lon'] = [y for x in observation_data[cur_obstype]['lon'] for y in x]
    
    return observation_data
    
    