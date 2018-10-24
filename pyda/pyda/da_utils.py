import numpy
import sympy

def round_exp(expr, sig_figures=1):
    ex2 = expr
    for a in sympy.preorder_traversal(expr):
        if isinstance(a, sympy.Float):
            ex2 = ex2.subs(a, round(a, sig_figures))
    return ex2


def convert_observation_to_vector(observation_data_dict):
    data_vector = []
    lat_vector = []
    lon_vector = []
    type_vector = []
    
    for data_type in observation_data_dict:
        data_length = len(observation_data_dict[data_type]['lat'])
        for i in range(0, data_length):
            data_vector.append(observation_data_dict[data_type]['value'][i])
            lat_vector.append(observation_data_dict[data_type]['lat'][i])
            lon_vector.append(observation_data_dict[data_type]['lon'][i])
            type_vector.append(data_type)

    obs_data_vector = sympy.Matrix(data_vector)
    obs_lat_vector = sympy.Matrix(lat_vector)
    obs_lon_vector = sympy.Matrix(lon_vector)
    obs_type_vector = sympy.Matrix(type_vector)
    
    return obs_data_vector, obs_lat_vector, obs_lon_vector, obs_type_vector

def convert_background_to_vector(background_data_dict):
    """convert background data from 2d array to vector
        the input must be a dict, and including at least three keys:
        lat, lon and "data_type" (e.g., T2)
    """
    data_vector = []
    type_vector = []
    lat_vector = []
    lon_vector = []
    
    for data_type in background_data_dict:
        data_shape = background_data_dict['lat'].shape
        if data_type == 'lat' or data_type == 'lon':
            continue
        for i in range(0, data_shape[0]):
            for j in range(0, data_shape[1]):
                data_vector.append(background_data_dict[data_type][i, j])
                lat_vector.append(background_data_dict['lat'][i, j])
                lon_vector.append(background_data_dict['lon'][i, j])
                type_vector.append(data_type)
                
    background_data_vector = sympy.Matrix(data_vector)
    background_lat_vector = sympy.Matrix(lat_vector)
    background_lon_vector = sympy.Matrix(lon_vector)
    background_type_vector = sympy.Matrix(type_vector)

    return (background_data_vector, background_lat_vector, 
            background_lon_vector, background_type_vector)


def convert_analysis_vector_to_matrix(background_data, analysis_vector):
    """covert the vector from the analysis process 
        back to its original space (e.g., 2d or 3d)"""
    n = 0
    analysis_data = {}
    for data_type in background_data:
        data_shape = background_data['lat'].shape
        cur_analysis_data = numpy.empty(data_shape)
        if data_type == 'lat' or data_type == 'lon':
            continue
        for i in range(0, data_shape[0]):
            for j in range(0, data_shape[1]):
                cur_data = analysis_vector[n]
                cur_analysis_data[i, j] = cur_data
                n = n + 1
        
        analysis_data[data_type] = cur_analysis_data
        
    return analysis_data
