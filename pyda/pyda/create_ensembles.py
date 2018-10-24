import numpy

def calculate_ensembles(data_dict, perterbation_parameters):
    data_shape = data_dict['lat'].shape
    
    ensemble_data_dict = {}
    
    ensemble_data_dict['lat'] = data_dict['lat']
    ensemble_data_dict['lon'] = data_dict['lon']
    
    for data_type in data_dict:
        if data_type == 'lat' or  data_type == 'lon':
            continue
        if data_type not in ensemble_data_dict:
            ensemble_data_dict[data_type] = numpy.empty((
                perterbation_parameters['ensemble_size'],
                data_shape[0], data_shape[1]))
        
        for i in range(0, data_shape[0]):
            for j in range(0, data_shape[1]):
                cur_data_value = data_dict[data_type][i, j]
                cur_en = numpy.random.normal(cur_data_value, 
                                        perterbation_parameters[data_type]['sigma'], 
                                        perterbation_parameters['ensemble_size'])
                ensemble_data_dict[data_type][:, i, j] = cur_en
    
    return ensemble_data_dict
                