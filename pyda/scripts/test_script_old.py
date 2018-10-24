import matplotlib.pyplot as plt
from pyda import create_testdata, create_ensembles, obs_term
from pyda import cost_function, da_utils, da_vis
import os
import sympy
import math
import numpy

work_dir = '/tmp/pyda'
if_create_new_be = False
obs_influcial_range = 0.1

background_data_path = '/home/szhang/workspace/pyda/tests/wrf_hourly_nz8kmN-NCEP-var-39-3_d02_2018-05-11_10:00:00'

background_domain = {
    'left_bottom': [-35.1, 175.9],
    'right_top': [-35.0, 176],
    'fields': ['T2'],
    }
observation_data = {
    'T2': {
        'value': [200.5],
        'lat': [-35.1],
        'lon': [175.88]
        }}
'''
    'T2': {
        'value': [273.5, 271.0],
        'lat': [-35.0, -35.5],
        'lon': [175.0, 175.3]
        }}
'''
'''
    'dbz': {
        'value': [13.5, 8.5],
        'lat': [-35.0, -35.7], 
        'lon': [175.0, 175.5]},
    'U10': {
        'value': [3.5, -1.5],
        'lat': [-35.0, -35.7],
        'lon': [175.0, 175.5]},
'''
perterbation_parameters = {
    'ensemble_size': 1000,
    'T2': { 'sigma': 0.15},
    'dbz': { 'sigma': 0.5},
    'U10': { 'sigma': 0.1},
    }
direct_obs_operator = {
    #'T2': sympy.Symbol('T2') + sympy.Symbol('U10')*0.001,
    #'T2': sympy.exp(sympy.Symbol('T2')),
    'T2': sympy.Symbol('T2'),
    'dbz': sympy.Symbol('U10')*0.001 + sympy.Symbol('dbz'),
    'U10': sympy.Symbol('U10')
    }

# step 1: create the demo dataset
# 1.1: create the background data
background_data_dict = create_testdata.obtain_background_data(
        background_data_path, background_domain['left_bottom'], background_domain['right_top'],
        background_domain['fields'])


# 1.2: create the observation data
observation_data_dict = create_testdata.obtain_obs_data(observation_data)


# 1.3: create background vectors
(background_data_vector, 
 background_lat_vector, 
 background_lon_vector,
 background_type_vector) = da_utils.convert_background_to_vector(background_data_dict)

# 1.4: create observation vectors
(obs_data_vector, 
 obs_lat_vector, 
 obs_lon_vector,
 obs_type_vector) = da_utils.convert_observation_to_vector(observation_data_dict)

linearized_obs_operator, innovation = obs_term.setup_observation_term(background_data_dict,
                       obs_type_vector, obs_data_vector, obs_lat_vector, obs_lon_vector,
                       background_type_vector, background_data_vector, 
                       background_lat_vector, background_lon_vector,
                       direct_obs_operator)
r = obs_term.setup_observation_errors(obs_type_vector)

obs_gradient = (linearized_obs_operator.transpose()) * (r**(-1)) * innovation

cost_function.obs_term_gradient_optimization(obs_gradient, background_data_vector)


a=b
# observation operator shape:
# h(x) = (number of observation * length of background data vector)
#background_derived_obs_matrix = sympy.Matrix(numpy.zeros((len(obs_data_vector), 
#                             len(background_data_vector))))

#observation_operator = sympy.Matrix(numpy.zeros((len(obs_data_vector), len(background_data_vector))))
#linearized_observation_operator = sympy.Matrix(numpy.zeros((len(obs_data_vector), len(background_data_vector))))

'''
bkg_shape = background_data_dict['lat'].shape
for i_obs, cur_obs in enumerate(obs_data_vector):
    cur_obs_type = obs_type_vector[i_obs][0, 0]
    for i_bkg in range(0, len(background_type_vector)):
        observation_operator[i_obs, i_bkg]=direct_obs_operator[cur_obs_type]
        linearized_observation_operator[i_obs, i_bkg] = sympy.diff(direct_obs_operator[cur_obs_type], background_type_vector[i_bkg])
'''

for i_obs, cur_obs in enumerate(obs_data_vector):
    cur_obs_type = obs_type_vector[i_obs][0, 0]
    cur_obs_lat = obs_lat_vector[i_obs][0, 0]
    cur_obs_lon = obs_lon_vector[i_obs][0, 0]
    cur_dist_list = []
    
    # check if we have the required model fields to 
    #  calculate the observation operator
    cur_bk_derived_obs_operator = direct_obs_operator[cur_obs_type]
    required_bk_vars = cur_bk_derived_obs_operator.free_symbols
    
    for required_bk_var in required_bk_vars:
        if str(required_bk_var) not in background_data_dict.keys():
            raise Exception('we cannot find {} from background '
                            'to process the obs operator for {}'.format(
                                required_bk_var, cur_obs_type))
    
    '''
    for i_bkg in range(0, len(background_type_vector)):
        cur_obs_operator_input = {}
        for required_bk_var in required_bk_vars:
            cur_obs_operator_input[str(required_bk_var)] = background_data_dict[str(required_bk_var)][i, j]
        background_derived_obs = direct_obs_operator[cur_obs_type].subs(cur_obs_operator_input)
        cur_obs_operator = bservation_operator[i_obs, i_bkg]
    ''' 
    
    analysis_data_vector = numpy.matrix(numpy.ones(background_data_vector.shape))
    
    bkg_shape = background_data_dict['lat'].shape
    for i in range(0, bkg_shape[0]):
        for j in range(0, bkg_shape[1]):
            cur_bk_lat = background_data_dict['lat'][i, j]
            cur_bk_lon = background_data_dict['lon'][i, j]
            cur_distance = math.sqrt((cur_obs_lat - cur_bk_lat)**2 + (cur_obs_lon - cur_bk_lon)**2)
            cur_dist_list.append(cur_distance)

    total_distance = sum(cur_dist_list)
    n = 0
    for i in range(0, bkg_shape[0]):
        for j in range(0, bkg_shape[1]):
            '''
            cur_obs_operator_input = {}
            for required_bk_var in required_bk_vars:
                start_index = background_type_vector.tolist().index([str(required_bk_var)])
                cur_obs_operator_input[str(required_bk_var)] = background_data_vector[start_index + n][0, 0]
                linearized_observation_operator[i_obs, start_index + n] = sympy.diff(
                    distance_conversion*direct_obs_operator[cur_obs_type], background_type_vector[start_index + n][0, 0])
            '''
            background_derived_obs = sympy.Matrix(numpy.ones(obs_data_vector.shape))
            
            #cur_bk_lat = background_data_dict['lat'][i, j]
            #cur_bk_lon = background_data_dict['lon'][i, j]
            
            for i_obs, cur_obs in enumerate(obs_data_vector):
                cur_obs_lat = obs_lat_vector[i_obs][0, 0]
                cur_obs_lon = obs_lon_vector[i_obs][0, 0]
                cur_obs_operator_input = {}
                for required_bk_var in required_bk_vars:
                    start_index = background_type_vector.tolist().index([str(required_bk_var)])
                    cur_obs_operator_input[str(required_bk_var)] = background_data_vector[start_index + n][0, 0]
                    cur_distance0 = math.sqrt((cur_obs_lat - cur_bk_lat)**2 + (cur_obs_lon - cur_bk_lon)**2)
                    distance_conversion0 = cur_distance0/float(total_distance)
                    linearized_observation_operator[i_obs, start_index + n] = sympy.diff(
                        direct_obs_operator[cur_obs_type], background_type_vector[start_index + n][0, 0])
                    
                bkg_weight_0 = 0
                bkg_weight_1 = 0
                for ii in range(0, bkg_shape[0]):
                    for jj in range(0, bkg_shape[1]):
                        cur_bk_lat = background_data_dict['lat'][ii, jj]
                        cur_bk_lon = background_data_dict['lon'][ii, jj]
                        cur_distance = math.sqrt((cur_obs_lat - cur_bk_lat)**2 + (cur_obs_lon - cur_bk_lon)**2)
                        cur_distance_weight_0 = (direct_obs_operator[cur_obs_type].subs(cur_obs_operator_input))/float(cur_distance**2)
                        cur_distance_weight_1 = 1.0/float(cur_distance**2)
                        bkg_weight_0 = bkg_weight_0 + cur_distance_weight_0
                        bkg_weight_1 = bkg_weight_1 + cur_distance_weight_1
                        
                        
                        #distance_conversion = cur_distance/float(total_distance)
                        #bkg_derived_obs_sum = bkg_derived_obs_sum + distance_conversion * direct_obs_operator[cur_obs_type].subs(cur_obs_operator_input)
                
                background_derived_obs[i_obs] = bkg_weight_0/bkg_weight_1

            # linearized_observation_operator[i_obs, i_bkg]
            incremental_data = numpy.matrix(0.01*numpy.ones(background_data_vector.shape))
            innovation = obs_data_vector - background_derived_obs - linearized_observation_operator*incremental_data

            n = n + 1
    '''
    for i in range(0, bkg_shape[0]):
        for j in range(0, bkg_shape[1]):
            cur_obs_operator_input = {}
            background_data_vector_for_field = {}
            for required_bk_var in required_bk_vars:
                cur_obs_operator_input[str(required_bk_var)] = background_data_dict[str(required_bk_var)][i, j]
            
            # operator_conversion = cur_bk_derived_obs_operator.subs(cur_obs_operator_input)
            cur_bk_lat = background_data_dict['lat'][i, j]
            cur_bk_lon = background_data_dict['lon'][i, j]
            cur_distance = math.sqrt((cur_obs_lat - cur_bk_lat)**2 + (cur_obs_lon - cur_bk_lon)**2)
            
            distance_conversion = cur_distance/float(total_distance)
            
            cur_hx = distance_conversion*cur_bk_derived_obs_operator.subs(cur_obs_operator_input)
            
            background_derived_obs_matrix[i_obs, n] = cur_hx
            n = n + 1
    '''
# (number of observation * length of background data vector) * (length of background data vector * 1)
#  = (number of observation * 1)
# h_x = background_derived_obs_matrix*sympy.Matrix(numpy.ones((len(background_type_vector), 1)))
    
# innovation = obs_data_vector - h_x

a = 3
'''
    for i_bk, cur_bk in enumerate(background_data_vector):
        # get the total distances for all model grid to this obs location
        cur_bk_lat = background_lat_vector[i_bk]
        cur_bk_lon = background_lon_vector[i_bk]
        cur_distance = math.sqrt((cur_obs_lat - cur_bk_lat)**2 + (cur_obs_lon - cur_bk_lon)**2)
        cur_dist_list.append(cur_distance)
        
    total_dis = sum(cur_dist_list)
    
    
    
    for i_bk, cur_bk in enumerate(background_data_vector):
        cur_bk_data = background_data_vector[i_bk]
        cur_bk_type = background_type_vector[i_bk]
        cur_bk_lat = background_lat_vector[i_bk]
        cur_bk_lon = background_lon_vector[i_bk]
        
        # obtain the required background variables
        cur_bk_derived_obs = direct_obs_operator['T2']
'''
# 1.4 obtain background errors
# 1.4.1 create a new be if it is required
if if_create_new_be:
    cost_function.create_background_errors(
        background_data_dict, len(background_data_vector), perterbation_parameters,
        os.path.join(work_dir, 'be.data'))

# 1.4.2 load the be file
background_error_matrix = cost_function.load_background_errors(os.path.join(work_dir, 'be.data'))

# 1.5 process the observation term


# 1.5 solve the background term:
analysis_data_vector = cost_function.background_term_gradient_optimization(
    background_error_matrix, background_data_vector)

analysis_data = da_utils.convert_analysis_vector_to_matrix(background_data_dict, analysis_data_vector)

da_vis.compare_background_and_analysis(background_data_dict, analysis_data)


print 'done'
'''
create_ensembles.calculate_ensembles(data_array, 100)
'''
'''
obs_data = create_testdata.create_1d_normal_distribution_data(11.2, 10, data_size=100)


plt.subplot(121)
plt.hist(background_data, 20, normed=True)
plt.subplot(122)
plt.hist(obs_data, 20, normed=True)

plt.show()
'''