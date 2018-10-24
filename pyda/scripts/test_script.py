import matplotlib.pyplot as plt
from pyda import create_testdata, create_ensembles, obs_term
from pyda import cost_function, da_utils, da_vis
import os
import sympy
import math
import numpy, scipy

import time
start_time = time.time()

work_dir = '/tmp/pyda'
if_create_new_be = True
obs_influcial_range = 0.1


print 'start'
os.environ["MKL_THREADING_LAYER"] = "GNU"

background_data_path = '/home/szhang/workspace/pyda/tests/wrf_hourly_nz8kmN-NCEP-var-39-3_d02_2018-05-11_10:00:00'

background_domain = {
    'left_bottom': [-35.0, 177.0],
    'right_top': [-34.0, 178.0],
    'fields': ['T', 'U'],
    }
observation_data = {
    'T': {
        'value': [298.5, 287.0],
        'lat': [-34.5,-34.2],
        'lon': [177.5,177.6]
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
    'T': { 'sigma': 0.15},
    'dbz': { 'sigma': 0.5},
    'U': { 'sigma': 0.1},
    }
direct_obs_operator = {
    #'T2': sympy.Symbol('T2') + sympy.Symbol('U10')*0.001,
    #'T2': sympy.exp(sympy.Symbol('T2')),
    'T': sympy.Symbol('T'),
    'dbz': sympy.Symbol('U')*0.001 + sympy.Symbol('dbz'),
    'U': sympy.Symbol('U')
    }

# step 1: create the demo dataset
# 1.1: create the background data
print 'obtain_background_data'
background_data_dict = create_testdata.obtain_background_data(
        background_data_path, background_domain['left_bottom'], background_domain['right_top'],
        background_domain['fields'])


# 1.2: create the observation data
print 'obtain_obs_data'
observation_data_dict = create_testdata.obtain_obs_data(observation_data)


# 1.3: create background vectors
print 'convert_background_to_vector'
(background_data_vector,
 background_lat_vector, 
 background_lon_vector,
 background_type_vector) = da_utils.convert_background_to_vector(background_data_dict)

# 1.4: create observation vectors
print 'convert_observation_to_vector'
(obs_data_vector, 
 obs_lat_vector, 
 obs_lon_vector,
 obs_type_vector) = da_utils.convert_observation_to_vector(observation_data_dict)


analysis_increment_vector_symbol = sympy.MatrixSymbol('analysis_increment_symbol', len(background_data_vector), 1)

print 'setup_observation_term'
linearized_obs_operator, innovation, background_derived_obs = obs_term.setup_observation_term(background_data_dict,
                       obs_type_vector, obs_data_vector, obs_lat_vector, obs_lon_vector,
                       background_type_vector, background_data_vector, 
                       background_lat_vector, background_lon_vector,
                       direct_obs_operator,
                       analysis_increment_vector_symbol)
r = obs_term.setup_observation_errors(obs_type_vector)

obs_term = -2 * (linearized_obs_operator.transpose()) * (r**(-1)) * innovation

# cost_function.obs_term_gradient_optimization(obs_term, background_data_vector)

# 1.4 obtain background errors
# 1.4.1 create a new be if it is required
print 'if_create_new_be'
if if_create_new_be:
    cost_function.create_background_errors(
        background_data_dict, len(background_data_vector), perterbation_parameters,
        os.path.join(work_dir, 'be.data'))

# 1.4.2 load the be file
print 'load_background_errors'
background_error_matrix = cost_function.load_background_errors(os.path.join(work_dir, 'be.data'))

print 'B size: {}'.format(background_error_matrix.shape)

background_error_array = numpy.asarray(background_error_matrix)
background_error_array_inv = numpy.linalg.inv(background_error_array)
background_error_matrix_inv = sympy.Matrix(background_error_array_inv)
bkg_term = 2 * (background_error_matrix_inv) * (analysis_increment_vector_symbol)
# bkg_term = 2 * (sympy.Matrix(background_error_matrix) ** (-1)) * (analysis_increment_vector_symbol)

# f_term = bkg_term + obs_term

# f_term2 = da_utils.round_exp(f_term, sig_figures=1)

print 'obs_term_gradient_optimization'
analysis_data_vector = cost_function.obs_term_gradient_optimization2(background_data_vector, obs_data_vector,
                                background_error_matrix_inv, r**(-1), linearized_obs_operator, background_derived_obs)

#analysis_data_vector = cost_function.obs_term_gradient_optimization(f_term, background_data_vector)


# 1.5 solve the background term:
#analysis_data_vector = cost_function.background_term_gradient_optimization(
#    background_error_matrix, background_data_vector)

analysis_data = da_utils.convert_analysis_vector_to_matrix(background_data_dict, analysis_data_vector)

da_vis.compare_background_and_analysis(background_data_dict, analysis_data)

end_time = time.time()

print 'takes: {}'.format((end_time - start_time))
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