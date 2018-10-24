import sympy
import numpy
import math

def setup_observation_errors(obs_type_vector):
    r = sympy.Matrix(numpy.matrix(0.1*numpy.identity(len(obs_type_vector))))
    return r

def setup_observation_term(background_data_dict,
                           obs_type_vector, obs_data_vector, obs_lat_vector, obs_lon_vector,
                           background_type_vector, background_data_vector, background_lat_vector, background_lon_vector,
                           direct_obs_operator,
                           analysis_increment_vector_symbol,
                           increment_data=0.001):
    """set up the observation term symbol matrix"""
    linearized_obs_operator = sympy.Matrix(
        numpy.ones((len(obs_type_vector), len(background_type_vector))))
    background_derived_obs = sympy.Matrix(
        numpy.ones((len(obs_type_vector), 1)))

    for i_obs in range(0, len(obs_data_vector)):
        cur_obs_type = str(obs_type_vector[i_obs])
        cur_obs_lat = float(obs_lat_vector[i_obs])
        cur_obs_lon = float(obs_lon_vector[i_obs])

        cur_bk_derived_obs_operator = direct_obs_operator[cur_obs_type]
        required_bk_vars = cur_bk_derived_obs_operator.free_symbols
        
        # 1. calculate h(x):
        bkg_shape = background_data_dict['lat'].shape
        bkg_weight_0 = 0
        bkg_weight_1 = 0
        for ii in range(0, bkg_shape[0]):
            for jj in range(0, bkg_shape[1]):
                cur_bk_lat = background_data_dict['lat'][ii, jj]
                cur_bk_lon = background_data_dict['lon'][ii, jj]

                cur_obs_operator_input = {}
                for required_bk_var in required_bk_vars:
                    cur_obs_operator_input[str(required_bk_var)] = background_data_dict[str(required_bk_var)][ii, jj]

                cur_distance = math.sqrt((cur_obs_lat - cur_bk_lat)**2 + (cur_obs_lon - cur_bk_lon)**2)
                cur_distance_weight_0 = (direct_obs_operator[cur_obs_type].subs(cur_obs_operator_input))/float(cur_distance**2)
                cur_distance_weight_1 = 1.0/float(cur_distance**2)
                bkg_weight_0 = bkg_weight_0 + cur_distance_weight_0
                bkg_weight_1 = bkg_weight_1 + cur_distance_weight_1
        
        background_derived_obs[i_obs, 0] = bkg_weight_0/bkg_weight_1
        final_bkg_weight_1 = bkg_weight_1
        
        # 2. find linearized observation operator
        # n = 0
        for i_bkg in range(0, len(background_type_vector)):
            cur_bk_lat = float(background_lat_vector[i_bkg])
            cur_bk_lon = float(background_lon_vector[i_bkg])
            cur_bk_type = str(background_type_vector[i_bkg])
            
            cur_distance = math.sqrt((cur_obs_lat - cur_bk_lat)**2 + (cur_obs_lon - cur_bk_lon)**2)
            cur_distance_weight_0 = cur_distance ** 2
            distance_term = 1.0/(cur_distance_weight_0 * final_bkg_weight_1)
            
            l = sympy.diff(direct_obs_operator[cur_obs_type], cur_bk_type)
            linearized_obs_operator[i_obs, i_bkg] = distance_term*l.subs(
                {cur_bk_type: float(background_data_vector[i_bkg])})
            
    linearized_obs_operator_with_increments = \
        linearized_obs_operator*analysis_increment_vector_symbol
    
    # get the innovation:
    innovation = obs_data_vector - background_derived_obs - \
        linearized_obs_operator_with_increments
    return linearized_obs_operator, innovation, background_derived_obs

'''
cur_bkg_t2_index = background_type_vector.tolist().index(['T2'])
cur_bkg_u10_index = 100
#cur_bkg_u10_index = background_type_vector.tolist().index(['U10'])
if n < cur_bkg_u10_index:
    l = sympy.diff(direct_obs_operator['T2'], 'T2')
    linearized_obs_operator[i_obs, i_bkg] = distance_term*l.subs({'T2': t2_haha[i_bkg]})
    # l_obs_operator_to_fill[i_obs, i_bkg] = distance_term*l
    
else:
    l = sympy.diff(direct_obs_operator['T2'], 'U10')
    # l_obs_operator_to_fill[i_obs, i_bkg] = distance_term*l.subs({'U10': bkg_data[n]})
    linearized_obs_operator[i_obs, i_bkg] = distance_term*l
'''

def setup_observation_term2(background_data_dict,
                           obs_type_vector, obs_data_vector, obs_lat_vector, obs_lon_vector,
                           background_type_vector, background_data_vector,
                           direct_obs_operator,
                           increment_data=0.001):
    
    linearized_observation_operator = sympy.Matrix(
        numpy.zeros((len(obs_data_vector), len(background_data_vector))))
    
    bkg_shape = background_data_dict['lat'].shape
    
    n = 0
    increment_data = sympy.MatrixSymbol('delatX', len(background_type_vector), 1)
    j_obs = sympy.Matrix(numpy.ones((len(background_type_vector), 1)))
    
    for i in range(0, bkg_shape[0]):
        for j in range(0, bkg_shape[1]):
            cur_bk_lat = background_data_dict['lat'][i, j]
            cur_bk_lon = background_data_dict['lon'][i, j]
            
            background_derived_obs = sympy.Matrix(numpy.ones(obs_data_vector.shape))
            
            for i_obs, _ in enumerate(obs_data_vector):
                cur_obs_type = obs_type_vector[i_obs][0, 0]
                cur_obs_lat = obs_lat_vector[i_obs][0, 0]
                cur_obs_lon = obs_lon_vector[i_obs][0, 0]
                
                cur_bk_derived_obs_operator = direct_obs_operator[cur_obs_type]
                required_bk_vars = cur_bk_derived_obs_operator.free_symbols
                cur_obs_operator_input = {}
                for required_bk_var in required_bk_vars:
                    start_index = background_type_vector.tolist().index([str(required_bk_var)])
                    cur_obs_operator_input[str(required_bk_var)] = background_data_vector[start_index + n][0, 0]
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
            # incremental_data = numpy.matrix(increment_data*numpy.ones(background_data_vector.shape))
            innovation = obs_data_vector - background_derived_obs - linearized_observation_operator*increment_data

            r = setup_observation_errors(obs_type_vector)
            j_obs[n] = -2* linearized_observation_operator.transpose() * (r**(-1)) * innovation
            n = n + 1
    
    a =3
    
        
                
                