import numpy
from numpy import float16
import da_utils
import create_ensembles
import pickle
import os
import sympy
from sympy.printing.theanocode import theano_function
from sympy import blockcut, block_collapse
from scipy import arange

def obs_term_gradient_optimization2(background_data_vector, observation_data_vector,
                                    actual_B_1, actual_R_1, actual_H, actual_hx,
                                    smallest_gradient=0.01,
                                    iteration_step=0.001,
                                    blockn=5,
                                    enable_parallel=False):
    """http://matthewrocklin.com/blog/work/2013/04/05/SymPy-Theano-part-3"""
    len_background_data_vector = len(background_data_vector)
    len_observation_data_vector = len(background_data_vector)
    # background_error_matrix_reverse = sympy.MatrixSymbol('B_1', len(background_data_vector), len(background_data_vector))
    background_error_matrix_reverse = sympy.MatrixSymbol('B_1', len_background_data_vector, len_background_data_vector)
    observation_error_matrix_reverse = sympy.MatrixSymbol('R_1', len_observation_data_vector, len_observation_data_vector)
    linearized_observation_operator = sympy.MatrixSymbol('H', len_observation_data_vector, len_background_data_vector)
    background_derived_obs_matrix = sympy.MatrixSymbol('hx', len_observation_data_vector, 1)
    observation_data_matrix = sympy.MatrixSymbol('y', len_observation_data_vector, 1)
    analysis_increment_vector_symbol = sympy.MatrixSymbol('delta_x', len_background_data_vector, 1)
    
    obs_term_gradient_value = numpy.matrix(999.0*numpy.ones(background_data_vector.shape))
    analysis_increment_vector = numpy.matrix(numpy.zeros(background_data_vector.shape))
    
    inputs = [background_error_matrix_reverse, 
              observation_error_matrix_reverse, 
              linearized_observation_operator, 
              background_derived_obs_matrix, 
              observation_data_matrix, 
              analysis_increment_vector_symbol]
    cost_function_gradient = [2*background_error_matrix_reverse*analysis_increment_vector_symbol - 
                              2*linearized_observation_operator.transpose()*observation_error_matrix_reverse*(
                                  observation_data_matrix - background_derived_obs_matrix - 
                                  linearized_observation_operator*analysis_increment_vector_symbol)]
    
    dtypes = {inp: 'float16' for inp in inputs}
    
    print 'create func'
    if enable_parallel:
        blocksizes = {
                background_error_matrix_reverse: [tuple(blockn*[len_background_data_vector/blockn]), 
                                                  tuple(blockn*[len_background_data_vector/blockn])],
                observation_error_matrix_reverse:     [tuple(blockn*[len_observation_data_vector/blockn]), 
                                                       tuple(blockn*[len_observation_data_vector/blockn])],
                linearized_observation_operator:     [tuple(blockn*[len_observation_data_vector/blockn]), 
                                                      tuple(blockn*[len_background_data_vector/blockn])],
                background_derived_obs_matrix:    [tuple(blockn*[len_observation_data_vector/blockn]), (1,)],
                observation_data_matrix:  [tuple(blockn*[len_observation_data_vector/blockn]), (1,)],
                analysis_increment_vector_symbol:  [tuple(blockn*[len_background_data_vector/blockn]), (1,)],
                }
        blockinputs = [blockcut(i, *blocksizes[i]) for i in inputs]
        blockoutputs = [o.subs(dict(zip(inputs, blockinputs))) for o in cost_function_gradient]
        collapsed_outputs = map(block_collapse, blockoutputs)

        f = theano_function(inputs, collapsed_outputs, dtypes=dtypes)
    else:
        f = theano_function(inputs, cost_function_gradient, dtypes=dtypes)
        #f=sympy.lambdify(analysis_increment_vector_symbol, obs_gradient, 'numpy')
        #f=sympy.lambdify(analysis_increment_vector_symbol, obs_gradient, 'sympy')
    print 'created func'
    n = 0
    while (obs_term_gradient_value.sum() <= -smallest_gradient or 
               obs_term_gradient_value.sum() > smallest_gradient):
        ninputs = [numpy.array(actual_B_1).astype(float16), 
                   numpy.array(actual_R_1).astype(float16), 
                   numpy.array(actual_H).astype(float16),  
                   numpy.array(actual_hx).astype(float16), 
                   numpy.array(observation_data_vector).astype(float16), 
                   numpy.array(analysis_increment_vector).astype(float16)]
        # [2*B_1*delta_x - 2*H.T*R_1*(-hx - H*delta_x + y)]
        # obs_term_gradient_value = 2*actual_B_1*analysis_increment_vector - 2*actual_H.transpose()*actual_R_1*(observation_data_vector-actual_hx-actual_H*analysis_increment_vector)
        #obs_term_gradient_value = 2*numpy.asarray(actual_B_1)*numpy.asarray(analysis_increment_vector) - 2*actual_H.transpose()*actual_R_1*(observation_data_vector-actual_hx-actual_H*analysis_increment_vector)
       
        obs_term_gradient_value = f(*ninputs)
        analysis_increment_vector = analysis_increment_vector - iteration_step * obs_term_gradient_value
        
        print n, obs_term_gradient_value.sum()
        
        n = n + 1
        
        if n > 2000:
            break
    
    # print analysis_increment_vector
    # print background_data_vector + analysis_increment_vector
    return background_data_vector + analysis_increment_vector

def obs_term_gradient_optimization(obs_gradient, background_data_vector,
                                   smallest_gradient=0.001,
                                   iteration_step=0.001):
    # def _get_haha(x):
    #    a = obs_gradient.subs({analysis_increment_vector_symbol: x})
    #    return a
    obs_term_gradient_value = numpy.matrix(999.0*numpy.ones(background_data_vector.shape))
    #analysis_increment_vector = sympy.Matrix(numpy.zeros(background_data_vector.shape))
    analysis_increment_vector = numpy.matrix(numpy.zeros(background_data_vector.shape))

    analysis_increment_vector_symbol = sympy.MatrixSymbol('analysis_increment_symbol', len(background_data_vector), 1)
    # from scipy import optimize
    # optimize.minimize(_get_haha, numpy.zeros(background_data_vector.shape), method="CG")  
    
    # a = _get_haha(x)
    dtypes = {inp: 'float8' for inp in analysis_increment_vector_symbol}
    
    print 'create func'
    f = theano_function([analysis_increment_vector_symbol], [obs_gradient], dtypes=dtypes)
    #f=sympy.lambdify(analysis_increment_vector_symbol, obs_gradient, 'numpy')
    print 'created func'
    #f=sympy.lambdify(analysis_increment_vector_symbol, obs_gradient, 'sympy')
    n = 0
    while (obs_term_gradient_value.sum() <= -smallest_gradient or 
               obs_term_gradient_value.sum() > smallest_gradient):
        # obs_term_gradient_value =  obs_gradient.subs({analysis_increment_vector_symbol: analysis_increment_vector}).doit()
        # f=sympy.lambdify(analysis_increment_vector_symbol, obs_gradient, 'sympy')
        #obs_term_gradient_value = f(analysis_increment_vector)
        obs_term_gradient_value = f(analysis_increment_vector)
        
        analysis_increment_vector = analysis_increment_vector - iteration_step * obs_term_gradient_value
        
        print n, obs_term_gradient_value.sum()
        
        n = n + 1
        
        if n > 2000:
            break
    
    # print analysis_increment_vector
    # print background_data_vector + analysis_increment_vector
    return background_data_vector + analysis_increment_vector
        
def background_term_gradient_optimization(background_error, background_data_vector, 
                                       iteration_step=0.001,
                                       max_iteration_steps=99999,
                                       smallest_gradient=0.001,
                                       method='steepest_descent'):
    """
        this function is only used to solve the background term, 
        since there is no observation term involved, basically we just expect to 
        look at the impacts from the background error on the analysis
        * cost function: J_b = [(x-xb)_t]*[B**(-1)]*[(x-xb)]
        * gradient of the cost function d(J_b)/d(x)=(2)*B**(-1)*(x-xb)
    """
    
    if method == 'steepest_descent':
        background_term_gradient = numpy.matrix(999.0*numpy.ones(background_data_vector.shape))
        
        # the initial guess for the analysis is set to a 
        # vector that all elements equal to the mean value from the background
        analysis_data_vector = numpy.matrix(numpy.ones(background_data_vector.shape)) * numpy.mean(background_data_vector)
        
        n = 0
        while (numpy.matrix.sum(background_term_gradient) <= -smallest_gradient or 
               numpy.matrix.sum(background_term_gradient) > smallest_gradient):
            try:
                background_term_gradient =  2 * (background_error ** (-1)) * (analysis_data_vector - background_data_vector)
            except ValueError:
                print('background error shape: {}; '
                      'background_data_vector shape: {}: '
                      'you may need to recreate your be data'.format(
                          background_error.shape, analysis_data_vector.shape))
            cur_solution = (analysis_data_vector - background_data_vector).transpose()* \
                            (background_error**2)* \
                            (analysis_data_vector - background_data_vector)
            analysis_data_vector = analysis_data_vector - iteration_step * background_term_gradient
            n = n + 1
            if n > max_iteration_steps:
                break
            print n, numpy.matrix.sum(background_term_gradient), cur_solution
    
    # print 'background: {}'.format(background_data_vector)
    # print 'final analysis is: {}'.format(analysis_data_vector)
    return analysis_data_vector
    

def convert_background_to_vector(background_data_dict):
    """convert background data from 2d array to vector
        the input must be a dict, and including at least three keys:
        lat, lon and "data_type" (e.g., T2)
    """
    data_vector = []
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


    return data_vector, lat_vector, lon_vector

def load_background_errors(be_path):
    """load background error from previous calculations"""
    background_error_matrix = pickle.load(open(be_path, "rb" ))
    return background_error_matrix


def create_background_bias_vectors_at_each_grid_for_each_ens_member(
        background_data_dict, perterbation_parameters):
    # get the background ensemble data
    # (this step is used to prepare the inputs to calculate standard deviation for each grid/field )
    #      we create the ensembles for each background grid/variables using the 
    #      perterbation parameters defined above
    # * the output, ensemble_diff_vector, shows the bias between (x) - (x_mean)
    #   over all ensemble members
    background_ensemble_data_dict = create_ensembles.calculate_ensembles(
            background_data_dict, perterbation_parameters)
    ensemble_diff_vector = []
    for data_type in background_ensemble_data_dict:
        data_shape = background_ensemble_data_dict['lat'].shape
        if data_type == 'lat' or  data_type == 'lon':
            continue
        for i in range(0, data_shape[0]):
            for j in range(0, data_shape[1]):
                cur_ensemble_value = background_ensemble_data_dict[data_type][:, i, j]
                ensemble_size = len(cur_ensemble_value)
                cur_ensemble_mean = numpy.mean(cur_ensemble_value)
                cur_ensemble_diff = []
                for ie in range(0, ensemble_size):
                    cur_ensemble_diff.append((cur_ensemble_value[ie] - cur_ensemble_mean))
                ensemble_diff_vector.append(cur_ensemble_diff)
    
    return ensemble_diff_vector

def create_background_errors(
                            background_data_dict,
                            length_background_data_vector, perterbation_parameters,
                            be_path):
    # 1. get the background ensemble data
    #      we create the ensembles for each background grid/variables using the 
    #      perterbation parameters and the current background data
    # in reality (e.g., a real 3dvar), we get this from historical model runs
    ensemble_diff_vector = create_background_bias_vectors_at_each_grid_for_each_ens_member(
        background_data_dict, perterbation_parameters)
    ensemble_size = perterbation_parameters['ensemble_size']
    
    # 2. calculate background errors
    background_error = numpy.empty((length_background_data_vector,
                                    length_background_data_vector))
    for i, cur_ens1 in enumerate(ensemble_diff_vector):
        for j, cur_ens2 in enumerate(ensemble_diff_vector):
            cur_ens_sum = 0
            
            # go through all members and get the sum of (x-x_bias)(y-y_bias)
            for ie in range(0, ensemble_size):
                cur_ens_sum = cur_ens_sum + cur_ens1[ie]*cur_ens2[ie]
            
            # average the sum of (x-x_bias)(y-y_bias) and get the variance
            cur_var = float(cur_ens_sum)/ensemble_size
            background_error[i, j] = cur_var
    background_error_matrix = numpy.matrix(background_error)
    
    # 3. save be
    pickle.dump(background_error_matrix, open(be_path, "wb" ))
    
    
    