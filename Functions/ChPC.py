import numpy as np
import os
from scipy.stats import norm
from tqdm import tqdm

class Unit:
    def __init__(self, size, batch):
        self.sz                = int(size)
        self.batch_sz          = int(batch[-1])
        self.prev_batch_sz     = int(batch[0])
        self.r                 = np.zeros((self.batch_sz, (self.sz * self.prev_batch_sz )))
        self.er                = np.zeros((self.batch_sz, (self.sz * self.prev_batch_sz )))
        self.weights           = None
        self.pre_units_indices = None
        # Initialize histories for this unit
        self.weight_history    = []
        self.activity_history  = []
        
    def reset_errors(self):
        self.er = np.zeros((self.batch_sz, (self.sz * self.prev_batch_sz)))
        
class Layer:
    def __init__(self, ltype, bacth_size, n_units, s_units, lr_r=None, lr_u=None, sigma_r=None, prior_r=None, f_r_type=None, prior_u=None, scale_u=None, norm_eps=1e-3, save=False):
        self.ltype    = ltype
        self.lr_r     = lr_r
        self.lr_u     = lr_u
        self.sigma_r  = sigma_r
        self.prior_u  = prior_u
        self.norm_eps = norm_eps
        self.scale_u  = scale_u
        self.save     = save  
        
        if ltype !='input':
            self.lr_rate  = self.lr_u['initial']
            self.prior_r  = prior_r
            
            if f_r_type == 'linear':
                self.f_r       = lambda x: x
                self.f_prime_r = lambda x: np.ones_like(x)  
            elif f_r_type == 'tanh':
                self.f_r       = lambda x: np.tanh(x)
                self.f_prime_r = lambda x: 1 - np.tanh(x) ** 2
            else:
                raise ValueError("Invalid f_r_type. Supported types are 'linear' and 'tanh'.")


            prior_type = prior_r['type']
            alpha      = prior_r['alpha']
            if prior_type == 'gaussian':
                self.p_r       = lambda r: alpha * np.sum(r**2)  # Gaussian prior: alpha * r^2
                self.p_prime_r = lambda r: 2 * alpha * r  # Derivative: 2 * alpha * r
            elif prior_type == 'kurtotic':
                self.p_r       = lambda r: alpha * np.sum(np.log(1 + r**2))  # Kurtotic prior: alpha * log(1 + r^2)
                self.p_prime_r = lambda r: 2 * alpha * r / (1 + r**2)  # Derivative: 2 * alpha * r / (1 + r^2)
            else:
                raise ValueError("Invalid prior_r type. Supported types are 'gaussian' and 'kurtotic'.")

            self.p_u       = lambda u: prior_u * np.sum(u**2)  # Gaussian prior: alpha * u^2
            self.p_prime_u = lambda u: 2 * prior_u * u  # Derivative: 2 * alpha * u
        
        self.units   = [Unit(size=s_units, batch=bacth_size) for _ in range(n_units)]

    def ff_connect(self, post_layer, connectivity_type, connectivity_matrix=None):
        pre_units  = len(self.units)
        post_units = len(post_layer.units)
        np.random.seed(1337)
        
        if connectivity_type == 'one-to-one':
            if pre_units != post_units:
                raise ValueError("One-to-one connectivity requires the same number of units in both layers.")
            for i in range(pre_units):
                tmp = np.random.rand(int(self.units[i].sz * self.units[i].prev_batch_sz), post_layer.units[i].sz)
                U   = tmp.astype(np.float32) * np.sqrt(2 / ((self.units[i].sz * self.units[i].prev_batch_sz) + post_layer.units[i].sz))
                post_layer.units[i].weights = [self.scale_u*U]
                post_layer.units[i].pre_units_indices = [i]
        
        elif connectivity_type == 'all-to-all':
            for j in range(post_units):
                post_layer.units[j].weights           = []
                for i in range(pre_units):
                    tmp = np.random.rand(int(self.units[i].sz * self.units[i].prev_batch_sz), post_layer.units[j].sz)
                    U   = tmp.astype(np.float32) * np.sqrt(2 / ((self.units[i].sz * self.units[i].prev_batch_sz)  + post_layer.units[j].sz))
                    post_layer.units[j].weights.append(self.scale_u*U)
                post_layer.units[j].pre_units_indices = list(range(pre_units))
                
        elif connectivity_type == 'defined':
            if connectivity_matrix is None:
                raise ValueError("A connectivity matrix must be provided for 'defined' connectivity type.")
            if connectivity_matrix.shape != (pre_units, post_units):
                raise ValueError("Connectivity matrix must have the shape (pre_units, post_units).")

            for j in range(post_units):
                post_layer.units[j].weights           = []
                post_layer.units[j].pre_units_indices = []
                for i in range(pre_units):
                    if connectivity_matrix[i, j] == 1:
                        tmp                     = np.random.randn(int(self.units[i].sz * self.units[i].prev_batch_sz), post_layer.units[j].sz)
                        U                       = tmp.astype(np.float32) * np.sqrt(2 / ((self.units[i].sz * self.units[i].prev_batch_sz) + post_layer.units[j].sz))
                        post_layer.units[j].weights.append(self.scale_u*U)
                        post_layer.units[j].pre_units_indices.append(i)
        else:
            raise ValueError("Invalid connectivity type. Supported types are 'one-to-one' and 'all-to-all'.")
                          
    def init_activity(self, input_data=None):
        if self.ltype == 'input':
            if input_data is None:
                raise ValueError("Input data must be provided for the input layer.")
            
            num_units = input_data.shape[1]
            if len(self.units) != num_units:
                raise ValueError(f"Input layer should have {num_units} units for the provided 3D input data.")
            
            batch_size = input_data.shape[0]
            if self.units[0].batch_sz != batch_size:
                raise ValueError(f"Input layer batch size should be {batch_size}.")
            
            inp_size = input_data.shape[-1]
            if self.units[0].sz != inp_size:
                raise ValueError(f"Input layer size should be {inp_size}.")
            
            for i, unit in enumerate(self.units):
                unit.r = input_data[:, i, :]
                    
        else:
            for unit in self.units:
                new_activity   = np.zeros((unit.batch_sz, (unit.sz * unit.prev_batch_sz)))
                
                for i, pre_index in enumerate(unit.pre_units_indices):
                    new_activity += np.dot(input_data[pre_index], unit.weights[i]).reshape(unit.batch_sz, -1)
                
                unit.r = self.f_r(new_activity)
                               
    def calculate_errors(self, next_layer):
        for unit in next_layer.units:
            if unit.pre_units_indices is not None:
                for i, pre_index in enumerate(unit.pre_units_indices):
                    self.units[pre_index].reset_errors()
                    for j in range(self.units[pre_index].batch_sz):
                        self.units[pre_index].er[j, :] += self.units[pre_index].r[j, :] - next_layer.f_r(np.dot(unit.r[:, j:j+unit.sz], unit.weights[i].T)).flatten()

    def update_activity(self, lower_layer=None, next_layer=None):
        if self.ltype == 'input':
            return
        dr_arr = []
        for j, unit in enumerate(self.units):
            dr = np.zeros((unit.batch_sz, (unit.sz * unit.prev_batch_sz )))

            for i, pre_index in enumerate(unit.pre_units_indices):
                if lower_layer is not None:
                    pre_unit       = lower_layer.units[pre_index]
                    weighted_input = np.zeros((lower_layer.units[0].batch_sz, lower_layer.units[0].sz * lower_layer.units[0].prev_batch_sz))
                    for k in range(lower_layer.units[0].batch_sz):
                        weighted_input[k, :] = np.dot(unit.r[:, k:k+self.units[0].sz], unit.weights[i].T).flatten()
                    dr            += self.sigma_r * np.dot(np.multiply(self.f_prime_r(weighted_input), pre_unit.er), unit.weights[i]).reshape(unit.batch_sz, -1)

            if next_layer is not None:
                dr -= next_layer.sigma_r * unit.er

            dr -= 0.5*self.p_prime_r(unit.r)

            # Update the unit's activity
            unit.r += self.lr_r * dr
            dr_arr.append(dr)
            
            if self.save:
                unit.activity_history.append(unit.r.copy())

        return dr_arr
    
    def update_weight(self, lower_layer=None, iter=0):
        if self.ltype == 'input':
            return

        for j, unit in enumerate(self.units):
            for i, pre_index in enumerate(unit.pre_units_indices):
                du = np.zeros_like(unit.weights[i])
                if lower_layer is not None:
                    pre_unit       = lower_layer.units[pre_index]
                    weighted_input = np.zeros((lower_layer.units[0].batch_sz, lower_layer.units[0].sz * lower_layer.units[0].prev_batch_sz))
                    for k in range(lower_layer.units[0].batch_sz):
                        weighted_input[k, :] = np.dot(unit.r[:, k:k+self.units[0].sz], unit.weights[i].T).flatten()
                        
                    du            += self.sigma_r * np.dot(unit.r.reshape(unit.prev_batch_sz,-1).T, np.multiply(self.f_prime_r(weighted_input), pre_unit.er)).T

                    du -= 0.5*self.p_prime_u(unit.weights[i])
                    
                    if iter % self.lr_u['round'] == (self.lr_u['round']-1):
                        self.lr_rate /= self.lr_u['scale']
                    # Update the unit's activity
                    tmp_w           = unit.weights[i] + self.lr_rate * du
                    unit.weights[i] =  tmp_w
                    
                    if self.save:
                        unit.weight_history.append(unit.weights[i].copy())
                    


            
class hPC:
    def __init__(self, input_data=None):
        self.input_data = input_data
        self.n_input    = np.shape(input_data)[0]
        self.layers     = []
               
    def add_layer(self, ltype, bacth_size, n_units, s_units, lr_r=None, lr_u=None, sigma_r=None, prior_r=None, f_r_type=None, prior_u=None, scale_u=None, norm_eps=1e-3, save=False):
        new_layer = Layer(ltype, bacth_size, n_units, s_units, lr_r, lr_u, sigma_r, prior_r,  f_r_type, prior_u, scale_u, norm_eps, save)
        self.layers.append(new_layer)
        return new_layer
        
    def init_network(self, input):
        self.layers[0].init_activity(input)
        for i in range(1, len(self.layers)):
            self.layers[i].init_activity([unit.r for unit in self.layers[i-1].units])

    def prediction_errors(self):
        for i in reversed(range(len(self.layers) - 1)):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            current_layer.calculate_errors(next_layer)
            
    def update_activities(self):
        norm_dr = []
        for i in range(1, len(self.layers)):
            lower_layer   = self.layers[i - 1]
            current_layer = self.layers[i]
            next_layer    = self.layers[i + 1] if i + 1 < len(self.layers) else None
            dr            = current_layer.update_activity(lower_layer=lower_layer, next_layer=next_layer)
            norm_dr.append(np.linalg.norm(np.array(dr).flatten(), ord=2)<current_layer.norm_eps)
        return norm_dr
    
    def update_weights(self, i):
        for i in range(1, len(self.layers)):
            lower_layer   = self.layers[i - 1]
            current_layer = self.layers[i]
            next_layer    = self.layers[i + 1] if i + 1 < len(self.layers) else None
            current_layer.update_weight(lower_layer=lower_layer, iter=i)
                    
    def training(self, data, scale=40, iteration=1000):
        n_data    = np.shape(data)[0]
        error_arr = np.zeros((4, n_data))
        for i in tqdm(range(n_data)):
            instance = data[i, :, :]*scale
            self.init_network(instance)
            for j in range(iteration):
                self.prediction_errors()
                norm_dr = self.update_activities()
                
                if sum(norm_dr) == (len(self.layers)-1):
                    self.update_weights(i)
                    break

                if j >= iteration-2: 
                    print("Error at patch:", i)
                    break
                
            error_arr[:, i] = self.calculate_total_error()
            
            if i % 1000 == 999:  
                print("iter: "+str(i+1)+"/"+str(n_data)+", Moving error:", np.mean(error_arr[0, i-999:i]))
        return error_arr

    def calculate_total_error(self):
        total_recon_error = 0
        total_sparsity_r  = 0
        total_sparsity_U  = 0
        
        for i in range(1, len(self.layers)):
            prev_layer       = self.layers[i-1]
            current_layer    = self.layers[i]
            for unit in prev_layer.units:
                total_recon_error   += current_layer.sigma_r * np.sum(unit.er**2)
            for unit in current_layer.units:
                total_sparsity_r    += current_layer.prior_r['alpha'] * np.sum(unit.r**2)
                for j, _ in enumerate(unit.pre_units_indices):
                    total_sparsity_U += current_layer.prior_u*np.sum(unit.weights[j]**2)
        
        total_error = total_recon_error + total_sparsity_r + total_sparsity_U
        return total_error, total_recon_error, total_sparsity_r, total_sparsity_U
