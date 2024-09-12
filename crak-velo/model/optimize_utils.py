import tensorflow as tf
import numpy as np


exp = tf.math.exp
pow = tf.math.pow
square = tf.math.square
sum = tf.math.reduce_sum
abs = tf.math.abs
mean = tf.math.reduce_mean
log = tf.math.log
sqrt = tf.math.sqrt

def col_minmax(matrix, gene_id=None):
    if gene_id != None:
        if (np.max(matrix, axis=0) == np.min(matrix, axis=0)):
            
            return matrix

    return (matrix - np.min(matrix, axis=0)) \
        / (np.max(matrix, axis=0) - np.min(matrix, axis=0))


class Model_Utils():
    def __init__(
        self, 
        adata=None,
        M_acc=None,
        B=None,
        Ms=None,
        Mu=None,
        
        config=None,
        logger=None
    ):
        self.adata = adata
        self.Ms, self.Mu = Ms, Mu
        self.M_acc = tf.constant(M_acc, dtype=tf.float32)
        self.B = tf.Variable(B, dtype=tf.float32, trainable=False)
        self.config = config
        self.logger = logger

    def init_vars(self):
        ngenes = self.Ms.shape[1]
        nregions = self.M_acc.shape[1]

        self.log_beta = tf.Variable(tf.zeros((1, ngenes), dtype=tf.float32), name='log_beta')
        self.intercept = tf.Variable(tf.zeros((1, ngenes), dtype=tf.float32), name='intercept')

        self.t = tf.Variable(tf.ones((1, ngenes), dtype=tf.float32) * 0.5, name='t') #! mean of Gaussian
        self.log_a = tf.Variable(tf.zeros((1, ngenes), dtype=tf.float32), name='log_a') #! 1 / scaling of Gaussian
        self.offset = tf.Variable(tf.zeros((1, ngenes), dtype=tf.float32), name='offset')

        self.log_h = tf.Variable(log(tf.math.reduce_max(self.Ms, axis=0, keepdims=True)), name='log_h')

        init_gamma = self.adata.var['velocity_gamma'].values
        self.log_gamma = tf.Variable(log(tf.reshape(init_gamma, (1, self.adata.n_vars))), name='log_gamma')
        self.log_gamma = tf.Variable(tf.where(tf.math.is_finite(self.log_gamma), self.log_gamma, 0), name='log_gamma')

        if self.config['velocity_genes']['vgenes'] == 'offset':
            init_inter = self.adata.var['velocity_inter'].values
            self.intercept = tf.Variable(tf.reshape(init_inter, (1, self.adata.n_vars)), name='intercept')
        
        self.log_region_weights = tf.Variable(tf.zeros((nregions, ngenes), dtype=tf.float32) , name='log_weights')
        self.log_region_weights.assign(tf.multiply(self.B, self.log_region_weights))

        self.B_genes_nr = tf.Variable(tf.cast(sum(self.B, axis=0, keepdims=True) != 0, tf.float32), trainable=False)
        
        self.indices = tf.cast(tf.where(self.B_genes_nr == 0, 0, 1), dtype=tf.float32)

        self.log_etta = tf.Variable(tf.zeros((1, ngenes), dtype=tf.float32) , name='log_etta')
        self.log_etta.assign(tf.multiply(self.B_genes_nr, self.log_etta))

    def velo_regions_matrices(self):
         self.B_tensor = tf.convert_to_tensor(self.B)
         self.M_velo_acc = self.M_acc

      
    
    def init_weights(self):
        nonzero_s, nonzero_u = self.Ms > 0, self.Mu > 0
        weights = np.array(nonzero_s & nonzero_u, dtype=bool)

        self.weights = tf.cast(weights, dtype=tf.float32)
        self.nobs = np.sum(weights, axis=0)

    def get_fit_s(self, args, t_cell):
        self.fit_s = exp(args[5]) * exp(-exp(args[3]) * square(t_cell - args[4])) + args[2]
        return self.fit_s
    
    def get_s_deri(self, args, t_cell):
        self.s_deri = (self.fit_s - args[2]) * (-exp(args[3]) * 2 * (t_cell - args[4]))
        return self.s_deri

    def get_fit_u(self, args):
        self.fit_u = (self.s_deri + exp(args[0]) * self.fit_s) / exp(args[1]) + args[6]
        return self.fit_u
    
    def get_u_deri(self, args, t_cell):
        self.u_deri = (( - args[3] * 2 * (t_cell - args[4]) + exp(args[0])) * self.s_deri - args[3] * 2 *self.fit_s) / square(exp(args[1]))
        self.u_deri *= self.indices
        return self.u_deri
    
    def get_alpha_unit(self, args, t_cell):
        u_dot = self.get_u_deri(args, t_cell)
        u = self.get_fit_u(args)
        alpha_unit = u_dot + exp(args[1])*u
        return alpha_unit
    
    def get_s_u(self, args, t_cell):
        s = self.get_fit_s(args, t_cell)
        s_deri = self.get_s_deri(args, t_cell)
        u = self.get_fit_u(args)
        
        if self.config['fitting_option']['assign_pos_u']:
            s = tf.clip_by_value(s, 0, 1000)
            u = tf.clip_by_value(u, 0, 1000)

        return s, u

    def max_density(self, dis):
        if self.config['fitting_option']['density'] == 'SVD':
            s, u, v = tf.linalg.svd(dis)
            s = s[0:50]
            u = u[:, :tf.size(s)]
            v = v[:tf.size(s), :tf.size(s)]
            dis_approx = tf.matmul(u, tf.matmul(tf.linalg.diag(s), v, adjoint_b=True))
            return tf.cast(mean(dis_approx, axis=1), tf.float32)

        if self.config['fitting_option']['density'] == 'Raw':
            return tf.cast(mean(dis, axis=1), tf.float32)

    def match_time(self, Ms, Mu, s_predict, u_predict, u_deri_func, u_deri_atac, x):
        val = x[1, :] - x[0, :]
        cell_time = np.zeros((Ms.shape[0], Ms.shape[2]))

        self.index_list = np.squeeze(tf.where(self.idx))

        for index in range(Ms.shape[2]):
            if index in self.index_list:
                sobs = Ms[:, :, index:index + 1] # n * 1 * 1
                uobs = Mu[:, :, index:index + 1]
                u_deri_atac_ = u_deri_atac[:, :, index:index + 1]

                spre = s_predict[:, :, index:index + 1] # 1 * 3000 * 1
                upre = u_predict[:, :, index:index + 1]
                u_deri_func_ = u_deri_func[:, :, index:index + 1]

                u_r2 = square(uobs - upre) # n * 3000 * 1
                s_r2 = square(sobs - spre)
                u_deri_r2 = square(u_deri_atac_ - u_deri_func_ )
                euclidean = sqrt(u_r2 + s_r2 + u_deri_r2)
                
                assign_loc = tf.math.argmin(euclidean, axis=1).numpy() # n * 1
                
                if self.config['fitting_option']['reorder_cell'] == 'Soft_Reorder':
                    cell_time[:, index:index + 1] = \
                        col_minmax(self.reorder(assign_loc), self.adata.var.index[index])
                if self.config['fitting_option']['reorder_cell'] == 'Soft':
                    cell_time[:, index:index + 1] = \
                        col_minmax(assign_loc, self.adata.var.index[index])
                if self.config['fitting_option']['reorder_cell'] == 'Hard':
                    cell_time[:, index:index + 1] = \
                        x[0, index:index + 1] + val[index:index + 1] * assign_loc

        if self.config['fitting_option']['aggregrate_t']:
            return self.max_density(cell_time[:, self.index_list]) #! sampling?
        else:
            return cell_time

    def reorder(self, loc):
        new_loc = np.zeros_like(loc)

        for gid in range(loc.shape[1]):
            sorted_vals = np.sort(loc[:, gid])
            ranks = np.searchsorted(sorted_vals, loc[:, gid])
            new_loc[:, gid] = ranks

        return new_loc

    def init_time(self, boundary, shape=None):
        x = tf.linspace(boundary[0], boundary[1], shape[0])
        x = tf.reshape(x, (-1, 1))
        x = tf.broadcast_to(x, shape)
        return tf.cast(x, dtype=tf.float32)
    
    def region_dynamics_matrix(self, latent_time_):
        ######order each region according to cell time (order the rows)#####
        
        
        time_order = tf.argsort(latent_time_)
        time_order = tf.cast(time_order, dtype=tf.int32)
        time_order = tf.expand_dims(time_order, axis=1)
        M_acc_ordered = tf.gather_nd(self.M_acc,
                   indices= time_order)
        
        return M_acc_ordered
    
    def compute_alpha(self, args, latent_time_):
        
        M_acc_oredered_smoothed = self. region_dynamics_matrix(latent_time_)
        M_acc_oredered_smoothed = tf.convert_to_tensor(M_acc_oredered_smoothed)

       
        log_region_weights = args[7]
        log_etta = args[8]
        
        x_exp = self.B * exp(log_region_weights)
        x_etta_exp = self.B_genes_nr * exp(log_etta)
        
        wr = tf.matmul(M_acc_oredered_smoothed, x_exp) # ncells by n genes
        alpha = x_etta_exp * wr
        
        return alpha
         
    def compute_u_deri_atac(self, args, t_cell):
         alpha = self.compute_alpha(args, t_cell)
         u_deri_atac = alpha - exp(args[1]) * self.Mu
         return u_deri_atac
    
    

    def get_opt_args(self, iter, args):
        remain = iter % 400
        
        if self.config['fitting_option']['mode'] == 1:
            
            if iter < self.config['base_trainer']['epochs'] / 2:
                args_to_optimize = [args[2], args[3], args[4], args[5], args[7], args[8]] if remain < 200 else [args[0], args[1], args[6],args[7], args[8]]
             
            else:
                args_to_optimize = [args[0], args[1], args[2], args[3], args[4], args[5], args[6],args[7], args[8]]

        
        if self.config['fitting_option']['mode'] == 2:
            
            if iter < self.config['base_trainer']['epochs'] / 2:
                args_to_optimize = [args[3], args[5], args[7], args[8]] if remain < 200 else [args[0], args[1], args[7], args[8]]
             
            else:
                args_to_optimize = [args[0], args[1], args[3], args[5], args[7], args[8]]

            
        
        return args_to_optimize
    
    def get_stop_cond(self, iter, pre, obj):
        stop_s = tf.zeros(self.Ms.shape[1], tf.bool)
        stop_u = tf.zeros(self.Ms.shape[1], tf.bool)
        remain = iter % 400

        if remain > 1 and remain < 200:
            stop_s = abs(pre - obj) <= abs(pre) * 1e-4
        if remain > 201 and remain < 400:
            stop_u = abs(pre - obj) <= abs(pre) * 1e-4

        return tf.math.logical_and(stop_s, stop_u)

    def get_interim_t(self, t_cell, idx):
        if self.config['fitting_option']['aggregrate_t']:
            return t_cell
        
        else:
            #? modify this later for independent mode
            t_interim = np.zeros(t_cell.shape)

            for i in range(t_cell.shape[1]):
                if idx[i]:
                    temp = np.reshape(t_cell[:, i], (-1, 1))
                    t_interim[:, i] = \
                        np.squeeze(col_minmax(temp, self.adata.var.index[i]))

            t_interim = self.max_density(t_interim)
            t_interim = tf.reshape(t_interim, (-1, 1))
            t_interim = tf.broadcast_to(t_interim, self.adata.shape)
            return t_interim