import tensorflow as tf
import numpy as np
# import torch
# import gpytorch
# from skorch.probabilistic import ExactGPRegressor

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
            print (gene_id)
            return matrix

    return (matrix - np.min(matrix, axis=0)) \
        / (np.max(matrix, axis=0) - np.min(matrix, axis=0))

# class RbfModule(gpytorch.models.ExactGP):
#     def __init__(self, likelihood, noise_init=None):
#         
#         super().__init__(train_inputs=None, train_targets=None, likelihood=likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.RBFKernel()

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# def GPR(device, max_epochs = 1, lr = 0.00001 ):
          # gpr = ExactGPRegressor(
            # RbfModule,
            # optimizer=torch.optim.Adam,
            # lr = lr,
            # max_epochs = max_epochs,
            # device = device,
            # batch_size = -1,
            # )
        # return gpr
    

class Model_Utils():
    def __init__(
        self, 
        adata=None,
        #M_acc=None,
        #B=None,
        Ms=None,
        Mu=None,
        
        config=None,
        logger=None
    ):
        self.adata = adata
        self.Ms, self.Mu = Ms, Mu
        #self.M_acc = M_acc
        #self.B = B
        self.config = config
        self.logger = logger

    def init_vars(self):
        ngenes = self.Ms.shape[1]
        #nregions = self.M_acc.shape[1]
        self.log_beta = tf.Variable(tf.zeros((1, ngenes), dtype=tf.float32), name='log_beta')
        self.intercept = tf.Variable(tf.zeros((1, ngenes), dtype=tf.float32), name='intercept')

        self.t = tf.Variable(tf.ones((1, ngenes), dtype=tf.float32) * 0.5, name='t') #! mean of Gaussian
        self.log_a = tf.Variable(tf.zeros((1, ngenes), dtype=tf.float32), name='log_a') #! 1 / scaling of Gaussian
        self.offset = tf.Variable(tf.zeros((1, ngenes), dtype=tf.float32), name='offset')

        self.log_h = tf.Variable(log(tf.math.reduce_max(self.Ms, axis=0, keepdims=True)), name='log_h')

        init_gamma = self.adata.var['velocity_gamma'].values
        self.log_gamma = tf.Variable(
            log(tf.reshape(init_gamma, (1, self.adata.n_vars))), 
            name='log_gamma'
        )

        # for id in np.where(init_gamma <= 0)[0]:
        #     self.logger.info(f'name: {self.adata.var.index[id]}, gamma: {init_gamma[id]}')

        self.log_gamma = tf.Variable(
            tf.where(tf.math.is_finite(self.log_gamma), self.log_gamma, 0), 
            name='log_gamma'
        )

        if self.config['velocity_genes']['vgenes'] == 'offset':
            init_inter = self.adata.var['velocity_inter'].values
            self.intercept = tf.Variable(
                tf.reshape(init_inter, (1, self.adata.n_vars)), 
                name='intercept'
            )
        
        #self.region_weights = tf.Variable(tf.ones((nregions,ngenes) * 0.01, dtype=tf.float32), name='log_weights')
        #self.etta = tf.Variable(tf.ones((1, ngenes), dtype=tf.float32) * 0.5, name='log_etta')
       
    
    #def velo_gene_regions_binary_matrix(self, B):
        ########## This function is written just un case we wanted to take regions associated with velocity genes#########
         #self.B_velo_genes = B[:, self.velocity_genes]
         #kept_regions = B_velo_genes.sum(axis=1)!=0
         #self.adata_atac.var["velo_gene_region"] = kept_regions
         #B_velo_genes = B_velo_genes[kept_regions,:]
         
         #return B_velo_genes, kept_regions    

    #def cell_velo_regions_matrix(self, kept_regions):
         #M_velo_acc = self.M_acc[kept_regions,:]
         # return M_velo_acc

    #def velo_regions_matrices(self):
         #self.B_tensor = tf.convert_to_tensor(self.B)
         #self.M_velo_acc = self.M_acc

         #######If we ant velocity regions######
         ####self.B, kept_regions = self.velo_gene_regions_binary_matrix(B)
         ####self.M_velo_acc = self.cell_velo_regions_matrix(kept_regions)
    
         
         
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
    
    # def get_u_deri(self, args, t_cell):
    #     self.u_deri = (( - args[3] * 2 * (t_cell - args[4]) + exp(args[0])) * self.s_deri - args[3] * 2 *self.fit_s) / square(exp(args[1]))
    #     return self.u_deri
    
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

    def match_time(self, Ms, Mu, s_predict, u_predict, x):
        val = x[1, :] - x[0, :]
        cell_time = np.zeros((Ms.shape[0], Ms.shape[2]))

        self.index_list = np.squeeze(tf.where(self.idx))

        for index in range(Ms.shape[2]):
            if index in self.index_list:
                sobs = Ms[:, :, index:index + 1] # n * 1 * 1
                uobs = Mu[:, :, index:index + 1]
                spre = s_predict[:, :, index:index + 1] # 1 * 3000 * 1
                upre = u_predict[:, :, index:index + 1]

                u_r2 = square(uobs - upre) # n * 3000 * 1
                s_r2 = square(sobs - spre)
                euclidean = sqrt(u_r2 + s_r2)
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
    
    #def region_dynamics_matrix(self):
    #######order each region according to cell time (order the rows)#####
         # time_order = np.argsort(self.t_cell)
         # M_acc_ordered = self.M_velo_acc[time_order, :]
         # return M_acc_oredered
    

    #def smooth_acc_dynamics(self):
    
         #M_acc_oredered = self.region_dynamics_matrix()
        #######smooth the accessibility (after ordering) using gaussian process#####
         #M_acc_oredered_smoothed = np.empty_like(M_acc_oredered.numpy()) 
         #n_regions = M_acc_oredered.shape[1]
         #M_acc_oredered = torch.tensor(M_acc_oredered)
         
         #torch.manual_seed(0)
         #torch.cuda.manual_seed(0)
         #device = 'cuda' if torch.cuda.is_available() else 'cpu' 
         #gpr = GPR(device, max_epochs = 1, lr = 0.00001 )
         #time = self.t_cell

         #for i in np.arange(0,n_regions):
           #gpr.fit(time, M_acc_oredered[:,i])
           #M_acc_oredered_smoothed[:,i] = gpr.predict(time)
        
        #return M_acc_oredered_smoothed
    
    #def compute_alpha(self, args):
         #M_acc_oredered_smoothed = self.smooth_acc_dynamics()
         #M_acc_oredered_smoothed = tf.convert_to_tensor(M_acc_oredered_smoothed)
         #exp(args[7]) = np.multiply(self.B_tensor, exp(args[7]))
         #exp(args[7]) = tf.cast(exp(args[7]), tf.float32)
         #wr = np.matmul(M_acc_oredered_smoothed, exp(args[7])) 
         #alpha =  exp(args[8]) * wr
         #return alpha
         
    #def compute_u_deri_atac(self, args):
         #alpha = self.compute_alpha(args)
         #u_deri_atac = alpha - tf.multiply(exp(args[1]),self.Mu) 
         #return u_deri_atac

    def get_opt_args(self, iter, args):
        remain = iter % 400
        
        if self.config['fitting_option']['mode'] == 1:
            if iter < self.config['base_trainer']['epochs'] / 2:
                args_to_optimize = [args[2], args[3], args[4], args[5]] if remain < 200 else [args[0], args[1], args[6]]#, args[7], args[8]]

            else:
                args_to_optimize = [args[0], args[1], args[2], args[3], args[4], args[5], args[6]]#, args[7], args[8]]
        
        if self.config['fitting_option']['mode'] == 2:
            if iter < self.config['base_trainer']['epochs'] / 2:
                args_to_optimize = [args[3], args[5]] if remain < 200 else [args[0], args[1]]#, args[7], args[8]]
                    
            else:
                args_to_optimize = [args[0], args[1], args[3], args[5]]#, args[7], args[8]]
        
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