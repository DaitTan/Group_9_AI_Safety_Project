
from partx.gprInterface import GaussianProcessRegressorStructure
from sklearn.preprocessing import StandardScaler
import math
import torch
import numpy as np
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from botorch import fit_gpytorch_model
import botorch

botorch.settings.debug()

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class ExternalGPR(GaussianProcessRegressorStructure):
    def __init__(self, random_state = 12345):
        self.scale = StandardScaler()
        self.scaley = StandardScaler()

        k_long_term_RBF = MaternKernel()
        k_long_term_RBF.lengthscale = 1

        k_long_term = ScaleKernel(k_long_term_RBF)
        k_long_term.outputscale = 1
        self.kernel = k_long_term_RBF
        print(self.kernel)

    def fit_gpr(self, X, Y):
        """Method to fit gpr Model

        Args:
            x_train: Samples from Training set.
            y_train: Evaluated values of samples from Trainig set.

        
        """


        X_scaled = self.scale.fit_transform(X)
        Y_scaled = self.scaley.fit_transform(np.array([Y]).T)[:,0]
        x_tensor = torch.from_numpy(X_scaled)
        y_tensor = torch.from_numpy(Y_scaled)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # self.likelihood.initialize(noise=1e-4)

        self.model = ExactGPModel(x_tensor, y_tensor, self.likelihood, self.kernel)

        # Find optimal model hyperparameters
        self.model.train()
        self.likelihood.train()

        
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood,self.model)

        fit_gpytorch_model(self.mll)

        # n_iter = 50
        # for i in range(n_iter):
        #     self.optimizer.zero_grad()
        #     output = self.model(x_tensor)
        #     loss = -self.mll(output, y_tensor).sum()
        #     loss.backward()
        #     print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
        #     self.optimizer.step()

        # Set into eval mode
        


    def predict_gpr(self, X):
        """Method to predict mean and std_dev from gpr model

        Args:
            x_train: Samples from Training set.
            

        Returns:
            mean
            std_dev
        """
        x_scaled = self.scale.transform(X)
        x_tensor = torch.from_numpy(x_scaled)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(x_tensor))
            mean = predictions.mean
            std = predictions.stddev
            # lower, upper = predictions.confidence_region()
        
        mean = mean.numpy()
        std = std.numpy()
        # print(mean.shape)
        # print(std.shape)
        # return mean, std
        return mean*np.sqrt(self.scaley.var_[0])+ self.scaley.mean_[0], std*(self.scaley.var_[0])**0.5

# def optimizer_lbfgs_b(obj_func, initial_theta):
#     with catch_warnings():
#         warnings.simplefilter("ignore")
#         params = fmin_l_bfgs_b(
#             obj_func, initial_theta, bounds=None, maxiter=30000, maxfun=1e10
#         )
#     return params[0], params[1]


# class InternalGPR(GaussianProcessRegressorStructure):
#     def __init__(self, random_state = 12345):
#         self.gpr_model = GaussianProcessRegressor(
#             # kernel=1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)), alpha=1e-6, normalize_y=True, n_restarts_optimizer=5, random_state = random_state
#             kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True, n_restarts_optimizer=5, random_state = random_state
#         )
#         print(Matern(nu=2.5))
#         self.scalex = StandardScaler()
#         self.scaley = StandardScaler()

#     def fit_gpr(self, X, Y):
#         """Method to fit gpr Model

#         Args:
#             x_train: Samples from Training set.
#             y_train: Evaluated values of samples from Trainig set.

        
#         """
#         X_scaled = self.scalex.fit_transform(X)
#         # Y_scaled = self.scaley.fit_transform(np.array([Y]).T)[:,0]
        
#         with catch_warnings():
#             warnings.simplefilter("ignore")
#             # self.gpr_model.fit(X_scaled, Y_scaled)
#             self.gpr_model.fit(X_scaled, Y)

#     def predict_gpr(self, X):
#         """Method to predict mean and std_dev from gpr model

#         Args:
#             x_train: Samples from Training set.
            

#         Returns:
#             mean
#             std_dev
#         """
#         x_scaled = self.scalex.transform(X)
        

#         with catch_warnings():
#             warnings.simplefilter("ignore")
#             yPred, predSigma = self.gpr_model.predict(x_scaled, return_std=True)

#         # return yPred*np.sqrt(self.scaley.var_[0])+ self.scaley.mean_[0], predSigma*(self.scaley.var_[0])**0.5
#         return yPred, predSigma

