
from bbmodel_3d import CartpoleModel
import numpy as np

from partx.partxInterface.staliroIntegration import PartX
from partx.bayesianOptimization.internalBO import InternalBO
# from partx.gprInterface.internalGPR import InternalGPR
from External_gpr import ExternalGPR as InternalGPR

from staliro.staliro import staliro
from staliro.options import Options

from staliro.options import Options, SignalOptions
from staliro.staliro import staliro, simulate_model
from staliro.specifications import TLTK, RTAMTDense

oracle_func = None

# Define Signals and Specification
class Benchmark_Cartpole():
    def __init__(self, itera) -> None:
        
        benchmark = f"Cartpole_model_v1_PPO_CP_{itera}_gpytorch_1"
        results_folder = "Cartpole_PartxRes"
        
        # phi = "F(G((pos <= 0.2) and (angle <= 0.03)))"
        # self.specification_rtamt = RTAMTDense(phi, {"pos":0, "angle": 1})

        phi = "F(G(pos <= 1)) and F(G(angle <= 0.157)) and F(G(abs_mom <= 1))"
        self.specification_rtamt = RTAMTDense(phi, {"pos":0, "angle": 1, "abs_mom": 2})

        self.results_folder = results_folder
        
        static = np.array([[-2,2], [-0.05,0.05], [-0.2,0.2], [-0.05, 0.05], [0.05, 0.15], [0.4, 0.6], [0,10]])

        self.MAX_BUDGET = 2000
        self.NUMBER_OF_MACRO_REPLICATIONS = 10
        self.model = CartpoleModel(itera)

        self.optimizer = PartX(
            BENCHMARK_NAME=f"{benchmark}_budget_{self.MAX_BUDGET}_{self.NUMBER_OF_MACRO_REPLICATIONS}_reps",
            oracle_function = oracle_func,
            num_macro_reps = self.NUMBER_OF_MACRO_REPLICATIONS,
            init_budget = 50,
            bo_budget = 15,
            cs_budget = 100,
            n_tries_randomsampling = 1,
            n_tries_BO = 1,
            alpha=0.05,
            R = 20,
            M = 500,
            delta=0.001,
            fv_quantiles_for_gp=[0.5,0.05,0.01],
            branching_factor = 2,
            uniform_partitioning = True,
            seed = 12345,
            gpr_model = InternalGPR(),
            bo_model = InternalBO(),
            init_sampling_type = "lhs_sampling",
            cs_sampling_type = "lhs_sampling",
            q_estim_sampling = "lhs_sampling",
            mc_integral_sampling_type = "uniform_sampling",
            results_sampling_type = "uniform_sampling",
            results_at_confidence = 0.95,
            results_folder_name = results_folder,
            num_cores = 1,
        )

        

        self.options = Options(runs=1, iterations=self.MAX_BUDGET, interval=(0, 400), signals=[], static_parameters=static)

    def run(self):
        result = staliro(self.model, self.specification_rtamt, self.optimizer, self.options)

import sys
itera = int(sys.argv[1])
opt = Benchmark_Cartpole(itera)
opt.run()