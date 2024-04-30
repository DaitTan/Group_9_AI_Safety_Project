import numpy as np
from numpy.typing import NDArray
from staliro.core import Interval
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult, Trace
import gymnasium as gym
from cartpoleenv import CartPoleEnv
from stable_baselines3 import PPO
import cv2
import imageio

from staliro.options import Options, SignalOptions
from staliro.staliro import staliro, simulate_model
from staliro.specifications import TLTK, RTAMTDense

from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from partx.utils import OracleCreator
import csv
import os
from partx.coreAlgorithm import PartXOptions, run_single_replication
from External_gpr import ExternalGPR
import time
import sys
import pathlib 
import pickle
import matplotlib as mpl



CartPoleDataT = NDArray[np.float_]
CartPoleResultT = ModelResult[CartPoleDataT, None]

def run_episode(init_state, init_env_vars, env, model, max_steps_per_episode = 400, render = False):    
    RANDOM_SEED = 0
    
    np.random.seed(RANDOM_SEED*10)
    
    env.action_space.seed(RANDOM_SEED*100)
    env.reset(seed=RANDOM_SEED*1000)

    states = []
    state = init_state
    # num_actions = env.action_space.n
    times = []
    # if render:
    images = []
    
    for count in range(max_steps_per_episode):
        if render:
            img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("test", img)
            cv2.waitKey(20)
        pi, _ = model.predict(state, deterministic=True)
        
        nstate, reward, done, info, _ = env.step(pi)
        states.append([np.abs(state[0]), np.abs(state[2]), np.abs(state[1]*(init_env_vars[0]+1)), count, reward])
        
        times.append(count)
        state = nstate

        if render:
            img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            images.append(img)
        if done:
            break        
    
    # if render:
    #     imageio.mimsave("lander_a2c.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
    return times, np.vstack(states)



class CartpoleModel(Model[CartPoleDataT, None]):

    def __init__(self, itera) -> None:
        print(f"Using PPO_CP_{itera}")
        self.model = PPO.load(f"logs_CartpoleEnv/rl_model_{itera}_steps")
        
        # self.envs = gym.make("CartPole-v1", render_mode="rgb_array")

    def simulate(self, inputs: ModelInputs, intrvl: Interval) -> CartPoleResultT:
        init_state = inputs.static
        self.envs = CartPoleEnv(init_state[0], init_state[1], init_state[2], render_mode="rgb_array")
        init_env_vars = init_state
        check_env(self.envs, warn=True)
        # print(init_state)
        times, traj = run_episode([1e-3, 1e-3, 1e-3, 1e-3], init_env_vars, self.envs, self.model, render=False)
        # print(traj.shape)
        trace = Trace(times, traj)

        return BasicResult(trace)



static = np.array([[0.05, 0.15], [0.4, 0.6], [0,10]])


options = Options(runs=1, iterations=1, interval=(0, 400), signals=[], static_parameters=static)



phi = "F(G(pos <= 1)) and F(G(angle <= 0.157)) and F(G(abs_mom <= 1))"
# phi = "G[0,10000]((pos <= 0.1) and (angle <= 0.20944))"
specification_rtamt = RTAMTDense(phi, {"pos":0, "angle": 1, "abs_mom": 2, "count":3})


def generateRobustness(sample, inModel, options: Options, specification, itera):
    
    result = simulate_model(inModel, options, sample)
    return specification.evaluate(result.trace.states, result.trace.times), result




num_samples = 2000
num_iterations = 10
iters_all = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
BENCHMARK_NAMES = [
    "Cartpole_model_v1_PPO_CP_10000_gpytorch_budget_2000_10_reps",
    "Cartpole_model_v1_PPO_CP_20000_gpytorch_budget_2000_10_reps",
    "Cartpole_model_v1_PPO_CP_30000_gpytorch_budget_2000_10_reps",
    "Cartpole_model_v1_PPO_CP_40000_gpytorch_budget_2000_10_reps",
    "Cartpole_model_v1_PPO_CP_50000_gpytorch_budget_2000_10_reps",
    "Cartpole_model_v1_PPO_CP_60000_gpytorch_budget_2000_10_reps",
    "Cartpole_model_v1_PPO_CP_70000_gpytorch_1_budget_2000_10_reps",
    "Cartpole_model_v1_PPO_CP_80000_gpytorch_budget_2000_10_reps",
    ]
folder_name = "Cartpole_PartxRes"
ITER_NAMES = [int(x)/1000 for x in iters_all]


all_rewards = []
all_robust = []
for iteration_train in iters_all:

    Benchmark_name = f"Cartpole_7d_UR_numsamples_{num_samples}_iters_{num_iterations}_1"
    base_path = pathlib.Path()
    result_directory_1 = base_path.joinpath(Benchmark_name)
    result_directory_1.mkdir(exist_ok=True)

    result_directory_2 = result_directory_1.joinpath(f"Policy_iterations_{iteration_train}")
    result_directory_2.mkdir(exist_ok=True)

    rewards = []
    robust = []
    for j in range(num_iterations): 
        
        with open(result_directory_2.joinpath(f"{Benchmark_name}_UR_rep_{j}"), "rb") as f:
            s, r, traces = pickle.load(f)
        
        rew = []
        
        for i in range(s.shape[0]):
            rew.append(np.stack(traces[i].trace.states)[:,-1].sum())
        rewards.append(np.array(rew).mean())
        robust.append(r.mean())
    all_rewards.append(rewards[0])
    all_robust.append(robust[0])


mpl.rcParams["font.size"] = 9

plt.plot(iters_all, all_rewards, label = "Rewards")
plt.xlabel("Iterations")
plt.ylabel("Mean Rewards")
plt.savefig("Fig3A.png", bbox_inches='tight')
plt.close()

plt.plot(iters_all, all_robust, label = "Robustness")

plt.xlabel("Iterations")
plt.ylabel("Mean Robustness")
plt.savefig("Fig3B.png", bbox_inches='tight')

# plt.plot(iters_all, all_robust, label = "Rewards")

oracle_info = OracleCreator(None, 1, 1)
rng = np.random.default_rng(123456)





res_final_dict = {}

import sys



res_arrs = []
falsificationpts = []
# iters = int(sys.argv[1])
for iters in range(len(BENCHMARK_NAMES)):
    benchmark_name = BENCHMARK_NAMES[iters]


    options_path = os.path.join(f"{folder_name}", f"{benchmark_name}", f"{benchmark_name}_result_generating_files", f"{benchmark_name}_options.pkl")
    with open(options_path, "rb") as f:
        options:PartXOptions = pickle.load(f)
    options.gpr_model = ExternalGPR()
    rep = 10

    results_at_confidence = 0.95

    res_dict_path = os.path.join(f"{folder_name}", f"{benchmark_name}", f"{benchmark_name}_result_generating_files", f"{benchmark_name}_all_result.pkl") 

    with open(res_dict_path, "rb") as f:
        res = pickle.load(f)

    ind = np.argmin(np.stack(res["best_falsification_points"])[:,-1])
    falsificationpts.append(res["best_falsification_points"][ind][1])
    x = res['fv_stats_with_gp'].to_numpy()[1,:].tolist()
    res_arrs.append([iters, x[0], x[1], x[2], x[3]])

    today = time.strftime("%m/%d/%Y")
    file_date = today.replace("/","_")
    values = []
    with open(f"{folder_name}_results{ITER_NAMES[iters]}_3d.csv", 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in res.items():
            writer.writerow([key, value])
            values.append(value)
    print("Done")



mpl.rcParams.update({'figure.autolayout': True})
mpl.rcParams["font.size"] = 9

fig = plt.figure()

ax = fig.add_subplot(111)
res_arrs = np.array(res_arrs)

ax.plot((res_arrs[:,0]+1)*10000, res_arrs[:,1], "-", label="Mean Falsification Volume")
ax.fill_between((res_arrs[:,0]+1)*10000, res_arrs[:,3], res_arrs[:,4], alpha=0.5, label=r"95% confidence interval")
ax.set_xlabel("PPO Training Iterations")
ax.set_ylabel("Falsification Volume")
ax.legend()
# plt.tight_layout()
plt.savefig(f"Fig4B.png")


