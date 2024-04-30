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
from partx.sampling import uniform_sampling
import sys
import pathlib 
import pickle


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
iters_all = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

all_rewards = []
all_robust = []
for iteration_train in iters_all:

    Benchmark_name = f"Cartpole_3d_UR_numsamples_{num_samples}_iters_{num_iterations}_1"
    base_path = pathlib.Path()
    result_directory_1 = base_path.joinpath(Benchmark_name)
    result_directory_1.mkdir(exist_ok=True)

    result_directory_2 = result_directory_1.joinpath(f"Policy_iterations_{iteration_train}")
    result_directory_2.mkdir(exist_ok=True)
    cartpole_blackbox = CartpoleModel(iteration_train)
    rewards = []
    robust = []
    for j in range(1): 
        rng = np.random.default_rng(12345 + num_iterations)
        oracle_info = OracleCreator(None, 1, 1)
        samples = uniform_sampling(num_samples, static, static.shape[0], oracle_info, rng)
        robs = []
        traces = []
        for sample_iter, sample in enumerate(samples):
            if sample_iter%500 == 0:
                print(f"Iteration {j}: {sample_iter}/{num_samples} Completed")
            rob1, trace = generateRobustness(sample,  cartpole_blackbox, options, specification_rtamt, iteration_train)
            robs.append(rob1)
            traces.append(trace)
        with open(result_directory_2.joinpath(f"{Benchmark_name}_UR_rep_{j}"), "wb") as f:
            pickle.dump((samples, np.array(robs), traces), f)
        
        