import numpy as np
from numpy.typing import NDArray
from staliro.core import Interval
from staliro.core.model import BasicResult, Model, ModelInputs, ModelResult, Trace
import gymnasium as gym
from cartpoleenv import CartPoleEnv
from stable_baselines3 import PPO
import cv2
from matplotlib.pyplot import cm
import imageio
from staliro.options import Options, SignalOptions
from staliro.staliro import staliro, simulate_model
from staliro.specifications import TLTK, RTAMTDense

import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env

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
    images = []
    for count in range(max_steps_per_episode):
        # if render:
        #     img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
        #     cv2.imshow("test", img)
        #     cv2.waitKey(50)
        pi, _ = model.predict(state, deterministic=True)
        
        nstate, reward, done, info, _ = env.step(pi)
        states.append([np.abs(state[0]), np.abs(state[2]), np.abs(state[1]*(init_env_vars[0]+1)), count])
        
        times.append(count)
        state = nstate
        if render:
            img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            images.append(img)
        if done:
            break        
    if render:
        imageio.mimsave("7d_ex2.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
    return times, np.vstack(states)



class CartpoleModel(Model[CartPoleDataT, None]):

    def __init__(self, itera) -> None:
        print(f"Using PPO_CP_{itera}")
        self.model = PPO.load(f"logs_CartpoleEnv/rl_model_{itera}_steps")
        
        # self.envs = gym.make("CartPole-v1", render_mode="rgb_array")

    def simulate(self, inputs: ModelInputs, intrvl: Interval) -> CartPoleResultT:
        init_state = inputs.static
        self.envs = CartPoleEnv(init_state[4], init_state[5], init_state[6], render_mode="rgb_array")
        init_env_vars = init_state[4:]
        check_env(self.envs, warn=True)
        # print(init_state)
        times, traj = run_episode(init_state[:4], init_env_vars, self.envs, self.model, render=True)
        # print(traj.shape)
        trace = Trace(times, traj)

        return BasicResult(trace)



# static = np.array([[-2,2], [-0.05,0.05], [-0.2,0.2], [-0.05, 0.05], [0.05, 0.15], [0.4, 0.6], [0,10]])


# options = Options(runs=1, iterations=1, interval=(0, 400), signals=[], static_parameters=static)



# phi = "F(G(pos <= 1)) and F(G(angle <= 0.157)) and F(G(abs_mom <= 1))"
# # phi = "G[0,10000]((pos <= 0.1) and (angle <= 0.20944))"
# specification_rtamt = RTAMTDense(phi, {"pos":0, "angle": 1, "abs_mom": 2, "count":3})


# def generateRobustness(sample, inModel, options: Options, specification, itera):
    
#     result = simulate_model(inModel, options, sample)
#     # print(result.trace.times)
#     # res = np.vstack(result.trace.states)
#     # plt.plot(result.trace.times, res[:,0], "g-", label = "Abs Pos")
#     # plt.plot(result.trace.times, res[:,1], "k-", label = "Abs Ang")
#     # plt.plot(result.trace.times, res[:,2], "r-", label = "Abs Mom")
#     # # plt.hlines(1, 0, 500, color = "g", label="pos thresh")
#     # # plt.hlines(0.15708, 0, 500, color = "k", label="ang thresh")
#     # # plt.hlines(1, 0, 500, color = "r", label="mom thresh")
#     # plt.legend()
#     # plt.savefig(f"new_Modified_{itera}.png")
#     # plt.close()
#     return specification.evaluate(result.trace.states, result.trace.times)




# # import pickle

# # with open("Cartpole_PartxRes_best_f_points.pkl", "rb") as f:
# #     data = pickle.load(f)


# # ff_points = []

# # for itera, point in enumerate(data):
# #     local_ff = []
# #     for itera in [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]:
# #         cartpole_blackbox = CartpoleModel(itera)
# #         rob1 = generateRobustness(point,  cartpole_blackbox, options, specification_rtamt, itera)
# #         local_ff.append(rob1)

# #         print(f"Rob. Sample for policy {itera} = {rob1}")
# #     ff_points.append(local_ff)

# # ff_points = np.array(ff_points)

# # fig = plt.figure()
# # ax = fig.add_subplot(111)

# # color = iter(cm.rainbow(np.linspace(0, 1, ff_points.shape[0])))

# # p = [i*10000 for i in range(1,9)]
# # for itera, ff in enumerate(ff_points):
# #     c = next(color)
# #     ax.plot(p ,ff, "-", color = c, label = f"Policy @ {(itera+1)*10}k")
# #     ax.plot(p[itera] ,ff[itera], ".", color = c,ms=10)
# # ax.hlines(0, 0, 80000, linestyles="dashed", colors="k")

# # plt.legend()
# # plt.tight_layout()
# # plt.show()
# # # plt.savefig("8d_example.pdf")

# # # plt.savefig("3d_example.png")

# sample1 =  [1.2432317 ,  0.02121187, -0.14460155,  0.01294964,  0.09345286,
#                0.6,  0.73098185]
# cartpole_blackbox = CartpoleModel(80000)
# rob1 = generateRobustness(sample1,  cartpole_blackbox, options, specification_rtamt, 100000)
# print(rob1)
# # # import sys

# # # itera = int(sys.argv[1])

# # for itera in [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]:
# #     cartpole_blackbox = CartpoleModel(itera)
# #     rob1 = generateRobustness(sample1,  cartpole_blackbox, options, specification_rtamt, itera)


# #     print(f"Rob. Sample for policy {itera} = {rob1}")

