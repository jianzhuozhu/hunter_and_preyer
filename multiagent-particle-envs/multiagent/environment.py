# -- coding: utf-8 --
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
import time
# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!

def array_n(array_In):
    array_Out = [0,0]
    array_Out[0] = array_In[0] / np.sqrt(np.square(array_In[0]) + np.square(array_In[1]))
    array_Out[1] = array_In[1] / np.sqrt(np.square(array_In[0]) + np.square(array_In[1]))
    return array_Out
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    } # metadata：元数据，用于支持可视化的一些设定，改变渲染环境时的参数，如果不想改变设置，可以无

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)    #需要操控的agent总数
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True  # 切换连续和离散
        # self.discrete_action_space = False
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        # self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        self.force_discrete_action = False
        # if true, every agent has the same reward   ##hasattr() 函数用于判断对象是否包含对应的属性。
        # self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.shared_reward = False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1) # 5
                # print('****u_action_space:', u_action_space)
                # os.system('pause')
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
                # u_action_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(world.dim_p,), dtype=np.float32)
                # print('u_action_space:',u_action_space)
            if agent.movable:
                total_action_space.append(u_action_space)
                # print('=========total_action_space',total_action_space)
            # communication action space

            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                    # chh 添加
                    # act_space = spaces.Discrete([[0, act_space.n - 1] for act_space in total_action_space])

                else:
                    act_space = spaces.Tuple(total_action_space)
                # print('\\\\\\', act_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # print('********action_space',self.action_space)
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))


        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    # step()用于编写智能体与环境交互的逻辑；它接受一个动作（action）的输入，
    # 根据action给出下一时刻的状态（state）、当前动作的回报（reward）、
    # 探索是否结束（done）及调试帮助信息信息。
    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents # 所有需要进行控制的agent
        # set action for each agent

        sensitivity_0 = self.agents[0].accel  # 加速度
        sensitivity_1 = self.agents[1].accel

        d = [0,0]
        d_n = [0,0]
        v = [0,0]
        v_n = [0,0]
        d[0] = self.agents[1].state.p_pos[0]-self.agents[0].state.p_pos[0]
        d[1] = self.agents[1].state.p_pos[1] - self.agents[0].state.p_pos[1]
        v[0] = self.agents[1].state.p_vel[0]-self.agents[0].state.p_vel[0]
        v[1] = self.agents[1].state.p_vel[1] - self.agents[0].state.p_vel[1]
        if d[0]==0 and d[1]==0:
            d_n = [0,0]
        else:
            d_n = array_n(d)
        if v[0]==0 and v[1]==0:
            v_n = [0,0]
        else:
            v_n = array_n(v)
        print('position:   ',d_n)
        action_n = [([1., 0., 0., 0., 0.]), ([1., 0., 0, 0, 0.])]
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
            if i==0:
                tmp = [0, 0]
                tmp_n = [0, 0]
                tmp[0] = (d_n[0]-0.01)
                tmp[1] = d_n[1]
                if tmp[0]==0 and tmp[1]==0:
                    tmp_n = [0,0]
                else:
                    tmp_n = array_n(tmp)
                agent.action.u[0] = tmp_n[0]*sensitivity_0
                agent.action.u[1] = tmp_n[1]*sensitivity_0

            elif i==1:
                temp = [0,0]
                temp_n = [0,0]
                p1 = 2
                p2 = 2
                if d_n[0]*v_n[0]+d_n[1]*v_n[1]>0:
                    p2 = -0.8
                else:
                    p2 = 0.8

                temp[0] = p1*d_n[0]-p2*v_n[1]
                temp[1] =p1*d_n[1]+p2*v_n[0]
                temp_n = array_n(temp)* sensitivity_1
                agent.action.u[0] = temp_n[0]
                agent.action.u[1] = temp_n[1]
        print('vel0:   ', self.agents[0].state.p_vel)
        print('vel1:   ', self.agents[1].state.p_vel)

        # advance world state
        delta_pos = np.abs(self.agents[0].state.p_pos - self.agents[1].state.p_pos)
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        #capture_distance = self.agents[0].size + self.agents[1].size
        capture_distance = 0.5
        if dist < capture_distance:
            done_n.append(True)

        self.world.step()  # core.py step()
        # record observation for each agent
        for i, agent in enumerate(self.agents):
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))

        # chh 10/21
        # 当两者距离小于抓捕范围时，done_n==true
        delta_pos = np.abs(self.agents[0].state.p_pos - self.agents[1].state.p_pos)
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # print('dist',dist)
        capture_distance = self.agents[0].size + self.agents[1].size
        capture_distance = 0.5
        if dist < capture_distance:
            done_n.append(True)
        # if any(done_n) == True:
        #     print("done_n:", done_n)

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward: # 非合作场景，shared_reward默认为false
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self): # reset()：用于在每轮开始之前重置智能体的状态
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)  # 调用的时候传入的是simple_tag的函数observation

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)  # core.py line 91 定义 self.dim_p=2 ##agent.action.u为2维数据[0,0]

        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]
        # print('a',action)
        if agent.movable:
            # physical action
            # print('ddd:',agent.action.u)
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action  #离散动作空间中 action 为1*5维的向量，action[0][0]为NOOP，即无操作，
                # 其余包括x轴正负向变化量大小，y轴正负向变化量大小
                # 在multi_discrite.py文件里面有说明
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    # agent.action.u[0] += action[0][1] - action[0][2]
                    # agent.action.u[1] += action[0][3] - action[0][4]
                    # chh 修改
                    # 在本项目中不需要累加
                    agent.action.u[0] = action[0][1] - action[0][2]
                    agent.action.u[1] = action[0][3] - action[0][4]

                    # print(('actionu',agent.action.u))
                    # print(('action1', action[0][1]))
                    # print(('action3', action[0][3]))
                    # uuu=np.sqrt(np.sum(np.square(agent.action.u)))
                    # if uuu<=1:
                    #     print('****uuu',uuu)
                else:
                    # 连续环境的输出
                    agent.action.u = action[0]
                    print('qqq:', action[0])
                # print("ccc:", agent.action.u)  # 查看信息

            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel # 加速度
            agent.action.u *= sensitivity
            # print("action", action)
            action = action[1:]  # action为空列表
            # print("ddd:",agent.action.u) #查看信息
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                for other in self.world.agents:
                    if other is agent: continue
                    word = '_'
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(800, 800)  # 修改显示框大小

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering

            # 设置显示框
            self.viewers[i].set_bounds(-30, +30, -30, +30)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results
