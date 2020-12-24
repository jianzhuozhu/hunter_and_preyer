# -- coding: utf-8 --
import numpy as np
from multiagent.core import World, Agent
from multiagent.scenario import BaseScenario
import random

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        acc_ran = random.uniform(1,1.5)
        num_good_agents = 1
        num_adversaries = 1
        num_agents = num_adversaries + num_good_agents

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents): # agent的设置
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.25 if agent.adversary else 0.2
            # 加速度 enviroment.py line 188
            agent.accel = acc_ran if agent.adversary else 1
            # agent.max_speed = 1.0 if agent.adversary else 1.0
            agent.initial_mass = 1.0 if agent.adversary else 1.0 # chh 10.19  质量的差别影响很大，红球总是能追上小球

        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents): # agent颜色
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # agent.state.p_pos = np.asarray([0.0, 0.0]) if agent.adversary else np.random.uniform(-0.5, +0.5, world.dim_p)
            # agent初始位置状态 [x,y]=[0,0]是在视野中心
            agent.state.p_pos = np.asarray([0.0, -4.0]) if agent.adversary else np.asarray([0.0, 0.0])
            # agent.state.p_pos = np.random.uniform(-4, +4, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)  # agent初始速度
            # print('aaa:',agent.state.p_pos)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        # return True if dist < dist_min else False
        return True if dist < 0.5 else False  # 小于0.5就认为已经抓住了

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = True
        # shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
                # rew += 0.1 * max([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        # shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10

        return rew

    # 返回一个1*8的数据
    def observation(self, agent, world):
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            # 默认策略用的是绝对位置绝对速度， 神经网络训练时用的是相对位置，绝对位置训练困难。
            other_pos.append(other.state.p_pos - agent.state.p_pos) # 相对位置
            # other_pos.append(other.state.p_pos)  # chh  修改 使用默认策略时，返回的是绝对位置
            other_vel.append(other.state.p_vel)  # 绝对速度
            # other_vel.append(other.state.p_vel - agent.state.p_vel) # 相对速度 原文并未使用相对速度，只考虑了相对位置，故在此统一使用绝对速度

        #     if not other.adversary:# if  other.adversary ==False:
        #         other_vel.append(other.state.p_vel)
        #         print('agent.state.p_vel:',agent.state.p_vel)  # 1*2
        #         print('other_vel:',other_vel) # 1*2
        #         print('*'*30)
        #     print('other_vel2 :',other_vel ) # 1*2
        # print('agent.state.p_vel:',agent.state.p_vel)  # 1*2
        # print("other_pos:", other_pos)  # 1*2
        # print("entity_pos:",entity_pos) # 空
        # print("agent.state.p_pos:", agent.state.p_pos) # 1*2

        return np.concatenate([agent.state.p_pos] + other_pos + [agent.state.p_vel] + other_vel)
        # return np.concatenate( [agent.state.p_pos]  + other_pos )
