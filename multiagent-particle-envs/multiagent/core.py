# -- coding: utf-8 --
import numpy as np

# physical/external base state of all entites
class EntityState(object):  #Entity类中调用
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


# action of the agent Agent类中调用Action类
class Action(object):
    def __init__(self):
        # physical action agent移动信息，1*2维数据,移动后的位置坐标
        self.u = None


# properties and state of physical world entity
# 是landmark，agent，border的父类
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.250
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0  # 什么意思
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None  # 表示加速度 environment.py line 188调用
        # state  包括位置和速度
        self.state = EntityState()
        # mass  质量
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass


# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True

        # physical motor noise amount动作噪声
        self.u_noise = None

        # control range
        self.u_range = 1.0
        # self.u_range = 1.0*30

        # action
        self.action = Action()
        # agent.action包含Action类中的physical action ：agent.action.u 和communication action ：agent.action.c
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []

        # position dimensionality
        self.dim_p = 2
        # color dimensionality颜色维度
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping 物理阻尼，对速度的一个影响 integrate_state函数中
        self.damping = 0.5
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):  # 在environment.py的step函数中调用
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        # p_force = self.apply_environment_force(p_force)

        self.integrate_state(p_force)
        # print('p_force:', p_force)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                # 在本次项目中不需要添加噪声
                p_force[i] = agent.action.u  # + noise # agent的action.u=action.u*accel  environment.py line 190
                # print(f"force", p_force)
        return p_force

    # 将agent的运动加在状态上，需要通过get_collision_force判断是不是碰撞
    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if(b <= a): continue  # a,b均为索引值, entity_a为adversary，entity_b为evader
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force


    # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable: continue # 如果是运动的实体则执行循环体，否则continue
            # entity.state.p_vel = entity.state.p_vel * (1 - self.damping)  # 更新agent速度：速度×（1-阻尼）
            # print('v',entity.state.p_vel)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass - entity.state.p_vel * self.damping / entity.mass) * self.dt  # 牛二：v=v0+f/m*t 速度=力/质量×时间
                # entity.state.p_vel += (p_force[i] / entity.mass ) * self.dt # 原论文物理模型:v=v0+f/m*t 速度=力/质量×时间
                # print(f'p_force{i}',p_force[i])
                # print('ss',p_force[i] / entity.mass -entity.state.p_vel * self.damping)

            entity.state.p_pos += entity.state.p_vel * self.dt

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        # numpy.logaddexp(x1, x2[, out])==log(exp(x1)+exp(x2)) CHH
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
