import gym
from gym import spaces
import numpy as np


class VRPEnv(gym.Env):
    """This is an environment for vehicle routing problems written in the style
    of the OpenAI gym library for easy compatibility with RL implementations"""
  
    def __init__(self, customer_count = 11, vehicle_count = 10, vehicle_capacity = 2):
        # customer count ('0' is depot) 
        self.customer_count = customer_count
        # the number of vehicles
        self.vehicle_count = vehicle_count
        # the capacity of vehicles
        self.vehicle_capacity = vehicle_capacity
        # create a list of unserved customers
        self.unserved_customers = []
      
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0,high=1, shape=(3,1), dtype=np.float64)
        self.VRP = np.array((self.customer_count,6))
        self._max_episode_steps = 1000
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.route = []
        self.route.append(0)

    def reset(self, seed):
        """Reset the environment to generate a new instance of the VRP"""  
        np.random.seed(seed) # Set the random seed to specified number if desired for fair evaluation
        x_locations = (np.random.rand(self.customer_count)).reshape((self.customer_count,1))
        y_locations = (np.random.rand(self.customer_count)).reshape((self.customer_count,1))
        demand = (np.random.randint(1,9,self.customer_count).reshape((self.customer_count,1))).reshape((self.customer_count,1))/10
        capacity = np.repeat(self.vehicle_capacity,self.customer_count).reshape((self.customer_count,1))
        VRP = np.concatenate((np.concatenate((np.concatenate((x_locations,y_locations), axis=1),demand),axis=1),capacity),axis=1)
        self.VRP = VRP.reshape((self.customer_count,4))
        self.VRP[0,2] = 0
        self.state = np.array((self.VRP[0,0],self.VRP[0,1],self.vehicle_capacity))
        self.unserved_customers = []
        for i in range(1, self.customer_count):
          self.unserved_customers.append(i)
        self.route = []
        self.route.append(0)
        return self.state 
  
    def step(self, action, customer):
        # Calculate the reward as the negative euclidean distance
        reward = -((self.state[0]-action[0])**2+(self.state[1]-action[1])**2)**0.5
        self.state[0:2] = action[0:2]
        self.state[2] = self.state[2] - action[2]
        self.unserved_customers.remove(customer)    
        done = False
        self.route.append(customer)
        if len(self.unserved_customers) > 0:
          done = False
        else:
          done = True
          reward += -((action[0]-self.VRP[0,0])**2+(action[1]-self.VRP[0,1])**2)**0.5
          self.route.append(0)
        reward = reward
        return self.state, reward, done
