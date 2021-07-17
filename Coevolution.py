from FFSNeuralNetwork import FFSNNetwork
from CVRPEnv import VRPEnv
import numpy as np


class Coevolution:

    def __init__(self, customer_count = 11, vehicle_count = 5, vehicle_capacity = 2, n_hidden = [20], SwarmSize = 50, Evaluations = 15):
        
        self.VRP = VRPEnv(customer_count,vehicle_count,vehicle_capacity)
        self.state = self.VRP.reset(1)
        self.n_input = 7
        self.n_output = 1
        self.n_hidden = n_hidden
        self.SwarmSize = SwarmSize
        self.Evaluations = Evaluations
        # Generate the NN and its associated structure
        self.ffsn_multi = FFSNNetwork(self.n_input, self.n_hidden)
        # Read the weights
        self.W = self.ffsn_multi.W
        self.B = self.ffsn_multi.B
        # Generate the weight and bias vectors
        self.weights = self.dict2weights(self.W)
        self.bias = self.dict2weights(self.B)
        # Combine weights and bias into a single particle
        self.particle = np.concatenate((self.weights, self.bias))
        # Break up the particle into weight and bias components
        self.weights = self.particle[0:len(self.weights)]
        self.bias = self.particle[len(self.weights):len(self.particle)]


    def Convert(self, lst, W):
        """Convert from a list data type to a dictionary data type"""
        res_dct = {lst[i]: W[i] for i in range(0, len(lst))} 
        return res_dct 
    
    
    def dict2weights(self, W):
        """Convert from dictionary data type to a weight vector"""
        data = list(W.items())
        an_array = np.array(data)
        array = an_array.flatten()
        for i in range(int(len(array)/2)):
            array1 = array[i*2+1]
            array1 = array1.flatten()
            if i == 0:
                vector = array1
            else:
                vector = np.concatenate((vector, array1))    
        return vector
    
    
    def weights2dict(self, V, n_input, n_hidden):
        """Convert from a weight vector data type to a dictionary data type"""
        architecture = np.concatenate(([n_input], n_hidden, [1]))
        W = []
        for i in range(len(architecture)-1):
            if i == 0:
                W1 = V[0:(architecture[i]*architecture[i+1])]
                W1 = W1.reshape((architecture[i], architecture[i+1]))
                count = (architecture[i]*architecture[i+1])
            else:
                W1 = V[count:(count+(architecture[i]*architecture[i+1]))]
                W1 = W1.reshape((architecture[i], architecture[i+1]))
                count = (count+(architecture[i]*architecture[i+1]))
            W.append(W1)
        lst = range(1, len(architecture)) 
        dictionary = self.Convert(lst, W)
        return dictionary
    
    
    def bias2dict(self, V, n_hidden):
        """Convert from a weight vector data type to a dictionary data type"""
        architecture = np.concatenate((n_hidden, [1]))
        W = []
        for i in range(len(architecture)):
            if i == 0:
                W1 = V[0:(architecture[i])]
            else:
                W1 = V[architecture[i-1]:((architecture[i-1]+architecture[i]))]
            W.append(W1)
        lst = range(1, (len(architecture)+1)) 
        dictionary = self.Convert(lst, W)
        return dictionary
    
    
    def GenerateSwarm(self, SwarmSize, n_input, n_output, n_hidden):
        for i in range(SwarmSize):
            # Initialise a new random neural network
            ffsn_multi = FFSNNetwork(n_input, n_hidden)
            # Recover the weights and bias from the random network
            W = ffsn_multi.W
            B = ffsn_multi.B
            # Convert the dictionary values to vectors
            weights = self.dict2weights(W)
            bias = self.dict2weights(B)
            # Concatenate the weight vectors into a particle
            particle = np.concatenate((weights, bias))
            # Combine the particles into a swarm
            if i == 0:
                Swarm = particle
            else:
                Swarm = np.concatenate((Swarm,particle))
        # Reshape the swarm to the right output format
        Swarm = Swarm.reshape((SwarmSize,(len(weights)+len(bias))))
        return Swarm
    
    
    def Objective(self, Swarm, weights, bias, n_input, n_output, n_hidden):
        Results = np.zeros((len(Swarm), self.Evaluations))
        ObjValue = np.zeros(len(Swarm))
        for j in range(self.Evaluations):
            for i in range(len(Swarm)):
                # Determine the weights and bias components of each particle
                NNweights = Swarm[i, 0:weights]
                NNbias = Swarm[i, weights:int(weights + bias)]
                # Convert the weights and bias to dictionary format
                Wupdated = self.weights2dict(NNweights, n_input, n_hidden)
                Bupdated = self.bias2dict(NNbias, n_hidden)
                # Update the values within the neural network
                self.ffsn_multi.W = Wupdated
                self.ffsn_multi.B = Bupdated     
                # Perform the prediction on the train set
                state = self.VRP.reset(j)
                routes = self.SolveVRP(self.ffsn_multi, state)
                Results[i, j] = self.DetermineDistance(routes)
        for j in range(len(Swarm)):
            ObjValue[j] = np.mean(Results[j, :])
        return ObjValue
    
    
    def Distance(self, cust_1, cust_2):
        dist = ((self.VRP.VRP[cust_1,0]-self.VRP.VRP[cust_2,0])**2+(self.VRP.VRP[cust_1,1]-self.VRP.VRP[cust_2,1])**2)**0.5
        return dist
    
    
    def DetermineDistance(self, routes):
        total_distance = 0
        for j in range(len(routes)):
            for i in range(len(routes[j])-1):
                total_distance = total_distance + self.Distance(routes[j][i],routes[j][i+1])
        return total_distance
    
    
    def SolveVRP(self, fssn_multi, state):
        routes = []
        route = []
        route.append(0)
        while len(self.VRP.unserved_customers) > 0:
            if len(self.VRP.unserved_customers) > 1:
                actions = np.zeros(len(self.VRP.unserved_customers))
                for i in range(len(self.VRP.unserved_customers)):
                    action = np.array((self.VRP.VRP[self.VRP.unserved_customers[i],0],self.VRP.VRP[self.VRP.unserved_customers[i],1],self.VRP.VRP[self.VRP.unserved_customers[i],2]))
                    state_action = np.append(state,action)
                    state_action = np.append(state_action,len(self.VRP.unserved_customers))
                    state_action = state_action.reshape((1,7))
                    actions[i] = fssn_multi.predict(state_action)
                    if action[2] > state[2]: # Vehicle cannot serve customer
                        actions[i] = 0
                if max(actions) == 0: # Vehicle does not have sufficient capacity to serve any customer
                    route.append(0)
                    routes.append(route)
                    route = [] # Start a new route
                    route.append(0)
                    state = np.array((self.VRP.VRP[0,0],self.VRP.VRP[0,1],self.VRP.vehicle_capacity)) # Set state to depot
                else:
                    action_num = np.argmax(actions) # Select action with maximum value
                    action = np.array((self.VRP.VRP[self.VRP.unserved_customers[action_num],0],self.VRP.VRP[self.VRP.unserved_customers[action_num],1],self.VRP.VRP[self.VRP.unserved_customers[action_num],2]))
                    customer = self.VRP.unserved_customers[action_num]
                    route.append(customer)
                    self.VRP.unserved_customers.remove(customer) # Remove customer from list of unserved customers
                    state[0:2] = action[0:2]
                    state[2] = state[2] - action[2]
            if len(self.VRP.unserved_customers) == 1:
                action = np.array((self.VRP.VRP[self.VRP.unserved_customers[0],0],self.VRP.VRP[self.VRP.unserved_customers[0],1],self.VRP.VRP[self.VRP.unserved_customers[0],2]))
                if action[2] < state[2]: # Vehicle can serve customer
                    route.append(self.VRP.unserved_customers[0])
                    self.VRP.unserved_customers.remove(self.VRP.unserved_customers[0])
                    route.append(0) # Return to depot after serving last customer
                    routes.append(route) #Add route to list of routes
                    state[0:2] = action[0:2]
                    state[2] = state[2] - action[2]
                else:
                    route.append(0) # Finish current route
                    routes.append(route) # Add current route to routes
                    route = []  # Create a new route
                    route.append(0) # Starting from the depot
                    route.append(self.VRP.unserved_customers[0]) # Add the final customer
                    state[0:2] = action[0:2]
                    state[2] = state[2] - action[2]
                    self.VRP.unserved_customers.remove(self.VRP.unserved_customers[0]) # Remove customer from list of unserved customers
                    route.append(0) # End the route at the depot
                    routes.append(route) # Add the route to the total number of routes
        return routes
            
    
    def PSO(self):
        """Execute the Particle Swarm Optimisation algorithm"""
        # Initialisation
        w = 0.7     # inertia weight
        c1 = 1.4
        c2 = 1.4
        rho = 1
        architecture = np.concatenate(([self.n_input], self.n_hidden, [1]))
        size = 0
        weights = 0
        bias = 0
        normal = True # If you want to apply no velocity clamping
        normalised = False # If you want to apply normalised velocity clamping
        component = False # If you want to apply componenet-based velocity clamping
        for i in range(len(architecture)-1):
            weights = weights + architecture[i]*architecture[i+1]
            bias = bias + architecture[i+1]
        size = weights + bias
        Velocity = np.zeros((self.SwarmSize, size))
        MaxIterations = 200
        Swarm = self.GenerateSwarm(self.SwarmSize, self.n_input, self.n_output, self.n_hidden)
        GBest = np.zeros(size+1)
        GBest[size] = 100000000
        PBest = np.zeros((self.SwarmSize, size+1))
        ObjValue = self.Objective(Swarm, weights, bias, self.n_input, self.n_output, self.n_hidden)
        print("Objective ", ObjValue)
        PBest = np.column_stack([Swarm, ObjValue])
        SwarmValue = np.column_stack([Swarm, ObjValue])
        
        # Determine GBest
        for i in range(self.SwarmSize):
            if PBest[i, size] < GBest[size]:
                for j in range(size+1):
                    GBest[j] = PBest[i, j]
    
        MeanVelocity = []
        iterations = 0
        Tracker = 0
        Vmax = 0.75
        while iterations <= MaxIterations and Tracker <= 50:
            iterations += 1
            Tracker += 1
            print("Search iterations: ", iterations)
            for i in range(self.SwarmSize):
                if normal == True:
                    # No velocity clamping
                    r1 = np.random.random(size)
                    r2 = np.random.random(size)
                    if SwarmValue[i, size] == GBest[size]:
                        Velocity[i, :] = 2*rho*(np.random.random(size)) - rho
                    else:
                        Velocity[i, :] = w*Velocity[i, :] + c1*r1*(PBest[i, 0:size]-SwarmValue[i, 0:size]) + c2*r2*(GBest[0:size]-SwarmValue[i, 0:size])
                if normalised == True:
                    # Nomalised velocity clamping
                    r1 = np.random.random(size)
                    r2 = np.random.random(size)
                    if SwarmValue[i, size] == GBest[size]:
                        Velocity[i, :] = 2*rho*(np.random.random(size)) - rho
                    else:
                        Velocity[i, :] = w*Velocity[i, :] + c1*r1*(PBest[i, 0:size]-SwarmValue[i, 0:size]) + c2*r2*(GBest[0:size]-SwarmValue[i, 0:size])
                    if np.linalg.norm(Velocity[i, :]) > Vmax:
                        Velocity[i, :] = (Vmax/np.linalg.norm(Velocity[i, :]))*Velocity[i, :]    
                if component == True:
                     # Component-based velocity clamping
                    r1 = np.random.random(size)
                    r2 = np.random.random(size)
                    if SwarmValue[i, size] == GBest[size]:
                        Velocity[i, :] = 2*rho*(np.random.random(size)) - rho
                    else:
                        Velocity[i, :] = w*Velocity[i, :] + c1*r1*(PBest[i, 0:size]-SwarmValue[i, 0:size]) + c2*r2*(GBest[0:size]-SwarmValue[i, 0:size])
                    Velocity[i, Velocity[i,:]>Vmax] = Vmax
                    Velocity[i, Velocity[i,:]<-Vmax] = -Vmax    
             
            # Move particles
            Swarm = Swarm + Velocity
    
            # Determine new objective value
            ObjValue = self.Objective(Swarm, weights, bias, self.n_input, self.n_output, self.n_hidden)
            SwarmValue = np.column_stack([Swarm, ObjValue])
            
            # Determine PBest
            for i in range(self.SwarmSize):
                if PBest[i, size] > SwarmValue[i, size]:
                    PBest[i, :] = SwarmValue[i, :]
                    #print("Updating PBest")
                    if GBest[size] > SwarmValue[i, size]:
                        GBest = SwarmValue[i, :]
                        Tracker = 0
                        print("GBest = ", GBest[size])

            MeanVelocity.append(np.mean(np.absolute(Velocity)))
        MeanVelocity = np.array(MeanVelocity)
        print("GBest = ", GBest[size])
        return GBest, PBest
    
    
    def evaluate(self, num_evaluations):
        OptimalNetwork, PBest = self.PSO()
        results = np.zeros(num_evaluations)
        # Break up the GBest particle into weight and bias components
        weights = OptimalNetwork[0:len(self.weights)]
        bias = OptimalNetwork[len(self.weights):len(self.particle)]
                
        Wupdated = self.weights2dict(weights, self.n_input, self.n_hidden)
        Bupdated = self.bias2dict(bias, self.n_hidden)
                
        self.ffsn_multi.W = Wupdated
        self.ffsn_multi.B = Bupdated
            
        for i in range(num_evaluations):
            print("This is evaluation number ", i)
            state = self.VRP.reset(i)
            distan = self.SolveVRP(self.ffsn_multi, state)
            print(distan)
            print(self.DetermineDistance(distan))
            results[i] = self.DetermineDistance(distan)
        
        return(results)
