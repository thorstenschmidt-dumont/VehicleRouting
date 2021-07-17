from Coevolution import Coevolution

# Generate an instance of the coevolution solver
coevoution = Coevolution(customer_count = 11, vehicle_count = 5, vehicle_capacity = 2, n_hidden = [50], SwarmSize = 50, Evaluations = 15)

# Run the solver and evaluate on 100 instances of the specified VRP
result = coevoution.evaluate(100)