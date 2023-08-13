# Multi-robot-Cleaning-Task-Allocation
Benchmark for Multi-robot Cleaning Task Allocation

The code is currently under construction, the data has been released. 






## Details of implementation detils

Codes are in the 'src' folder.

We introduce three intelligent algorithms, Simulated Annealing Algorithm (SA), Genetic Algorithm (GA) and Particle Swarm Optimization Algorithm (PSO), and a reinforcement learning based solvers as baseline methods.
Details of the implementation are provided in the following section.

In our implement, the representation of a feasible solution is a vector.
The first part of the vector is the service order of the cleaning zones, and if we have several cleaning works (vacuuming, mopping, ...), we list service order of all works in sequence.
The second part of the vector is the workload of each cleaning robot, that is the number of cleaning zones robots need to perform
For example, given a environment of three cleaning zones ($1,2,3$) and each cleaning zone has two cleaning works, vacuuming and mopping.
The heterogeneous robust cluster has two types of robots (vacuuming and mopping work), with two robots for each. 
We indicate these robots as R1, R2, R3 and R4, of which "R1, R2" represent vacuuming robots and "R3, R4" represent mopping robots. 
A feasible solution vector "$1,2,3|3,2,1|2,1|1,2$" means robot R1 vacuums $1$ $2$ zones orderly, R2 vacuums zone $3$, R3 mops zone $3$ and R4 mops zone $2$ and $1$ orderly.
The solution representation above is also used for other intelligent optimization algorithms (GA and PSO).

### Simulated Annealing Algorithm
Simulated Annealing (SA) is an intelligence optimization algorithm for approximating the global optimum of a given system.
Simulated annealing is inspired by annealing in metallurgy where a material is heated to a high temperature and then slowly cool down to obtain strong crystalline structure.
The optimal solution in optimization corresponding to the minimum energy state in the annealing process.  
The search process in SA is similar as stochastic hill climbing but it gives the probability to get out of the local optima.
Every iteration, a new feasible neighbor solution y is generated randomly.
If the new neighbor solution is better than current solution x, this solution will be accepted.
Otherwise, it will go through metropolis criteria to judge whether to accept this sub-optimal solution or not.
The probability $p$ to accept a worse solution depends on the current temperature $T$ and energy degradation $-f(y)-f(x)$ of the objective value, as shown in the following equation.
A random value $r$ between $0$ and $1$ is generated every iteration. 
If $p>r$, the solution $y$ is accepted, otherwise it is rejected. 

$p=\exp \frac{-f(y)-f(x)}{T}.$

The operations to generate neighbor solutions are swap and reversion of the solution vector elements randomly.
In order to get feasible neighbor solutions, the algorithm repeats the random generation process until the solutions are feasible.
The SA intelligent algorithm uses four parameters, Iter, T0, Ts, and $\alpha$. 
$Iter$ denotes the number of iterations for which the search proceeds at a particular temperature, while $T0$ represents the initial temperature, and $Ts$ represents the final temperature, below which the SA procedure is stopped. 
Finally, $\alpha$ is the coefficient controlling the cooling schedule. 

### Genetic Algorithm

Genetic Algorithm (GA) is one of the most basic evolutionary algorithms, which simulates Darwin's theory of biological evolution to get the optimum solution.
GA algorithm contains four main steps: Reproduction, Mutation, Crossover and Selection, and repeats these four steps for every iteration until convergence (the solution does not change).

GA algorithm starts from a set of individuals of a population, each individual is a feasible solution vector mentioned above.
We randomly generate these individuals until get feasible solutions.
The probability that an individual will be selected for reproduction is based on its makespan time $C_{max}$.
We use crossover and mutation operations to reproduce best solutions.
The crossover operation exchanges parts of two solutions with each other at a given crossover rate $G_c$.
The mutation operation randomly change some codes to others in a given solution at the mutation rate $G_m$.
The mutation and crossover operations are also repeated until we can get feasible new solutions.
Every iteration, new solutions are generated through the crossover and mutation of species in the population, and better solutions are selected for the next round of competition.

The GA intelligent algorithm contains four parameters, Iter, $P_s$, $G_c$, and $G_m$. 
$Iter$ denotes the number of iterations, while $P_s$ represents the population size, $G_c$ represents the crossover rate, which is the probability of crossover operator, and $G_m$ is the probability of mutation operator. 

### Particle Swarm Optimization Algorithm
Particle swarm optimization (PSO) is an evolutionary computation optimization algorithm, which was inspired by the behaviour of flocks of birds and herds of animals. 
The basic idea of PSO is to find the optimal solution through collaboration and information sharing between individuals in a population.
PSO algorithm starts from a set of particles, with the number of $N_p$.
In PSO, each particles in the population only has two properties: speed $v^{t}_{i}, i \in N_p$ and position $p_{i}^{t}, i \in N_p$, with speed represents the speed of particle's movement and position represents the direction of movement.
Each particle individually searches for the local optimal solution (pBest) in its search space, which is the best solution so far by that particle.
And particles share the individual best position with the other particles in the whole swarm to get the current global optimal solution (gBest) for the whole swarm. 
And all particles in the swarm adjust their speeds (under the Maximum velocity $V_{max}$) and positions according to gBest and pBest.
Particles' velocities $V_{t}$ is updated as following euqation.

$v^{t+1}_{i} = W \dot v^{t+1}_{i} + c_1 U_{1}^{t}(pBest^t_{i}-p_{i}^{t}) + c_2 U_{2}^{t}(gBest-p_{i}^{t}),$


where $W$ is the Inertia weight, $c_1$ is the cognitive constant, $c_2$ is the social constant, $U_1$ and $U_2$ are random numbers. 
And moving particles to their new positions as following equation.

$p_{i}^{t+1} = p_{i}^{t} + v^{t+1}_{i},$

PSO is convergent if the velocity of the particles will decrease to zero or stay unchanged.

For cross-border processing of particles, we simply set these particles to the border values.
Parameters of PSO algorithm are listed as follows: The number of particle $N_p$, The number of iteration $Iter$, and the Maximum velocity $V_{max}$.


### Reinforcement learning

Given the success of deep neural networks in computer vision and natural language processing, recently there has been a trend to use reinforcement learning (RL) to tackle optimization problems.
Due to the properties of our task allocation problem, we choose to use an end-to-end reinforcement learning based solver as our baseline.

Our network is a Transformer model, which use an encoder-decoder architecture.
The encoder takes Task Data, Robot Data and Environment Embedding as input. 
The Task Data contains main attributes of cleaning zones (region area and distance between two regions), which is a vector, and we use a linear function to embed the structure data into a high dimension vector($N \times 128 $ in our experiments).
The Robot Data contains contain robot's cleaning efficiency, travel speed and the service ability, and we use the same way to get the robot feature.
The job precedence constraint is implicitly declared through the calculation of reward function.

The whole process is similar to the human decision-making process.
The network takes the state from the environment (Env Embedding) and gathers all problem information to make a decision of current step.
Then previous decision would yield a new environment state, and the network use the updated information to make a decision of next step.
But the origin policy network is designed for VRP problem, which only has one vehicle.
In their implements, the network would produce all nodes sequentially until there is no node remains.
However, for our heterogeneous cleaning robot task allocation, it is unsuitable to produce tour sequentially.
We let the input of the decoder is an embedding of the current region property and the robot's embedding.
Every step, the network choose a robot in turn and yield a task for this robot.









