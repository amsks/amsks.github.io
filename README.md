## Aditya Mohan

<img style="float:left;display:inline-block;padding-right: 16px" src="./static/meinfoto.jpg" width="100px">

- Reach me: <a href='mailto:adityak735@gmail.com'> `adityak735@gmail.com` </a>
- [Github](http://github.com/amsks)
- [Linkedin](https://www.linkedin.com/in/amsks/)
- [Curriculum Vitae](CV/CV_Aditya_Mohan.pdf)
- [Reading list](reading-list.html)


### About Me
- I am a Master student in the [EIT Digital Master's course in Autonomous systems](https://masterschool.eitdigital.eu/programmes/aus/), working towards a specialization at the intersection of Robotics and Reinforcement Learning. I have previously worked in various fields in hardware and software, and on projects that range from cognitive radios to RTOS stacks, and now I aim to combine them and become a researcher in the field of Robotics and AI. I'm always up for collaborating on interesting projects.

- **Fun Fact :** I am a musician - multiinstrumentalist and singer - and can sing in around 8 languages so far. Checkout some of my song covers on my [instagram page](https://www.instagram.com/melodic.musings/)

#### Education
- [EURECOM](http://www.eurecom.fr/en)
- [TU-Berlin](https://www.tu.berlin/)
- [Manipal Institute of Technology](https://manipal.edu/mit.html)

#### Technologies

- **Robotics :** ROS, MORSE, FlexBE
- **Programming :** C++, Embedded C, Python, JAVA
- **RL Software :** OpenAI Gym, PyBullet, PhysX, RAI
- **Deep Learning :** Pytorch, Keras, Tensorflow

#### Zettelkasten

# Optimization: Unconstrainted Optimization


# Clouds: Fundamentals of Cloud Computing

## Cloudonomics: Quantative Analysis of cloud
In 2012, Joe weinman came up with the economic theory to estimate the business value of cloud computing, calling it Cloudonomics. The two major 




# Clouds: Introduction to Cloud Technologies
The best way to look at the development of the cloud is to look at the lifecycle for major utilities throughout history. Take the case of water, initially, the people procured water themselves which was very intensive in terms of effort and time. However, models were developed to separate the process of procurement of water and its usage. Thus, the market moved towards some players procuring the water and delivering it to the populace who could use it. However, this also went ahead and developed into a system where water was delivered through pipelines and a user would be charged on a pro-rate basis, depending on their usage. The same thing happened with electricity. This trend can be generalized to the lifecycle shown in the figure below:

<img width=500 height=200 src="static/Clouds/general-cycle.png">

If we look at IT from the lens of this cycle, then the innovation phase would be the phase where new kinds of products and services were introduced into the market, the product phase would be when companies maintained these on a growing user base, and the service phase would be when companies started addressing the growing demand and user-base by trying to achieve economies of scale through the cloud.

## Defining Cloud Computing
Cloud Computing can be defined in the following three ways:
1. It is the delivery of computing as a service rather than a product
2. It is a method to offer shared resources, software, and information to computers and other devices
3. It is a metered service over the network

### IT as a Service
There are 3 primary ways in which IT as a service can be offered:
1. **Software-as-a-Service (SaaS) :** These are applications running on a browser
2. **Platform-as-a-Service (Paas) :** These are software platforms made available to developers through APIs, to build applications
3. **Infrastructure-as-a-Service (Iaas) :** These are basic computing resources like CPU, Memory, Disk, etc. made  available to users in the form of Virtual Machine Instances

Some other models that are also possible are **Hardware-as-a-Service (Haas)** where users can get access to barebones hardware machines, do whatever they want with them (E.g clusters), and **X-as-a-service (Xaas)** which might extend to Backend, Desktop, etc.

### Cloud Infrastructure
Servers are computers that provide to user machines - the client - and the main idea behind these is that they can be designed for reliability and to service a high number of requests. Dual socket servers are the fundamental building blocks of cloud infrastructure. Organizations usually require many physical servers, like a web server or database server, to provide various services. These servers are grouped, organized, and placed into racks. For standardization, 1 Rack Unit (RU) is defined as 4.45 cm.

<img width=500 height=200 src="static/Clouds/RU.png">

A data center is a facility that is used to house a large number of servers. It needs to provide Air-Conditioning to cool the servers, Power supply to all the servers and needs to implement monitoring, network, and security mechanisms for these servers. Now the companies all have the option of privately owned data centers, but these are certain problems associated with this:
- These are expensive to set-up with a high CAPEX for real-estate, servers, and peripherals
- They have high OPEX in energy and administration costs
- It is difficult to grow or shrink applications. If the company initially budgets a small number of servers, and then there a demand surge, sometimes even overnight for companies like FaceApp, they would have to expand the area abruptly, which is very difficult. Now, let us say they are able to expand the area and resource pool, they would not be able to shrink these if they demand tapers off. These things are simply not possible for smaller companies, as much as they are for bigger companies like Dropbox
- Servers can also suffer from the problem of low utilization. This can be caused by uneven usage of applications, where one application might be exhausting one resource while leaving the others stranded-off. Another reason for this is sudden demand spikes, which taper off even more suddenly

Thus, the idea behind cloud infrastructure is to alleviate this problem by separating the server infrastructure from the end-users. The servers can be grouped into a large resource pool and then access can be given to applications based on their demand and the pricing can be set-up based on the usage of this resource pool. Hence, the applications don't need to worry about the usage statistics as far as to look into load balancing. Moreover, the sudden demand spikes and shrinks can be easily adjusted by changing the user requests. However, to offer such a service two requirements need to be met:
- Means for rapidly and dynamically satisfying application's fluctuating resource need. This is provided by **Virtualization**
- Means for servers to Quickly and reliable access shared and persistent data. This is done by programming models and distributed file/storage/database systems

This resource pool can also be defined based on its location:
- **Single-Site Cloud :** This would be the collection of hardware and software that the vendors use to offer computing resources and services to users.
- **Geographically Distributed Cloud :** This is a resource pool that is spread across multiple locations and has a composition of different structures and services

### Cloud Hardware and Software stack
The full stack for clouds has 9 components, as shown in the figure below:

<img width=200 height=300 src="static/Clouds/cloud-stack.png">

- **Applcations :** These are applications like Web-apps or Scientific Coputation Jobs etc.
- **Data :** These are the database systems like Old SQL (Oracle, SQLServer), No SQL (MongoDB, Cassandra), and  New SQL (TimesTen, Impala, Hekaton) systems
- **Runtime Environment :** These are runtime platforms like Hadoop, Spark, etc. to support cloud programming models.
- **Middleware :** These are platforms for Resource Management, Monitoring, Provisioning, Identity Management, and Security.
- **Operating Systems :** These are operating systems like Linux used on a personal machine, but they can also be packed with libraries and software for quick deployment. For example, Amazon Machine Images (AMI) contain OS as well as required software packages as a “snapshot” for instant deployment
- **Virtualization :** This layer is the key enabler of the cloud services. It creates a mapping between the lower hardware layers and the upper applications and OS layers and contributes towards multi latency. For example, the Amazon EC2 is based on the Xen virtualization platform, and Microsoft Azure is based on HyperV

The stuff below virtualization has already been discussed. However, one thing that can now b understood is how does this stack help in differentiating between the offered services. As shown in the figure below, in the case of Saas the user has only access to the applications offered by the cloud. In the case of Paas, the user manages the application and Data layer of the stack. In the case of Iaas, the user has access to all the layers above the virtualization layer, so that they can build their own application on the offered resources.

<img width=800 height=300 src="static/Clouds/stack-resources.png">

### Types of Cloud
There are three basic types of clouds:
1. **Public (external) Cloud :** This is a resource pool that serves as an open market for on-demand computing and IT resources. However, the availability, reliability, security, trust, and SLAs can have limitations.
2. **Private (Internal) Cloud :** This is the same set of services of cloud, but devoted to the functions of a large enterprise with the budget of large-scale IT
3. **Hybrid Cloud :** This is the best of both worlds. The private cloud is extended by connecting it to other public cloud vendors to make use of their available cloud services. So, a company can use their private cloud, and when the resources surge they can also extend usage to the public cloud, of course paying pro-rata

### Applications Enabled by the Cloud
The applications that can be enabled by the cloud are of 4 types
1. **High-Growth Applications :** This the same case as FaceAPP that was discussed previously. Imagine a startup that is growing. They would need a dynamic resource usage mechanism, that as discussed previously, is comfortably offered by the cloud. The risk of not setting up a distributed resource management method is losing on customer experience. This was the case with Friendster(2001), that had a similar offering as Facebook but could not keep up with the user growth.
2. **Aperiodic Applications :** These are applications that face sudden demand peaks and need a way to handle this. The cloud enables them comfortably, and again the risk is user experience. For example, Flipkart offered the 'Big-Billion Day' sale in a similar manner to Amazon's Prime Day, but initially, they could not handle the load and the customer experience was ruined. However, they did fix it over time
3. **On-off Applications :** These are one-off applications that for which extending private resources makes no sense. for example, scientific simulations requiring 1000s of computers
4. **Periodic Applications :** These are applications that will have a periodic demand surge, like stock market analysis tools or HFT tools, and thus, dynamic, flexible infrastructure can reduce costs, improve performance

### Advantages Offered by Cloud Computing
1. Pay-as-you-go economic model
2. Simplified IT Management
3. Quick adn Effortless scalability
4. Flexible options
5. Improved Resource Utilization
6. Decrease in Carbon Footpriint

# MobMod: Vehicular Flow Modelling

# MobMod: Palm Calculus
Palm calculus is a way to reconsile differences in metrics that arise from sampling differences. To simply explain this, the rudimentary example is that of a cyclist going through the



# MobMod: Random Mobility
As with any analysis, the basics start from idealized scenarios. In terms of modeling, this would be random mobility. The historical viewpoint on this comes from Brownian Motion, which is the model of the movement of particles suspended in a liquid or gas caused by collisions with molecules of the surrounding medium. The two most basic models of random mobility are:
- **Random Walk :** For every new interval $t$, each node randomly and uniformly chooses its new direction $\theta(t)$ from $(0, 2\pi]$. The new speed follows a uniform distribution or a Guassian distribution from $[0, V_{max}]$. Therefore, during time interval t, the node moves with the velocity vector $(v(t)cos(\theta(t)),v(t)sin(\theta(t)))$. If the node moves according to the above rules and reaches the boundary of simulation field, the leaving node is bounced back to the simulation field with the angle of $\pi - \theta(t)$ or $\theta(t)$, respectively. This effect is called border effect.
- **Random Waypoint :** As the simulation starts, each mobile node randomly selects one location in the simulation field as the destination, which can be bounded $[-X^{max},+X^{max}],[-Y^{max},+Y^{max}]$. It then travels towards this destination with constant velocity chosen uniformly and randomly from $[0,V_{max}]$. The velocity and direction of a node are chosen independently of other nodes, and the direction is also sampled uniformly and randomly from $[0,2\pi]$. Upon reaching the destination, the node stops for a duration defined by the ‘pause time’ parameter. If $T_{pause} =0$, this leads to continuous mobility.


<img width=300 height=200 src="static/MobMod/rwm-rwp.png">

#### Limitations of the Random Waypoint and Walk models
These models are not able to capture a lot of realistic scenarios, the major ones listed as follows:
1. **Temporal Dependency of Velocity :** In these models the velocity of the mobile node is a memoryless random process since the values at each epoch are independent of the previous one. Thus, sudden mobility behaviors are possible in these models, including sharp turns, sudden acceleration, or sudden stop. However, in real situations, these values change smoothly
2. **Spatial Dependency of Velocity :** in these models, each node moves independently of all the other nodes. However, in real scenarios, like battlefield communication or museum touring, these values may be correlated in different ways, which is not taken into account in these models
3. **Geographic Restriction of Movement :** In these models the mobile nodes are allowed to move freely without any restrictions, but this might not be the case in real life scenarios, like driving for instance, where the agents are contained in their movement by the roads, obstacles, etc.

## More Realistic Models
#### Manhattan Model
This model addresses the drawback of Geographic restriction on movement. The general idea is that initially, the nodes are placed randomly on the edges of the graph. Then for each node, a destination is randomly chosen and the node moves towards this destination through the shortest path along the edges. Upon arrival, the node pauses for T time and again chooses a new destination for the next movement. This procedure is repeated until the end of the simulation. In the Manhattan Model, these edges are in the form of a grid. Thus, this is just an extension of the Random Waypoint idea, but with added constraints on movement

<img width=300 height=200 src="static/MobMod/Mahattan-model.png">

#### Reference-point Group Mobility Model (RPGM)

This model addresses the drawback of spatial dependency of the velocity in the random models. Nodes are categorized into groups. Each group has a center, which is either a logical center or a group leader node. For the sake of simplicity, we assume that the center is the group leader. Thus, each group is composed of one leader and many members. The movement of the group leader determines the mobility behavior of the entire group. The movement of group leader at time t can be represented by motion vector $V^{t}_{group}$. The motion vector can be randomly chosen or carefully designed based on certain predefined paths. Each member of this group deviates from this general motion vector to some degree, for example, each mobile node could be randomly placed in the neighborhood of the leader. The velocity of each member can be expressed as $V^{t}_{group} + R_i$, where the second term is the deviation of each member, indexed by i, from the group leader's velocity.

<img width=300 height=200 src="static/MobMod/Ref-pt-model.png">

#### Gauss-Markov Model
This model addresses the correlation of velocities. In this model, the velocity of mobile node is assumed to be correlated over time and modeled as a Gauss-Markov stochastic process:
$$
\begin{aligned}
R(t) = \bar{\alpha}.R(t-1) + (1-\bar{\alpha}).R + \sqrt{1 -  \bar{\alpha}^2 }.\bar{X}_{t-1}\\

& R(t) \rightarrow speed/direction\\
&\bar{X} \text{\textasciitilde} I.I.D Gaussian (0, \sigma)\\
&\bar{\alpha} = e^{-\beta} \text{\textasciitilde} [0,1]\\
&\bar{\alpha} = 0 \rightarrow Brownian Motion \\
&\bar{\alpha} = 1 \rightarrow Linear Motion \\
\end{aligned}
$$

Based on these equations, we observe that the velocity mobile node at time slot $t$ is dependent on the velocity-time slot $t-1$. Therefore, the Gauss-Markov model is a temporally dependent mobility model where the degree of dependency is determined by the memory level parameter $\bar{\alpha}$, a parameter to reflect the randomness of the Gauss-Markov process. By tuning this parameter, this model is capable of duplicating different kinds of mobility behaviors lying on the spectrum of Linear and Brownian motion.

#### Smooth Motion Model
Another mobility model considering the temporal dependency of velocity over various time slots is the Smooth Random Mobility Model. Here, instead of the sharp turns and accelerations as proposed in the Random Waypoint Model, these values are changed smoothly. It is observed that mobile nodes in real life tend to move at certain preferred speeds, rather than at speeds purely
uniformly distributed in the range. Therefore, in Smooth Random
Mobility model, the speed within the set of preferred speed values has a high probability, while a uniform distribution is assumed on the remaining part of the entire interval
$$
V^{pref} = Random(V_1, V_2, .... , V_n)
$$
The frequency of speed change is assumed to be a Poisson process. Upon an event of speed change, a new target speed is chosen according to the probability distribution function of speed above and the speed of the mobile node is changed
incrementally from the current speed to the targeted new speed
by acceleration or deceleration a(t). The probability distribution function of acceleration or deceleration a(t) is uniformly distributed among $[0, a_{max}]$ and $[a_{min} ,0]$, respectively. The new speed depends on the previous speed:
$$
V_t = V_{t-1} + a(t)(V_t - V_{t-1})
$$

A similar approach is followed for the direction update with angular accelerations

## Problem of Unsteady Values
One of the most important factors that play a role in the analysis of the various parameters associated with these networks is the stability of values. In the evolution of network states, there is usually an initial phase where the process variables change over time, and this phase is called a **Transient Phase**. During this transient phase, analyzing these variables is not possible as any value that they predict for the system is not a good indicator. As this phase passes, these values start to stabilize and as they stop varying over time units on an average, a **Steady State** is reached. This is the point at which a feasible network analysis starts becoming possible. in case the network transitions into something else over time, another steady state is reached after an intermediate transient state. The problem comes when these transient states start lasting longer or start happening more frequently since then stable analysis of the system becomes increasingly difficult. Thus, one of the major areas of research had been in figuring out a way to effectively predict these steady-state values and use them in analyzing the models. The seminal work on this was done by J-Y Le Boudec at EPFL, where the team developed a method called **Palm Calculus** and used it to predict the steady-state distributions of all major random mobility models.

# MobMod: Introduction to Mobility Modelling
The fundamental abstraction that is needed to understand inter-connected phenomena is a way to describe the different relationships between the various participating entities. For example, imagine a scenario where a disaster management Mobile Ad-hoc network of drones is deployed to triangulate sensitive points, the key to successful execution lies in how these various drones interact and share knowledge. Thus, the **Network** that is shared between these drones - or, **Nodes** - would have a certain way of establishing communication and its performance can be analyzed in terms of various performance parameters. One central aspect of analyzing this network would be the simulating how these nodes might move and what kind of impact that might have on the network. It is exactly for this kind of analysis and understanding that we create **Mobility Models**: They allow mimicking the behavior of mobile nodes when network performance is simulated. The simulation results are strongly correlated with the mobility model.

#### Case of Connected Cars
The way this analysis fits into the driving scenario is in terms of modeling the uncertainty in the traffic scenario. In the case of autonomous driving, the limitation for an individual agent is more from the side of the sensors. Usually, we would use cameras and Radars to work stuff out, but Radars can only see around 10m. So, how would this agent work in a highly uncertain scenario? Imagine a car driving on an Indian road in a 2-tier city, with pedestrians walking on all sides and multiple vehicles going in all directions.

<img width=600 height=400 src="static/MobMod/Indian-road.png">

One way to approach this solution would be to shift a bit of the load from the individual agent to the network. This could be done if the cars are connected through a network based on 4G/5G. Thus, in this case, we could view the cars as nodes communicating with each other and moving in some way. For each car, the regular functions - movement, mapping, obstacle detection, and avoidance, etc. - would be performed on the edge of this network while the co-ordinating function would be performed on the network. Thus, the issue of modeling the Mobility becomes important in this case. Modeling of the traffic flow, especially in the case of more uncertain scenarios, becomes extremely important.

#### Need for Modelling
1. The performance evaluation of such a large-scale network is bounded by simulation: Raw analytical analysis is too complex while conducting experiments is too expensive
2. Real-world traces are hard to find, either because they do not exist or are not publicly available (E.g City-Data).
3. Trivial representations of mobility might bias the simulation results. The available races might not represent real-world situations very effectively, and might have a lot of residual effects that might render them useful to represent only specific conditions, and thus, not generalizable.

#### The safe and sustainable mobility conundrum
Now if we were to go about this modeling, one of the central issues that would come up is that of optimality. The optimization here would be in terms of maximizing the usage of the road capacity and the driver and pedestrian safety. To better understand this, imagine we implement a model where a car is made to drive slow to increase safety. Now, this is fine for the individual driver, since the car always maintains a safe distance from the car in front of it. However, two problems come-up here:
1. If the safe distance is too large, then a pedestrian might walk-in between leading to the car to halt
2. The speed of the car is slow and so, the general traffic speed is also slow. This might create problems on traffic bottlenecks, like a 7 lane highway leading to a 2 lane road, etc.

If we use the same policy on high-volume traffic conditions, then slow speeds nad sudden halts can easily lead to shockwave propagation and Ghost Jams. Let us formalize this by looking at traffic in terms of flow

$$
Flow = Density . Speed
$$

Thus, looking at this model we can see some safe directions to analyze this situation would be by controlling this flow through either reducing the speed or maybe keeping it and reducing the flow through density control. But more importantly, if we were to model the mobility by simulating this condition, we could develop interesting ways of cooperative navigation.

#### Vehicular Models
To analyze this network, we start looking at vehicles as nodes. Thus, the traffic becomes a MANET and our methods of simulation enable us to better understand this network.

<img width=1000 height=400 src="static/MobMod/car-node.png">

The impact of mobility is even more pronounced in the case of vehicular networks. The three factors that differentiate these networks from other networks are:
1. The speed of each node is not bounded in small intervals, and is not smooth. It is highly variable
2. The movement is far from random. Thus, we cannot directly sample from standard distributions in realistic scenarios
3. The nodes do not move independently, and in  fact, have strong reciprocal dependencies

Consequently, the abstractions that this network viewpoint offers can be on three levels:
1. Vehicular Traffic Model: Abstraction of the large scale trajectories employed by the vehicles
2. Vehicular Flow Model: Abstractions of the physical inter-dependencies
3. Vehicular Driver Model: Abstractions of the actions of individual nodes, like breaks, turns, etc.

# MALIS: Introdution to Machine Learning 



# RL: Model-Free Control

While prediction is all about estimating the value function in an environment for which the underlying MDP is not known, Model-Free control deals with otimizing this value function.




# RL: Model-Free Prediction
One of the problems with DP is that it assumes a full knowledge of the MDP, and consequently, the environment. While this holds true for a lot of applications, it might not hold true for all cases. In fact, the upper limit does turn out to be the ability to be accurate about the underlying MDP. Thus, if we don't know the true MDP behind a process, the next best thing would be to try to approximate them. One of the ways to go about this is **Model-Free RL**. 

## Monte-Carlo Methods
The first that comes to my mind when someone speaks fo approximation is Monte-carlo methods. The core idea behind the monte-carlo approach, which has its root in gambling, is to use probaility to approximate quantities. suppose we have to approximate the area of a circle relative to a rectangle inside which it is inscribed (This is a classic exmaple and an easy experiment too), the experiment-based approach would be to make a board and randomly throw some balls on it. In a true random throw, let's call the creation of a spot on the board as a simulation ( experiment, whatever!), after each simulation, record the number of sots inside the circular area and he total number of spots including the circular area and the the recangluar area. A ratio of these two quantities would give us an estimate of the relative area of the circle and the rectangle. Now as we conduct more such experiments, this estimate would actually get better since the underlying probability of a spot appearing inside the circle is proportional to the amount of area that the circle occupies inside the rectangle. Thus, in the infinity limit, we could get the exact value of this ratio. 

### Applying MC idea to RL
Another way to put the monte-carlo approach would be to say that Monte-Carlo methods only require experience. In the case of RL, this would translate to sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. An experiment in this sense would be a full rollout of an episode which will create a sequence of states and reqards. when multiple such experiments are conducted, we get better approximations of our MDP. To be specific, our goal here would be to learn $v_{\pi}$ from episodes of experience under policy $\pi$. The value function is the expectation fo reward

$$
v_{\pi}(s) = E_{\pi}[G_t | S_t = s]
$$

So, all we have to do is estimate this expectation using the empirical mean of the returns for the experiments. 

### First-Visit MC Evaluation
To evaluate state s, at the first time-step t at which s is visited: 
1. Increment counter $N(s) \gets  N(s) + 1$
2. Increment total return $S(s) \gets S(s) + 1 $
3. Estimate value by the mean return $V(s) = S(s)/N(s) $

As we repreat more experiments and update the values at the  first visit, convergence to optimal values $V(s) \to  v_{\pi}$
as $N(s) \to \infin$

### Every-Visit MC Evaluation
This is same as first visit evaluation, excep  we update at every visit: 
1. $N(s) \gets  N(s) + 1$
2. $S(s) \gets S9(s) + 1 $
3. $V(s) = S(s)/N(s) $


### Incremental MC updates
The empirical mea can be expressed as an incremental update as follows: 

$$
{\mu}_k = \frac{1}{k} \displaystyle\sum_{j=1}^n x_j = \frac{1}{k} (x_k + \displaystyle\sum_{j=1}^{k-1} x_j) = \frac{1}{k} (x_k + (k-1)\mu_{k-1})
$$

Thus, for the state updates, we can follow a similar pattern and for each state $S_t$ and return $G_t$ express it as: 
1. $N(S_t) \gets N(S_t) + 1$
2. $V(S_t) = V(S_t) + \frac{1}{N(S_t)}(G_t - V(S_t))$

An another useful way would be to track a running mean: 

$$
V(S_t) \gets V(S_t) + \alpha (G_t - V(S_t)
$$



## Temporal-Difference (TD) Learning
The MC method of learning needs an episode to terminate in order to work its way backward. In TD, the idea is work the way forwrad by replacing the remainder of the states with an estimate. This is one method that is considered central adn novel to RL (According to Sutton adn Barto). Like MC methods, TD methods can learn directly from raw experience and like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome, something which is called Bootstrapping - updating a guess towards aguess (I know!).

### Concept of Target and Error
If we look at the previous equation of Incremental MC, the general form that can be extrapolated is 

$$
V(S_t) \gets V(S_t) + \alpha (T - V(S_t)
$$

Here, the quantity $T$ is called the **Target** and the quantity $T - V(S_t)$ is called the **error**. Now, in the MC version, the target is the return $G_t$, means the MC method has to wait for this return to be propagated backwards to then see the error if its current value function from this return, and improve. This is where TD methods show their magic; At time t+1 they immediately form a target and make a useful update using the observed reward and the current estimate of the value function. The simplest TD method, TD(0), thus has the following form 

$$
V(S_t) \gets V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

This is why bootstrapping is a guess of guess, since the TD method bases its update in part on an existing estimate. 

## Comparing TD, MC and DP
Another way to look these algorithms would be through the bellman optimality equation:

$$
v_{\pi}(s) = E [ R_{t+1} + \gamma v_{\pi}(S_{t+1})| S_t = s]
$$

The MC method is an estimate because it does not have a model of the environment and thus, needs to sample in order to get an estimate of the mean. The DP method is an estimate because it does not know the future values of states and thus, uses the current state value estimate in its place. The TD method is an estimate because it does both of these things. Hence, it is combination of both. However, unlike DP, MC and TD do not require a model of the environment. Moreover, the online nature of these algorithms is something that allows them to work with samples of backups, whereas DP requires full backup.TD and MC can be differentiated in the nature of the samples that they work with: TD requires shallow backups since it is inherently online in nature, while MC requires deep backups due to the  nature of its search. Another way to look at the inherent difference is to realize that DP inherently dies a breadth first search, while MC does a depth first search. TD(0) only looks one step ahead and forms a guess. David silver summarizes the differences on the following spectrum, which I find really helpful: 

<img width=600 height=500 src="static/Reinforcement Learning/TD-MC-DP.png">

## Extending TD to n-steps 

The next natural step for something like TD, would be to extend it to further steps. For this, we generalize the target and define the it as follows:

$$
G^{(n)}_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+ n} + \gamma^n V(S_{t+n})
$$

And so the equation again follows the same format:

$$
V(S_t) \gets V(S_t) + \alpha (G^{(n)}_t - V(S_t)
$$

One interesting thing to note here is that if n is increased all the way till the terminal state, then we essentially get the same equation as MC methods! 

### Averaging over n returns 
To get the best out of all n's, one improvement could be to avergae teh returns over certain number of states. For example, we could combine 2-step and 4-step returns and take teh average :

$$
G_avg = \frac{1}{2} [ G^{(2)} + G^{(4)} ]
$$

This has been shown to work  better in many cases, but only incrementally.

### $\lambda$-Return

This is a method to combine the returns from all the n-steps:

$$
G^{(\lambda)}_t = (1 - \lambda ) \displaystyle\sum_{n=1}^{\infin} \lambda^{n-1} G^{(n)}_t

$$

And ths, is als called **Forward-view TD($\lambda$)**

### Backward View 

To understand the backward view, we need a way to see how we are going to judge the causal relationships between events and outcomes (Returns). There are two heuristics:
1. **Frequency Heuristic :** Assign credit to the most frequent states
2. **Recency Heuristic :** Assign credit to the most recent states.

The way we keep track of how each states fares on these two heuristic is through **Eligibility Traces** :

$$
E_t(s) = \gamma \lambda E_{t-1}(s) + \bold{1}(S_t = s)
$$

These traces accumulate as the frequency increases, and are higher for more recent states. If the frequency drops, they also drop. This is evident in the figure below:

<img width=300 height=100 src="static/Reinforcement Learning/ET.png">

So, all we need to do it scale the TD-error $\delta_t$ according to the trace function:

$$
V(S) \gets  V(S) + \alpha \delta_t E_t(s)
$$

Thus, when $\lambda = 0$, we get TD(0) and when $\lambda = 1$, the credit is deferred to the end of the episode nand we get MC



# RL: Planning and Dynamic Programming

Dynamic programming (DP) is a method that solves a problem by breaking it down into sub-problems and then solving each sub-problem individually, and then combining them into a solution. A godo example is the standard fibonacci sequence calculation problem, where traditionally the way to solve ti would be through recursion

```cpp
int fib(int x)
{
    if (x < 2)
        return 1;

    return fib(x-1) + fib(x-2);
}

```
However, the way DP would go about this would be to cache the variables after teh first call, so that the same call is not made again, making the program more efficient:

```cpp
int fib(int x)
{
    static vector<int> cache(N, -1);

    int& result = cache[x];

    if (result == -1)
    {
        if (x < 2)
            result = 1;
        else
            result = fib(x-1) + fib(x-2);
    }

    return result;
}

```

The 2 characteristics that a problems need to have fo DP to solve it are: 
1. **Optimal Substructure :** Any problem has optimal substructure property if its overall optimal solution can be constructed from the optimal solutions of its subproblems i.e the property $Fib(n) = Fib(n-1) + Fib(n-2)$ in fibonacci numbers
2. **Over-lapping Sub-problems :** The problem involves sub-problems which need to be solved recursively many tiimes

Now, in teh case of an MDP, we have already seen that these properties are fulfilled: 
1. The Bellman equation gives a recursive relation that satisfies the overlap requirement
2. The valu function is able to store and re-use the solutions from each state-visit, and thus, the we can exploit it as an optimal substructure

Hence, DP can be used for making solutions to MDPs more tractable, and thus, is a good tool to solve the planning problem in an MDP. The plannign problem, as discussed before, is of two types: 
1. **Predicton Problem :** **How do we evaluate a policy ?** or, Using the MDP tuple as an input, the output is a value function $V_{\pi}$ and/or a policy $\pi$
2. **Control Problem :** **How do we optimie the policy ?** Using the MDP tuple as an input, the output is a an optimal value function $V_{*}$ and/or a policy $\pi_{*}$ 

## Iterative Policy Evaluation

The most basic way is to iteratively apply the bellman equation, using hte old values to calculate a new estimate, and then using this new estimate to calculate new values. IN the bellman equation for the state-value function 

$$
v_{\pi}(s) = \sum_{\substack{a \in A}} \pi (a|s) [ R^{a}_s  +  \gamma \sum_{\substack{s' \in S}} P^{a}_{ss'} v_{\pi} (s) ]
$$

As long as either $\gamma < 1$ or eventual termination is guaranteed from all states under the policy $\pi$, the uniqueness of the value function is guaranteed. Thus, we can consider a sequence of approximation functions $v_0, v_1, v_2, ...$ each mapping states to Real numbers, start with an arbitrary estimate of $v_0$, and obstain successive approximations using bellman equation, as follows: 

$$
v_{k+1}(s) = \sum_{\substack{a \in A}} \pi (a|s) [ R^{a}_s  +  \gamma \sum_{\substack{s' \in S}} P^{a}_{ss'} v_{k} (s') ]
$$

The sequence $v_k$ can be shown to converge as $ k \rightarrow \infin $

the process, is basically a propogation towards the root of the decision tree from the roots.

<img width=300 height=200 src="static/Reinforcement Learning/It-pol-eval.png">

This update operation is applied to each state in the MDP at each step, and so, is called **Full-Backup**, ad so in a computer program we would have two cache arrays - one for $v_k(s)$ and one for $v_{k+1}(s)$

## Policy Improvement
Once we have a policy, the next question is do we follow this policy or shift to a new improved policy ? one way to answer this problem is to take an action that this policy does not suggest and then evluate the smae policy after that action. If the returns are higher than the we can say that taking that action is better than following the current policy. The way we evaluate the action is throught the action value function: 

$$
q_{\pi}(s,a) = R^{a}_s  +  \gamma \sum_{\substack{s' \in S}} P^{a}_{ss'} v_{\pi} (s')
$$

If this value is greater than the value function of a state S, then that means that it is better to select this action than follow the policy $\pi$, and by extension, it would mean that anytime we encounter state S, we would like to take this action. So, let's call the schema of taking action a everytime we encounter s as a new policy ${\pi}'$, and so, we can now say 

$$
q_{\pi}(s,{\pi}'(s)) \geqslant v_{\pi}(s)
$$


This implies that the policy ${\pi}'$ must be at-least as good as the policy $\pi$

$$
v_{{\pi}'} \geqslant v_{\pi}
$$

Thus, if we extend this idea to multiple possible actions at any state s,  the net incentive is to go full greedy on it and select best out of all those possible actions: 

$$
{\pi}'(s) = \argmax_a q_{\pi}(s,a)
$$

The greedy policy, thus, takes the action that looks best in the short term i.e after one step of lookahead. The point at which the new policy stops becoming better than the old one, is the conveergence point, and we can conclude that optimality has been reached. This idea also applies in the general case of stochastic policies, with the addition tha in the case of multiple actions with the maximum value, a portion of the stochastic probabiltiy can be given to each.

## Policy Iteration

Following the greedy policy improvement process, we can bstain a sequency of policies: 

$$
{\pi}_0 \rightarrow v_{\pi_0} \rightarrow {\pi_1} \rightarrow v_{\pi_1} .... \rightarrow {\pi}_* \rightarrow v_{{\pi}_*}
$$

Since a finite MDP has a finite number of policies, this process must converge at some point to an optimal value. This process is called **Policy Iteration**. The algorithm, thus, follows the process: 
1. **Evaluate** the policy using the Bellman equation
2. **Improve** the policy using greedy policy improvement.

A natural question that comes up at this point is that do we actually need to follow this optimiztaion procedure till the end ? It does sound like a lot of work, and in fact, as  seen in the Gridworld example (Sutton and Barto), the workably optimal policy is actually reached much before the final iteration step, where basically teh final three steps actualy are redundant. Thus, we can include stopping conditions to tackle this: 
1. $\epsilon$-convergence
2. Stop after k iteratiokns
3. Value Iteration


## Value Iteration

in this algorithm, the evaluation is truncated to one sweep, or one backup of each state. to understand this, the first step is to understand something called the **Principle of Optimality**. The basic idea is that an optimal policy can be subdivided into two parts:
- An optimal first action $A_*$
- An optimal policy from the successor state S'

So, if we know the the solution to $v_*(s')$ for all s' succeeding a state, s , then the solution can be found with just a one-step lookahead

$$
v_*(s) \gets \max_{a \isin A} R^a_s + \gamma \sum_{\substack{s' \in S}} P^{a}_{ss'} v_{*} (s')
$$

The intuition is to start from teh final reward and work your way backward. There is no explicit update of policy, only values. This also opens up the possibility that the intermediate values might not correspond to any policies, and so interpreting anything midway will have some residue in addition to the greedy policy.  In practice, we stop once the value function changes by only a small amount in a sweep. A  summary of synchronous methods for DP is given by David Silverman:


<img width=500 height=150 src="static/Reinforcement Learning/sync_DP_summary.png">


# RL: Markov Processes
These are random processes indexed by time and are used to model systems that have limited memory of the past. The fundamental intuition behind Markov processes is the property that the future is independent of the past, given the present. In a general scenario, we might say that to determine the state of an agent at any time instant, we only have to condition it on a limited number of previous states, and not the whole history of its states or actions. The size of this window determines the order of the Markov process.

To better explain this, one primary point that needs to be addressed is that the complexity of a Markov process greatly depends on whether the time axis is discrete or topological. When this space is discrete, then the Markov process is a Markov Chain. A basic level understanding of how these processes play out in the domain of reinforcement learning is very clear when analyzing these chains. Moreover, the starting point of analysis can be further simplified by limiting the order of Markov Processes to first-order. This means that at any time instant, the agent only needs to see its previous state to determine its current state, or its current state to determine its future state. This is called the **Markov Property**

$$
\Rho(S_{t+1}|S_t) = \Rho(S_{t+1}|S_1, ..., S_t) )
$$

## Markov Process
tthe simplest process is a tuple $<S,P>$ of states and Transitions. The transitions can be represented in the form of a Matrix $P = [P_{ij}]$, mapping the states from which the transition originates, the i index, and the states to which the transition goes, the j index.

$$
\begin{bmatrix}
  P_{11} & . & . & . & P_{1n}\\
  . & . & . & . & . \\ 
  . & . & . & . & . \\
  . & . & . & . & . \\
  P_{n1} & . & . & . & P_{nn}
\end{bmatrix}
$$

Another way to visualize this would be in the form of a graph, as shown below, courtsey David Silver. 

<img width=800 height=500 src="static/Reinforcement Learning/MP.png">

This is a basic chain that represents the actions a student can take in the class, with associated probabilities of taking those actions. Thus, in the state - Class 1 - the student has an equal chance of going to the next class or browsing Facebook. Once they start browsing Facebook, then they have a 90% chance of continuing to browse since it is addictive. Similarly, other states can be seen too. 

## Markov Reward Process
Now if we add another parameter of rewards to the Markov processes, then the scenario changes to the one in which entering each state have an associated expected immediate reward. This, now, becomes a Markov Reward Process. 

<img width=800 height=500 src="static/Reinforcement Learning/MRP.png">

To fully formalize this, one more thing needs to be added - the discounting factor $\gamma$. This is a hyperparameter that intuitively represents the amount of weightage we give to future rewards. The use of Gamma can be seen in computing the return $G_t$ on a state:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... 
$$




The reasons for discounting are as follows:
- To account for uncertainty in the future, and thus, better balance our current decisions. The larger the value, the more weightage we give to the 'shadow of the future'
- To make the math more convenient, since only when we discount the successive terms, we can get convergence on an infinite GP
- Avoid Infinite returns, which might be possible in loops within the reward chain
- This is similar to how biological systems behave 

Thus, the reward process becomes can be characterized by the tuple $<S, P, R, \gamma >$ . Now, to better analyze this chain, we will also define a way to estimate the value of a state - **Value function** - as an expectation of the Return on that state. Thus,

$$
V(S) = E[ G_t| S_t = s ]
$$

An intuitive way to think of this is in terms of betting. Each state is basically a bet that our agent needs to make. Thus, the reward process represents the agent's understanding of each of these bets, and to qualify them, the agent has to think in terms of the potential returns that these bets can give. This is what we qualify here as the expectation. But the magic comes when we apply it recursively, and this is called the **Bellman Equation**

$$
V(S) = E[R_{t+1} + \gamma V(S_{t+1}| S_t = s]
$$

This basically means that the value of the current state can be seen in terms of the value of the next state and so on, and thus, we can have a correlated relationship between states. To better see how this translates to the whole MDP, we can also express this as a Matrix Operation

$$
\begin{bmatrix} 
   V_1 \\
   .  \\
   .  \\
   V_n
\end{bmatrix} 
= 
\begin{bmatrix} 
   R_1 \\
   .  \\
   .  \\
   R_n
\end{bmatrix} 
+ 
\begin{bmatrix}
  P_{11} & . & . & P_{1n}\\
  . & . & . & . \\ 
  . & . & . & . \\
  P_{n1} & . & . & P_{nn}
\end{bmatrix}
\begin{bmatrix} 
   V_1 \\
   .  \\
   .  \\
   V_n
\end{bmatrix} 
$$

And so, the numerical way to solve this would be to invert the matrix (assuming it is invertible) and the solution, then would be:
$$
\bar{V} = (1 - \gamma \bar{P})^{-1} \bar{R}
$$

However, as anyone familiar with large dimensions knows, this becomes intractable pretty easily. Hence, the whole of RL is based on figuring out ways to make this tractable, using majorly three kinds of methods: 
1. Dynamic Programming
2. Monte-Carlo Methods
3. Temporal Difference Learning

We will delve deeper into these in the next parts! 

## Markov Decision Process

If we add actions to the Markov Reward Process, then there can multiple states that the agent can reach by taking action to each state. Thus, the agent now has to decide which action to take. This is called a Markov Decision Process.

<img width=800 height=500 src="static/Reinforcement Learning/MDP.png">

Thus, the MDP can be summarized by the tuple $<S, A, P, R, \gamma >$. Here, we can also define the transitions and rewards in terms of the actions:

$$
\begin{aligned}
R^{a}_{s} = E[ R_{t+1} | S_t=s, A_t=a ] \\
P^{a}_{ss'} = P[ S_{t+1}=s'| S_t=s, A_t=a ]
\end{aligned}
$$

Now, the important thing is how should the agent make these decisions. The schema that the agent follows for this is called a **Policy**, which can be seen as the probability of taking an action, given the state: 

$$
\pi (a|s) = P [ A_t = a | S_t = s ] 
$$

Under a particular policy $\pi$, the Markov chain that results is nothing but an MRP, since we don't consider the actions that the agent did not take. This can be characterized by $<S, P^{\pi}, R^{\pi}, \gamma>$, and the respective transitions and rewards can be described as:

$$
\begin{aligned}
R^{\pi} = \sum_{\substack{a \in A}} \pi (a|s) R^{a}_{s}\\
P^{\pi}_{ss'} = \sum_{\substack{a \in A}} \pi (a|s) P^{a}_{ss'}
\end{aligned}
$$

Another important thing that would need to be distinguished here is the value function, which can be defined for both states and actions: 
- **State-Value Function ($v_{\pi}$):** Values for states when policy $\pi$ is followed 

$$
v_{\pi}(s) = E_{\pi}[G_t | S_t = s]
$$

- **Action-Value Function ($q_{\pi}$):** Expected return onj starting from state s, following policy $\pi$ and taking action a

$$
q_{\pi}(s, a) = E_{\pi}[G_t | S_t = s, A_t = a] 
$$


### Bellman Equation for MDPs
We can extend the bellman formulation to recursively define the qualities of state and actions:

$$
\begin{aligned}
v_{\pi}(s) = E_{\pi} [R_{t+1} + \gamma v_{\pi}(s')| S_t = s, S_{t+1} = s'] \\

q_{\pi}(s, a) = E_{\pi} [R_{t+1} + \gamma q_{\pi}(s', a')| S_t = s, S_{t+1} = s', A_t = a, A_{t+1} = a']

\end{aligned}
$$

However, a better way is to look at the inter-dependencies of these two value functions. The value of the state can be viewed as the sum of the value of the actions that can be taken from this state, which can in-turn be viewed as the weigted sum of values of the states that can result from each action.

#### Bellman Expectation in second recursive form

The expectation for the value of states is the sum of the values of the actions that can result from that state

<img width=400 height=200 src="static/Reinforcement Learning/sve.png">

Thus, under the policy $\pi$ this value is the sum of the q-values of the actions: 

$$
v_{\pi}(s) = \sum_{\substack{a \in A}} \pi (a|s) q_{\pi} (s,a)
$$

Now, the action can be viewed in a similar manner as a sum over the value fo the states that can result from it

<img width=300 height=200 src="static/Reinforcement Learning/ave.png">

and written in the same manner

$$
q_{\pi}(s,a) = R^{a}_s  +  \gamma \sum_{\substack{s' \in S}} P^{a}_{ss'} v_{\pi} (s)
$$

And, if we put these equations together, we can get a self-recursive formulation of the bellman expectation. So, for the state this would be

$$
v_{\pi}(s) = \sum_{\substack{a \in A}} \pi (a|s) [ R^{a}_s  +  \gamma \sum_{\substack{s' \in S}} P^{a}_{ss'} v_{\pi} (s) ]
$$

A Visualization for this would basically be a combination of the above two trees

<img width=300 height=200 src="static/Reinforcement Learning/sveave.png">

A similar process can be done for the action value function, and the result comes out to be

$$
q_{\pi}(s,a) = R^{a}_s  +  \gamma \sum_{\substack{s' \in S}} P^{a}_{ss'} \sum_{\substack{a' \in A}} \pi (a'|s') q_{\pi} (s',a')
$$

<img width=300 height=200 src="static/Reinforcement Learning/avesve.png">


#### Bellman Optimality Equation

With the recursive forms, the question really comes on how do we go about creating a closed-loop optimality criterion. Here, the key point that needs to be taken into account is **The agent is free to choose the action that it can take in each state, but it can't choose the state that results from that action**. This means, we start from a state, and maximize the result by choosing the action with the maximum action value. This is the first step of lookahead. Now, each of those actions has the associated action value that needs to be determined. In the case where the action can only lead to one state, it's all well and good. However, in the case where multiple states can result out of the action, the value of the action can be determined by basically rolling a dice and seeing which state the action leads to. Thus, the value of the state that the action leads to determines the value of the action. This happens for all the possible action from our first state, and thus, the value of the state is determined. Hence, with this **Two-step lookahead**, we can formulate the decision as maximizing the action values.

$$
v_{*}(s) = \max_{\substack{a}} [ R^{a}_s + \gamma \sum_{\substack{s' \in S}} P^{a}_{ss'} v_{*} (s)]
$$

Now, the question arises as to how can this equation be solved. The thing to note here is the fact that it is not linear. Thus, in general, there exists no closed-form solution. However, a lot of work has been done in developing iterative solutions to this, and the primary methods are:
- **Value Iteration :** Here methods solve the equation by iterating on the value function, going through episodes, and recursively working backward on value updates
- **Policy iteration :** Here, the big idea is that the agent randomly selects a policy and finds value function corresponding to it. Then it finds a new and improved policy based on the previous value function, and so on.
- **Q Learning :** This is a model-free way in which the agent is guided through the quality of actions that it takes, wit the aim of selecting the best ones
- **SARSA :** Here the idea is to iteratively try to close the loop by selecting a **S**tate, **A**ction, and **R**eward and then seeing the **S**tate and **A**ction that follows.

## Extensions to MDP

MDPS, as a concept, has been extended to make them applicable to multiple other kinds of problems that could be tackled. Some of these extensions are: 
1. **Infinite and Continuous MDPs :** In this extension, the MDP concept is applied to infinite sets, mainly countably infinite state or action spaces, Continuous Spaces (LQR), continuous-time et. al  
2. **Partially Observable MDPs (POMDP) :** A lot of scenarios exist where there are limits on the agent's ability to fully observe the world. These are called Partially-Observable cases. Here, the state is formalized in terms of the belief distribution over the possible observations and encoded through the history of the states. The computations become intractable in theory, but many interesting methods have been devised to get them working. Eg. DESPOT
3. **Undiscounted and Average Reward MDP :** These are used to tackle ergodic MDPs - where there is a possibility that each state can be visited an infinite number of times ( Recurrence), or there is no particular pattern in which the agent visits the states (Aperiodicity) - and to tackle this, the rewards are looked at as moving averages that can be worked with on instants of time.


# RL: Introduction to Reinforcement learning
The way I like to think about Reinforcement learning is by imagining myself as a ring-master who has to train a certain creature. One of the reasons why I like this visualization is because RL has historical roots in trial and error and the psychology of animal learning. So, let's say I want to train this creature to perform some tricks around a hula-hoop, one of the ways I might go about this is by creating a scheme where the creature would be punished for not jumping through the hoop. As the training goes on, the punishment guides it to go to places where it does not receive punishment, and over time, it learns to jump through the hoop.

A similar thing happens in RL. The thing that I need to train is called an **Agent**. My language of communication with this agent is through numbers, encoded in processes that I create for it to understand and interact with the world around it. The way this agent interacts with the world around it is through **Actions (A)** and the way it understands the world is through **Observations (O)**. Now, my task is to define these actions and observations and train this agent to achieve a certain task by creating a closed-loop control of feedback for the actions it takes. This feedback is the **reward (R)** that agent receives for each of its actions. So, the key is to devise a method to guide the agent in such a way that it 'learns' to reach the goal by selecting actions with the highest **Expected Rewards (G)**, updating these values by observing the environment after taking that action. Thus, the agent first takes random actions and updates its reward values, and slowly, it starts to favor actions with higher rewards, which eventually lead to the goal.

<img width=800 height=500 src="static/Reinforcement Learning/agent-env.svg">

The way I define observations is through formalizing it as a **State (S)** in which this agent exists, or can exist. This state can either be the same as the observation, in case the agent can see everything about its environment, for example, in an extreme case imagine if you were able to see all the atoms that constitute your surroundings, or the state can be defined in terms of **Beliefs (b)** that agent the might have based on its observation. More on this in the next sections!

A standard testbed in RL, and an overly used example, is the Mountain Car scenario. As shown in the figure below, the car exists in a valley and the goal is at the top. The car needs to reach this goal by accelerating, but it is unable to reach the top by simply accelerating from the bottom. Thus, it must learn to leverage potential energy by driving up the opposite hill before the car is able to make it to the goal at the top of the rightmost hill.

<!-- ![image](Reinforcement_Learning/Pics/mountain-car.jpg) -->
<img width=700 height=400 src="static/Reinforcement Learning/mountain-car.jpg">

One way to define the values for the agent - the car - would be to define the state as the (position, velocity) of the car, the actions as (Do nothing, Push the car left, Push the car right), and rewards as -1 for each step that leads to a position that is not the goal and 0 for reaching the goal.

To characterize the agent, the following components are used in the RL vocabulary:
- **Policy $(\pi : S \rightarrow A)$:** This is the behavior of the agent that i.e the schema it follows while navigating in the environment it observes by taking actions. Thus, it is a mapping from state to action
- **Value Function (V):** This is the agent's prediction of future rewards. The way this fits into the picture is that at each step the agent predicts the rewards that it can get in the future by following a certain set of actions under a policy. This expectation of reward is what determines which actions the agent should select.
- **Model :** The agent might make a model of the world that it observes around itself. Then it can use this model to extract information that it can use to better decide the actions that the it can take. There are two types of models that are used, Reward Model and Transition
- **Reward Model :** Model to predict the next immediate reward. This is defined in terms of Expectation fo reward conditioned on a state and action : $R^{a}_{s} = \Epsilon[ R | S=s, A=a ]$
- **Transition Model :** Model to predict the next state using the dynamics of the environment. This is deinfed in terms of probability of a next state, conditioned on the current state and actions : $P^{a}_{ss'} = \Rho[ S'=s'| S=s, A=a ]$

Thus, using the above components learning can be classified into three kinds:
1. **Value-Based RL :**  In this type, the agent uses a value function to track the quality of states and thus, follows trends in the value functions. For example, in a maze with discretized boxes as steps, the agent might assign values to each step and keep updating them as it learns, and thus, end up creating a pattern where a trend of following an increase int eh value would inevitably lead to the way out of the maze
2. **Policy-Based RL:** In this case, the agent would directly work with the policy. So, in the case of the maze example, each step might be characterized by four directions in which the agent can traverse ( up, down, left, right) and for each box, the agent might assign a direction it will follow once it reaches that, and as it learns it can update these directions t create a clear path to the end of the maze
3. **Actor-Critic :** If two ideas are well-established in the scientific community, in this case, the value-based, and policy-based approach, then the next best step could be to try and merge them to get the best of both worlds. This is what the actor-critic does; it tries to merge both these ideas by splitting the model into two parts. The actor takes the state as an input and outputs the best actions by following a learned optimal policy ( policy-based learning ). The critic generates the value for this action by evaluating the value function ( value-based learning ). These both compete in a game to improve their methods and overall the agent learns to perform better.

The learning can also be distinguished based on whether the agent has a model of the world, in which case the learning is **Model-Based RL**, or whether the agent operates without a model of the world i.e **Model-Free RL**. This will be explored in more detail in the next sections.

Finally, certain paradigms are common in RL which recurs regularly and thus, it might be good to list them down:
- **Learning and Planning :** In learning the rules of the game are unknown and are learned by putting the agent in the environment. For example, I remember some people once told me how some coaches teach the basics of swimming by asking the learner to directly jump into the semi-deep water and try to move their hands and legs in a way so that they can float. Irrespective of whether this actually happens or not, if someone learned this way I could think of it as a decent enough analogy. Planning, on the other hand, is driven by a model of the rules that need to be followed, which can be used by the agent to perform a look-ahead search on the actions that it can take.
- **Exploration and Exploitation :** This is the central choice the agent needs to make every time it takes an action. At any step, it has certain information about the world and it can go on exploiting it to eventually reach a goal (maybe), but the problem is it might not know about the most optimal way to reach this goal if it just acts on the information it already has. Thus, to discover better ways of doing things, the agent can also decide to forego the path it 'knows' will get it the best reward according to its current knowledge and take a random action to see what kind of reward it gets. Thus, in doing so the agent might end up exploring other ways of solving a problem that it might not have known, which might lead to higher rewards than the path it already knows. Personally, the most tangible way I can visualize it is by thinking of a tree of decisions, and then imagining that the agent knows one way to reach the leaf nodes with the maximum reward. However, there might exist another portion of the tree that has higher rewards, but the agent might not ever go to if it greedily acts on its current rewards.
- **Prediction and Control :** Prediction is just finding a path to the goal, while control is optimizing this path to the goal. Most of the algorithms in RL can be distinguished based on this.

# LIS: Setting up RAI on HPC

## List of RPMs:
```
- ann-devel-1.1.2-3.el7.x86_64.rpm        
- gflags-2.1.1-6.el7.x86_64.rpm          
- jsoncpp-0.10.5-2.el7.x86_64.rpm      
- poly2tri-0.0-10.20130501hg26242d0aa7b8.el7.x86_64.rpm
- assimp-devel-3.1.1-2.el7.x86_64.rpm             
- gflags-devel-2.1.1-6.el7.x86_64.rpm    
- lapack-3.4.2-8.el7.x86_64.rpm        
- proj-4.8.0-4.el7.x86_64.rpm
- atlas-3.10.1-12.el7.x86_64.rpm                  
- glfw-3.2.1-2.el7.x86_64.rpm            
- libann-1.1.2-alt5.x86_64.rpm         
- pybind11-devel-2.4.3-2.el8.aarch64.rpm
- atlas-devel-3.10.1-12.el7.x86_64.rpm            
- glibc-2.17-307.el7.1.x86_64.rpm        
- libassimp3-3.3.1-alt1_5.x86_64.rpm   
- qhull-2003.1-20.el7.x86_64.rpm
- ceres-solver-1.12.0-5.el7.x86_64.rpm            
- glibc-devel-2.17-307.el7.1.x86_64.rpm  
- libgcc-4.8.5-39.el7.x86_64.rpm       
- suitesparse-4.0.2-10.el7.x86_64.rpm
- ceres-solver-devel-1.12.0-5.el7.x86_64.rpm      
- glibc-static-2.17-307.el7.1.i686.rpm   
- libgeotiff-1.2.5-14.el7.x86_64.rpm   
- suitesparse-devel-4.0.2-10.el7.x86_64.rpm
- ceres-solver-devel-1.13.0-12.el8.x86_64.rpm     
- glog-0.3.3-8.el7.x86_64.rpm            
- libGLEW-1.10.0-5.el7.x86_64.rpm      
- tbb-4.1-9.20130314.el7.x86_64.rpm
- eigen3-devel-3.3.4-6.el7.noarch.rpm             
- glog-devel-0.3.3-8.el7.x86_64.rpm      
- libstdc++-4.8.5-39.el7.x86_64.rpm    
- zlib-1.2.7-18.el7.x86_64.rpm
- f2c-20160102-1.el7.x86_64.rpm                   
- gnuplot-4.6.2-3.el7.x86_64.rpm         
- libX11-1.6.7-2.el7.x86_64.rpm
- freeglut-3.0.0-8.el7.x86_64.rpm                 
- graphviz-2.30.1-21.el7.x86_64.rpm      
- libX11-devel-1.6.7-2.el7.x86_64.rpm
- gcc-x86_64-linux-gnu-4.8.5-16.el7.1.x86_64.rpm  
- irrXML-1.8.1-3.el7.2.x86_64.rpm        
- minizip-1.2.7-18.el7.x86_64.rpm
```
## Initial Set-up:
```
- module load comp/gcc/7.2.0  
- module load python/3.6.8_tf-cpu
- module load nvidia/cuda/9.2.88
- export C_INCLUDE_PATH=$C_INCLUDE_PATH:$HOME/usr/include:$HOME/usr/local/bin/include/python3.6m
- export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$HOME/usr/include:$HOME/usr/local/bin/include/python3.6m
- export PATH=$PATH:$HOME/usr/local/bin/bin
- alias python3=python3.6
```
## Problems: 
These were the list of problems faced during the set-up

### Eigen and Assimp Issue (Type = Not Linked )
  - ln -sf eigen3/Eigen Eigen
  - ln -sf assimp/ Assimp 
### Cannot find -ljsoncpp and cannot find -llapack ( Type = .so file not present)
```
- ld -ljsoncpp --verbose
- ld -llapack --verbose
- Re-installed the rpm, the normal binary, and not the devel one, and set the symbolic link to what was being searched for 

- Added to generic.mk the following:
    CXXFLAGS += -L/home/users/a/amsks1996/usr/lib64
    CXXFLAGS += -Wl,-rpath=/home/users/a/amsks1996/usr/lib64

- Symbolic Links $HOME/usr/lib64
- ln -sf libjsoncpp.so.0 libjsoncpp.so
- ln -sf liblapack.so.3 liblapack.so
```
-  Assimp issue ( Type = include ) : download the devel rpm and place the files in the include folder

- Ceres not found ( Type = include ) : download the devel rpm adn place the files in the include folder

-  glog/logging.h not found ( Type = include ): Downloaded the rpm for 

- gflags/gflags.h ( Type = include ) : same 

-  qhull/qhull_a.h ( Type = include ) : Same for qhull-devel-2003.1-20.el7.x86_64.rpm 

-  **-lglew**, **-lqhull**, **-lGeo**, **-lglfw**, **-lcgraph**, **-lgvc**     **-ljsoncpp** **-llapack**  -lOptim  **-lann** : Compiled from the code

### libspqr.so libtbbmalloc.so libtbb.so libcholmod.so libccolamd.so libcamd.so libcolamd.so libamd.so liblapack.so libf77blas.so libatlas.so libsuitesparseconfig.so librt.so libcxsparse.so liblapack.so libf77blas.so libatlas.so libsuitesparseconfig.so librt.so libcxsparse.so libgflags.so.2.2.1 libglog.so 

These are related to the Ceres solver, and are defined in rai/buil/defines.ml

- Change the location to the usr/home

```
-lceres -lglog -lcholmod -llapack -lblas -lpthread  $(HOME)/usr/lib/libspqr.so $(HOME)/usr/lib/libtbbmalloc.so $(HOME)/usr/lib/libtbb.so $(HOME)/usr/lib/libcholmod.so $(HOME)/usr/lib/libccolamd.so $(HOME)/usr/lib/libcamd.so $(HOME)/usr/lib/libcolamd.so $(HOME)/usr/lib/libamd.so $(HOME)/usr/lib/liblapack.so $(HOME)/usr/lib/libf77blas.so $(HOME)/usr/lib/libatlas.so $(HOME)/usr/lib/libsuitesparseconfig.so $(HOME)/usr/lib/librt.so $(HOME)/usr/lib/libcxsparse.so $(HOME)/usr/lib/liblapack.so $(HOME)/usr/lib/libf77blas.so $(HOME)/usr/lib/libatlas.so $(HOME)/usr/lib/libsuitesparseconfig.so $(HOME)/usr/lib/librt.so $(HOME)/usr/lib/libcxsparse.so $(HOME)/usr/lib/libgflags.so.2.2.1 -lpthread $(HOME)/usr/lib/libglog.so"
```
- Recursively the dependencies


### Random Shared Issue 

g++ -L/home/users/a/amsks1996/git/ceres-solver/build/lib -L/home/users/a/amsks1996/opt/physx3.4/lib -L/beegfs/home/users/a/amsks1996/git/rai-python/rai/lib -L/home/users/a/amsks1996/opt/lib -L/usr/local/lib -L/home/users/a/amsks1996/usr/lib64 -L/home/users/a/amsks1996/usr/include -L/home/users/a/amsks1996/usr/bin  -o libOptim.so ./BayesOpt.o ./GlobalIterativeNewton.o ./GraphOptim.o ./Graph_Problem.o ./KOMO_Problem.o ./RidgeRegression.o ./benchmarks.o ./constrained.o ./convert.o ./gradient.o ./lagrangian.o ./newOptim.o ./newton.o ./opt-ceres.o ./optimization.o ./primalDual.o -lCore -lceres -lglog -lcholmod -llapack -lblas -lpthread  /usr/lib/x86_64-linux-gnu/libspqr.so /usr/lib/x86_64-linux-gnu/libtbbmalloc.so /usr/lib/x86_64-linux-gnu/libtbb.so /usr/lib/x86_64-linux-gnu/libcholmod.so /usr/lib/x86_64-linux-gnu/libccolamd.so /usr/lib/x86_64-linux-gnu/libcamd.so /usr/lib/x86_64-linux-gnu/libcolamd.so /usr/lib/x86_64-linux-gnu/libamd.so /usr/lib/x86_64-linux-gnu/liblapack.so /usr/lib/x86_64-linux-gnu/libf77blas.so /usr/lib/x86_64-linux-gnu/libatlas.so /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so /usr/lib/x86_64-linux-gnu/librt.so /usr/lib/x86_64-linux-gnu/libcxsparse.so /usr/lib/x86_64-linux-gnu/liblapack.so /usr/lib/x86_64-linux-gnu/libf77blas.so /usr/lib/x86_64-linux-gnu/libatlas.so /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so /usr/lib/x86_64-linux-gnu/librt.so /usr/lib/x86_64-linux-gnu/libcxsparse.so /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1 -lpthread /usr/lib/x86_64-linux-gnu/libglog.so -lpthread -lrt -lPhysX3Extensions -lPhysX3_x64 -lPhysX3Cooking_x64 -lPhysX3Common_x64 -lPxFoundation_x64 -lBulletSoftBody -lBulletDynamics -lBulletCollision  -lLinearMath -lrt -shared


### Path Variable Backup 

```
/beegfs/home/cluster/python/3.6.8/bin:/cluster/comp/binutils/2.29/bin:/cluster/comp/gcc/7.2.0/bin:/home/users/a/amsks1996/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/home/users/a/amsks1996/.local/bin:/home/users/a/amsks1996/bin:/cluster/cuda/cuda-9.2.88//bin/:/cluster/cuda/cuda-9.2.88//open64/bin/:/cluster/cuda/cuda-9.2.88//nvvm/:/cluster/cuda/cuda-9.2.88//samples/bin/x86_64/linux/release/:/home/users/a/amsks1996/usr/include
```

### Linking step error

AAA -lceres -lglog -lcholmod -llapack -lblas -lpthread  $(HOME)/usr/lib/libspqr.so $(HOME)/usr/lib/libtbbmalloc.so $(HOME)/usr/lib/libtbb.so $(HOME)/usr/lib/libcholmod.so $(HOME)/usr/lib/libccolamd.so $(HOME)/usr/lib/libcamd.so $(HOME)/usr/lib/libcolamd.so $(HOME)/usr/lib/libamd.so $(HOME)/usr/lib/liblapack.so $(HOME)/usr/lib/libf77blas.so $(HOME)/usr/lib/libatlas.so $(HOME)/usr/lib/libsuitesparseconfig.so $(HOME)/usr/lib/librt.so $(HOME)/usr/lib/libcxsparse.so $(HOME)/usr/lib/liblapack.so $(HOME)/usr/lib/libf77blas.so $(HOME)/usr/lib/libatlas.so $(HOME)/usr/lib/libsuitesparseconfig.so $(HOME)/usr/lib/librt.so $(HOME)/usr/lib/libcxsparse.so $(HOME)/usr/lib/libgflags.so.2.2.1 -lpthread $(HOME)/usr/lib/libglog.so -lceres -lglog -cholmod ld: final link failed: Nonrepresentable section on output

### PYTHON not shareable issue 

python is not built with share option

```
Re-install python 3.6.8 from source locally and build the project with that. 

Commands: 
    - curl -O https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz
    -  tar -xzf Python-3.6.8.tgz
    -  cd Python-3.6.8/
    -  ./configure --enable-optimizations --enable-shared --prefix=$HOME/usr/local/bin/
    -  make altinstall

```
### Python Linking Issue
Have to specify the path variables and  link the stuff

```
g++ -g -O3 -Wall -DRAI_PYBIND `python3-config --cflags` -DRAI_PHYSX -D_DEBUG -DPX_DISABLE_FLUIDS -DCORELIB -DPX32 -DLINUX -DRAI_BULLET -DBT_USE_DOUBLE_PRECISION -Wno-terminate -fPIC -static -L /home/users/a/amsks1996/usr/lib64/ -std=c++14 -L/home/users/a/amsks1996/usr/lib64 -o ry-Feature.o -c ry-Feature.cpp
/bin/sh: python3-config: command not found
        - Solved via adding python to the C_INCLUDE_PATH, and CPLUS_INCLUDE_PATH , and creating alias
```

## libassimp.so issue (Total Bitch!)

```
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `std::__cxx11::basic_stringstream<char, std::char_traits<char>, std::allocator<char> >::basic_stringstream()@GLIBCXX_3.4.26'
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `p2t::CDT::CDT(std::vector<p2t::Point*, std::allocator<p2t::Point*> > const&)'
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `std::__cxx11::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::basic_ostringstream()@GLIBCXX_3.4.26'
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `log@GLIBC_2.29'
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `powf@GLIBC_2.27'
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `pow@GLIBC_2.29'
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `p2t::CDT::AddHole(std::vector<p2t::Point*, std::allocator<p2t::Point*> > const&)'

libIrrXML.so.1()(64bit) -
libc.so.6(GLIBC_2.14)(64bit) ------ found libc.so.6 at /home/users/a/amsks1996/usr/lib64//libc.so.6
libgcc_s.so.1()(64bit) ------  found libgcc_s.so.1 at /cluster/comp/gcc/7.2.0/lib64/libgcc_s.so.1
libgcc_s.so.1(GCC_3.0)(64bit) ------  ound libgcc_s.so.1 at /cluster/comp/gcc/7.2.0/lib64/libgcc_s.so.1
libm.so.6()(64bit)  ------   found libm.so.6 at /home/users/a/amsks1996/usr/lib64//libm.so.6
libm.so.6(GLIBC_2.2.5)(64bit)   - 
libminizip.so.1()(64bit) ------  found libminizip.so.1 at /home/users/a/amsks1996/usr/lib64//libminizip.so.1
libpoly2tri.so.1.0()(64bit) ------- found libpoly2tri.so.1.0 at /home/users/a/amsks1996/usr/lib64//libpoly2tri.so.1.0
libpthread.so.0()(64bit)    ------- found libpthread.so.0 at /home/users/a/amsks1996/usr/lib64//libpthread.so.0
libpthread.so.0(GLIBC_2.2.5)(64bit) -
libstdc++.so.6()(64bit) ------ found libstdc++.so.6 at /cluster/comp/gcc/7.2.0/lib64/libstdc++.so.6
libstdc++.so.6(CXXABI_1.3)(64bit)   -
libstdc++.so.6(GLIBCXX_3.4)(64bit)  -
libstdc++.so.6(GLIBCXX_3.4.11)(64bit)   -
libstdc++.so.6(GLIBCXX_3.4.15)(64bit)   -
libstdc++.so.6(GLIBCXX_3.4.9)(64bit)    -
libz.so.1()(64bit)  -
rtld(GNU_HASH)



attempt to open ./libassimp.so succeeded
-lassimp (./libassimp.so)
libz.so.1 needed by ./libassimp.so --->  found libz.so.1 at /home/users/a/amsks1996/usr/lib64//libz.so.1
libminizip.so.1 needed by ./libassimp.so ---> found libminizip.so.1 at /home/users/a/amsks1996/usr/lib64//libminizip.so.1
libpoly2tri.so.1.0 needed by ./libassimp.so ---> found libpoly2tri.so.1.0 at /home/users/a/amsks1996/usr/lib64//libpoly2tri.so.1.0
libstdc++.so.6 needed by ./libassimp.so ---> found libstdc++.so.6 at /home/users/a/amsks1996/usr/lib64//libstdc++.so.6
libm.so.6 needed by ./libassimp.so ---> found libm.so.6 at /home/users/a/amsks1996/usr/lib64//libm.so.6
libgcc_s.so.1 needed by ./libassimp.so ---> found libgcc_s.so.1 at /home/users/a/amsks1996/usr/lib64//libgcc_s.so.1
libc.so.6 needed by ./libassimp.so ---> found libc.so.6 at /home/users/a/amsks1996/usr/lib64//libc.so.6
libGL.so.1 needed by /home/users/a/amsks1996/usr/lib64//libpoly2tri.so.1.0 ---> found libGL.so.1 at /usr/lib64/libGL.so.1
ld-linux-x86-64.so.2 needed by /home/users/a/amsks1996/usr/lib64//libstdc++.so.6 ---> found ld-linux-x86-64.so.2 at /home/users/a/amsks1996/usr/lib64//ld-linux-x86-64.so.2


libexpat.so.1 needed by /usr/lib64/libGL.so.1 ---> found libexpat.so.1 at /usr/lib64/libexpat.so.1
libxcb-dri3.so.0 needed by /usr/lib64/libGL.so.1 ---> found libxcb-dri3.so.0 at /usr/lib64/libxcb-dri3.so.0
libxcb-present.so.0 needed by /usr/lib64/libGL.so.1 ---> found libxcb-present.so.0 at /usr/lib64/libxcb-present.so.0
libxcb-sync.so.1 needed by /usr/lib64/libGL.so.1 ---> found libxcb-sync.so.1 at /usr/lib64/libxcb-sync.so.1
libxshmfence.so.1 needed by /usr/lib64/libGL.so.1 ---> found libxshmfence.so.1 at /usr/lib64/libxshmfence.so.1
libglapi.so.0 needed by /usr/lib64/libGL.so.1 ---> found libglapi.so.0 at /usr/lib64/libglapi.so.0
libselinux.so.1 needed by /usr/lib64/libGL.so.1 ---> found libselinux.so.1 at /usr/lib64/libselinux.so.1
libXext.so.6 needed by /usr/lib64/libGL.so.1 ---> found libXext.so.6 at /usr/lib64/libXext.so.6
libXdamage.so.1 needed by /usr/lib64/libGL.so.1 ---> found libXdamage.so.1 at /usr/lib64/libXdamage.so.1
libXfixes.so.3 needed by /usr/lib64/libGL.so.1 ---> found libXfixes.so.3 at /usr/lib64/libXfixes.so.3
libX11-xcb.so.1 needed by /usr/lib64/libGL.so.1 ---> found libX11-xcb.so.1 at /home/users/a/amsks1996/usr/lib64//libX11-xcb.so.1
libX11.so.6 needed by /usr/lib64/libGL.so.1 ---> found libX11.so.6 at /home/users/a/amsks1996/usr/lib64//libX11.so.6
libxcb.so.1 needed by /usr/lib64/libGL.so.1 ---> found libxcb.so.1 at /usr/lib64/libxcb.so.1
libxcb-glx.so.0 needed by /usr/lib64/libGL.so.1 ---> found libxcb-glx.so.0 at /usr/lib64/libxcb-glx.so.0
libxcb-dri2.so.0 needed by /usr/lib64/libGL.so.1 ---> found libxcb-dri2.so.0 at /usr/lib64/libxcb-dri2.so.0
libXxf86vm.so.1 needed by /usr/lib64/libGL.so.1 ---> found libXxf86vm.so.1 at /usr/lib64/libXxf86vm.so.1
libdrm.so.2 needed by /usr/lib64/libGL.so.1 ---> found libdrm.so.2 at /usr/lib64/libdrm.so.2
libpthread.so.0 needed by /usr/lib64/libGL.so.1 ---> found libpthread.so.0 at /home/users/a/amsks1996/usr/lib64//libpthread.so.0
libdl.so.2 needed by /usr/lib64/libGL.so.1 ---> found libdl.so.2 at /home/users/a/amsks1996/usr/lib64//libdl.so.2
libXau.so.6 needed by /usr/lib64/libxcb-dri3.so.0 ---> found libXau.so.6 at /usr/lib64/libXau.so.6
libpcre.so.1 needed by /usr/lib64/libselinux.so.1 ---> found libpcre.so.1 at /usr/lib64/libpcre.so.1


ld: warning: cannot find entry symbol _start; not setting start address
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `p2t::CDT::AddHole(std::vector<p2t::Point*, std::allocator<p2t::Point*> > const&)'
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `log@GLIBC_2.29'
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `std::__cxx11::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::basic_ostringstream()@GLIBCXX_3.4.26'
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `p2t::CDT::CDT(std::vector<p2t::Point*, std::allocator<p2t::Point*> > const&)'
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `std::__cxx11::basic_stringstream<char, std::char_traits<char>, std::allocator<char> >::basic_stringstream()@GLIBCXX_3.4.26'
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `powf@GLIBC_2.27'
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `pow@GLIBC_2.29'
```

### Relocation Error : Third Party Software

```
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/usr/lib64


readlink -f /path/file


/beegfs/home/cluster/comp/gcc/7.2.0/lib64/libstdc++.so.6.0.24

```




# Misc: Setting up Envrironment

#### Installing VS Code 

- Install the .deb link from 

```
https://code.visualstudio.com/Download
```
- Navigate to the Downloads folder

```
cd ~/Downloads
```
- Install using dpkg 

```
sudo dpkg -i Name_of_file
```

#### Installing Sublime Text

- Open Terminal and install the key

```
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
```

- Add the apt repository

```
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
```

- Finally, check updates and install sublime-text via apt

```
sudo apt update

sudo apt install sublime-text
```

#### Installing CUDA

- Remove all NVIDIA traces in the system

```
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*
```

- Setup the correct CUDA PPA on the system

```
sudo apt update
sudo add-apt-repository ppa:graphics-drivers

sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
```

- Install CUDA 10.1 packages

```
sudo apt update
sudo apt install cuda-10-1
sudo apt install libcudnn7
```

- To specify PATH to CUDA in ‘.profile’ file, open it :

```
sudo gedit ~/.profile
```

- Then add this to the end of the file

```
# set PATH for cuda 10.1 installation
if [ -d "/usr/local/cuda-10.1/bin/" ]; then
    export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```

- Restart and check the versions for the installation. For CUDA, NVIDIA and libcudnn

```
nvcc  – version

nvidia-smi

/sbin/ldconfig -N -v $(sed ‘s/:/ /’ <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn
```

#### Setting up Virtual Environment for DL

- Install virtual env using apt

```
sudo apt update

sudo apt install virtualenv
```

- Install virtualenvwrapper

```
sudo apt install virtualenvwrapper
```

- Check the installation paths ( Should be in /usr/bin/ )

```
which virtualenv

which virtualenvwrapper
```

- Create the new environment for keras and tensorflow

```
mkvirtualenv keras_tf -p python3
```

- Check if the global commands work

```
workon keras_tf 

deactivate
```

- Install and Check tf

```
pip install --upgrade tensorflow

python
>>> import tensorflow as tf
>>> tf.__version__

```

- Install Keras related dependencies

```
pip install numpy scipy
pip install scikit-learn
pip install pillow
pip install h5py
```

- Install keras

```
pip install keras
```

#### Setting up Jekyll for local website
- First check if ruby and gem are already installed on your system

```
ruby -v
gem -v 
```

- if no, then install ruby

```
sudo apt-get install ruby-full
```

- Install other dependencies

```
build-essential zlib1g-dev
```

- Configure gem installation related stuff in the bashrc

```
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

- Install jeyll and bundler

```
gem install jekyll bundler
```

- Clone the github pages repo and navigate to it. Then make an orphan branch

```
git checkout --orphan gh-pages
```

- To create a new Jekyll site, use the jekyll new command, replacing VERSION with the current dependency version for Jekyll

```
bundle exec jekyll VERSION new .
```

- Update the gemfile with the sources

```
gem "github-pages", "~> VERSION", group: :jekyll_plugins
```

- Check if any other related dependencies are missing

```
bundle install
```

- Run the localhost:4000 website

```
bundle exec jekyll serve
```