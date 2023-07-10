---
layout: post
title: Introduction to Reinforcement learning
date: 2023-07-09 08:57:00-0400
categories: reinforcement-learning
giscus_comments: false
related_posts: false
---

One way to look at the behavior of an organism is by looking at how it interacts with its environment, and how this interaction allows it to behave differently over time through the process of learning. In this view, the behavior of the organism can be modeled in a closed-loop manner through a fundamental description of the action and sensation loop. It receives input - sensation - from the environment through sensors, acts on the environment through actuators, and observes the impact of its action on its understanding of the environment through a mechanism of quantification in the form of the rewards it receives for its action. A similar thing happens in RL. The thing that we need to train is called an Agent. The language of communication with this agent is through numbers, encoded in processes that we create for it to understand and interact with the world around it. The way this agent interacts with the world around it is through Actions (A) and the way it understands the world is through Observations (O). Now, our task is to define these actions and observations and train this agent to achieve a certain task by creating a closed-loop control of feedback for the actions it takes. This feedback is the Reward (R) that agent receives for each of its actions. So, the key is to devise a method to guide the agent in such a way that it 'learns' to reach the goal by selecting actions with the highest Expected Rewards (G), updating these values by observing the environment after taking that action. Thus, the agent first takes random actions and updates its reward values, and slowly, it starts to favor actions with higher rewards, which eventually lead to the goal.

<div class="col-sm">
    {% include figure.html path="assets/img/Reinforcement-Learning/agent-env.svg" class="img-centered rounded z-depth-0" %}
</div>


The way we define observations is through formalizing it as a **State (S)** in which this agent exists, or can exist. This state can either be the same as the observation, in case the agent can see everything about its environment, for example, in an extreme case imagine if you were able to see all the atoms that constitute your surroundings, or the state can be defined in terms of **Beliefs (b)** that agent the might have based on its observation. this distinction is important for problems of partial observability, a topic for the future. A standard testbed in RL is the Mountain Car scenario. As shown in the figure below, the car exists in a valley and the goal is at the top. The car needs to reach this goal by accelerating, but it is unable to reach the top by simply accelerating from the bottom. Thus, it must learn to leverage potential energy by driving up the opposite hill before the car is able to make it to the goal at the top of the rightmost hill.

<!-- <img width=700 height=400 src="static/Reinforcement Learning/mountain-car.jpg"> -->
<div class="col-sm">
    {% include figure.html path="assets/img/Reinforcement-Learning/mountain-car.jpg" class="img-centered rounded z-depth-0" %}
</div>

One way to define the values for the agent - the car - would be to define the state as the (position, velocity) of the car, the actions as (Do nothing, Push the car left, Push the car right), and rewards as -1 for each step that leads to a position that is not the goal and 0 for reaching the goal. To characterize the agent, the following components are used in the RL vocabulary:

- **Policy ($$\pi: S \rightarrow A$$):** This is the behavior of the agent that i.e the schema it follows while navigating in the environment it observes by taking actions. Thus, it is a mapping from state to action
- **Value Function (V):** This is the agent's prediction of future rewards. The way this fits into the picture is that at each step the agent predicts the rewards that it can get in the future by following a certain set of actions under a policy. This expectation of reward is what determines which actions the agent should select.
- **Model:** The agent might make a model of the world that it observes around itself. Then it can use this model to extract information that it can use to better decide the actions that it can take. There are two types of models that are used, Reward Model and Transition
- **Reward Model:** Model to predict the next immediate reward. This is defined in terms of Expectation fo reward conditioned on a state and action :

$$
R^{a}_{s} = \mathbb{E}[ R | S=s, A=a ]
$$

- **Transition Model:** Model to predict the next state using the dynamics of the environment. This is defined in terms of probability of a next state, conditioned on the current state and actions :

    $$P^{a}_{ss'} = \mathbb{P}[ S'=s'| S=s, A=a ]$$

Thus, using the above components learning can be classified into three kinds:

1. **Value-Based RL:**  In this type, the agent uses a value function to track the quality of states and thus, follows trends in the value functions. For example, in a maze with discretized boxes as steps, the agent might assign values to each step and keep updating them as it learns, and thus, end up creating a pattern where a trend of following an increase in the value would inevitably lead to the way out of the maze
2. **Policy-Based RL:** In this case, the agent would directly work with the policy. So, in the case of the maze example, each step might be characterized by four directions in which the agent can traverse (up, down, left, right) and for each box, the agent might assign a direction it will follow once it reaches that, and as it learns it can update these directions t create a clear path to the end of the maze
3. **Actor-Critic:** If two ideas are well-established in the scientific community, in this case, the value-based, and policy-based approach, then the next best step could be to try and merge them to get the best of both worlds. This is what the actor-critic does; it tries to merge both these ideas by splitting the model into two parts. The actor takes the state as an input and outputs the best actions by following a learned optimal policy (policy-based learning). The critic generates the value for this action by evaluating the value function ( value-based learning). These both compete in a game to improve their methods and overall the agent learns to perform better.

The learning can also be distinguished based on whether the agent has a model of the world, in which case the learning is **Model-Based RL**, or whether the agent operates without a model of the world i.e **Model-Free RL**.  This will be explored in more detail in the next sections. Finally, certain paradigms are common in RL which recurs regularly, and thus, it might be good to list them down:

- **Learning and Planning:** In learning the rules of the game are unknown and are learned by putting the agent in the environment. For example, I remember some people once told me how some coaches teach the basics of swimming by asking the learner to directly jump into the semi-deep water and try to move their hands and legs in a way so that they can float. Irrespective of whether this actually happens or not, if someone learned this way I could think of it as a decent enough analogy. Planning, on the other hand, is driven by a model of the rules that need to be followed, which can be used by the agent to perform a look-ahead search on the actions that it can take.
- **Exploration and Exploitation:** This is the central choice the agent needs to make every time it takes an action. At any step, it has certain information about the world and it can go on exploiting it to eventually reach a goal (maybe), but the problem is it might not know about the most optimal way to reach this goal if it just acts on the information it already has. Thus, to discover better ways of doing things, the agent can also decide to forego the path it 'knows' will get the best reward according to its current knowledge and take a random action to see what kind of reward it gets. Thus, in doing so the agent might end up exploring other ways of solving a problem that it might not have known, which might lead to higher rewards than the path it already knows. Personally, the most tangible way I can visualize it is by thinking of a tree of decisions, and then imagining that the agent knows one way to reach the leaf nodes with the maximum reward. However, there might exist another portion of the tree that has higher rewards, but the agent might not ever go to if it greedily acts on its current rewards.
- **Prediction and Control:** Prediction is just finding a path to the goal, while control is optimizing this path to the goal. Most of the algorithms in RL can be distinguished based on this.
