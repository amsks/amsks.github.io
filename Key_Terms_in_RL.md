# Explanations of Key Terms used in Reinforcement Learning Vocabulary

## MDP
A **Markov Decision Process (MDP)** is a way to model decision making in discrete, stochastic and/or sequential processes. The key here is the markov property, which basically states that the probability of enbtering a certain state, when conditioned upon the previous state is same as the probability of entering that state when conditioned on all the history of states. The significance here is that if we had state A and state B and A precedes B, then we can assume that all the states that preceded B in an episode can be  ignored, and A can just be conditioned on B because of the way that B is linked to all previous states. So, we don't really need to look at all the history since the sequence of transitions makes sure that there would be a particular way to reach B, and thus, A. If we assume this about the environment, then we can create a map of transitions between states as our model of the environment and then use that to assign rewards for reaching a state. This is a markov reward process. 




## Markov Chain Monte-Carlo

## Parameters of agent and environment
* Reward
* Sate
* Policy
* Discounting factor
* Trajectory
* State-space
* Tansition function

## Bellman Equation

## Dynamic Programming

##  Value function

## Policy Gradient

## Q-Learning

## Model-Based and Model-Free Learning

## Exploration and Exploitation

## Common RL types
* Inverse RL
* Imitation RL
* APprenticeship learning
* Meta Learning


## Reward Shaping

## Actor-Critic

## Monte-Carlo Tree search

## Human in the loop

## Deep RL

## Differentiable AI

## Zero-Shot, One-Shot, Few-shot
