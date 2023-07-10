---
layout: post
title: Policy Gradients
date: 2023-07-10 12:57:00-0400
categories: reinforcement-learning
giscus_comments: false
related_posts: false
---

The core idea of Reinforcement learning is to learn some kind of behavior through optimizing for rewards. The behavior learned by an agent i.e. the schema it follows while going through this process is the learned policy that it uses to decide which action to take and thus, the transition from one state to another. One way to close the loop for the agent to learn is by evaluating the states and actions through value functions and thus, our way to measure the learned policy is seen through these value functions, approximated by lookup tables, linear combinations, Neural Networks e.t.c. Policy Gradient methods take a different approach where they bypass the need for a value function by parameterizing the policy directly. While the agent can still use a value function to learn, it need not use it for selecting actions. The advantages that policy gradient methods offer are 3 fold: 

1. Approximating the policy might be simpler than approximating action values and a policy-based method might typically learn faster and yield a superior asymptotic policy. One very good example that illustrates this is the work by [Simsek et. al](http://proceedings.mlr.press/v48/simsek16.pdf) on the game of Tetris where they showed that it is possible to choose amongst actions without really evaluating them.
2. Policy gradient methods can handle stochastic policies. The case of card games with imperfect information, like poker, is a direct example where the optimal play might be to do 2 different things with certain probabilities. If we are maximizing the actions based on value approximations, we don't really have a natural way of finding stochastic policies. Policy Gradient methods can do this.
3. Policy gradient methods offer stronger convergence guarantees since with continuous policies the action probabilities change smoothly. This is not the case with the fixed $$\epsilon$$ -greedy evaluation since there is always a probability to do something random.  
4. The choice of policy parameterization is sometimes a good way of injecting prior knowledge about a desired form of the policy into the system. This is especially helpful when we look at introducing Meta-Learning strategies into Reinforcement Learning.

In the following sections, I first use the theoretical treatment done in Sutton and Barto's book since I was better able to understand policy gradients' essence through this. However, I again do the derivation by looking at the whole thing from the viewpoint of trajectories, since I find it more intuitive.

## Policy Gradient Theorem

The issue with the parameterization of the policy is that the policy affects both the action selections and the distribution of states in which those selections are made. While going from state to action is straightforward, going the other way round involves the environment and thus, the parameterization is typically unknown. Thus, with this unknown effect of policy changes on the state distributions, the issue is evaluating the gradients of the performance. This is where the policy gradient theorem comes into the picture, as it shows that the gradient of the policy w.r.t its parameters does not involve the derivative of the state distribution. For episodic tasks, if we assume that every episode starts in some particular state $$s_0$$, then we can write a performance measure as the value of the start state of the episode

$$J(\mathbf{\theta}) = v_{\pi_\theta} (s_0)$$

For simplicity, we remove the $$\bf{\theta}$$  from the subscript of $$\pi$$. Now, to get a derivative of this measure, we start by differentiating this value function w.r.t $$\mathbf{\theta}$$:

 

$$
\begin{aligned}
\nabla_{\mathbf{\theta}} J(\mathbf{\theta}) & =  \nabla v_\pi (s) \\
& = \nabla \bigg [ \sum_{a \in \mathcal{A}}\pi(a|s) q_\pi(s,a)   \bigg ] \\
& = \sum_a \bigg[ \nabla\pi(a|s) q_\pi (s, a)  + \pi(a|s) \nabla q_\pi(s,a)    \bigg ] \\
& = \sum_a \bigg[ \nabla\pi(a|s) q_\pi (s, a)  + \pi(a|s) \,\, \nabla \sum _{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a) ( r + v_\pi (s') )     \bigg ]
\end{aligned}
$$

we now extend $$q_\pi (s,a)$$ in the second term on the right to the rollout for the new state $$s'$$.

$$
\nabla v_\pi (s) = \sum_a \bigg[ \nabla\pi(a|s) q_\pi (s, a)  + \pi(a|s) \,\, \nabla \sum _{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a) ( r + v_\pi (s') ) \bigg ] 
$$

The reward is independent of the parameters $\theta$, so we can set that derivative inside the sum to 0, and so, we get:

$$
\nabla v_\pi (s) = \sum_a \bigg[ \nabla\pi(a|s) q_\pi (s, a)  + \pi(a|s) \,\, \nabla \sum _{s' \in \mathcal{S} } p(s' | s, a) v_\pi (s') \bigg ] 
$$

Thus, we now have a recursive formulation of $$\nabla v_\pi (s)$$ in terms of $$\nabla v_\pi (s')$$. To calculate this derivative in the infinite horizon episodic case we just need to unroll this infinitely many times, which can be written as 

$$\sum_{x \in \mathcal{s}} \sum _{k=0}^\infty P(s \rightarrow x, k , \pi )  \sum_{a \in \mathcal{A}} \nabla \pi (a|s)  q_\pi (x, a) $$

Here, $$P(s \rightarrow x, k, \pi )$$ is the probability of transitioning from state $$s$$ to state $$x$$ in $$k$$  steps under policy $$\pi$$. To estimate this probability we use something called the stationary distribution of the Markov chains. This term comes from the [Fundamental Theorem of Markov Chains](http://www.math.uchicago.edu/~may/VIGRE/VIGRE2008/REUPapers/Plavnick.pdf) which intuitively says that in very long random walks the probability of ending up at some state is independent of where you started. When we club all these probabilities into a distribution over the states, then we have a stationary distribution, denoted by $$\mu(s)$$. In on-policy training, we usually estimate this distribution by the fraction of time spent in a state. In our case of episodic tasks, if we let $$\eta(s)$$ denote the total time spent in a state in an episode, then we can calculate $$\mu(s)$$ as 

$$\mu(s) = \frac{\eta(s)}{\sum_s \eta(s)}$$

In our derivation, $$P(s \rightarrow x, k, \pi )$$  for very long walks can be estimated by the total time spent in the state $$s$$. Thus, we can inject $$\eta(s)$$  into our equation as follows:

$$\begin{aligned}
\nabla_{\mathbf{\theta}}J(\mathbf{\theta})  & = \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla\pi(a|s)  q_\pi(s,a) \\
& = \sum_{s \in \mathcal{S}} \eta(s) \sum_{s \in \mathcal{S}} \frac{\eta(s)}{\sum_{s \in \mathcal{S}} \eta(s)} \sum_{a \in \mathcal{A}} \nabla\pi(a|s)  q_\pi(s,a) \\
& \propto \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}} \nabla\pi(a|s)  q_\pi(s,a) 
\end{aligned}$$

Thus, we get the form of the theorem as 

$$\nabla_{\mathbf{\theta}}J(\mathbf{\theta}) \propto \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}}  q_\pi(s,a) \nabla\pi(a|s) $$

The proportionality is the average length of an episode. In the case of continuous tasks, this is 1. Thus, now we see that we have a gradient over a parameterized policy, which allows us to move in the direction of maximizing this gradient i.e gradient ascent. We can estimate this gradient through different means.

## REINFORCE: Monte-Carlo Sampling

For Monte-Carlo sampling of the policy, our essential requirement is sampling from a distribution that allows us to get an estimate of the policy. From the equation of the policy gradient theorem, we can write this again as an expectation over a sample of states $$S_t \sim s$$ in the direction of the policy gradient

$$\nabla_{\mathbf{\theta}}J(\mathbf{\theta})  = \mathbb{E}_\pi \bigg [\sum_{a \in \mathcal{A}}  q_\pi(S_t,a) \nabla_{\mathbf{\theta}} \pi(a|S_t, \mathbf{\theta}) \bigg ]$$

The expectation above would be an expectation over the actions if we were to include the probability of selecting the actions as the weight. Thus, we can do that to remove the sum over actions too:

$$\begin{aligned}
\nabla_{\mathbf{\theta}}J(\mathbf{\theta})  & = \mathbb{E}_\pi \bigg [\sum_{a \in \mathcal{A}}  q_\pi(S_t,a) \nabla_{\mathbf{\theta}} \pi(a|S_t, \mathbf{\theta}) \frac{\pi(a|S_t, \mathbf{\theta})}{\pi(a|S_t, \mathbf{\theta})} \bigg ] \\
& = \mathbb{E}_\pi \bigg [\sum_{a \in \mathcal{A}}   \pi(a|S_t, \mathbf{\theta}) q_\pi(S_t,a) \frac{\nabla_{\mathbf{\theta}} \pi(a|S_t, \mathbf{\theta})}{\pi(a|S_t, \mathbf{\theta})} \bigg ] \\
& =\mathbb{E}_\pi \bigg [ q_\pi(S_t,A_t) \frac{\nabla_{\mathbf{\theta}} \pi(A_t|S_t, \mathbf{\theta})}{\pi(A_t|S_t, \mathbf{\theta})} \bigg ] 
\end{aligned}$$

The expectation over $$q_\pi (S_t, A_t)$$ is essentially the return $$G_t$$. Thus, we can replace that in the above equation to get: 

$$\nabla_{\mathbf{\theta}}J(\mathbf{\theta})  = \mathbb{E}_\pi \bigg [ G_t\frac{\nabla_{\mathbf{\theta}} \pi(A_t|S_t, \mathbf{\theta})}{\pi(A_t|S_t, \mathbf{\theta})} \bigg ] $$

We now have a full sampling of the states and actions conditioned on our parameters in the gradients. This can be considered a sample from the policy and we can update our parameters using this quantity to get our update rule as: 

$$
\begin{aligned}
\mathbf{\theta}_{t+1} &= \mathbf{\theta}_t + \alpha \nabla_{\mathbf{\theta}}J(\mathbf{\theta}) \\ & =\mathbf{\theta}_t + \alpha \bigg ( \, G_t\frac{\nabla_{\mathbf{\theta}} \pi(A_t|S_t, \mathbf{\theta})}{\pi(A_t|S_t, \mathbf{\theta})} \bigg )
\end{aligned}
$$

This is the REINFORCE Algorithm! We have each update which is simply the learning rate $$\alpha$$ multiplied by a quantity that is proportional to the return and a vector of gradients of the probability of taking a certain action in a state. From the gradient ascent logic, we can see that this vector is the direction of maximizing the probability of taking action $A_t$ again, whenever we visit $S_t$. Moreover, The update is increasing the parameter vector in this direction proportional to the return, and inversely proportional to the action probability. Since the return is evaluated till the end of the episode, this is a Monte-Carlo Algorithm. We can further refine this using the identity of log differentiation and adding the discount factor to get the update as:

$$\mathbf{\theta}_{t+1} = \mathbf{\theta} + \alpha \gamma^t G_t \nabla_{\mathbf{\theta}} \ln \pi(A_t| S_t , \mathbf{\theta}) $$

## Looking at Trajectories

Another way to look at the above formulation is through sampled trajectories, as done in [Sergey Levine's slides](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf). While theoretically, this treatment is similar to the one done above, I just find the notation more intuitive. Recall, a trajectory is a sequence of states and actions over time, and the rewards accumulated in this sequence qualify this trajectory. Thus, we can say that the utility objective $$J(\mathbf{\theta})$$  is the sum of the accumulated rewards of some number of trajectories sampled from a policy $$\pi_\theta$$ 

$$J(\mathbf{\theta}) = \mathbb{E}_{\tau \sim \pi_{\mathbf{\theta}} (\tau)} \bigg [  \sum_t r(s_t, a_t)  \bigg ] \approx \frac{1}{N} \sum_i \sum_t r(s_{i:t}, a_{i:t}) $$

Let this sum of reward be denoted by $$G(\tau)$$  for a trajectory $$\tau$$. Thus, we can re-write the above equation as

$$
J(\mathbf{\theta}) =  \mathbb{E}_{\tau \sim \pi_{\mathbf{\theta}}(\tau) } \big [ G(\tau)  \big ]
$$

This expectation is essentially sampling a trajectory from a policy and weighing it with the accumulated rewards. Thus, we can write it as 

$$
J(\mathbf{\theta}) =  \int \pi_{\mathbf{\theta}} (\tau) G(\tau) d\tau
$$

Now, we just differentiate this objective, and add a convenient policy term to make it an expectation:

$$
\begin{aligned}
\nabla_{\mathbf{\theta}} J(\mathbf{\theta})  & = \nabla_{\mathbf{\theta}} \int \pi_{\mathbf{\theta}} (\tau) G(\tau) d\tau   \\
& = \int \nabla_{\mathbf{\theta}}  \pi_{\mathbf{\theta}} (\tau) G(\tau) d\tau \\
& = \int \pi_{\mathbf{\theta}}(\tau) \frac{\nabla_{\mathbf{\theta}}  \pi_{\mathbf{\theta}}(\tau) }{\pi_{\mathbf{\theta}}(\tau)} G(\tau) d\tau

\end{aligned}
$$

Now, we just use an identity $$\frac{dx}{x} = d \log x$$ and get

$$\nabla_{\mathbf{\theta}} J(\mathbf{\theta})   = \int \pi_{\mathbf{\theta}} (\tau) \nabla_{\mathbf{\theta}}  \log \pi_{\mathbf{\theta}} (\tau ) G(\tau) d\tau   $$

Thus, we can write this as:

$$\nabla_{\mathbf{\theta}} J(\mathbf{\theta}) = \mathbb{E}_{\tau \sim \pi_{\mathbf{\theta}} (\tau)} \big [ \nabla_{\mathbf{\theta}}  \log \pi_{\mathbf{\theta}} (\tau ) G(\tau)   \big ] $$

Hence, we get the same final result as before: the gradient depends on the gradient of the log policy weighted by the rewards accumulated over the trajectory. Now, we can translate this to states and actions over the trajectory by simply considering what the policy of the trajectory represents: 

$$\begin{aligned}
& \pi_{\mathbf{\theta}} (s_1, a_1, ..., s_T, a_T) = \mu(s) \prod_{t=1}^{T} \pi_{\mathbf{\theta}}(a_t|s_t)p(s_{t+1}|s_t, a_t) \\
\implies & \log \pi_{\mathbf{\theta}} (s_1, a_1, ..., s_T, a_T) = \log \mu(s) + \sum_{t=1}^{T} \log \pi_{\mathbf{\theta}}(a_t|s_t) + \log p(s_{t+1}|s_t, a_t)
\end{aligned}$$

When we differentiate this log policy w.r.t $$\mathbf{\theta}$$, we realize that $$\mu(s)$$ and $$p(s_{t+1}|s_t, a_t)$$ do not depend on this parameter, and would be set to 0. Thus, our utility expression would end up looking like

$$\nabla_{\mathbf{\theta}} J(\mathbf{\theta}) = \mathbb{E}_{\tau \sim \pi_{\mathbf{\theta}} (\tau)} \big [  \sum_{t=1}^{T} \nabla_{\mathbf{\theta}} \log \pi_{\mathbf{\theta}}(a_t|s_t) \sum_{t=1}^{T}r(s_t, a_t)   \big ] $$

Then we take average of the samples as the expectation, we get 

$$\nabla_{\mathbf{\theta}} J(\mathbf{\theta}) \approx \frac{1}{N} \sum _{i=1}^N \bigg [  \sum_{t=1}^{T} \nabla_{\mathbf{\theta}} \log \pi_{\mathbf{\theta}}(a_t|s_t) \sum_{t=1}^{T}r(s_t, a_t)   \bigg ] $$

And this is the key formula behind REINFORCE again and the update can thus, be written as

$$\mathbf{\theta} \leftarrow \mathbf{\theta} + \alpha \nabla_{\mathbf{\theta}} J(\mathbf{\theta}) $$

This is where I find it more intuitive to just use $$\tau$$ for trajectories since now we can just write our REINFORCE algorithm as : 

1. Sample a set of trajectories $$\{ \tau ^i\}$$ from the policy $$\pi_{\mathbf{\theta}}(a_t|s_t)$$ 
2. Estimate $$\nabla_{\mathbf{\theta}} J(\mathbf{\theta})$$ 
3. Update the parameters $$\mathbf{\theta} \leftarrow \mathbf{\theta} + \alpha \nabla_{\mathbf{\theta}} J(\mathbf{\theta})$$


## Reducing Variance

The idea of parameterizing the policy and working directly with the sampled trajectories has an intuitive appeal due to its clarity. However, this approach suffers from high variance. This is because when we compute the expectation over trajectories, we are essentially sampling $$N$$ different trajectories and then taking the average of the accumulated rewards. If we take this sampling to be uniform, we can easily imagine scenarios, where the trajectories sampled, have wildly different accumulated rewards. Thus, the chances of getting a high variance in the values that we are averaging over increase. If we were to scale $$N$$  to $$\infty$$ then our average becomes closer to the true expectation. However, this is computationally expensive. There are multiple ways to reduce this variance.

### Rewards to go

In the trajectory formulation, we are accumulating all of the rewards from $$t=0$$ to $$t = N$$. However, one way to make this online would be to just consider the rewards from the timestep at which we take the policy log value till the end of the horizon. This can be written as

$$\nabla_{\mathbf{\theta}} J(\mathbf{\theta}) \approx \frac{1}{N} \sum _{i=1}^N \bigg [  \sum_{t=1}^{T} \nabla_{\mathbf{\theta}} \log \pi_{\mathbf{\theta}}(a_t|s_t) \sum_{t'=t}^{T}r(s_{t'}, a_{t'})   \bigg ] $$

Thus, by reducing the number of  rewards we consider for each policy evaluation, we are essentially better able to control the variance up to a certain extent

### Baselining

Another way to control the variance is to realize that the actions in a state are the quantities that create a variance for each state in trajectory. However, we see that our return $$G(\tau)$$  is dependent on both states and actions. Thus, if we could compare this value to a baseline value $$b(s)$$, we eliminate the variance resulting from fixed state selection. In the policy gradient theorem, this would look like

$$\nabla_{\mathbf{\theta}}J(\mathbf{\theta}) \propto \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}}  \big ( q_\pi(s,a) - b(s)  \big )  \nabla\pi(a|s) $$

since $$b(s)$$  does not vary with actiosn, it would not have an effect on our summation since its sum over all actions would be 0:

$$\begin{aligned}
\sum_a b(s) \nabla_{\mathbf{\theta}}\pi(a|s, \mathbf{\theta}) & = b(s) \nabla_{\mathbf{\theta}} \sum_a \pi(a|s, \mathbf{\theta}) \\
& = b(s) \nabla_{\mathbf{\theta}} 1 \\
& = 0
\end{aligned}$$

Thus, we can effectively update our REINFORCE update with this baseline to get

$$\mathbf{\theta}_{t+1} =   =\mathbf{\theta}_t + \alpha \big ( \, G_t - b(s) \big ) \frac{\nabla_{\mathbf{\theta}} \pi(A_t|S_t, \mathbf{\theta})}{\pi(A_t|S_t, \mathbf{\theta})}$$

Or, in the trajectory formulation 

$$\mathbf{\theta}_{t+1} =   =\mathbf{\theta}_t + \alpha \nabla_{\mathbf{\theta}}  \log \pi_{\mathbf{\theta}} (\tau ) \big (G(\tau)  - b(s) \big ) $$

One good function for $$b(s)$$ could be the estimate of the state value $$\hat{v}(s_t, \mathbf{w})$$ . Doing something like this may seem like going to the realm of actor-critic methods, where we are parameterizing the policy and using the value function, but this is not the case here since we are not using the value function to bootstrap. We are stabilizing the variance by using the estimate of the value function as a baseline. Baselines are not just limited to this. We can inject all kinds of things into the baseline to try scaling up our policy gradient. For example, an [interesting paper](https://arxiv.org/abs/2102.10362) published recently talks about using functions that take into account causal dependency as a Baseline. There are many other extensions like Deterministic Policy Gradients, Deep Deterministic Policy Gradients, Proximal Optimization e.t.c that look deeper into this problem.

