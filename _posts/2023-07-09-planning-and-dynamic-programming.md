---
layout: post
title: Planning and Dynamic Programming
date: 2023-07-09 08:57:00-0400
categories: reinforcement-learning
giscus_comments: false
related_posts: false
---

Dynamic programming (DP) is a method that solves a problem by breaking it down into sub-problems and then solving each sub-problem individually, after which it combining them into a solution. A good example is the standard Fibonacci sequence calculation problem, where traditionally the way to solve it would be through recursion

```cpp
int fib(int *x*) {
	
	if (x < 2) {
		return 1;
	}
	return fib(x-1) + fib(x-2);	

}
```

However, the way DP would go about this would be to cache the variables after the first call, so that the same call is not made again, making the program more efficient:

```cpp
int fib(int *x*) {

	static vector<int> cache(N, -1);
	int& result = cache[x];
	
	if (result == -1) {
	
		if (x < 2) result = 1;
	
		else result = fib(x-1) + fib(x-2);
	}
	
	return result;
}
```

The 2 characteristics that a problems need to have fo DP to solve it are:

1. **Optimal Substructure :** Any problem has optimal substructure property if its overall optimal solution can be constructed from the optimal solutions of its subproblems i.e the property $$F(n) = F(n-1) + F(n-2)$$ in fibonacci numbers
2. **Overlapping Sub-problems:** The problem involves sub-problems that need to be solved recursively many times

Now, in the case of an MDP, we have already seen that these properties are fulfilled:

1. The Bellman equation gives a recursive relation that satisfies the overlapping sub-problems requirement
2. The value function is able to store and re-use the solutions from each state-visit, and thus, we can exploit it as an optimal substructure

Hence, DP can be used for making solutions to MDPs more tractable, and thus, is a good tool to solve the planning problem in an MDP. The planning problem, as discussed before, is of two types:

1. **Prediction Problem:** **How do we evaluate a policy ?** or, Using the MDP tuple as an input, the output is a value function $$v_\pi$$ and/or a policy $$\pi$$
2. **Control Problem:** **How do we optimize the policy ?** Using the MDP tuple as an input, the output is an optimal value function $$v_*$$ and/or a policy $$\pi_*$$

## Iterative Policy Evaluation

The most basic way is to iteratively apply the Bellman equation, using the old values to calculate a new estimate, and then using this new estimate to calculate new values. In the Bellman equation for the state-value function

$$v_\pi(s) = \sum_{a \in A} \pi (a|s) \big[ R^{a}s  +  \gamma \sum{s' \in S} P^{a}{ss'} v_\pi(s) \big]$$

As long as either $$\gamma < 1$$ or the eventual termination is guaranteed from all states under the policy $$\pi$$, the uniqueness of the value function is guaranteed. Thus, we can consider a sequence of approximation functions $$v_0, v_1, v_2, ...$$  each mapping states to Real numbers, start with an arbitrary estimate of $v_0$, and obtain successive approximations using Bellman equation, as follows:

$$v_{k+1}(s) = \sum_{a \in A} \pi (a|s) \big[ R^{a}_s  +  \gamma \sum_{s' \in S} P^{a}_{ss'} v_{k} (s') \big]$$

The sequence $$v_k$$ can be shown to converge as $$k \rightarrow \infty$$. The process is basically a propagation towards the root of the decision tree from the roots.

<div class="col-sm">
    {% include figure.html path="assets/img/Reinforcement-Learning/It-pol-eval.png" class="img-centered rounded z-depth-0" %}
</div>

This update operation is applied to each state in the MDP at each step, and so, is called **Full-Backup**. Thus, in a computer program, we would have two cached arrays - one for $$v_k(s)$$ and one for $$v_{k+1}(s)$$.

## Policy Improvement
Once we have a policy, the next question is do we follow this policy or shift to a new improved policy? one way to answer this problem is to take any action that this policy does not suggest and then evaluate the same policy after that action. If the returns are higher then we can say that taking that action is better than following the current policy. The way we evaluate the action is through the action-value function:

$$
q_\pi(s,a) = R^a_s  +  \gamma \sum_{s' \in S} P^{a}_{ss'} v_\pi(s')
$$

If this value is greater than the value function of a state S, then that essentially means that it is better to select this action than follow the policy $$\pi$$ , and by extension, it would mean that anytime we encounter state $$S$$, we would like to take this action. So, let's call the schema of taking action $$a$$ every time we encounter $$S$$ as a new policy $$\pi'$$, and so, we can now say

$$q_\pi(s,\pi'(s)) \geq v_\pi(s)$$

This implies that the policy $$\pi'$$ must be **at-least** as good as the policy $$\pi$$:

$$v_{\pi'} \geq v_\pi$$

Thus, if we extend this idea to multiple possible actions at any state $$S$$,  the net incentive is to go full greedy on it and select the best out of all those possible actions:

$$\pi'(s) = \argmax_a q_\pi(s,a)$$

The greedy policy, thus, takes the action that looks best in the short term i.e after one step of lookahead. The point at which the new policy stops becoming better than the old one is the convergence point, and we can conclude that optimality has been reached. This idea also applies in the general case of stochastic policies, with the addition that in the case of multiple actions with the maximum value, a portion of the stochastic probability can be given to each.

## Policy Iteration

Following the greedy policy improvement process, we can obtain a sequence of policies:

$$\pi_0 \rightarrow v_{\pi_0} \rightarrow {\pi_1} \rightarrow v_{\pi_1} .... \rightarrow \pi_* \rightarrow v_{\pi_*}$$

Since a finite MDP has a finite number of policies, this process must converge at some point to an optimal value. This process is called **Policy Iteration**. The algorithm, thus, follows the process:

1. **Evaluate** the policy using the Bellman equation
2. **Improve** the policy using greedy policy improvement.

A natural question that comes up at this point is that do we actually need to follow this optimization procedure all the way to the end? It does sound like a lot of work, and in a lot of cases, a workably optimal policy is actually reached much before the final iteration step, where the steps after achieving this policy add minimal improvement and thus, are somewhat redundant. Thus, we can include stopping conditions to tackle this, as follows:

1. $$\epsilon$$-convergence
2. Stop after $$k$$ iteratiokns
3. Value Iteration


## Value Iteration

In this algorithm, the evaluation is truncated to one sweep → one backup of each state. To understand this, the first step is to understand something called the **Principle of Optimality**. The idea is that an optimal policy can be subdivided into two parts:

- An optimal first action $$A_*$$
- An optimal policy from the successor state $$S'$$

So, if we know the solution to $$v_*(s')$$ for all $$s'$$ succeeding the state $$s$$, then the solution can be found with just a one-step lookahead

$$v_*(s) \gets \max_{a \isin A} R^a_s + \gamma \sum_{\substack{s' \in S}} P^{a}_{ss'} v_{*} (s')$$

The intuition is to start from the final reward and work your way backward. There is no explicit update of policy, only values. This also opens up the possibility that the intermediate values might not correspond to any policy, and so interpreting anything midway will have some residue in addition to the greedy policy.  In practice, we stop once the value function changes by only a small amount in a sweep. A summary of synchronous methods for DP is given by David Silverman:


```markdown
| Problem      | Bellman Equation                   | Algorithm    |
|--------------|------------------------------------|--------------|
| Prediction   | Expectation Equation               | right 1      |
| Control      | Expectation Equation + Greedy Eval.| right 2      |
| Control      | Optimality Equation                | right 3      |
```