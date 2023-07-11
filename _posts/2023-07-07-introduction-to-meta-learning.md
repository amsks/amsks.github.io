---
layout: post
title: Introduction to Meta-Learning 
date: 2023-07-07 10:57:00-0400
categories: meta-learning
giscus_comments: false
related_posts: false
---

The motivating research that I personally find very interesting is a paper by Deepmind in 2018, that looks at the prefrontal cortex as a meta-reinforcement learning problem. The core idea here is that there seems to be an added function that dopamine serves: it is not just adjusting the 'weights' of the neurons in the brain, but also carrying some relevant information about the rules pertaining to the tasks that are being performed. This means that at the most fundamental level when given rewards for different kinds of tasks, it might be possible that there is an abstraction happening that allows learning underlying patterns between these tasks. Thus, if the tasks share some kind of commonalities in this meta-space, then it could be that the mind is formulating a meta-representation between these tasks which is allowing the person to treat any new task that shares some kind of underlying structure, no matter how unapparent that may be at a superficial level, as a problem of generalizing to new scenarios using priors from previous experiences. The core idea behind Meta-Learning is to develop a way for an agent to learn to generalize to new unseen tasks. One crucial way to see this, which I personally found very helpful, is to pit this notion against transfer learning or multi-task learning. A task like distinguishing between cats and birds, or riding a bicycle with trainer wheels is something that humans can learn with very few samples, while the traditional ML agent trained on a Neural Network approximation would require a great number of samples to be able to do this. 

## Basic Formulation

The intuition about what exactly is the meta-learning problem is very well seen through probabilistic lenses. Let's denote a dataset as $$\mathcal{D} = \{ (\mathbf{x}, \mathbf{y})_k\}$$ . In a standard supervised learning problem, we are essentially learning a loss function over this dataset for a function approximator that takes a vector $$\mathbf{x}_i$$ as input and produces some kind of an output  $$y_i$$. Let us denote these parameters by $$\theta$$  and so, we can say that our learning objective is

$$\min_\theta \mathcal{L}(\theta, \mathcal{D})$$

Depending on our requirement, we take take this loss function to be something like a negative log likelihood e.t.c. Now, given this loss function, we can define a task as

$$\mathcal{T}_i := \{ p_i(\mathbf{x}), p_i(\mathbf{y}|\mathbf{x}), \mathcal{L}_i \} $$

Thus, we are taking as a reference as data generating distribution and sampling from it with the indicated probability $$p_i$$ and our task is to predict the posterior distribution over labels given our inputs sampled from a prior distribution by minimizing a particular loss function. The way we go about this is to segregate our data into a training set $$\mathcal{D}^{train}_i$$ and a test set $$\mathcal{D}_i^{test}$$. 

If our objective is to learn multiple tasks then we need to train our function approximator to work on multiple such tasks $$\mathcal{T}_i$$ for a total set of tasks $$i = 1, 2, ..., T$$. To do this, we figure out a way to inculcate a task descriptor $$\mathbf{z}_i$$ into our function approximation so that the approximator can adjust its output based on the task, and then we essentially minimize over a sum fo multiple such losses to get the following formulation

$$\min_\theta \sum_{i=1}^T \mathcal{L}_i (\theta, \mathcal{D}_i)$$

If our objective is to learn something meta, then we can view this parameter $$\theta$$  as a prior that is being used by our approximator to learn an optimal set of parameters $$\phi$$  and predict the posterior distribution over $$\mathbf{y}$$ a given set of inputs $$\mathbf{x}$$ for a dataset $$\mathcal{D}$$. A good way to understand this is to say that we have some metadata for training $$\mathcal{D}_{meta-train}$$ that we are using along with the inputs from a dataset to get our posterior prediction, and $$\theta$$  is essentially parameterizing this meta training data to give us some meta-parameters that we can use for our training. This can be mathematically formulated as

$$\begin{aligned}
            % {\subseteq}
& \theta^* \in \underset{\theta}{\text{argmax}} \{\log p(\theta|\mathcal{D}_{meta-train} \} \\
& \phi^*   \in \underset{\phi}{\text{argmax}} \{ \log p(\phi | \mathcal{D}^{tr}, \theta^*) \} \\
  & \,\,\,\,\,\,\,\,\,\ \therefore \,\,\,\phi^* = f_{\theta^*}(\mathcal{D}^{tr})

\end{aligned}$$

Thus, we can see this learning as two problems: 

1. Meta-learning problem → Understand the model agnostic parameters $$\theta^*$$ 
2. Adaptation Problem → Use the meta-parameters to adapt your learning curve and get the test-specific parameters $$\phi^*$$

### Mechanistic Formulation of Meta-Learning Problem

In supervised learning, an instance of meta-learning is a **few-shot classification,** where our task is to train the approximator on different datasets so that it learns to classify totally new and unseen data with only a few shots. The way we implement this is by having test data for each training dataset so that the approximator can be tested for each abstraction that it is learning. These are also called support set $$S$$ and prediction set $$B$$ 

$$\mathcal{D} = \big < S, B\big >$$

For a support set that contains $$k$$ labeled examples for $$N$$ classes, we call this a $$k$$-shot-$$N$$-classification task. Thus, a 5-shot-2-classification task on images would consist of datasets that have a total of 10 example images divided into one or the other class equally, and our classifier needs to learn to classify images in an unseen dataset with unseen classes but having the same configuration.  We consider data  over which w  need to do the prediction as $$\mathcal{D}_i = \{ (\mathbf{x}, \mathbf{y} )_j \}$$ and the meta-data as a set of these datasets $$\mathcal{D}_{meta-train} = \{\mathcal{D}_i \}$$. Our training data contains $$K$$ input-output pairs and so can be written as 

$$\mathcal{D}^{tr} =  \{ (\mathbf{x}, \mathbf{y} )_{1:K}\} $$

This is a $$K$$-shot prediction problem on a test input $$\mathbf{x}_{test}$$ and thus our problem is to predict $$\mathbf{y}_{test}$$ using these inputs, training datasets, and parameters

$$\mathbf{y}_{test} = f (\mathcal{D}^{tr}, \mathbf{x}_{test}; \theta ) $$

This translates the problem of meta-learning as a design and optimization problem, similar to how we would go about 

Thus, the general way to go about solving the meta-learning problem is: 

1. Choose a function to output task-specific parameters $$p(\phi_i | \mathcal{D}_i^{tr}, \theta)$$
2. Use the training data $$\mathcal{D}_{meta-train}$$ to create a max-likelihood objective that can then be used to optimize $$\theta$$



