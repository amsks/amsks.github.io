---
layout: post
title: Parametric Methods for Meta-Learning
date: 2023-07-07 11:57:00-0400
categories: meta-learning
giscus_comments: false
related_posts: false
---

## Black-Box Adaptation

These are a set of approaches that treat step 1 as an inference problem and thus train a neural network to represent $$p(\phi_i \mid \mathcal{D}^{tr}, \theta)$$, i.e., a way to estimate $$\phi_i$$ and then use that as a parameter to optimize for a new task. The deterministic way to go about it would be to take point estimates:

$$
\phi_i = f_\theta (\mathcal{D}^{tr}_i)
$$

Thus, we can treat $$f_\theta(.)$$ as a neural network parameterized by $$\theta$$ which takes the training data as an input, sequential or batched, and outputs the task-specific parameters $$\phi_i$$, which are then used by another neural network $$g_{\phi_i} (.)$$ to predict the outputs on a new dataset. Thus, we can essentially treat this as a supervised learning problem, with our optimization being

$$\begin{aligned}
& \max_\theta \sum_{\mathcal{T_i}} \sum_{(x,y) \sim \mathcal{D_i}^{test}} \log g_{\phi_i} (y\mid x) \\
= & \max_\theta \sum_{\mathcal{T_i}} \mathcal{L}(f_\theta(\mathcal{D^{tr}_i}), \mathcal{D_i^{test}})
\end{aligned}$$

To make this more tractable, $$\phi$$ can be replaced by a sufficient statistic $$h_i$$ instead of all the parameters. Some ANN architectures that work well with this approach are LSTMs, as shown in the work of [Santoro et al.](http://proceedings.mlr.press/v48/santoro16.pdf), feedforward networks with averaging, as shown by [Ramalho et al.](https://arxiv.org/abs/1807.01613), and architectures with inner task learners and outer meta-learners, i.e., [Meta-Networks by Munkhdalai](https://arxiv.org/abs/1703.00837), etc. I am personally fascinated by the use of transformer architectures in this domain.

The advantage of this approach is that it is expressive and easy to combine with other techniques like supervised learning, reinforcement learning, etc. However, the optimization bit is challenging and not the best solution from the onset for every kind of problem. Thus, our step-by-step approach would be:

1. Sample task $$\mathcal{T}_i$$ (a sequential stream or mini-batches)
2. Sample disjoint datasets $$\mathcal{D^{tr}_i}$$, $$\mathcal{D^{test}_i}$$ from $$\mathcal{D}_i$$
3. Compute $$\phi_i \leftarrow f_\theta(\mathcal{D^{tr}_i})$$
4. Update $$\theta$$ using $$\nabla_\theta\mathcal{L}(\phi_i, \mathcal{D^{test}_i})$$

## Optimization-Based Approaches

This set treats the prediction of $$\phi_i$$ as an optimization procedure and then differentiates through that optimization process to get a $$\phi_i$$ that leads to good performance. The method can be summarized as the sum of two surrogate objectives: the maximization of observing the training data given $$\phi_i$$, and the maximization of getting $$\phi_i$$ given our model parameters $$\theta$$.

$$\max_{\phi_i} \log p(\mathcal{D^{tr}_i} \mid \phi_i) + \log p(\phi_i \mid \theta)$$

The second part of the above summation is our prior and the first part is a likelihood. Thus, our next question is the form of this prior that might be useful. In deep learning, one good way to incorporate priors is through the initialization of hyperparameters, or fine-tuning. Thus, we can take $$\theta$$ as a pre-trained parameter and run gradient descent on it:

$$\phi \leftarrow \theta - \alpha \nabla_\theta \mathcal{L} (\theta, \mathcal{D^{tr}})$$

One popular way to do this for image classification is to have a feature extractor pre-trained on some dataset like ImageNet and then fine-tune its output to our problem. The aim in optimization-based approaches is to get to a sweet spot in the multidimensional parameter space $$\mathbf{\Phi} = \{\phi_1, \phi_2, \dots, \phi_n\}$$ such that our model becomes independent of the loss function and the training data, and this is called Model-Agnostic Meta-Learning (MAML). Thus, our procedure now becomes:

1. Sample task $$\mathcal{T}_i$$ (a sequential stream or mini-batches)
2. Sample disjoint datasets $$\mathcal{D^{tr}_i}$$, $$\mathcal{D^{test}_i}$$ from $$\mathcal{D}_i$$
3. Optimize $$\phi_i \leftarrow f_\theta(\mathcal{D^{tr}_i})$$
4. Update $$\theta$$ using $$\nabla_\theta\mathcal{L}(\phi_i, \mathcal{D^{test}_i})$$

For our optimization process, let's define our final task-specific parameter as

$$\phi = u(\theta, \mathcal{D^{tr}})$$

And now, our optimization target becomes

$$\begin{aligned}
& \min_\theta \mathcal{L}(\phi, \mathcal{D^{test}}) \\
= & \min_\theta \mathcal{L} \big(u(\theta, \mathcal{D^{tr}}), \mathcal{D^{test}} \big)
\end{aligned}$$

This optimization can be achieved by differentiating our loss w.r.t. our meta-parameters $$\theta$$ and then performing an inner differentiation w.r.t. $$\phi$$:

$$\frac{d\mathcal{L} (\phi, \mathcal{D^{test}})}{d \theta} = \nabla _{\bar{\phi}} \mathcal{L} (\bar{\phi}, \mathcal{D^{test}}) \mid_{\bar{\phi} = u(\theta, \mathcal{D^{tr}})} d_\theta \big( u(\theta, \mathcal{D^{tr}}) \big)$$

Now, if we use our optimization update for $$u(.)$$, then we get:

$$\begin{aligned}
& u(\theta, \mathcal{D^{tr}}) = \theta - \alpha \,\, d_\theta \big( L(\theta, \mathcal{D^{tr}}) \big) \\
\implies & d_\theta \big( u(\theta, \mathcal{D^{tr}}) \big) = \mathbf{1} - \alpha \, d^2_\theta \big(L(\theta, \mathcal{D^{tr}}) \big)
\end{aligned}$$

Thus, when we substitute the Hessian in the derivative equation, we get:

$$\begin{aligned}
\frac{d\mathcal{L} (\phi, \mathcal{D^{test}})}{d \theta} & = \bigg(\nabla _{\bar{\phi}} \mathcal{L} (\bar{\phi}, \mathcal{D^{test}}) \mid_{\bar{\phi} = u(\theta, \mathcal{D^{tr}})} \bigg) . \bigg( \mathbf{1} - \alpha \, d^2_\theta \big(L(\theta, \mathcal{D^{tr}}) \big) \bigg) \\
& = \nabla _{\bar{\phi}} \mathcal{L} (\bar{\phi}, \mathcal{D^{test}}) \mid_{\bar{\phi} = u(\theta, \mathcal{D^{tr}})}
- \alpha\,\, \bigg( \nabla _{\bar{\phi}} \mathcal{L} (\bar{\phi}, \mathcal{D^{test}})
. d^2_\theta \big(L(\theta, \mathcal{D^{tr}}) \big) \bigg)
\mid_{\bar{\phi} = u(\theta, \mathcal{D^{tr}})}
\end{aligned}$$

We now have a matrix product on the right, which can be made more efficient and turns out to be easier to compute than the full Hessian of the network. Thus, this process is tractable.

One really interesting thing that comes out of this is that we can also view this model-agnostic approach and the optimization update as a computation graph! Thus, we can say

$$\phi_i = \theta - f(\theta, \mathcal{D_i^{tr}}, \nabla_\theta \mathcal{L})$$

Now, we can train an ANN to output the gradient $$f(.)$$, and this allows us to mix the optimization procedure with the black-box adaptation process. Moreover, MAML approaches show better performance on the Omniglot dataset since they are optimizing for the model-agnostic points. It has been shown by [Finn and Levine](https://arxiv.org/abs/1710.11622) that MAML can approximate any function of $$\mathcal{D_i^{tr}}$$ and $$x^{ts}$$, given:

- Non-zero $$\alpha$$
- A loss function gradient that does not lose information about the label
- Data points in $$\mathcal{D_i^{tr}}$$ that are unique

Thus, MAML is able to inject inductive bias without losing expressivity.

### Inference

To better understand why MAML works well, we need to look through probabilistic lenses again. The meta-parameters $$\theta$$ are inducing some kind of prior knowledge into our system, and so our learning objective would be to maximize the probability of observing the data $$\mathcal{D}_i$$, given our meta-parameters $$\theta$$:

$$\max_\theta \log \prod_i p(\mathcal{D}_i \mid \theta)$$

This can be further written as the sum of the probabilities of $$\mathcal{D_i}$$ given our model-specific parameters $$\phi_i$$, and the probability of seeing each $$\phi_i$$ given our prior knowledge $$\theta$$:

$$\max _\theta \prod_i \int p(\mathcal{D_i} \mid \phi_i) p(\phi_i \mid \theta) d\phi_i$$

And now, we can estimate the probability of seeing each $$\phi_i$$ given our prior knowledge $$\theta$$ using a Maximum A-Posteriori (MAP) estimate $$\hat{\phi}$$, so that

$$\max_\theta \log \prod_i p(\mathcal{D}_i \mid \theta) \approx \max_\theta \log \prod_i p(\mathcal{D}_i \mid \hat{\phi}_i) p(\hat{\phi}_i \mid \theta)$$

[It has been shown](https://regijs.github.io/papers/laa96.pdf) that, for likelihoods that are Gaussian in $$\phi_i$$, gradient descent with early stopping corresponds exactly to maximum a-posteriori inference under a Gaussian prior with mean initial samples. This estimation is exact in the linear case, and the variance in non-linear cases is determined by the order of the derivative. Thus, by limiting the computation to second derivatives, MAML is able to maintain a fairly good MAP inference estimate, and so MAML approximates hierarchical Bayesian inference. We can also use other kinds of priors, like:

- [Explicit Gaussian prior](https://arxiv.org/abs/1909.04630): $$\phi \leftarrow \min_{\phi'} \mathcal{L} (\phi', \mathcal{D^{tr}}) + \frac{\lambda}{2} \vert \vert  \theta - \phi'\vert \vert ^2$$
- [Bayesian linear regression](https://arxiv.org/abs/1807.08912) on learned features
- Convex optimization on learned features
- Ridge or logistic regression
- Support vector machines

### Challenge 1: Choosing Architecture

The major bottleneck in this process is the inner gradient step, and so we want to choose an architecture that is effective for this inner step. One idea, called [Auto-Meta](https://arxiv.org/abs/1806.06927), is to adopt progressive neural architecture search to find optimal architectures for meta-learners, i.e., combine AutoML with gradient-based meta-learning. The interesting results of this were:

- They found highly non-standard architectures, both deep and narrow.
- They found architectures very different from the ones used for supervised learning.

### Challenge 2: Handling Instabilities

Another challenge comes from the instability that can come from the complicated bi-level optimization procedure. One way of mitigating this is to learn the inner vector learning rate and then tune the outer learning rate:

- [Meta-Stochastic Gradient Descent](https://arxiv.org/abs/1707.09835) is a meta-learner that can learn initialization, learner update direction, and learning rate, all in a single closed-loop process.
- [AlphaMAML](https://arxiv.org/abs/1905.07435) incorporates an online hyperparameter adaptation scheme that eliminates the need to tune meta-learning and learning rates.

Another idea is to optimize only a subset of parameters in the inner loop:

- [DEML](https://arxiv.org/abs/1802.03596) jointly learns a concept generator, a meta-learner, and a concept discriminator. The concept generator abstracts a representation of each instance of training data, the meta-learner performs few-shot learning on this instance, and the concept discriminator recognizes the concept pertaining to each instance.
- [CAVIA](https://arxiv.org/abs/1810.03642) partitions the model parameters into context parameters, which serve as additional input to the model and are adapted on individual tasks, and shared parameters, which are meta-trained and shared across tasks. Thus, during test time only the context parameters need to be updated, which is a lower-dimensional search problem compared to all the model parameters.

In [MAML++](https://arxiv.org/pdf/1810.09502.pdf) the authors ablate the various ideas and issues of MAML and then propose a new framework that addresses these issues. Some significant points were the decoupling of the inner loop learning rate and the outer updates, the addition of batch normalization to each and every step, and greedy updates.

### Challenge 3: Managing Compute and Memory

The backpropagation through many inner gradient steps adds computational and memory overhead that is hard to deal with. One idea to mitigate this is to approximate the derivative of $$\phi_i$$ w.r.t. $$\theta$$. This is a crude approximation and works well for few-shot learning problems, but fails in more complex problems like imitation learning.

Another direction is to try to not compute the gradient at all and use the [implicit function theorem](https://arxiv.org/abs/1909.04630). Let's take our function $$\phi$$ as the explicit Gaussian representation:

$$\phi = u(\theta, \mathcal{D^{tr}}) = \underset{\phi'}{\text{argmin}} \mathcal{L}(\phi', \mathcal{D^{tr}}) + \frac{\lambda}{2} \vert \vert \phi' - \theta \vert \vert ^2$$

Let our optimization function be

$$G(\phi', \theta) = \mathcal{L}(\phi', \mathcal{D^{tr}}) + \frac{\lambda}{2} \vert \vert \phi' - \theta \vert \vert ^2$$

Finding the $$\text{argmin}$$ of this function implies that the gradient w.r.t. $$\phi$$ is $$0$$, i.e.,

$$\begin{aligned}
& \nabla_{\phi'} G(\phi', \theta) \big\vert _{\phi' = \phi} = 0 \\
\implies & \nabla_\phi L(\phi) + \lambda(\phi - \theta) = 0 \\
\implies & \phi = \theta - \frac{1}{\lambda} \nabla_\phi L(\phi)
\end{aligned}$$

Thus, our derivative now becomes

$$\begin{aligned}
& \frac{d \phi}{d \theta} = \mathbf{1} - \frac{1}{\lambda} \nabla_\phi^2 L(\phi) \frac{d \phi}{d \theta} \\
\therefore\,\,\,& \frac{d \phi}{d \theta} = \bigg[\mathbf{1} + \frac{1}{\lambda} \nabla_\phi^2 L(\phi) \bigg]^{-1}
\end{aligned}$$

Thus, we can compute this without going through the inner optimization process, and it works only on the assumption that our function $$G(\phi', \theta)$$ has an $$\text{argmin}$$ to begin with.
