---
layout: post
title: Maximum Likelihood Estimation
date: 2023-07-05 10:57:00-0400
categories: machine-learning
giscus_comments: false
related_posts: false
---

## Main idea

1. Make an explicit assumption about what distribution the data was modeled from.
2. Set the parameters of this distribution so that the data we observe is most likely, i.e., **maximize the likelihood of our data**.

For a simple example of a coin toss, we can see this as maximizing the probability of observing heads from a binomial distribution:

$$p(z_1, ..., z_n) = p(z_1, ..., z_n\vert \theta)$$

We assume the i.i.d. condition, and so we should be able to break this down into:

$$p(z_1\vert \theta)p(z_2\vert \theta) \cdots p(z_n\vert \theta)$$

Formally, let us define a likelihood function as:

$$L(\theta) = \prod_{i=1}^N p(z_i\vert \theta)$$

Now, our task is to find the $$\hat{\theta}$$ that maximizes this likelihood:

$$\hat{\theta}_{MLE} = \underset{\theta}{\text{argmax}} \prod_{i=1}^N p(z_i\vert \theta)$$

Instead of maximizing a product, we can also view this problem as minimizing a sum if we take the log of all values:

$$\hat{\theta}_{MLE} = \underset{\theta}{\text{argmax}} \sum_{i=1}^N \log(p(z_i\vert \theta))$$

## MLE for linear regression

Let us use this idea for the regression problem. We assume that our outputs are distributed in a Gaussian manner around the line we have to find. This basically means that our $$\epsilon$$ is Gaussian noise that is messing up our outputs from the fundamental distribution.

<div class="col-sm">
    {% include figure.html path="assets/img/MALIS/MLE.png" class="img-centered rounded z-depth-0" %}
</div>

Thus, our equation for the probability of our output would be:

$$p(y_i\vert x_i; w_i, \sigma^2) = \frac{1}{\sigma \sqrt{2\pi}} \exp\{ \frac{ - (y_i - x_iw)^2 }{2 \sigma^2} \}$$

Our task is to estimate $$\hat{w}$$ such that the likelihood $$p$$ is maximized. In other words, we need to find the value of $$w$$ that maximizes the above expression:

$$\begin{aligned}
& \hat{w} = \underset{w}{\text{argmax}} \prod_{i=1}^N p(y_i\vert x_i; w_i, \sigma^2) \\
\implies & \hat{w} = \underset{w}{\text{argmax}} \prod_{i=1}^N \frac{1}{\sigma \sqrt{2\pi}} \exp\{ \frac{ - (y_i - x_iw)^2 }{2 \sigma^2} \}
\end{aligned}$$

We can again use the log trick to make this a sum maximization:

$$\begin{aligned}
&\hat{w} = \underset{w}{\text{argmax}} \sum_{i=1}^N \log(\frac{1}{\sigma \sqrt{2\pi}} \exp\{ \frac{ - (y_i - x_iw)^2 }{2 \sigma^2} \}) \\
\implies &\hat{w} = \underset{w}{\text{argmax}} \{ \sum_{i=1}^N \log(\frac{1}{\sigma \sqrt{2\pi}}) + \sum_{i=1}^N \log(\exp\{ \frac{ - (y_i - x_iw)^2 }{2 \sigma^2} \}) \} \\
\implies &\hat{w} = \underset{w}{\text{argmax}} -\sum_{i=1}^N \frac{(y_i - x_iw)^2}{2 \sigma^2}
\end{aligned}$$

This is basically the same as the normal expression we had, the only difference being the normalizing factor $$\sigma$$. If we change the negative maximization to minimization:

$$\hat{w} = \frac{1}{2 \sigma^2} \underset{w}{\text{argmin}} \sum_{i=1}^N (y_i - x_iw)^2$$

This is the same as minimizing:

$$\hat{w} = \underset{w}{\text{argmin}} \sum_{i=1}^N (y_i - x_iw)^2$$

And the solution is:

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T\mathbf{y}$$

Surprise, surprise!!
