---
layout: post
title: Linear Regression
date: 2023-07-05 08:57:00-0400
categories: machine-learning
giscus_comments: false
related_posts: false
---

**Main ideas:**

- Use least squares to fit a line to data.
- Use R-squared.
- Use the p-value.

## Fitting the Line

Try to minimize a metric that represents the fit:

- Let the line be $$y(x) = w_0 + w_1x$$.
- Now, our optimization goal is to find the values of $$w_0, w_1$$ so that the variation around this line is minimal → we do this by minimizing the squared errors.

To know if taking the samples into account actually improves anything or not, all we have to do is calculate the variance around the fit, compare it with the variance around the mean of the y values of the points, and give an answer in percentages! This is called the $$R^2$$ value:

$$
R^2 = \frac{\text{Var}(\text{mean}) - \text{Var}(\text{fit})}{\text{Var}(\text{mean})}
$$

Thus, if this value is 0.6, we get a 60% improvement in the variance by taking the x features into account.

Let's go to the interesting stuff: the math of it all.

## Math of Regression

Let's take the case of a set of multidimensional features $$\mathbf{X} \in \mathbb{R}^D$$, where $$i = 1, \dots, N$$. For each of these D-dimensional inputs, we have one output $$\mathbf{y} \in \mathbb{R}$$. Thus, we have our data as $$\{(X_i,y_i)\}$$, to which we have to fit a D-dimensional hyperplane so that the variance around this hyperplane is minimal. Let's start by defining our model:

$$
\begin{aligned}
&y_i(X_i) = f(X_i) + \epsilon \\
&\hat{y_i} = \hat{f}(X_i) \\
\end{aligned}
$$

Here, the actual data is a function $$f: \mathbb{R}^D \rightarrow \mathbb{R}$$ and our hyperplane is a function $$\hat{f}: \mathbb{R}^D \rightarrow \mathbb{R}$$, which produce the targets $$y$$ and predictions $$\hat{y}$$, respectively. So, we can check the error between our actual data and the predicted data, which we call the sum-of-squares error:

$$
\begin{aligned}
&\mathbf{e} = (\mathbf{y} - \mathbf{\hat{y}})^2 \\
&\mathbf{e} = (\mathbf{y} - \mathbf{\hat{y}})^T(\mathbf{y} - \mathbf{\hat{y}})\\
\end{aligned}
$$

Here, I have used bold to represent vector notation. Since our model is linear, we can define it as:

$$
\hat{f}(\mathbf{X}) = \mathbf{X}\mathbf{w}
$$

- **Note:** to make this work by taking the bias into account, we let $$\mathbf{w} \in \mathbb{R}^{D+1}$$, where the D weights correspond to the D features and the extra weight is the bias. Thus, $$\mathbf{X} \in \mathbb{R}^{N \times (D+1)}$$, which basically means that our N observations are stacked vertically and each observation is of D dimensions, but to make the notation work, we add a 1 at the start, which will be the multiplier for our bias term, and thus, have D+1 as the dimension of the row.

Thus, our error now becomes:

$$
\begin{aligned}
&\mathbf{e} = (\mathbf{y} -\mathbf{X}\mathbf{w}  )^T(\mathbf{y} - \mathbf{X}\mathbf{w}) \\
\implies &\mathbf{e} = (\mathbf{y} -\mathbf{w}^T\mathbf{X}^T  )(\mathbf{y} - \mathbf{X}\mathbf{w}) \\
\implies &\mathbf{e} = \mathbf{y}^T\mathbf{y} -  \mathbf{y}^T\mathbf{X}\mathbf{w} - \mathbf{w}^T\mathbf{X}^T\mathbf{y} + \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w} \\
\end{aligned}
$$

Now, to get our optimal weights we follow the method to get the minima of e, i.e., differentiate e w.r.t. $$\mathbf{w}$$ and then set it to 0:

$$
\begin{aligned}
&\nabla_w\mathbf{e} = 0 \\
\implies &\nabla_w(\mathbf{y}^T\mathbf{y} -  \mathbf{y}^T\mathbf{X}\mathbf{w} - \mathbf{w}^T\mathbf{X}^T\mathbf{y} + \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w} ) = 0 \\
\implies &\nabla_w(\mathbf{y}^T\mathbf{y}) -  \nabla_w(\mathbf{y}^T\mathbf{X}\mathbf{w}) - \nabla_w(\mathbf{w}^T\mathbf{X}^T\mathbf{y}) + \nabla_w(\mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w}) = 0 \\
\implies &-2\mathbf{y}^T\mathbf{X} - 2\mathbf{w}^T\mathbf{X}^T\mathbf{X} = 0 \\
\implies &(\mathbf{X}^T\mathbf{X})\mathbf{w}^T = \mathbf{y}^T\mathbf{X} \\
\therefore \,\, &\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T\mathbf{y}\\
\end{aligned}
$$

Hence, all we need to do is plug $$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T\mathbf{y}$$ into our original equation and we get the solution.

Of course, this is the optimization variant of our regression problem, and gradient descent goes around this by computing the solution iteratively: it takes an initial guess of $$\mathbf{w}$$ and then moves in the direction of decrease, proportionally to the rate of decrease. However, the solution to which it should end up converging is the same!

We can also do all sorts of gymnastics around this solution to make the variance go down even further. For example, we could transform our input $$\mathbf{X}$$ to a new space by $$\mathbf{\phi(\mathbf{X})}$$, in which, subject to a 1-1 mapping, our solution would simply become:

$$
\mathbf{w} = (\mathbf{\phi(\mathbf{X})}^T\mathbf{\phi(\mathbf{X})})^{-1} \mathbf{\phi(\mathbf{X})}^T\mathbf{y}
$$

The essence of regression remains the same. In the case where $$D = 2$$, we use this same technique on 2D matrices and get those simplistic equations for the starting points of regression that we see in most places.
