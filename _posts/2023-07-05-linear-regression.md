---
layout: post
title: Linear Regression
date: 2023-07-05 08:57:00-0400
categories: machine-learning
giscus_comments: false
related_posts: false
---

**Main ideas :**

- Use Least squares to fit a line to Data
- Use R square
- Use p value

**Fitting the Line →** Try to minimize a metric that represents the fit

- Let the Line be $$y(x) = w_0 + w_1x$$
- Now, our optimization goal is to find the values of $$w_0, w_1$$ so that the variation around this line is minimal → We do this by minimizing the squared Errors

To know if taking into the samples actually improves anything or not, all we have to do is calculate the variance around the fit and compare it with variance around the mean of the y values of the point, and give an answer in percentages! This is called the $$R^2$$ value: 

$$
R^2  = \frac {\text{Var}(mean) - \text{Var}(fit)}{\text{Var}(\text{mean})}
$$

Thus, if this value is 0.6 , we get a 60% improvement in the variance by taking the x features into account. 

Let's go to the interesting stuff → The Math of this all 

## Math of Regression 

Let's take the  case of a set of multidimensional features $$\mathbf{X} \in \mathbb{R}^D \,\,\,$$ where, $$i= 1,...,N$$. For each of these set of D dimensional inputs, we have one output $$\mathbf{y} \in \mathbb{R}$$. Thus, we have our data as $\{(X_i,y_i)\}$ to which we have to fit a D dimensional hyperplane so that the variance around this hyperplane is minimal. Let's start by defining our model:

$$
\begin{aligned}
&y_i(X_i) = f(X_i) + \epsilon \\
&\hat{y_i} = \hat{f}(X_i) \\
\end{aligned}
$$

Here, the actual data is a function $$f: \mathbb{R}^D \mathbb{R}ightarrow  \mathbb{R}$$ and our hyperplane is function $$\hat{f}: \mathbb{R}^D \mathbb{R}ightarrow \mathbb{R}$$ which produces the targets $$y$$ and prediction $$\hat{y}$$, respectively. So, we can check the error between our actual data and the predicted data, which  we call the Sum-of-square error:

$$
\begin{aligned}
&\mathbf{e} = (\mathbf{y} - \mathbf{\hat{y}})^2 \\
&\mathbf{e} = (\mathbf{y} - \mathbf{\hat{y}})^T(\mathbf{y} - \mathbf{\hat{y}})\\
\end{aligned}
$$

Here, I have used bold to represent vector notation. since our model is linear, we can define it as:

$$
\hat{f}(\mathbf{X}) = \mathbf{X}\mathbf{w} 
$$

- **Note:** to make this work by taking bias into account we let $$\mathbf{w} \in \mathbb{R}^{D+1}$$ where the D weights are corresponding to D features and the extra weight is the bias. Thus, $$\mathbf{X} \in \mathbb{R}^{NX(D+1)}$$ which basically means that our N observations are stacked vertically and each observation is of D dimensions, but to make the notation work, we add a 1 at the start, which will be the multiplier for our bias term, and thus, have D+1 as the dimension of the row.

Thus, our error now becomes

$$
\begin{aligned}
&\mathbf{e} = (\mathbf{y} -\mathbf{X}\mathbf{w}  )^T(\mathbf{y} - \mathbf{X}\mathbf{w}) \\
\implies &\mathbf{e} = (\mathbf{y} -\mathbf{w}^T\mathbf{X}^T  )(\mathbf{y} - \mathbf{X}\mathbf{w}) \\
\implies &\mathbf{e} = \mathbf{y}^T\mathbf{y} -  \mathbf{y}^T\mathbf{X}\mathbf{w} - \mathbf{w}^T\mathbf{X}^T\mathbf{y} + \mathbf{w}^T\mathbf{X}^T\mathbf{X}\mathbf{w} \\
\end{aligned}
$$

Now, to get our optimal weights we follow the method to get the minima of e i.e differentiate e w.r.t $$\mathbf{w}$$ and then set it to 0:

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

Hence, all we need to do is plug-in $$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T\mathbf{y}$$ into our original equation and we get the solution. Of course, this is the optimization variant of our regression problem and gradient descent goes around this by computing this solution iteratively by taking an initial guess of \mathbf{w} and then moving towards the direction of decrease, and moving proportionally to the rate of decrease. However, the solution to which it should end up converging is the same! We can also do all sorts of gymnastics around this solution to make the variance go down even further. For example, we could transform our input $$\mathbf{X}$$ to a new space by $$\mathbf{\phi(\mathbf{X})}$$, in which subject to 1-1 mapping, our solution would simply become

$$
\mathbf{w} = (\mathbf{\phi(\mathbf{X})}^T\mathbf{\phi(\mathbf{X})})^{-1} \mathbf{\phi(\mathbf{X})}^T\mathbf{y}
$$

The essence of regression remains the same. in the case where $$D= 2$$, we use this same technique on 2D matrices and those simplistic equation for the starting points of regression that we see in most places.
