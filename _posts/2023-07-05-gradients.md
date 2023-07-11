---
layout: post
title: Gradients
date: 2023-07-05 18:57:00-0400
categories: machine-learning
giscus_comments: false
related_posts: false
---

## Gradient Descent

- **predicted value** $$= \text{intercept} + \text{slope}$$ 
- **metric** → $$y = C + \frac{dy}{dx} x$$

let $$y'$$ be the expected value:
- residual $$= y' - y $$ 
- sos = $$(y' - y)^2$$. 
- Thus, our optimization target becomes : $$\frac{1}{2} \sum (y' - (C + Mx))^2$$

How gradient Descent works is by taking steps towards the optimal target. This is different from least-squares since in the latter we numerically compute the optimal solution by differentiating the target w.r.t C and setting it to 0 to find the inflection, which will be the minimal point. Gradient descent, on the other hand, works by first selecting a random value of intercept, say  $$C_1$$ , and then moving a step in the direction of decrease in value. This step is determined by the learning rate $$\alpha$$ which is a hyperparameter. So, at $$C_1$$, we differentiate the SOS target w.r.t C and calculate the value by putting $$C = C_1$$ and then multiply this value by alpha to get the next intercept point → Thus, when we are far away from the inflection, we take larger steps and when we are closer to the inflection, we take smaller steps since the slope is saturating. evident in N-dimensional metrics → the same thing is happening on hyperplanes
- The learning rate $$\alpha$$ determines the size of the steps we take and tuning this is important, since if it is too small, our time to convergence is slower, while if it is too large, we overshoot the solution → Classic control phenomenon!
- One solution is to start with a large learning rate and make it smaller with each step! → **Schedule the Learning Rate**


## Stochastic Gradient Descent

The computations in the Gradient Descent step scale up pretty fast and thus, convergence becomes an issue. SGD resolves this by sampling points for the intercept residual calculation!  → Instead of using all points, we can randomly sample n points - **Mini-batch** - and use them → This is especially helpful when points are clustered in different clusters, since the points in one cluster wil more-or-less have similar residuals! 
- Again the sensitivity to $$\alpha$$ comes into picture and again we can adapt scheduling  to overcome this!
