---
layout: post
title: Generic ML Concepts
date: 2023-07-05 17:57:00-0400
categories: machine-learning
giscus_comments: false
related_posts: false
---

## Cross-Validation

Cross-validation allows us to compare different ML methods! When working with a dataset for training a learner:

- Naive idea → use all the data.
- Better idea → use x percent for training and y percent for testing.

The issue is: how do we know that the selection is good? The lord answers through **K-Fold Cross-Validation** → split the data into K blocks → for each block, train on the remaining K-1 blocks, test on that block, and log the metric → average out the performance and use this for comparison.

- **Leave-one-out CV** → use each sample as a block.

## Confusion Matrix

Plot the predicted positives and negatives vs. the ground truths:

<div class="col-sm">
    {% include figure.html path="assets/img/MALIS/Confusion.png" class="img-centered rounded z-depth-0" %}
</div>

**NOTE:**

- The diagonal always shows the true values.
- The non-diagonal elements are always false.

The major metrics are:

- **Sensitivity** → positive labels correctly predicted: $$\frac{TP}{TP + TN}$$
- **Specificity** → negative labels correctly predicted: $$\frac{TN}{TN + FN}$$

Let's say we test out logistic regression against random forests to classify patients with and without heart disease. Then the algorithm with the higher sensitivity should be chosen if our target is to classify patients with heart disease, while the algorithm with the higher specificity should be chosen if we want to classify patients without heart disease.

### What About Non-Binary Classification?

Calculate these values for each label by treating the values as label and !label. For example, if we have three labels, we take the true positives as all the classifications done for label i and the TN as all the misclassifications done for label i → this means that if the data actually belonged to the other classes and was still classified as belonging to i, then it is a false positive.

Similarly, we take true negatives as all the classifications done on all other classes except our current label, and the false negatives as the classifications. Let's take the following example:

<div class="col-sm">
    {% include figure.html path="assets/img/MALIS/Confusion-2.png" class="img-centered rounded z-depth-0" %}
</div>

Here, for the class Cat, we get:

- Sensitivity $$= 5/(5 + 3 + 0) = 5/8 = 0.625$$
- Specificity $$= (3 + 2 + 1 + 11)/(3 + 2 + 1 + 11 + 2) = 17/19 = 0.894$$

Other major metrics are:

- **Accuracy** → $$(TP+TN)/\text{total} = (100+50)/165 = 0.91$$
- **Misclassification Rate** → $$(FP+FN)/\text{total} = (10+5)/165 = 0.09 = 1 - \text{accuracy}$$
- **Precision** → $$TP/\text{predicted yes} = 100/110 = 0.91$$
- **TP Rate** → $$TP/\text{yes} = 100/105 = 0.95$$
- **FP Rate** → $$FP/\text{no} = 10/60 = 0.17$$

The idea is to strike a balance between these and get the hang of how our classifier is actually performing!

## Bias and Variance

- **Bias** → the inability of an algorithm to capture the true relationship in the data. Formally, it is the inherent error that we obtain from the model even with infinite training data, due to the classifier being biased to a particular solution.
- **Variance** → the difference in fits between the training and the testing data, i.e., the error caused by sensitivity to fluctuations in the training set.

High bias means the learned model is simpler and might not fit the training data very well, and so it does not perform so well on the test set → **Underfitting**. High variance means that the learned model has a better fit to the training set but does not perform so well on the test set → **Overfitting**.

What is happening is that the training set can essentially be viewed as the true relationship curve plus some noise that scatters the data around the curve. This is the same for training and test sets. Now, if our model fits so well to the training set that it is able to pass exactly through each data point, it has actually fitted to the noise that scattered the data from the actual signal.

Thus, it has so much variability that it won't perform well on other datasets, which might inherently be sampled from the same curve with some random noise that scatters the data a bit differently. This is why the model has overfitted to the training set by adapting to the noise.

In general, the error depends on the square of the bias and varies directly with the variance and the noise:

$$
E = B^2 + V + N
$$

And this variation can be plotted as follows:

<div class="col-sm">
    {% include figure.html path="assets/img/MALIS/BVT.png" class="img-centered rounded z-depth-0" %}
</div>

## ROC and AUC

The whole idea of the ROC curve is adjusting our classification threshold (for example, in the case of logistic regression) to mess around with the rates of TP and FP. We plot these values for each threshold against each other on a graph, as shown:

<div class="col-sm">
    {% include figure.html path="assets/img/MALIS/ROC.png" class="img-centered rounded z-depth-0" %}
</div>

Here, we have plotted sensitivity, a.k.a. the true positive rate, against the FP rate, a.k.a. 1 - specificity. At point (1,1), we see that our classifier is classifying all samples as TP and FP. Now, let's say our problem is to predict whether the patient has a certain disease or not; then this is not acceptable, since the FP rate is high and we can't afford false positive classifications.

So, we adjust our threshold and see the sensitivity remain the same through the next two points on the left, but the FP rate decreases, which means our model is getting better. Then we see that both rates fall, and then, finally, our model reaches a level where the TP rate is positive while the FP rate is negative. This is a desirable performance for our purposes. In case we are willing to accept some FPs for a better TP classification, we can select points on the right that increase the TP but also end up having some misclassifications.

AUC (Area Under the Curve) is used to compare the performance of two classifiers, as shown below:

<div class="col-sm">
    {% include figure.html path="assets/img/MALIS/AUC.png" class="img-centered rounded z-depth-0" %}
</div>

Since the AUC is greater for the red curve, the model that it represents is better, since for the same levels of FP it delivers more TPs.
