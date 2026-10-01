---
layout: post
title: Random Forests and Adaboost
date: 2023-07-05 15:57:00-0400
categories: machine-learning
giscus_comments: false
related_posts: false
---

## Random Forests

The issue with Decision Trees is that they are not flexible enough to achieve high accuracy. So, we use Random Forests, which alleviate this problem by creating multiple trees from different starting points. The steps are as follows:

- Create a bootstrapped dataset of the same size by selecting random samples with replacement.
{% include figure.html path="assets/img/MALIS/RF/rf-1.png" class="img-centered rounded z-depth-0" %}
{% include figure.html path="assets/img/MALIS/RF/rf-2.png" class="img-centered rounded z-depth-0" %}

- Create a decision tree by selecting questions at random from this bootstrapped dataset, and use the impurity function to choose the metric to be used.
{% include figure.html path="assets/img/MALIS/RF/rf-3.png" class="img-centered rounded z-depth-0" %}
- Wherever a question needs to be asked, select the new metric randomly out of the metrics except the one already used, i.e., in this case, Good Blood Circulation.
{% include figure.html path="assets/img/MALIS/RF/rf-4.png" class="img-centered rounded z-depth-0" %}
- Go back to step 1 and repeat to create a new bootstrapped dataset, and repeat everything to create another tree → do this process a fixed number of times.

Thus, by creating a variety of trees → a forest → we are able to get trees with different performances that can predict the labels. For a new data item, run it down all the trees, keep track of the classifications (Yes and No), and then choose the classification with the bigger count → **Bagging = Bootstrapping + Aggregating**.

Thus, bagging is an ensemble technique where we train multiple classifiers on subsets of our training data and then combine them to create a better classifier.

### Evaluating RF

The entries that didn't end up in the bootstrapped dataset (out-of-bag data) are run through the trees to get the classification from all of them, and we again use bagging to see what the final classification is → for all out-of-bag samples, we evaluate the confusion matrix and calculate the precision, accuracy, sensitivity, and specificity.

### Hyperparameters in RF

The hyperparameters in the RF are:

1. m → the number of variables we are using out of the subset in the bootstrap to create the tree.
2. k → the number of trees we have in the forest.

We can do the out-of-bag evaluation on different random forests and select the one with the best accuracy.

## AdaBoost

Learners can be considered weak or strong as follows:

- **Weak** → error rate is only slightly better than random.
- **Strong** → error rate is highly correlated with the actual classification.

**AdaBoost combines a lot of weak learners to create a strong classifier!** This is characterized by 3 key points:

1. It creates an RF of **stumps** (trees with only one question used for classification), which act as weak learners.
2. All stumps in this forest don't have an equal **say**: some have more and some have less, and these are used as weights for the classification that each stump makes.
3. The errors made by the previous stumps are taken into account by the next stump to reduce misclassification, i.e., the stumps sequentially try to reduce misclassification, in contrast to a vanilla RF where the stumps are all separate.

The steps are as follows:

- Start with the dataset, but assign each data point a weight, i.e., create a new column with weights, which have to be normalized; at the start, all have equal values.
{% include figure.html path="assets/img/MALIS/ADA/ada-1.png" class="img-centered rounded z-depth-0" %}

- Use a weighted impurity to classify nodes → we use the same formula, but for each label we use the associated weights in the Gini calculation. Since all weights are the same, we ignore them for now and see that the Gini for patient weight is the lowest, so we use it for our first stump.
- Now we see how many errors this stump made → in this case, it is 1. We determine the say of this stump by summing the weights of the erroneously classified samples → $$E = \sum W_i$$ → and the total say as $$S = \frac{1}{2} \log(\frac{1 - E}{E})$$ → we get the say as 0.47 for this stump.
- Now we update the weight of the incorrectly classified sample using the formula $$w \leftarrow w * e^S$$, and so we get the new weight for the incorrect sample as
- Now we decrease the weights of all the correctly classified samples using the formula $$w \leftarrow w * e^{-S}$$, which gives us the new weights of all other labels as 0.05.
- Now we normalize the updated weights by dividing each weight by the sum of all weights.
{% include figure.html path="assets/img/MALIS/ADA/ada-2.png" class="img-centered rounded z-depth-0" %}
- We repeat the procedure using a weighted Gini index, or just by creating duplicates of the samples with large weights.

The main thing that characterizes this algorithm is boosting, which is a fancy name for training multiple weak classifiers to create a stronger classifier by taking the errors of the previous classifiers into account. AdaBoost creates a forest of stumps, but each new stump uses the normalized weights to determine which kinds of misclassifications to focus on and thus, in a sense, uses the errors of the previous stumps to improve.
