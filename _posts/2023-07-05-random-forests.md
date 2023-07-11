---
layout: post
title: Random Forests and Adaboost
date: 2023-07-05 15:57:00-0400
categories: machine-learning
giscus_comments: false
related_posts: false
---

## Random Forests

The issue with Decision Trees is that they are not flexible to achieve high accuracy. So, we use Random Forests, which alleviate this problem by creating multiple trees from different starting points. The steps are as follows: 

- Create a Bootstrapped Dataset of the same size by selecting random samples with replacement 
{% include figure.html path="assets/img/MALIS/RF/rf-1.png" class="img-centered rounded z-depth-0" %}
{% include figure.html path="assets/img/MALIS/RF/rf-2.png" class="img-centered rounded z-depth-0" %}
    
- Create a Decision tree by selecting questions at random from this bootstrapped dataset, and use the impurtiy function to segregate the metric to used
{% include figure.html path="assets/img/MALIS/RF/rf-3.png" class="img-centered rounded z-depth-0" %}
- Wherever a question needs to be asked, select the new metric randomly out of the metrics except the one used i.e in this case Good blood Circulation
{% include figure.html path="assets/img/MALIS/RF/rf-4.png" class="img-centered rounded z-depth-0" %}
- Go back to step 1 and repeat to create a new bootstrapped dataset and repeat everything to create another tree → Do this process a fixed number of times.

Thus, by creating a variety of trees → A forest → We are able to get trees with different performances that can predict the labels. For new data items, run it down all the trees and keep a track of the classification - Yes and No - and then choose the classification with the bigger number → **Bagging = Bootstrapping + Aggregating**. Thus, Bagging is an ensemble technique where we train multiple classifiers on subsets of our training data and then combine them to create a better classifier.

### Evaluating RF

the entries that didn't' end up in the bootstrap dataset - Out of bag Data - are run through the tree to get the classification from all the trees and again use Bagging to see what the final classification is → For all out of bag samples, we evaluate the confusion matrix and calculate the precision, accuracy, sensitivity, and specificity  

### Hyperparameters in RF
The hyperparameters in the RF are :

1. m → The number of variables we are using out of the subset in bootstrap to create the tree 
2. k → the number of trees we have in the forest 

We can do the Out-of-Bag stuff on different random forests and select the one with the best accuracy

## AdaBoost

Learners can be considered weak or strong as follows: 
- **Weak** → Error Rate is only slightly better than random
- **Strong** → Error Rate highly correlated with the actual classification

**Adaboost combines a lot of weak learners to create a strong classifier!**. This is characterized by 3 key points: 

1. It creates an RF of **stumps** → trees with only one question used for classification → which act as weak learners
2. All stumps in this forest don't have equal **say**, some have more and some have less and they are used as  weights for the classification that each stump makes
3. The errors made by the previous stumps are taken into account by the next stump to reduce misclassification i.e the stumps sequentially try to reduce misclassification in contrast to a vanilla RF where the stumps are all separate

The steps are a follows:
- Start with the dataset, but assign each data point a weight i.e create a new column with weights, which have to be normalized, and at the start, all have equal values
{% include figure.html path="assets/img/MALIS/ADA/ada-1.png" class="img-centered rounded z-depth-0" %}

- Use a weighted impurity to classify nodes → We use the same formula, but for each label we use the associated weights in the gini calculation. Since all weights are the same, we ignore them for now and see that the gini for patient weight is the lowest, so we use it for our first stump:
- Now we see how many errors this stump made → in this case it is 1. We determine the say of this stump by summing the weights of the erroneously classified samples → $$E = \sum W_i$$ → and the total say as $$S = \frac{1}{2} log(\frac{1 - E}{E})$$ → we get the say as 0.47 for this stump
- Now we update the weight of the incorrectly classified sample using the formula: $$w \leftarrow w * e^S$$ and so, we get the new weight for the incorrect sample as 
- Now we will decrease the amount of weights for all the correctly classified samples by using the formula : $$w \leftarrow w * e^{-S}$$ which gives us the new weights of all other labels as 0.05
- Now we normalize the updated weight by dividing each weight by the sum all weights
{% include figure.html path="assets/img/MALIS/ADA/ada-2.png" class="img-centered rounded z-depth-0" %}
- We repeat the procedure using a weighted gini index or just creating duplicates of the samples with the large weights.  

The main thing that characterizes this algorithm is Boosting, which is a fancy name for when we train multiple weak classifiers to create a stringer classifier by taking errors of the previous classifiers into account. AdaBoost is creating a forest of stumps, but each new stumps uses the normalized weights to determine which kind of mis-classifications to focus on and thus, in a sense, uses the errors of the previous stumps to improve. 
