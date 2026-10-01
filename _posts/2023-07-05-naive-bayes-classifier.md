---
layout: post
title: Naive Bayes Classifier
date: 2023-07-05 11:57:00-0400
categories: machine-learning
giscus_comments: false
related_posts: false
---

## Multinomial Naive Bayes

Let's take the case of spam classification → we have emails with combinations of words that we want to use to classify future emails as spam or not. Let's say they contain combinations of four words: **Dear, Friend, Lunch, Money**.

Now, we just count the frequency of these four words in the normal emails and then assign **likelihoods** to each as follows. Say D = 8, F = 5, L = 3, M = 1:

$$
\begin{aligned}
&P(D\vert N) = 8/17 = 0.47 \\
&P(F\vert N) = 5/17 = 0.29 \\
&P(L\vert N) = 3/17 = 0.18 \\
&P(M\vert N) = 1/17 = 0.06 \\
\end{aligned}
$$

Now, we do the same for 7 spam emails:

$$
\begin{aligned}
&P(D\vert S) = 2/6 = 0.29 \\
&P(F\vert S) = 1/7 = 0.17 \\
&P(L\vert S) = 0/7 = 0 \\
&P(M\vert S) = 4/7 = 0.57 \\
\end{aligned}
$$

Then, we define the ratios for Normal (N) and Spam (S):

$$
\begin{aligned}
&P(N) = 0.67 \\
&P(S) = 0.33 \\
\end{aligned}
$$

Now, for every word combination we get, we just multiply the priors with the likelihoods and compare. For example, if we get **Dear Friend**:

$$
\begin{aligned}
&P(N) * P(D\vert N) * P(F\vert N) = 0.09 \\
&P(S) * P(D\vert S) * P(F\vert S) = 0.01 \\
\end{aligned}
$$

The key realization is that the product of the prior and the likelihoods, according to Bayes' theorem, should be proportional to the likelihood of the email being normal given the words seen, i.e., P(N), and the same for spam. Thus, directly comparing the two values above tells us that the email has a higher chance of being normal → we classify it as normal.

### Unseen words

But what if a word not previously encountered in spam is seen in the email? Take the example of **Lunch Money Money Money Money**:

$$
\begin{aligned}
&P(N) * P(L\vert N) * P(M\vert N)^4 = 2e-5 \\
&P(S) * P(L\vert S) * P(M\vert S)^4 = 0 \\
\end{aligned}
$$

This is a problem since it limits our ability to classify → we alleviate this by introducing **placeholder observations** into the spam group. These observations are $$\alpha$$ in number and can be included in the counting process for frequentist likelihoods to eradicate this problem → so, for the value of 1 additional observation, we get:

$$
\begin{aligned}
&P(D\vert N) = 9/(17 + 4) = 0.43 \\
&P(F\vert N) = 6/(17 + 4) = 0.29 \\
&P(L\vert N) = 4/(17 + 4) = 0.19 \\
&P(M\vert N) = 2/(17 + 4) = 0.10 \\
&\\
&P(D\vert S) = 3/(7 + 4) = 0.27 \\
&P(F\vert S) = 2/(7 + 4) = 0.18 \\
&P(L\vert S) = 1/(7 + 4) = 0.09 \\
&P(M\vert S) = 5/(7 + 4) = 0.45 \\
\end{aligned}
$$

Using the same prior values, we get:

$$
\begin{aligned}
&P(N) * P(L\vert N) * P(M\vert N)^4 = 1e-5 \\
&P(S) * P(L\vert S) * P(M\vert S)^4 = 1.22e-5
\end{aligned}
$$

Now, we see that the email is more likely to be spam!

### Why naive?

Naive Bayes treats language as just a bag of words, and so the score for **Dear Friend** would be the same as for **Friend Dear**. In the general sense, for any such problem, Naive Bayes does not exploit inter-dependencies in values, as seen from the segregation of probabilities into disjoint sets during calculation.

## Gaussian Variant

We do the same process, but this time we create Gaussian distributions for the variables to represent likelihoods. Thus, we take the points on these Gaussian curves as the likelihood values that need to be multiplied by the prior to generate the score.

- To manage underflow, we take the log of all probabilities and add them to calculate the score.
- The score with the higher log value is the class into which the new observation should be classified.

One way to study the impact of different variables is to weight the different classes → weighted log-loss. Cross-validation helps in determining which class has more impact.
