---
layout: post
title: Using Language for (Meta-) RL
date: 2023-07-05 14:57:00-0400
categories: machine-learning
giscus_comments: false
related_posts: false
---

It has been argued that language can be a very powerful way to compress information about the world.
In fact, the learning of humans is significantly sped up around the time they start understanding and using language.
A natural question, then, is whether the same can be argued for sequential decision-making systems
that learn to optimize either a single task or multiple tasks. To this end, there has been a surge of
works exploring the use of language in Reinforcement Learning (RL) and Meta-RL.
The goal of this blog post is to try and explain some of the recent works in this sub-field and
help elucidate how language can help with incorporating structure about the environment to improve
learning generalization in (Meta) RL.

## Introduction

We live in a time where sequential decision-making techniques like Reinforcement Learning (RL)
are making increasingly larger strides, not just in robot manipulation, and in games once considered
to be the pinnacle of human intelligence, but also in an increasingly novel set of scenarios like
chemistry and logistics. While a lot of the theory in RL existed from classical times, the success
of integrating Deep Neural Networks (DNNs) as function approximators has created a sort of Cambrian
explosion in recent years. Traditionally, a major focus of the field was on developing techniques
that can learn to solve an inherent optimization problem, like learning a solution to a maze. As
the field has evolved in recent years, its scope has started to broaden to encompass bigger questions,
like whether a learned policy to solve a maze can generalize to other configurations (Generalization
in RL), or whether a policy can be transferred to scenarios where conditions differ slightly from
the training conditions (Robustness, Deployability), or how we can design agents in a data-driven
manner (AutoRL). Yet, a major bottleneck in current RL techniques is that they are, largely, not yet
ready for real-world deployment.

In parallel, another Cambrian explosion has been happening in the field of Natural Language
Processing (NLP). Language models have come a long way since the days of word embeddings and
sequence-to-sequence models to the age of attention and pre-trained models. Crucially, as this growth
continues with newer innovations like ChatGPT, it also leads us to innovative applications of these
language models in other fields of Machine Learning, including RL.

In this blog post, I will explore the connection between natural language and RL through some recent
and not-so-recent works that I find very interesting. Since this topic is vast, so much so that a
full survey has been written on it, I will limit the focus to how language can be used to augment
RL pipelines (Language-assisted RL), and not on the use of RL for language training (RL for
language). Through this blog, my hope is to visit two ideas in using Language for RL that exist at
two very different points in the Deep RL timeline, and yet hold significance in the way they use
language to augment the RL pipeline.

## RL Basics

To better cater to audiences beyond the RL community, I think it would be good to briefly revise
some core concepts. RL folks are more than welcome to skip to the next section. I am going to try
my best to keep it less math-oriented, but I apologize in advance for the symbols I
will use.

<div class="col-sm">
    {% include figure.html path="assets/img/MALIS/DT/dt-1.png" class="img-centered rounded z-depth-0" %}
</div>
