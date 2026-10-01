---
layout: post
title: Topological Manifolds
date: 2023-07-06 14:57:00-0400
categories: gauge-theory
giscus_comments: false
related_posts: false
---

We can define a Topological Manifold as a paracompact, Hausdorff topological space $$(M,O)$$ in which every point $$p \in M$$ has a neighborhood $$p \in U \in O$$ for which there exists a homeomorphism:

$$x : U \rightarrow x(U) \subseteq \mathbb{R}^d$$

In other words, a topological space is a manifold if it behaves like $$\mathbb{R}^d$$ locally. This manifold, as is obvious, is $$d$$-dimensional. For example, a circle and a square are homeomorphic and represent the $$S^2$$ manifold.

### Sub-Manifold

We can also extend this notion to subsets of our topological space → so, if we have a space $$N \subseteq M$$, then we call it a sub-manifold of $$M$$ if it is a manifold in its own right. All we need to check, thus, is whether the induced topology $$(N, O \mid_N)$$ is a manifold or not.

So, a circle can be considered a sub-manifold of $$\mathbb{R}^2$$, but two circles touching each other at exactly one point might not satisfy this, since the point at which they touch locally behaves like $$\mathbb{R}, \mathbb{R}$$, which is not the same as $$\mathbb{R}^2$$.

### Product Manifolds

We can also define a product manifold by taking two manifolds $$(M, O_M), (N, O_N)$$ and defining a topological manifold $$(M \times N, O_{M \times N})$$, which has dimension $$\dim(M) + \dim(N)$$. For example, a toroid is a product of two circles and can be written as $$T^2 = S^1 \times S^1$$, while a cylinder is $$C = S^1 \times \mathbb{R}$$.

<div class="wrapper">
    <div class="col-sm">
        {% include figure.html path="assets/img/GT/TopMan/TopMan-1.png" class="img-centered rounded z-depth-0" %}
    </div>
</div>

A Möbius strip is a curious case, since it cannot be written as a product manifold, even though locally it looks like one. To describe this, we need to define something new, called a bundle.

## Bundles

Bundles are pretty central to a lot of things. A bundle of topological manifolds can formally be defined as a triplet $$(E, \pi, M)$$ where:

- $$E$$ → Total space.
- $$M$$ → Base space.
- $$\pi$$ → A continuous surjective map from the base space to the total space, a.k.a. a projection.

For a point $$p \in M$$, the pre-image of the set containing only $$p$$ under the map $$\pi$$ is called a Fibre:

$$F := \mathrm{preim}_\pi (\{ p\}) \,\,\,\,\, \exists p \in M$$

For example, let's take a product manifold. For a Fibre Bundle $$F$$ and a base space $$M$$, we can define the total space as:

$$
\begin{aligned}
E = M \times F \\
\pi : M \times F \rightarrow M
\end{aligned}
$$

So, a Möbius strip can be thought of as a bundle constructed by taking a rectangle, identifying the sides going in opposite directions, and then projecting the points onto the center.

<div class="col-sm">
    {% include figure.html path="assets/img/GT/TopMan/TopMan-2.png" class="img-centered rounded z-depth-0" %}
</div>

We find that even though it is not a product manifold, we can say that the pre-image of every point maps to the interval $$[-1, 1]$$, and this makes it a bundle of $$S^1$$.

We can see that bundles are essentially a generalization of the idea of taking a product, by intuitively understanding that to make a bundle we basically take a base space and attach fibers to it in a certain way. However, the definition does not really mention any notion of the total space being built out of the base space, and this is the generalization bit.

### Fiber Bundle

We can be a bit more restrictive in our notion of a bundle and yet be more general than a simple product space. To better elucidate this, we can say that in a bundle the fiber need not be the same for all points → we are only interested in the existence of some fiber as per its definition. So, if we restrict the points to having the same fiber:

$$
F := \mathrm{preim}_\pi (\{ p\}) \,\,\,\,\, \forall p \in M
$$

then we call $$E \xrightarrow{\pi} M$$ a Fiber Bundle with the Typical Fiber $$F$$. We often write the map as:

$$
F \rightarrow E \xrightarrow{\pi} M
$$

Thus, fiber bundles lie between product manifolds and general bundles.

### Section

Once we have a fiber bundle, we can further define a section of the bundle as a map $$\sigma : M \rightarrow E$$ such that if we take a point $$p \in M$$ and map it to some point $$q \in E$$, and then use $$\pi : E \rightarrow M$$ to map $$q$$ back to $$M$$, then the projection of $$q$$ is the same point:

$$\pi \circ \sigma = \mathbf{I}_M$$

<div class="col-sm">
    {% include figure.html path="assets/img/GT/TopMan/TopMan-3.png" class="img-centered rounded z-depth-0" %}
</div>

A very good example of this is in quantum mechanics → the wave function $$\Psi$$ is a section of the complex line bundle ($$\mathbb{C}$$-line bundle) over some physical space, such as $$\mathbb{R}^3$$. This goes from the physical space to the complex space.

### Sub-Bundles

We can use the same logic as for sub-manifolds to create sub-bundles. We take a bundle $$E \xrightarrow{\pi} M$$ and then define another bundle $$E' \xrightarrow{\pi'} M'$$. This new bundle is a sub-bundle if it meets the following three conditions:

1. $$E' \subset E$$.
2. $$M' \subset M$$.
3. $$\pi \mid_{M'} = \pi'$$ → when we restrict the projection map of our parent bundle to the base space of the other bundle, we should essentially get the projection map of the other bundle.

### Isomorphism in Bundles

Suppose we have two bundles:

$$
\begin{aligned}
&E \xrightarrow{\pi_E} M \\
&F \xrightarrow{\pi_F} N
\end{aligned}
$$

and two maps:

$$
\begin{aligned}
&\varphi : E \rightarrow F \\
&f : M \rightarrow N
\end{aligned}
$$

Then we call this pair a bundle morphism. This can essentially be seen as true if the diagram below commutes.

<div class="col-sm">
    {% include figure.html path="assets/img/GT/TopMan/TopMan-4.png" class="img-centered rounded z-depth-0" %}
</div>

Now, suppose we also have $$(\varphi^{-1}, f^{-1})$$ as another bundle morphism such that:

$$
\begin{aligned}
&\varphi^{-1} : F \rightarrow E \\
&f^{-1} : N \rightarrow M
\end{aligned}
$$

Then the above two bundles are called isomorphic, since they clearly have the same fiber, and these isomorphic bundles are the structure-preserving maps.

The essence of bundles, thus, lies not only in the topology of the manifolds but also in the projection → we can have topological spaces that are homeomorphic to each other, but if the projection does not create an inverse mapping, then they won't be isomorphic as bundles. Bundles can also be **locally isomorphic** if we restrict the mapping and they still maintain the relationship.

### Common Terminology on Bundles

- **Trivial Bundle** → A bundle that is isomorphic to a product bundle.
- **Locally Trivial** → A bundle that is locally isomorphic to a product bundle. E.g., a cylinder is a trivial, and so locally trivial, bundle, while a Möbius strip is locally trivial but not a trivial bundle. Locally, any section of a bundle can be represented as a map from the base space to a fibre. Thus, in quantum mechanics, it is okay to talk about $$\Psi$$ locally as a function, but there might be places in the space where we cannot do so.
- **Pull-Back Bundle** → A fiber bundle that is induced by a map of its base space. It allows us to create a sort of yellow data from the white data. For example, if we have $$M' \xrightarrow{f} M$$ and $$E \xrightarrow{\pi} M$$, then we can find the pull-back bundle $$E' \xrightarrow{\pi'} M'$$ as:

    $$E' := \big \{(m', e) \in M \times E \,\, \big \vert  \, \pi(e) = f(m') \big \}$$

## Viewing Manifolds from Atlases

Let $$(M,O)$$ be a topological manifold of dimension $$d$$. Then a pair $$(U, x)$$, where

$$
\begin{aligned}
&U \in O \\
&x : U \rightarrow \mathbb{R}^d
\end{aligned}
$$

is called a chart of the manifold. This is just terminology formalizing the notion that the neighborhood of a point in a manifold that maps to some subset of $$\mathbb{R}^d$$ is called a chart.

However, since $$x$$ maps to $$\mathbb{R}^d = \mathbb{R} \times \mathbb{R} \times \dots$$, we can now say that the components of $$x$$ are essentially the coordinates of a point $$p \in U$$ w.r.t. the chart $$(U,x)$$. This is crucial to understand, since now we are realizing that on any topological manifold, we can only define coordinates based on a chart, and we can have different such charts. Thus, there has to exist a set of charts such that every point is covered, i.e.:

$$\cup_{(U,x) \in A} U = M$$

Thus, there will be many non-empty charts that overlap, and the collection of such charts is called an **Atlas**.

### Compatibility in Charts

Two charts $$(U,x)$$ and $$(V,y)$$ are called $$C^0$$-compatible if either of the following conditions is met:

1. $$U \cap V = \Phi$$.
2. $$U \cap V \neq \Phi$$, but $$y \circ x^{-1}$$ exists.

Now, this is essentially the case when we are looking at manifolds. However, as we can see, compatibility allows us to traverse between two charts without really worrying about the underlying manifold.

For example, in physics, we transform between coordinate systems (which are the charts in this case), but we work with the fundamental assumption that this transformation does not change the trajectory of the particle. In other words, we can see the trajectory as a curve on a manifold and the coordinate systems of measurement as charts that are $$C^0$$-compatible. When the charts in an atlas are pairwise compatible, we get a $$C^0$$-Atlas.
