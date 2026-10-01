---
layout: post
title: Differentiable Structures
date: 2023-07-06 16:57:00-0400
categories: gauge-theory
giscus_comments: false
related_posts: false
---

To get differentiable manifolds, we need to refine the notion of compatibility. We do this by saying that an atlas $$\mathcal{A}$$ is $$K$$-compatible if $$\forall (U,x), (V,y) \in \mathcal{A}$$ either of the following two conditions holds:

- $$U \cap V = \Phi$$.
- $$U \cap V \neq \Phi$$ → $$U,V$$ satisfy some condition $$K$$.

Usually, $$K$$ is defined to be one of:

1. $$C^0$$.
2. $$C^k$$ → Transition maps are $$k$$-times differentiable.
3. $$C^{\infty}$$ → Smooth transitions.
4. $$C^{\omega}$$ → Analytic, i.e., can be expanded through a Taylor expansion.
5. Complex → Transition functions satisfy the Cauchy-Riemann equations.

**Whitney Theorem** → The theorem says:

- Any maximal $$C^k$$ atlas contains a $$C^{\infty}$$ atlas.
- Any two $$C^k$$ atlases that contain the same $$C^{\infty}$$ atlas are identical.

This implies that once we have a $$C^1$$ atlas, we can essentially construct a smooth manifold. Thus, in terms of differentiable structure, we don't need to worry about varying values of $$k$$. **The only thing we need to look for is whether the function is differentiable once or not.**

## Smooth Manifold

A $$C^k$$ manifold is a triple $$(M,O,\mathcal{A})$$ where:

- $$(M,O)$$ → Topological space.
- $$\mathcal{A}$$ → $$C^k$$ atlas.

We can create this atlas by simply taking a $$C^0$$ atlas and removing the pairs that are not differentiable.

### Differentiable Maps

If we have two manifolds $$(M, O_M, \mathcal{A}_M)$$ and $$(N, O_N, \mathcal{A}_N)$$, then we can say that a map

$$\phi: M \rightarrow N$$

is differentiable if, for some points in some charts in the two atlases:

$$
\begin{aligned}
&p \in U : (U,x) \in \mathcal{A}_M \\
&q \in V : (V,y) \in \mathcal{A}_N \\
&q = \phi(p)
\end{aligned}
$$

the transformation map that results:

$$y \circ \phi \circ x^{-1}$$

is $$C^k$$ in $$\mathbb{R}^{\dim(M)} \rightarrow \mathbb{R}^{\dim(N)}$$.

Thus, the notion of differentiability, as expected, depends on the representation of $$U,V$$ in the chosen atlases, i.e., $$x(u \in U), y(v \in V)$$, and this raises the question → what if we had another chart? If this differentiability exists as a notion on all such charts, maybe we can also 'lift' this concept from the chart to the manifold level. To do this, let's consider another chart such that:

$$
\begin{aligned}
&p \in U : (U,x') \in \mathcal{A}_M \\
&q \in V : (V,y') \in \mathcal{A}_N \\
&q = \phi(p)
\end{aligned}
$$

Given our original maps $$(U,x), (V,y)$$, we don't really have a guarantee that the transformation on the new map:

$$y' \circ \phi \circ x'^{-1}$$

is a differentiable structure. However, if we look at a transformation from $$x,y$$ to $$x', y'$$, we can definitely write it as:

$$
\begin{aligned}
&x(U) \xrightarrow{x' \circ x^{-1}} x'(U) : \mathbb{R}^{\dim(M)} \rightarrow \mathbb{R}^{\dim(M)} \\
&y(V) \xrightarrow{y' \circ y^{-1}} y'(V) : \mathbb{R}^{\dim(N)} \rightarrow \mathbb{R}^{\dim(N)}
\end{aligned}
$$

And because this transformation exists, we can say that the second chart is also differentiable, since we can always transform it to the first chart and then impose differentiability. This is summarized in the figure below:

<div class="col-sm">
    {% include figure.html path="assets/img/GT/DS.png" class="img-centered rounded z-depth-0" %}
</div>

This relation would, by extension, be true for any chart for which this relation holds → a $$C^k$$-compatible chart. If the map $$\phi : M \rightarrow N$$ is $$C^{\infty}$$-compatible, then it is called a **diffeomorphism**, and the two manifolds $$(M, O_M, \mathcal{A}_M)$$ and $$(N, O_N, \mathcal{A}_N)$$ are called diffeomorphic if there exists a diffeomorphism between them:

$$M \cong_{\mathrm{diff}} N$$

Usually, we consider diffeomorphic manifolds the same as smooth manifolds.

### How many different differentiable structures can we put on a manifold up to diffeomorphism?

The answer depends on the dimension:

1. $$\dim = 1,2,3$$ → **Radó-Moise Theorems** → We can make a unique differentiable manifold from a topological manifold, since all the different ones are diffeomorphic, which allows us to work easily with differentiability.
2. $$\dim > 4$$ → **Surgery Theory** → We can essentially understand a higher-dimensional torus by using familiar structures like a sphere and a cylinder, which can be 'intelligently' combined. If we take a sphere, make a hole, and insert a cylinder (i.e., perform surgery) while controlling invariants like the fundamental group, homotopy groups, homologies, etc., then we can essentially understand the torus, since we understand the sphere and the cylinder. The assertion is that we can, similarly, understand all structures in higher dimensions by performing intelligent surgery. In the 1960s, it was shown that there are finitely many smooth manifolds one can make from a topological manifold in dimensions greater than 4. One practical application of this to physics, in principle, could be that if we assume spacetime is a differentiable manifold of higher dimensions, then pure math tells us that there exist finitely many ways in which this higher-dimensional structure could be projected to our 3D understanding, and we could conduct experiments to determine which one of these nature has chosen!
3. $$\dim = 4$$ → For the case of compact spaces, there are uncountably many different smooth manifolds that can be created! For the case of compact spaces, we look at partial results based on Betti numbers. Thus, when we look at Einstein's description of spacetime as $$\mathbb{R}^4$$, we can see that there are uncountably many smooth manifolds, as far as we know from the analysis. Thus, if our theories fail, they could very well fail because of our choice of the structure.

One of the key features of differentiable manifolds is tangent spaces. Since we are speaking of geometry intrinsically, we need to develop an intuition that is separate from the embedding space in which an object exists.
