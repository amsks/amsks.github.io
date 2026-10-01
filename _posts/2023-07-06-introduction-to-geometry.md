---
layout: post
title: Introduction to Geometry
date: 2023-07-06 12:57:00-0400
categories: gauge-theory
giscus_comments: false
related_posts: false
---

Euclidean geometry rests on Euclid's five postulates, which are basically the set of rules for doing anything in Euclidean space:

1. Draw a straight line from any point to any point.
2. Produce a finite straight line continuously in a straight line.
3. Describe a circle with any center and distance.
4. All right angles are equal to one another.
5. If a straight line falling on two straight lines makes the interior angles on the same side less than two right angles, the two straight lines, if produced indefinitely, meet on that side on which are the angles less than the two right angles.

The fifth postulate, also known as the parallel postulate, is the one that was violated to create the following variants of geometry:

- **Spherical/Elliptical Geometry** → This assumes that parallel lines converge, for example, at the poles of a sphere. So, imagine two parallel lines that start somewhere on the equator and converge at the poles of a sphere. If we hold this sphere by the poles and pull it, we get a surface that is not equally curved at all points (an ellipse), and the same idea of converging lines applies to it too. The rules that govern this 'world' would then need to change. For example, the interior angles of a triangle on this surface would no longer sum to $$\pi$$ radians, but to something like $$\pi(1+4f)$$, where $$f$$ is the fraction of the sphere's surface that is enclosed by the triangle.
- **Hyperbolic Geometry** → Here we violate the fifth postulate by assuming a scenario where parallel lines diverge. An example surface that obeys the rules of this geometry is the shape of a Pringles chip (a hyperbolic paraboloid), on which we can easily produce two parallel lines and see that they diverge along the surface.

To go abstract, we want to understand how we can generalize to any kind of surface geometry → abstract and define the ideas that are central to these geometries. So, we start by understanding some properties of spherical and hyperbolic geometry.

## Nature of the Curvature

If we draw concentric latitudes around the pole, take the radius of each circle as the length measured from the center of the sphere, and then measure the circumference, we would see that it would actually be the same as the standard circle formula → it would be somewhat less for all points not at the equator:

$$
\begin{aligned}
&C = 2 \pi R \sin(\frac{r}{R}) \\
\implies &C < 2 \pi R
\end{aligned}
$$

This is like saying that the sphere tends to curve towards the horizontal origin, the pole. For the saddle (the Pringles chip), on the other hand, we would see that the curvature is not tending towards anything.

To make this notion more precise, we can take the second derivative of the surface at each point by defining a vector that is normal to the surface at each point, and classify the curvature:

- **Positive Curvature** → Curvature that tends to curve in the same direction as the tangent, i.e., the second derivative is negative.
- **Negative Curvature** → Curvature that tends away from the tangent, i.e., the second derivative is negative.
- **Zero Curvature** → The notion of flat.

This is somewhat similar to how we would define the points of maxima and minima for functions, and a 2D equivalent is shown below:

<div class="col-sm">
    {% include figure.html path="assets/img/GT/Intuitive/Intuitive-1.png" class="img-centered rounded z-depth-0" %}
</div>

Thus, we now have a notion that allows us to say that the spherical curvature is positive, while the hyperbolic curvature is negative. The Euclidean curvature is, of course, flat.

Another way to think of this would be that there is a notion of finiteness associated with the positive curvature of spherical geometry, which relates to parallel lines focusing on one point when produced further, while there is a notion of infinity associated with negative curvature that makes parallel lines diverge in the hyperbolic case. Since the Euclidean plane is flat, parallel lines keep going on to infinity without ever meeting.

<div class="col-sm">
    {% include figure.html path="assets/img/GT/Intuitive/Intuitive-2.png" class="img-centered rounded z-depth-0" %}
</div>

## Generalizing Geometry

- The first step to creating a general notion of geometry is to understand the point of view from which we talk about surfaces. The classical way of looking at geometry is through a higher-dimensional space in which the surface is embedded. So, when I am looking at a place, I exist in $$\mathbb{R}^3$$, in which there is a surface in $$\mathbb{R}^2$$ that I can see and then comment on its properties like curvature, etc. This is an **Extrinsic View**, and so the curvature is the Extrinsic Curvature of the surface. However, this might not be the ideal way to look at curvature, since we would always need a higher-dimensional space to be able to study any space.
- Another view of geometry is the Intrinsic View, where we study the space from the perspective of the space itself. This is the same as saying we take a space, get some 'rulers' to measure something like a distance on this space, and 'protractors' to measure something like an angle. Using these tools, we create a system that allows us to understand the curvature of our space in and of itself. This curvature is called the **Intrinsic Curvature**.
- To demonstrate this, consider the figure shown below. One way to think of it would be as a Euclidean space that has been 'waved' a bit. The extrinsic picture from 3D is pretty clear. However, if we think from the point of view of a creature bound to this 2D space → to the creature, this is still a flat surface.

<div class="col-sm">
    {% include figure.html path="assets/img/GT/Intuitive/Intuitive-3.png" class="img-centered rounded z-depth-0" %}
</div>

### Normal and Tangential Vectors

To understand why, we need to use vectors. Let's take the simple example of a sphere. At any point, we can define two vectors:

- **Normal Vector** → Protrudes outwards from the sphere, and so comes out into the 3D space.
- **Tangential Vector** → Is tangent to the surface at every point, and so remains in the tangent plane.

We can use normal vectors to define the extrinsic curvature → consider a normal vector at a point $$A$$ on a sphere. If we parallel transport this vector to a point $$B$$, i.e., take this vector and move it to point $$B$$ along some path while keeping its original orientation intact, we can then compare it with the normal vector at $$B$$. The difference between these vectors defines the extrinsic curvature of the surface.

<div class="col-sm">
    {% include figure.html path="assets/img/GT/Intuitive/Intuitive-4.png" class="img-centered rounded z-depth-0" %}
</div>

We can use tangential vectors to study the intrinsic curvature of this surface → if we take a tangential vector at point $$A$$ on this sphere, move it around a loop on the sphere, and then compare how it has changed, the change should be proportional to the curvature of the region enclosed by the loop. For example, in the figure below, if we take the tangential vector, transport it through the upper hemisphere to the other end, and then come back to the original point along the equator, we actually get a $$\pi$$ radian shift.

<div class="col-sm">
    {% include figure.html path="assets/img/GT/Intuitive/Intuitive-5.png" class="img-centered rounded z-depth-0" %}
</div>

This would not be the case if the vector were parallel transported on a Euclidean space, since in any loop we would get back the same vector. If we use this procedure on the wavy surface, we can see that the tangential vector would not change in a loop, but the normal vector would. Hence, we say that the surface is extrinsically curved but intrinsically flat.

## Riemann's Geometry

We can use the ideas above to create some notions around any curved surface we want. To do this, we first need to assume that the surfaces are smooth, i.e., there are no abrupt changes. This idea is formalized further in topology into the notion of a manifold.

For now, let's go with the notion that if we zoom into this smooth surface, we end up encountering a Euclidean space, similar to how the Earth seems flat but is actually curved (Flat-Earthers?). Thus, if we zoom in a good enough amount, we get an infinitesimally small Euclidean space. On this space, we would not need to define the notion of a distance, which comes from the L2 norm, i.e., the Pythagorean theorem:

$$
ds^2 = dx_1^2 + dx_2^2
$$

### The Metric Tensor

If we were to scale $$x_1, x_2$$ by some constants $$a_1, a_2$$ and then change the right angle to an angle $$\theta$$ between $$a_1x_1$$ and $$a_2x_2$$, then our equation would be modified to:

$$
ds^2 = a_1^2dx_1^2 + a_2^2dx_2^2 + 2a_1a_2dx_1dx_2\cos(\theta)
$$

This is called a **Metric Tensor**. We can express this in a general matrix form to make it extendable to more dimensions and to any further information that might be required to characterize the surface:

$$
\begin{bmatrix}
   g_{11} & g_{12} \\
   g_{21} & g_{22}
\end{bmatrix} =
\begin{bmatrix}
   a_1^2 & a_1a_2\cos(\theta) \\
   a_1a_2\cos(\theta) & a_2^2
\end{bmatrix}
$$

Thus, in general, we can write:

$$
ds^2 = g_{ij}dx^idx^j
$$

### Geodesics

We can use this metric tensor to measure the distance between any two points by adding up all the small distances $$ds$$ along the way:

$$
S = \int_a^b \sqrt{g_{ij}dx^idx^j} ds
$$

This distance can be along any path between the two points. If we consider the set of all paths that connect two points, we can then be interested in the shortest path out of this set. This is called a **Geodesic**. These points are given by the Euler-Lagrange formulation for an energy function $$E$$ defined as:

$$
E = \frac{1}{2} \int_a^b g_{ij}dx^idx^j ds \,\,\,\,\,\,\,\,\, s.t. \,\,\,\,\,\,\,\,\, S^2 \leq 2(b-a)E
$$

And the final equation for the geodesic comes out to be:

$$
dt^n + \Gamma_{mr}^nt^rdx^m = 0
$$

where $$\Gamma_{mr}^n$$ is called the Christoffel symbol and is defined as:

$$
\Gamma_{mr}^n = \frac{g^{np}}{2}\bigg[ \frac{\partial g_{pm}}{\partial x^r} + \frac{\partial g_{pr}}{\partial x^m} - \frac{\partial g_{mr}}{\partial x^p}\bigg]
$$

### The Riemann Curvature Tensor

Now, as we discussed with parallel transport previously, Riemann formalized that idea through the Riemann tensor → take a vector $$V_s$$ and pass it through a loop on a curved surface back to its original point to get a vector $$V_p$$. This change is denoted by a vector $$D_rV_s$$, which characterizes the curvature and can be written as:

$$
D_rV_s = \partial_rV_s - \Gamma_{rs}^p V_p
$$

This characterizes the curvature of the surface, and the general form of the curvature tensor is:

$$
R^t_{srn} = \partial_r \Gamma_{sn}^t - \partial_s \Gamma_{rn}^t + \Gamma_{sn}^p \Gamma_{pr}^t - \Gamma_{rn}^p \Gamma_{ps}^t
$$

Thus, we just need to specify a point and two basis vectors along which the loop needs to move, i.e., a total of three vectors, and we get the curvature at this point computed by the Riemann curvature tensor. Since it exists for all points, we can also say that this is a field, i.e., the metric takes a value at each point, and based on where the points are, we can have values for the metric.

If we extend this idea further, we see that for a collection of two points we will always have values pertaining to paths between these points. We can call this a connection field. This connection, as we saw before, comes from the metric tensor. Thus, we can say that the metric gives the connection between two points, and the connection gives the curvature. Each path may have a different curvature depending on the nature of the surface.
