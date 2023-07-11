---
layout: post
title: Algebra of Genetic Algorithms
date: 2023-07-08 08:57:00-0400
categories: evolutionary-computation
---

## Formalizing genetic Algorithm Search

To formalize genetic algorithms we need to define inputs and outputs and formalize what this algorithm does through maps. So, let's consider $$X$$ to be the space of all solutions. From elements of this space of solutions, we can construct multisets by grouping them, so let's take $$P_M(X)$$ to be this maximal set of multi-sets. Thus, a population $$P$$ can  be thought of as one of these possible multisets and so 

$$P \in P_M(X)$$

Now, the process of mutation and cross-over can be formalized as a genetic operator that maps $$P_M(X)$$, which can also be thought of as mapping $$X^k$$ to $$X$$. Let $$\Omega$$  be a genetic operator. Thus, we can say: 

$$\Omega:P_M(X) \rightarrow P_M(X) \\
\Omega : X^k \rightarrow X  \,\,\,\, \,\,\,\,\forall k \in \mathbb{N} \\
\implies \Omega (x_1, ..., x_k \in X) = x \in X$$

Thus, we can now classify the operators as follows: 

- Recombination → $$\Omega : X^2 \rightarrow X$$
- Mutation → $$\Omega : X \rightarrow X$$
- Selection → $$\Omega: A \subseteq P_M(X) \rightarrow B \subseteq A$$

## Forma Analysis

### Schema Theorem

Forma analysis creates a rigorous framework to look at how genetic algorithms might work on different problems and comes from Schema Theorem. According to the original paper, the Schema Theorem can be essentially summarized as 

$$
\mathbb{E} \{N_\xi(t+1) \} \geq N_\xi(t) \frac{\hat{\mu}_\xi(t)}{\bar{\mu}(t)} \bigg[  1 - \sum_{\omega \in \Omega}p_\omega p_\omega ^\xi\bigg]
$$

The various elements in this equation are: 

- $$\xi$$  → Schema which is like a template of strings in which certain positions are fixed. For example, a schema for binary strings like 1XX0X1 means that the first, fourth, and sixth positions have to be 1, 0, and 1, respectively, and the other positions can be either 0 or 1
- $$N_\xi(t)$$  → Population size at time $$t$$ that belongs to a schema $$\xi$$
- $$\hat{\mu}_\xi(t)$$ → Average fitness of all the members of the population at time t that are instances (members) of the schema $$\xi$$
- $$\bar{\mu}(t)$$ → average fitness of the whole population at time t
- $$\Omega$$  → Set of genetic operators in Use
- $$p_\omega p_\omega ^\xi$$ → This signified the potentially disruptive effect when we apply the operators $$\omega \in \Omega$$  to members of schema $$\xi$$

The theorem is saying that the expectation of the next generation under a schema is proportional to the relative fitness of the schema and inversely to the disruptive potential of the genetic operators on this schema. This disruptive potential is directly proportional to the length of the schema, in addition to the probability of other operators like mutation and crossover. Thus, short and low-order schemata with above-average fitness increase exponentially in frequency in successive generations. The ability of the schema theorem, which governs the behavior of a simple genetic algorithm, to lead the search to interesting areas of the space is governed by the quality of the information it collects about the space through observed schema fitness averages in the population → If the schemata tend to collect together solutions with related performance, then the fitness-variance of schemata will be relatively low, and the information that the schema theorem utilizes will have predictive power for previously untested instances of schemata that the algorithm may generate. On the other hand, if the performances are not related in the schemata then the solutions generated cannot be assumed to bear any relation to the fitness of the parents i.e they might just be random. Thus, we need to incorporate domain-specific knowledge in the representations that we use for our algorithm since that signified the underlying distribution that might relate to similar performances in the future. Now, it was proven in the 1990s that this schema theorem applied to any subset of the schema $$\xi$$, and not just the whole schemata, under the constraint that we adjust the disruptive potential according to the subset. The generalized schema was termed Formae (Singular Forma) and this is how the theory around format came to be. Forma analysis allows us to develop genetic representations and operators that can maximize $$\hat{\mu_\xi(t)}$$ by selecting subsets of $$\xi$$ that are appropriate for the domain. This is done by constructing equivalence relations that partition the search space into appropriate equivalence classes which play the rôle of format. 

### Equivalence Relations → Basis

The first step to forma analysis is to define relations ($$\sim$$ ) on our search space $$X$$. This is simply saying that each element of our search space can have a property that is either true or false. For example, we can define a greater than relation $$> : X \rightarrow \{0,1\}$$ that compares our element to some integer. Now, these relations are called equivalence relations if they are 

- Reflexive → If each element of the domain is related to itself
- Symmetric → $$a \sim b \implies b \sim a$$
- Transitive → $$a \sim b , b \sim c \implies a \sim c$$

Equivalence relations are essentially partitions of $$X$$ since they partition it into equivalence classes. Going back to our example of a schema, if we are to consider our binary schema 1XX0X1 and generalize it to something like XXXXXX where X represents positions in a string that need to be specified and X represents the unspecified positions, then the equivalence relation here is that the 1, 4 and 6 positions need to be specified. Now, taking $$\{0,1\}$$  as our alphabet one of the equivalence classes that are induced by this equivalence relation is 1XX0X1, but there can be others like 0XX0X1, or 1XX0X0. Thus, our equivalence relation induces multiple equivalence classes that then form the schema. 

Let us denote the set of all equivalence relations on $$X$$ as $$E(X)$$ . So, if we have an equivalence relation $$\psi \in E(X)$$, then we can call $$E_\psi$$  to be the set of equivalence classes induced by $$\psi$$. This set of classes is called formae. Now, if we have a vector of relations, say $$\mathbf{\Psi} \in E(X)$$ , then we call $$\Xi_\mathbf{\Psi}$$  as the set of formae, given by: 

$$\Xi_\mathbf{\Psi} := \prod_{i=1}^\mathbf{\Psi} \Xi_{\psi_i}$$

And we can also denote the union of the formae as 

$$\Xi(\mathbf{\Psi}) := \bigcup_{\psi \in \mathbf{\Psi}} \Xi_\psi$$

Now, let's consider a relation that lies at the intersection of all the members of $$\mathbf{\Psi}$$ →  $$\phi := \bigcap \mathbf{\Psi}$$. This relation would induce equivalence classes that would be intersections of the classes induced by the elements of $$\mathbf{\Psi}$$, and this result can be mathematically written as: 

$$[x]_\phi = \bigcap \{ [x]_\psi \,\, |\,\, \psi \in \mathbf{\Psi} \}
$$

We can also define the span of $$E(X)$$  as a map from its power set onto itself

$$Span: \mathbb{P}(E(X)) \rightarrow \mathbb{P}(E(X))$$

If we have a condition where a set of relations $$B \in E(X)$$ has members that cannot be constructed by intersecting any other members of $$B$$, then $$B$$ is called an independent set of relations. Also, $$B$$ is said to be orthogonal to the order $$k$$ if given any $$k$$ equivalence classes induced by members of $$B$$, their intersection is non-empty. If $$k = |B|$$, then we call $$B$$ orthogonal. It has been shown that orthogonality implies independence, and so, we can use this concept to define a basis of $$\mathbf{\Psi}$$ → Any subset $$B$$ of $$\mathbf{\Psi} \subseteq E(X)$$ will constitute a basis iff: 

- $$B$$ in independent
- $$B$$ spans $$\mathbf{\Psi}$$

Thus, if $$B$$ is orthogonal then we have an orthogonal basis. Moreover, the number of elements in $$B$$ determines the dimensions of our basis. This notion of orthogonality of the set is important as it helps us ensure that our mapping from representations to solutions is fully defined.

### Representations through Basis

Once we have a basis, we can follow the vectorization procedure to vectorize $$\mathbf{\Psi}$$ in terms of the elements of $$B$$ → A general equivalence relation $$\mathbf{\Psi}$$ can be decomposed into component basic equivalence relations in $$B$$. Our first step would be to go from equivalence relations to representations, by defining a representation. We first define a partial representation function $$\rho$$  for an equivalence relation $$b \in E(X)$$: 

$$\rho_{b} : X \rightarrow \Xi_b$$

 Taking $$[x]_b$$ to be the equivalence class under the relation  $$b,$$ we can say 

$$\rho_b(x) := [x]_b$$

Thus, if we have a set $$B = \{ b_1, b_2, ..., b_n\}$$, we can define a genetic representation function as

$$\mathbf{\rho_B} := (\rho_{b_1}, ..., \rho_{b_2}) \,\,\,\,\, s.t \,\,\,\,\, \mathbf{\rho_B}: X \rightarrow \Xi_B \\
\implies \mathbf{\rho_B} (x) = ([x]_{b_1}, ..., [x]_{b_n})$$

Let $$C$$ be the space of chromosomes (Representations), we can call this set the image of $$X$$under $$\mathbf{\rho_B}$$ and if $$\mathbf{\rho}_B$$ is injective, we can define a growth function $$g:C \rightarrow X$$  as the inverse of the representation function: 

$$g: \Xi_b \rightarrow X \\
g(\mathbf{\xi})  := \mathbf{\rho}_B^{-1}(\mathbf{\xi})$$

We now have a vector space over which we have created a way to map representations to Chromosomes and back, which allow us to define genetic operations through these functions. 

### Coverage and Unique Basis

Our next step is to understand how these equivalence relations can generate representations, and how the Chromosomes relate to these equivalence relations. To go towards usefulness, we first have to define something called Coverage → A set of equivalence relations $$\mathbf{\Psi} \subset E(X)$$  is said to cover $$X$$ if, for each pair of solutions in $$X$$, there is at least one equivalence relation in $$\mathbf{\Psi}$$ under which the solutions in the pair are not equivalent. Formally, 

$$\forall x \in X, y \in X/\{x\} : \exists \psi  \in \mathbf{\Psi} : \psi(x,y) = 0$$

The significance of this notion is easy to understand → Coverage is important because if a set of equivalence relations covers $$X$$ then specifying to which equivalence class a solution belongs for each of the equivalence relations in the set suffices to
identify a solution uniquely. By this definition, we can also prove that any basis $$B$$ fo  $$\mathbf{\Psi}$$ would cover $$X$$ if it covers $$\mathbf{\Psi}$$ and extend it further to show that any orthogonal basis of $$X$$ that also covers it can be a faithful representation of $$X$$. This is the point that we have been trying to dig-into through formalism → The information that this orthogonal basis includes in its formulation is critical to the impact of genetic algorithm in search. 

### Genes and Alleles

We can define the Genes as the members of the basis $$B$$ of $$\mathbf{\Psi}$$ and the members of $$\Xi_B$$ will be called the formae, or the alleles. 

Using our basis we can track the information it transmits by checking the equivalence of the solutions generated under the relations in $$B$$. This is called the Dynastic Potential → Given a basis $$B$$ for a set of relations $$\mathbf{\Psi} \subset E(X)$$ that covers $$X$$, the dynastic potential $$\Gamma$$  of a subset $$L \subseteq X$$ is the set of all solutions in $$X$$ that are equivalent to at least one member of $$L$$ under the equivalence relations in $$B$$. 

$$\Gamma: P(X) \rightarrow P(X) \\
\Gamma(L) :=  \big \{ x \in X | \,\,\,  \forall b \in B : \exists l \subset L: b(l,x) = 1   \big \}$$

Thus, the dynastic potential of $$L$$ would be the set of all children that can be generated using only alleles available from the parent solutions in L. The solutions in $$L$$ belong to different equivalence classes or formae. Thus, by measuring how many formae include solutions in $$L$$. This is called the similarity set, formally defined as the intersection  of all the formae to which solutions in $$L$$ can belong: 

$$\Sigma(L) :=  \begin{cases} 
\bigcap \{ \xi \in \Xi \,\, | \,\, L \subset \xi \}, \,\,\,\, if\,\, \exists \xi \in \Xi: L \subset \xi \\
X, \,\,\,\, otherwise    
\end{cases}$$

Now, it has been proved that the dynastic potential is contained by the similarity set

$$\forall L\subset X: \Gamma(L) \subset \Sigma(L)$$

Thus, we now have a full  mathematical mechanism to se how the optimization process evolves: 

1. We have a representation of genes as our Basis of equivalence relations
2. These genes map to alleles through a vector of partial representations $$\mathbf{\rho}_B$$
3. The chromosomes then evolve to give a new set of genes through the growth function $$g$$ after applying genetic operators $$\Omega$$  to the representations 
4. The information that survives this process is quantified by the dynastic potential $$\Gamma$$  of the solution space hence generated.