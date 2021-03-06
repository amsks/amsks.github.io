## Aditya Mohan

<img style="float:left;display:inline-block;padding-right: 16px" src="./static/meinfoto.jpg" width="100px">

- Reach me: <a href='mailto:adityak735@gmail.com'> `adityak735@gmail.com` </a>
- [Github](http://github.com/amsks)
- [Linkedin](https://www.linkedin.com/in/amsks/)
- [Curriculum Vitae](CV/CV_Aditya_Mohan.pdf)
- [Reading list](reading-list.html)


### About Me
- I am a Master's student in the [EIT Digital Master's course in Autonomous systems](https://masterschool.eitdigital.eu/programmes/aus/). I am interested in exploring how learning systems can learn to abstract structure out of tasks and learn to reason about themselves. What excites me the most is the extent to which we can learn about our intelligence through the endeavor of creating one. I have worked on problems at the intersection of Robotics, Reinforcement Learning, and Meta-Learning in single-agent and Multi-Agent settings, and I have also dabbled with Evolutionary Algorithms and Optimization Literature. I am currently working on ad-hoc cooperation in the game of [Hanabi](https://arxiv.org/abs/1902.00506)


- **Fun Fact :** I am a musician - Multi-instrumentalist and Singer - and can sing in around 8 languages so far. Checkout some of my song covers on my [instagram page](https://www.instagram.com/melodic.musings/)

#### Education
- [EURECOM](http://www.eurecom.fr/en)
- [TU-Berlin](https://www.tu.berlin/)
- [Manipal Institute of Technology](https://manipal.edu/mit.html)

#### Technologies

- **Robotics :** ROS, MORSE, FlexBE
- **Programming :** C++, Embedded C, Python, JAVA
- **RL Software :** OpenAI Gym, PyBullet, PhysX, RAI
- **Deep Learning :** Pytorch, Keras, Tensorflow


#### Projects and Work Reports

- [Plan-Based Reward-Shaping for Goal-Focused Exploration in Reinforcement Learning](CV/LIS_Internship_Report.pdf)
- [Pulmonary Embolism Detection](CV/MALIS_Final_Report.pdf)




<!-- --------------------------------- Notes ----------------------------------------- -->
#### Notes

# AutoML: Bayesian Optimization

The general optimization problem can be stated as the task of finding the minimal point of some objective function by adhering to certain constraints. More formally, we can write it as

$$\min_x f(x) \,\,\,\, s.t \,\,\,\, g(x) \leq 0 \,\,\,, \,\,\, h(x) = 0  $$

We usually assume that our functions $f, g, h$ are differentiable, and depending on how we calculate the first and second-order gradients (The Jacobians and Hessians) of our function, we designate the different kinds of methods used to solve this problem. Thus, in a first-order optimization problem, we can evaluate our objective function as well as the Jacobian, while in a second-order problem we can even evaluate the Hessian.  In other cases, we impose some other constraints on either the form of our objective function or do some tricks to approximate the gradients, like approximating the Hessians in Quasi-Newton optimization. However, these do not cover cases where $f(x)$  is a black box. Since we cannot assume that we fully know this function our task can be re-formulated as finding this optimal point $x$ while discovering this function $f$. This can be written in the same form, just without the constraints

$$\min _x f(x) $$

## KWIK

To find the optimal $x$ for an unknown $f$  we need to explicitly reason about what we know about $f$. This is the **Knows What It Knows** framework. I will present an example from the paper that helps understand the need for this explicit reasoning about our function. Consider the task of navigating the following graph:

<img width=700 height=275 src="static/AutoML/KWIK.png">

Each edge in the graph is associated with a binary cost and let's assume that the agent does not know about these costs beforehand, but knows about the topology of the graph. Each time an agent moves from one node to another, it observes and accumulates the cost. An episode is going from the source on the left to the sink on the right. Hence, the learning task is to figure out the optimal path in a few episodes. The simplest solution for the agent is to assume that the costs of edges are uniform and thus, take the shortest path through the middle, which gives it a total cost of 13. We could then use a standard regression algorithm to fit a weight vector to this dataset and estimate the cost of the other paths, simply based on the nodes observed so far, which gives us 14 for the top, 13 for the middle, and 14 for the bottom paths. Hence, the agent would choose to take the middle path, even though it is suboptimal as compared to the top one.

Now, let's consider an agent that does not just fit a weight vector but reasons about whether it can obtain the cost of edges with the available data. Assuming the agent completed the first episode through the middle path and accumulated a reward of 13, the question it needs to answer is which path to go for next. In the bottom path cost of the penultimate node is 2, which can be figured out from the costs of nodes already visited 

$$3 - 1 = 2$$

This gives us more certainty than the uniform assumption that we started with. However, this kind of dependence does not really exist for the upper node since the linear combination does not work on the nodes already visited. If we incorporate a way for our agent to say that it is not sure about the answer to the cost of the upper nodes, we can essentially incentivize it to explore the upper node in the next round, allowing our agent to visit this node and discover the optimal solution. This is similar to how we discuss the exploration-exploitation dilemma in Reinforcement Learning.

## MDP framework

Motivated from the previous section and based on the treatment done [here](https://www.user.tu-berlin.de/mtoussai//teaching/Lecture-Maths.pdf), we can model our solver as an agent and the function as the environment. Our agent can sample the value of the function in a range of possible values and in a limited budget of samples, it needs to find the optimal $x$. The observation that comes after sampling from the environment is the noisy estimate of $f$, which can call $y$. Thus, we can write our function as the expectation over these outputs

$$f( x) = \mathbb{E}\big [ y |f(x) \big ]$$

We can cast this as a Markov Decision Process where the state is defined by the data the agent has collected so far. Let's call this data $S$. Thus, at each iteration $t$, our agent exists in a state $S_t$ and needs to make a decision on where to sample the next $x_t$. Once it collects this sample, it adds this to its existing knowledge

$$S_{t+1} = S_t \cup \{x_t, f_t \} $$

We can create a policy $\pi$  that our agent follows to take an action from a particular state

$$\pi : S_t \rightarrow x_t$$

Hence, the agent operates with a prior over our function $P(f)$  , and based on this prior it calculates a deterministic posterior $P_\pi (S|x_t, f)$  by multiplying it with the expectation over the outputs.

$$\pi ^* = \argmin_\pi  \int P(f) P( S|\pi , f) \mathbb{E}[y|f]$$

Since the agent does not know $f$ apriori, it needs to calculate a posterior belief over this function based on the accumulated data

$$P(f|S) = \frac{P(S|f) P(f)}{P(S)} $$

With the incorporation of this belief, we can define an MDP over the beliefs with stochastic transitions. The states in this MDP are the posterior belief $P(f|S)$ . Thus, the agent needs to simulate the transitions in this MDP and it can theoretically solve the optimal problem through something like Dynamic programming. However, this is difficult to compute.

## Bayesian Methods

This is where Bayesian methods come into the picture. They formulate this belief $P(f|S)$  as a Bayesian representation and compute this using a gaussian process at every step. After this, they use a heuristic to choose the next decision. The Gaussian process used to compute this belief is called **surrogate function** and the heuristic used is called an **Acquisition Function.** We can write the process as follows: 

1. Compute the posterior belief using a surrogate Gaussian process to form an estimate of the mean $\mu(x)$  and variance around this estimate $\sigma^2(x)$  to describe the uncertainty
2. Compute an acquisition function $\alpha_t(x)$  that is proportional to how beneficial it is to sample the next point from the range of values
3. Find the maximal point of this acquisition function and sample at that next location 

    $$x_t = \argmax_x \alpha_t(x) $$

This process is repeated a fixed number of iterations called the **optimization budget** to converge to a decently good point. Three poplar acquisition functions are

- **Probability of Improvement (MPI) →** The value of the acquisition function is proportional to the probability of improvement at each point. We can characterize this as the upper-tail CDF of the surrogate posterior

    $$\alpha_t( x) = \int_{-\infty}^{y_{opt}}\mathcal{N} \big (y|\mu(x), \sigma (x) \big ) dy $$

- **Expected Improvement (EI)** → The value is not just proportional to the probability, but also to the magnitude of possible improvement from the point.

    $$\alpha_t(x) = \int_{-\infty}^{y_{opt}}\mathcal{N} \big (y|\mu(x), \sigma (x) \big ) \big [   y_{opt} - y\big ] dy$$

- **Upper Confidence Bound (UCB)** → We control the exploration through the variance and control parameter and exploit the maximum values

    $$\alpha_t(x)  = -\mu(x)  + \beta\sigma(x) $$

The evaluation of this maximization of the acquisition function is another non-linear optimization problem. However, the advantage is that these functions are analytic and so, we can solve for jacobians and Hessians of these, ensuring convergence at least on a local level. To make this process converge globally, we need to optimize from multiple start points from the domain and hope that after all these random starts the maximum found by the algorithm is indeed the global one.


## Hyperparameter Tuning

One of the places where Global Bayesian Optimization can show good results is the optimization of hyperparameters for Neural Networks. So, let's implement this approach to tune the learning rate of an Image Classifier! I will use the KMNIST dataset and a small ResNet-9 Model with a Stochastic Gradient Descent optimizer. Our plan of attack is as follows:

1. Create a training pipeline for our Neural Network with the Dataset and customizable learning rate 
2. Cast the training and inference into a an objective function, which can serve as ou blackbox
3. Map the inference to an evaluation metric that can be used in the optimization procedure 
4. Use this function in a global bayesian optimization procedure. 

### Creating the training pipeline and Objective Function

I have used PyTorch and the lightning module to create a boilerplate that can be used to train our network. Since KMNIST and ResNet architectures are already available in PyTorch, all we need to do is customize the ResNet architecture for MNIST, which I have done as follows

```python
def create_resnet9_model() -> nn.Module:
    '''
        Function to customize the RESNET to 9 layers and 10 classes

        Returns
        --------
        torch.module
            Pytorch Module of the Model
    '''
    model = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=10)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model

# 
class ResNet9(pl.LightningModule):
    def __init__(self, learning_rate=0.005):
        '''
            Pytorch Lightning Module for training the RESNET with SGD optimizer

            Parameters
            -----------
            learning_rate: float 
                Learning rate to be used for training every time since it is an 
                optimization parameter
        '''
        super().__init__()
        self.model = create_resnet9_model()
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    @auto_move_data
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        loss = self.loss(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
```

Once this is done, our next step is to use the training pipeline and cast it into an objective function. For this, we need to evaluate our model somehow. I have used the balanced accuracy as an evaluation metric, but any other metric can also be used (like the AUC-ROC score) 

```python
def objective(  lr=0.1, 
                epochs=1, 
                gpu_count=1, 
                iteration=None, 
                model_dir='./outputs/models/', 
                train_dl=None,
                test_dl = None 
            ):

    '''
        The objective function for the optimization procedure 

        Parameters
        -----------
        lr: float 
            learning Rate 
        epochs: int 
            Epochs for training
        gpu_count: int 
            Number of GPUs to be used (0 for only CPUs)
        iteration: int 
            Current iteration
        model_dir: str
            directory to save model checkpoints 
        train_dl: Torch Dataloader 
            Dataloader for training
        test_dl: Torch Dataloader 
            Dataloader for inference

        Returns
        ---------
        float
            balanced Accuracy of the model after inference

    '''

    save = False
    checkpoint = "current_model.pt"
    model = ResNet9(learning_rate=lr)

    trainer = pl.Trainer(
        gpus=gpu_count,
        max_epochs=epochs,
        progress_bar_refresh_rate=20
    )

    trainer.fit(model, train_dl)
    trainer.save_checkpoint(checkpoint)

    inference_model = ResNet9.load_from_checkpoint(
        checkpoint, map_location="cuda")

    true_y, pred_y, prob_y = [], [], []
    for batch in tqdm(iter(test_dl), total=len(test_dl)):
        x, y = batch
        true_y.extend(y)
				model.freeze()
		    probabilities = torch.softmax(inference_model(x), dim=1)
		    predicted_class = torch.argmax(probabilities, dim=1)
        pred_y.extend(predicted_class.cpu())
        prob_y.extend(probabilities.cpu().numpy())

    if save is False:
        os.remove(checkpoint)

    return np.mean(balanced_accuracy_score(true_y, pred_y))
```

### Implementing Bayesian Optimization

As mentioned in the previous sections, we first need a Gaussian Process as a surrogate model. We can either write it from scratch or just use some open-sourced library to do this. Here, I have used sci-kit learn to create a regressor  

```python
# Create the Model
    m52 = sklearn.gaussian_process.kernelsConstantKernel(1.0) * Matern( length_scale=2.0, 
                                        nu=1.5
                                    )
    model = sklearn.gaussian_process.GaussianProcessRegressor(
                                        kernel=m52, 
                                        alpha=1e-10, 
                                        n_restarts_optimizer=100
                                    )
```

Once th Gaussian process is established, we now need to write the acquisition function. I have used the Expected Improvements acquisition function. The core idea can be re-written as proposed by Mockus 

$$EI(x) = \begin{cases} 
\big( \mu_t(x) - y_{max} - \epsilon \big ) \Phi(Z)   + \sigma_t (x) \phi(Z) &\sigma_t(x) > 0  \\
0  & \sigma_t(x) > 0

\end{cases}$$

Where

$$Z = \frac{\mu_t(x) - y_{max} - \epsilon}{\sigma_t(x) }$$

and $\Phi$ and $\phi$  are the PDF and CDF functions. This formulation is an analytical expression that achieves the same result as our earlier formulation and we have added $\epsilon$ as an exploration parameter. This can be implemented as follows

```python
def _acquisition(self, X, samples):
        '''
            Acquisition function using hte Expected Improvement method

            Parameters
            -----------
            X : N x 1 
                Array of parameter points observed so far

            X_samples : N x 1
                Array of Sampled points between the bounds

            Returns
            --------
            float
                Expected improvement

        '''

        # calculate the max of surrogate values from history
        mu_x_, _ = self.surrogate(X)
        max_x_ = max(mu_x_)

        # Get the mean and deviation of the samples 
        mu_sample_, std_sample_ = self.surrogate(samples)
        mu_sample_ = mu_sample_[:, 0]

        # Get the improvement
        with np.errstate(divide='warn'):
            z = (mu_sample_ - max_x_ - self.eps) / std_sample_
            EI_ = (mu_sample_ - max_x_ - self.eps) * \
                scipy.stats.norm.cdf(z) + std_sample_ * scipy.stats.norm.pdf(z)
            EI_[std_sample_ == 0.0] = 0

        return EI_
```

the `self.surrogate()` function is just predicting using the Gaussian process earlier written. Once we have our expected improvements, we need to optimize our acquisition by maximizing over these expected improvements

```python
def optimize_acq(self, X):
        '''
            Optimization of the Acquisition function using a maximization check of the outputs

            Parameters
            -----------
            X : N x 1 
                Array of parameter points

            Returns
            --------
            float
                Next location of the sampling point based on the Maximization

        '''
            
        # Calculate Acquisition value for each sample
        EI_ = self._acquisition(X, self.X_samples_)

        # Get the index of the largest Score
        max_index_ = np.argmax(EI_)

        return self.X_samples_[max_index_, 0]
```

### Putting it all together

Now that we have our optimization routines, we just need to combine them with our objective function into a loop and we are done. In my code, I have implemented the optimization as a class and I pass the paramters to this class. So, the main loop looks as follows: 

 

```python
budget = 10

train_data = KMNIST("kmnist", train=True, download=True, transform=ToTensor())
test_data = KMNIST("kmnist", train=False, download=True, transform=ToTensor())

train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8 )
test_dl = DataLoader(test_data, batch_size=batch_size, num_workers=8)

# sample the domain
X = np.array([np.random.uniform(0, 1) for _ in range(init_samples)])
y = np.array([objective(lr =x, 
                        epochs=init_epochs, 
                        gpu_count=gpu_count,
                        train_dl=train_dl,
                        test_dl=test_dl
                        ) for x in X])

X = X.reshape(len(X), 1)
y = y.reshape(len(y), 1)

for i in range(budget):
		# fit the model
		B.model.fit(X, y)
		
		# Select the next point to sample
		X_next = B.optimize_acq(X, y)
		
		# Sample the point from Objective
		Y_next = objective( lr=X_next, 
		                    epochs=epochs, 
		                    gpu_count=gpu_count,
		                    model_dir= output_dir+"/models/", 
		                    iteration=i+1,
		                    train_dl= train_dl,
		                    test_dl = test_dl
		                    )
		
		print(f"LR = {X_next} \t Balanced Accuracy = {Y_next*100} %")
		
		# Plots for second iteration onwards 
		B.plot(X, y, X_next, i+1)
		
		# add the data to History
		X = np.vstack((X, [[X_next]]))
		y = np.vstack((y, [[Y_next]]))
```

Here, I have used a budged to 10 function evaluations in the main loop and 2 function evaluations before the first posterior estimate. An exemplary plot of what comes out at the end is shown below

<img width=700 height=400 src="static/AutoML/plot-iter-3.png">

The vertical axis is the Balanced Accuracy and the horizontal axis is the learning rate. As can be seen, this is the third iteration of the main loop, with 2 points sampled as an initial estimate, and the acquisition function is the highest at the region with the balance of uncertainty and value of the mean.


# Meta: Non-Parametric Methods
The optimization-based methods are very useful for model-agnosticism and expression with sufficiently deep networks. However, as we have seen the main bottleneck is the second-order optimization which ends up being compute and memory intensive. Thus, the natural question is whether we can embed a learning procedure without the second-order optimization? one answer to this lies in the regime of data when it comes to the test time → during the meta-test time our paradigm of few-shot learning is a low data regime. Thus, methods that are non-parametric and have been shown to work well in these cases can be applied here! Specifically,

- We want to be parametric during hte trainng phase
- We can apply a non-parametric way to compare classes during test time

Thus, the question now becomes → Can we use parametric Learners to produce effective non-parametric learners? The straight answer to this would be something like K-Nearest Neighbors where we take a test sample and compare it against our training classes to see which one I the closest. However, now we have the issue of the notion of closeness! In the supervised case, we could simply use a L2-Norm, but this might not be the case for meta-learning strategies since direct comparison of low-level features might fail to take into account meta-information that might be relevant. Hence, we look to other approaches

### Siamese Networks
A siamese network is an architecture which was multiple sub-networks with identical configuration i.e same weights and hyperparameters. We train this architecture to output the similarity between two inputs using the feature vectors.

<img width=700 height=400 src="static/Meta/Meta-1.png">

In our case, we can train these networks to predict whether or not two images belong to the same class or not and thus, we have a black box similarity between our test data and our training classes. We can use this during the test to compare the input with all classes in our training set and output the class with the highest similarity.

### Matching Networks

The issue with the siamese network approach is that we are training the network on Binary classification but testing it on Multi-Class classification. [Matching Networks](https://arxiv.org/pdf/1606.04080.pdf) circumvent this problem by training the networks in such a way that the Nearest-Neighbors method produces good results in the learned embedding space.

<img width=800 height=400 src="static/Meta/Meta-2.png">

To do this, we learn an embedding space $g(\theta)$ for all input classes and also create an embedding space $h(\theta)$ for the test data. We then compare the $g(.)$  and $f(.)$  to predict each image class, which is then summed to create a test prediction. Each of the black dots in the above image corresponds to the comparison between training and test images, and our final prediction is the weighted sum of these individual labels

$$\hat{y}^{ts} = \sum _{x_k, y_k \in \mathcal{D^{tr}}} f _\theta(x^{ts}, x_k) y_k$$

This paper used a bi-directional LSTM to produce $g_\theta$ and a convolutional encoder to embed the images in $h_\theta$ and the model was trained end-end, with the training and test doing the same thing. The general algorithm for this is as follows:

1. Sample Task $\mathcal{T}_i$ ( a sequential stream or mini-batches )
2. Sample Disjoint Datasets $\mathcal{D^{tr}_i}$,$\mathcal{D^{test}_i}$ from $\mathcal{D}_i$
3. Compute $\hat{y}^{ts} = \sum _{x_k, y_k \in \mathcal{D^{tr}}} f _\theta(x^{ts}, x_k) y_k$
4. Update  $\theta$ using $\nabla_\theta\mathcal{L}(\hat{y}^{ts}, y^{ts})$

### Prototypical Networks

While Matching networks work well for one-shot meta-learning tasks, they do all this for only one class. If we had a problem of more than one shot prediction, then the matching network will do the same process for each class, and this might not be the most efficient method for this. [Prototypical Networks](https://arxiv.org/abs/1703.05175) alleviate this issue by aggregating class information to create a prototypical embedding → If we assume that for each class there exists an embedding in which points cluster around a single prototype representation, then we can train a classifier on this prototypical embedding space. This is achieved through learning a non-linear map from the input into an embedding space using a neural network and then taking a class’s prototype to be the mean of its support set in the embedding space.

<img width=600 height=400 src="static/Meta/Meta-3.png">

As shown in the figure above, the classes become seperable in this embedding space. 

$$\bm{c}_k = \frac{1}{|\mathcal{D_i^{tr}}|} \sum_{(x,y) \in \mathcal{D_i^{tr}}} f_\theta(x)$$

Now, if we have a metric on this space then all we have to do is find the near class cluster to a new query point and we can classify that point as belonging to this class by taking the softmax over the distances 

$$p_\theta(y = k|x) = \frac{\exp \big(  -d (f_\theta(x), \bm{c}_k) \big)}{ \sum_{k'} \exp \big(  -d (f_\theta(x), \bm{c}_{k'}) \big)}$$

If we want to reason more complex stuff about our data then we just need to create a good enough representation in the embedding space. Some approaches to do this are: 

- [Relation Network](https://arxiv.org/abs/1711.06025) → This is an approach where they learn the relationship between embeddings i.e instead of taking $d(.)$  as a pre-determined distance measure, they learn it inherently for the data
- [Infinite Mixture Prototypes](https://arxiv.org/pdf/1902.04552.pdf) → Instead of defining each class by a single cluster. they represent each class by a set of clusters. Thus, by inferring this number of clusters they are able to interpolate between nearest neighbors and the prototypical representations.
- [Graph Neural Networks](https://arxiv.org/abs/1711.04043) →  They extend the Matching network perspective to view the few-shot problem as the propagation of label information from labeled samples towards the unlabeled query image, which is then formalized as a posterior inference over a graphical model.

## Comparing Meta-Learning Methods

### Computation Graph Perspective

We can view these meta-learnin algorithms as computation graphs: 

- Black-box adaptation is essentially a sequence of inputs and outputs on a computational graph
- Optimization can be seen as a embedding an optimization routine into a computational graph
- Non-parameteric methods can be seen as computational graphs working with the embedding spaces

This viewpoint allows us to see how to mix-match these approaches to improve performance:

- [CAML](https://openreview.net/forum?id=BJfOXnActQ) → Ths approach creates an embedding space, which is also a metric space, to capture inter-class dependencies. once this is done, they run gradient descent using this metric
- [Latent Embedding Optimization (LEO)](https://arxiv.org/abs/1807.05960) → They learn a data-dependent latent generative representation of model parameters and then perform gradient-based meta-learning on this space. Thus, this approach decouples the need for GD on higher dimensional space by exploiting the topology of the data.

### Algorithmic Properties Perspective

We consider the following properties of the most importance for most tasks : 

- **Expressive Power** → Ability of our learned function $f$ to represent a range of learning procedures. This is important for scalability and applicability to a range of domains.
- **Consistency** → The learned learning procedure will asymptotically solve the task given enough data, regardless of the meta-training procedure. The main idea is to reduce the reliance on the meta-training and generate the ability to perform on Out-Of-Distribution (OOD) tasks
- **Uncertainty Awareness** → Ability to reason about ambiguity during hte learning process. This is especially important for things like active learning, calibrated uncertainty, and RL.

We can now say the following about the three approaches:

- **Black-Box Adaptation** → Complete Expressive Power, but not consistent
- **Optimization Approaches** → Consistent and expressive for sufficiently deep models, but fail in expressiveness for other kinds of tasks, especially in Meta-Reinforcement Learning.
- **Non-parametric Approaches** → Expressive for most architectures and consistent under certain conditions






<!-- %%% -->
# Meta: Parametric Methods for Meta-Learning

## Back-Box Adaptation

These are a set of approaches that treat step 1 as an inference problem and thus, training a Neural Network to represent $p(\phi_i|\mathcal{D}^{tr}, \theta)$ i.e a way to estimate $\phi_i$ and then use that as a parameter to optimize for a new task. The deterministic way to go about it would be to take point estimates 

$$\phi_i = f_\theta (\mathcal{D^{tr}_i})$$

Thus, we can treat $f_\theta(.)$ as a neural network parameterized by $\theta$  which takes the training data as an input, sequential or batched, and outputs the task-specific parameters $\phi_i$ which are then used by another neural network $g_{\phi_i} (.)$ to predict the outputs on a new dataset. Thus, we can essentially treat this as a supervised learning problem with our optimization being 

$$\begin{aligned}
& \max_\theta \sum_{\mathcal{T_i}} \sum_{(x,y) \sim \mathcal{D_i}^{test}} \log g_{\phi_i} (y|x) \\
= & \max_\theta \sum_{\mathcal{T_i}} \mathcal{L}(f_\theta(\mathcal{D^{tr}_i}), \mathcal{D_i^{test}})
\end{aligned}$$

To make this more tractable, $\phi$  can be replaced by a sufficient statistic $h_i$ instead of all the parameters. Some ANN architectures that work well with this approach are LSTMs, as shown in the work of [Santoro et. al](http://proceedings.mlr.press/v48/santoro16.pdf), feedforward networks with averaging as shown by [Ramalho et. al](https://arxiv.org/abs/1807.01613), Having inner task learners and outer meta-learners i.e [Meta-Networks byMukhdalai](https://arxiv.org/abs/1703.00837) e.t.c. I am personally fascinated by the use of transformer architectures in this domain. The advantage of this approach is that it is expressive and easy to combine with other techniques like supervised learning, reinforcement learning e.t.c. However, the optimization bit is challenging and not the best solution from the onset for every kind of problem. Thus, our step-by-step approach would be: 

1. Sample Task $\mathcal{T}_i$ ( a sequential stream or mini-batches )
2. Sample Disjoint Datasets $\mathcal{D^{tr}_i}$,$\mathcal{D^{test}_i}$ from $\mathcal{D}_i$
3. Compute $\phi_i \leftarrow f_\theta(\mathcal{D^{tr}_i})$
4. Update  $\theta$ using $\nabla_\theta\mathcal{L}(\phi_i, \mathcal{D^{test}_i})$

## Optimization-Based Approaches

This set treats the prediction of $\phi_i$  as an optimization procedure and then differentiates through that optimization process to get a $\phi_i$ that leads to good performance. The method can be summarized into the surrogates sums of maximization of observing the training data given $\phi_i$ and the maximization of getting $\phi_i$  given our model parameters $\theta$.

$$\max_{\phi_i} \log p(\mathcal{D^{tr}_i} | \phi_i )  + \log p(\phi_i | \theta)$$

The second part of the above summation is our prior and the first part is a likelihood. Thus, our next question is the form of this prior that might be useful. In deep learning, one good way to incorporate priors is through the initialization of hyperparameters, or fine-tuning. Thus, we can take $\theta$ as a pre-trained parameter and run gradient descent on it 

$$\phi \leftarrow \theta - \alpha \nabla_\theta \mathcal{L} (\theta, \mathcal{D^{tr}})$$

One popular way to do this for image classification is to have a feature extractor pre-trained on some datasets like ImageNet and then fine-tune its output to our problem. The aim in optimization-based approaches is to get to a sweet-spot in the multidimensional parameter space $\bm{\Phi}  = {\phi_1, \phi_2, .., \phi_n}$ such that our model becomes independent of the loss function and the training data, and this is called Model-Agnostic Meta-Learning. Thus, now our procedure becomes

1. Sample Task $\mathcal{T}_i$ ( a sequential stream or mini-batches )
2. Sample Disjoint Datasets $\mathcal{D^{tr}_i}$,$\mathcal{D^{test}_i}$ from $\mathcal{D}_i$
3. Optimize $\phi_i \leftarrow f_\theta(\mathcal{D^{tr}_i})$
4. Update  $\theta$ using $\nabla_\theta\mathcal{L}(\phi_i, \mathcal{D^{test}_i})$

For our optimization process, let's define our final task specific parameter as

$$\phi = u(\theta, \mathcal{D^{tr}}) $$

And now, our optimization target becomes

$$\begin{aligned}
& \min_\theta  \mathcal{L}(\phi, \mathcal{D^{test}}) \\
= & \min_\theta \mathcal{L} \big (u(\theta, \mathcal{D^{tr}}), \mathcal{D^{test}} \big)
\end{aligned}$$

This optimization can be achieved by differentiating our loss w.r.t our meta-parameters $\theta$ and then performing an inner differentiation w.r.t $\phi$:

$$\frac{d\mathcal{L} (\phi, \mathcal{D^{test}} ) }{d \theta} = \nabla _{\bar{\phi}} \mathcal{L} (\bar{\phi}, \mathcal{D^{test}} )  \bigg |_{\bar{\phi} = u(\theta, \mathcal{D^{tr}})  }  d_\theta \big (   u(\theta, \mathcal{D^{tr}} ) \big )$$

Now, if we use our optimization update for $u (.)$ then  we get:

$$\begin{aligned}
& u(\theta, \mathcal{D^{tr}} ) = \theta  - \alpha \,\, d_\theta \big( L(\theta, \mathcal{D^{tr}}) \big ) \\
\implies & d_\theta \big (   u(\theta, \mathcal{D^{tr}} ) \big ) = \bm{1}  - \alpha \, d^2_\theta  \big (L(\theta, \mathcal{D^{tr}}) \big )
\end{aligned}$$

Thus, when we substitute the hessian in the derivative equation we get:

$$\begin{aligned}
\frac{d\mathcal{L} (\phi, \mathcal{D^{test}} ) }{d \theta} & = \bigg (\nabla _{\bar{\phi}} \mathcal{L} (\bar{\phi}, \mathcal{D^{test}} )  \bigg |_{\bar{\phi} = u(\theta, \mathcal{D^{tr}})  } \bigg ). \bigg ( \bm{1}  -  \alpha \, d^2_\theta  \big (L(\theta, \mathcal{D^{tr}}) \big ) \bigg ) \\
& = \nabla _{\bar{\phi}} \mathcal{L} (\bar{\phi}, \mathcal{D^{test}} )  \bigg |_{\bar{\phi} = u(\theta, \mathcal{D^{tr}})  }

-

 \alpha\,\, \bigg( \nabla _{\bar{\phi}} \mathcal{L} (\bar{\phi}, \mathcal{D^{test}} )
. d^2_\theta  \big (L(\theta, \mathcal{D^{tr}}) \big ) \bigg )
\bigg |_{\bar{\phi} = u(\theta, \mathcal{D^{tr}})  } 
\end{aligned}$$

We now have a matrix product on the right which can be made more efficient and  turn out ot be easier to compute than the full hessian of the network. Thus, this process is tractable. one really interesting thing that comes out of this is that we can also view this model-agnostic approach and the optimization update as a computation graph! Thus, we can say

$$\phi_i = \theta - f(\theta, \mathcal{D_i^{tr}}, \nabla_\theta \mathcal{L} )$$

Now, we can train an ANN to output the gradient $f(.)$  , and thus, this allows us to mix the optimization procedure with the black-box adaptation process. Moreover, MAML approaches show a better performance on the omniglot dataset since they are optimizing for the model-agnostic points. It has been shown by [Finn and Levine](https://arxiv.org/abs/1710.11622) that MAML can approximate any function of $\mathcal{D_i^{tr}}$ and $x^{ts}$ give: 

- Non-zero $\alpha$
- Loss function gradient does not lose information about the label
- Data-points in $\mathcal{D_i^{tr}}$  are unique

Thus, MAML is able to inject inductive bias without losing expressivity. 

### Inferece

To better understand why MAML works well,  we need to look through probabilistic lenses again to say that the meta-parameters $\theta$  are inducing some kinds of prior knowledge into our system and so our learning objective would be to maximize the probability of observing the data $\mathcal{D}_i$, given our meta-parameters $\theta$ 

$$\max_\theta \log  \prod_i p(\mathcal{D}_i| \theta )
 $$

This can be further written as the sum of the probabilities of $\mathcal{D_i}$ given our model-specific parameters $\phi_i$, and the probability of seeing each $\phi_i$ given our prior knowledge $\theta$ :

$$\max _\theta \prod_i \int p(\mathcal{D_i} |\phi_i) p(\phi_i|\theta) d\phi_i$$

 And now, we can estimate the probability of seeing each $\phi_i$ given our prior knowledge $\theta$ using a Maximum A-Posteriori (MAP) estimate $\hat{\phi}$, so that

$$\max_\theta \log  \prod_i p(\mathcal{D}_i| \theta ) \approx \max_\theta \log  \prod_i p(\mathcal{D}_i|\hat{\phi}_i) p(\hat{\phi} | \theta)  
$$

[It has been shown](https://regijs.github.io/papers/laa96.pdf) that, for likelihoods that are Gaussian in $\phi_i$, gradient descent with early stopping corresponds exactly to maximum a-posteriori inference under a Gaussian prior with mean initial samples. This estimation is exact in the linear case, and the variance in non-linear cases is determined by the order of the derivative. Thus, by limiting the computation to second derivatives, MAML is able to maintain a fairly good MAP inference estimate and so, MAML approximates hierarchical Bayesian Inference. We can also use other kinds of priors like: 

- [Explicit Gaussian Prior](https://arxiv.org/abs/1909.04630): $\phi \leftarrow \min_{\phi'} \mathcal{L} (\phi', \mathcal{D^{tr}})  + \frac{\lambda}{2} || \theta - \phi'||^2$
- [Bayesian Linear Regression](https://arxiv.org/abs/1807.08912) on learned features
- Convex optimization on learned features
- Ridge or logistic regression
- Support Vector Machines

### Challenge 1: Choosing Architecture

The major bottleneck in this process is the inner gradient step and so, we want to chosse an architecture that is effective for this inner step. One idea, called [Auto-Meta](https://arxiv.org/abs/1806.06927) is to adopt the progressive neural architecture search to find optimal architectures for meta-learners i.e combine AutoML with Gradient-Based Meta-Learning. The interesting results of this were: 

- They found highlynon-standard architectures, both deep and narrow
- They found architectures very different from the ones used for supervised learning

### Challenge 2: Handling Instabilities

Another challenge comes from the instability that can come from the complicated Bi-Level optimization procedure. One way of mitigating this is to learn the inner vector learning rate and then tune the outer learning rate : 

- [Meta-Stochastic Gradient Descent](https://arxiv.org/abs/1707.09835) is a meta-learner that can learn initialization, learner update direction, and learning rate, all in a single closed-loop process
- [AlphaMAML](https://arxiv.org/abs/1905.07435) incorporates an online hyperparameter adaptation scheme that eliminates the need to tune meta-learning and learning rates

Another idea idea is to optimize only a subset of parameters in the innter loop: 

- [DEML](https://arxiv.org/abs/1802.03596) jointly learns a concept generator, a meta-learner, and a concept discriminator. The concept generator abstracts representation of each instance of training data, the meta-learner performs few-shot learning on this instance and the concept discriminator recognizes the concept pertaining to each instance
- [CAVIA](https://arxiv.org/abs/1810.03642) partitions the model parameters into context parameters that serve as additional input to the model and are adapted on individual tasks and shared parameters that are meta-trained and shared across tasks. Thus, during test time only the context parameters need to be updated, which is a lower-dimensional search problem compared to all the model parameters

In [MAML++](https://arxiv.org/pdf/1810.09502.pdf) the authors ablate the various ideas and issues of MAML and then propose a new framework that addresses these issues. Some significant points were the de-coupling of the inner loop learning rate and the outer updates, the addition of batch normalization to each and every step and greedy  updates.

### Challenge 3: Managing Compute and Memory

The backpropagation through many inner-gradient steps adds computational and memory overhead that is hard to deal with. One idea to mitigate this is to approximate the derivative of $\phi_i$ w.r.t $\theta$. This is a crude approximation and works well for few-shot learning problem, but fails in more complex problems like imitation learning. Another direction is to try to not compute the gradient at all and use the [implicit function theorem](https://arxiv.org/abs/1909.04630)→ Let's take our function $\phi$ as the explicit gaussian representation :

$$\phi = u(\theta, \mathcal{D^{tr}})  = \argmin_{\phi'} \mathcal{L}(\phi', \mathcal{D^{tr}}) + \frac{\lambda}{2} ||\phi' - \theta ||^2
$$

Let our optimization function be

$$G(\phi', \theta ) = \mathcal{L}(\phi', \mathcal{D^{tr}}) + \frac{\lambda}{2} ||\phi' - \theta ||^2$$

Finding the $\argmin$ of the this function implies that the gradient w.r.t $\phi$  is $0$ i.e

$$\begin{aligned}
& \nabla_{\phi'} G(\phi', \theta) \big|_{\phi' = \phi} = 0 \\
\implies & \nabla_\phi L(\phi) + \lambda(\phi - \theta )  = 0 \\
\implies & \phi = \theta - \frac{1}{\lambda} \nabla_\phi L(\phi)
\end{aligned}$$

Thus, our derivative now becomes

$$\begin{aligned}
 & \frac{d \phi}{d \theta } = \bm{1} -   \frac{1}{\lambda} \nabla_\phi^2 L(\phi) \frac{d \phi}{d \theta } \\
\therefore\,\,\,& \frac{d \phi}{d \theta } = \bigg [\bm{1} + \frac{1}{\lambda} \nabla_\phi^2 L(\phi) \bigg ] ^{-1} 
\end{aligned}$$

Thus, we can compute this without going through the inner optimization process and it works only on the assumption that the out function $G(\phi', \theta)$  has an $\argmin$ , to begin with.

<!-- %%% -->
# Meta: Intro to Meta-Learning
- Source : [Chelsea Finn's lectures](https://youtube.com/playlist?list=PLoROMvodv4rMC6zfYmnD7UG3LVvwaITY5)

The motivating research that I personally find very interesting is a paper by Deepmind in 2018, that looks at the prefrontal cortex as a meta-reinforcement learning problem. The core idea here is that there seems to be an added function that dopamine serves: it is not just adjusting the 'weights' of the neurons in the brain, but also carrying some relevant information about the rules pertaining to the tasks that are being performed. This means that at the most fundamental level when given rewards for different kinds of tasks, it might be possible that there is an abstraction happening that allows learning underlying patterns between these tasks. Thus, if the tasks share some kind of commonalities in this meta-space, then it could be that the mind is formulating a meta-representation between these tasks which is allowing the person to treat any new task that shares some kind of underlying structure, no matter how unapparent that may be at a superficial level, as a problem of generalizing to new scenarios using priors from previous experiences. The core idea behind Meta-Learning is to develop a way for an agent to learn to generalize to new unseen tasks. One crucial way to see this, which I personally found very helpful, is to pit this notion against transfer learning or multi-task learning. A task like distinguishing between cats and birds, or riding a bicycle with trainer wheels is something that humans can learn with very few samples, while the traditional ML agent trained on a Neural Network approximation would require a great number of samples to be able to do this. 

## Basic Formulation

The intuition about what exactly is the meta-learning problem is very well seen through probabilistic lenses. Let's denote a dataset as $\mathcal{D} = \{ (\bm{x}, \bm{y})_k\}$ . In a standard supervised learning problem, we are essentially learning a loss function over this dataset for a function approximator that takes a vector $\bm{x}_i$ as input and produces some kind of an output  $y_i$. Let us denote these parameters by $\theta$  and so, we can say that our learning objective is

$$\min_\theta \mathcal{L}(\theta, \mathcal{D})$$

Depending on our requirement, we take take this loss function to be something like a negative log likelihood e.t.c. Now, given this loss function, we can define a task as

$$\mathcal{T}_i := \{ p_i(\bm{x}), p_i(\bm{y}|\bm{x}), \mathcal{L}_i \} $$

Thus, we are taking as a reference as data generating distribution and sampling from it with the indicated probability $p_i$ and our task is to predict the posterior distribution over labels given our inputs sampled from a prior distribution by minimizing a particular loss function. The way we go about this is to segregate our data into a training set $\mathcal{D}^{train}_i$ and a test set $\mathcal{D}_i^{test}$. 

If our objective is to learn multiple tasks then we need to train our function approximator to work on multiple such tasks $\mathcal{T}_i$ for a total set of tasks $i = 1, 2, ..., T$. To do this, we figure out a way to inculcate a task descriptor $\bm{z}_i$ into our function approximation so that the approximator can adjust its output based on the task, and then we essentially minimize over a sum fo multiple such losses to get the following formulation

$$\min_\theta \sum_{i=1}^T \mathcal{L}_i (\theta, \mathcal{D}_i)$$

If our objective is to learn something meta, then we can view this parameter $\theta$  as a prior that is being used by our approximator to learn an optimal set of parameters $\phi$  and predict the posterior distribution over $\bm{y}$ a given set of inputs $\bm{x}$ for a dataset $\mathcal{D}$. A good way to understand this is to say that we have some metadata for training $\mathcal{D}_{meta-train}$ that we are using along with the inputs from a dataset to get our posterior prediction, and $\theta$  is essentially parameterizing this meta training data to give us some meta-parameters that we can use for our training. This can be mathematically formulated as

$$\begin{aligned}
& \theta^* = \argmax_\theta \{\log p(\theta|\mathcal{D}_{meta-train} \} \\
& \phi^* = \argmax_{\phi} \{ \log p(\phi | \mathcal{D}^{tr}, \theta^*) \} \\
  & \,\,\,\,\,\,\,\,\,\ \therefore \,\,\,\phi^* = f_{\theta^*}(\mathcal{D}^{tr})

\end{aligned}$$

Thus, we can see this learning as two problems: 

1. Meta-learning problem → Understand the model agnostic parameters $\theta^*$ 
2. Adaptation Problem → Use the meta-parameters to adapt your learning curve and get the test-specific parameters $\phi^*$

### Mechanistic Formulation of Meta-Learning Problem

In supervised learning, an instance of meta-learning is a **few-shot classification,** where our task is to train the approximator on different datasets so that it learns to classify totally new and unseen data with only a few shots. The way we implement this is by having test data for each training dataset so that the approximator can be tested for each abstraction that it is learning. These are also called support set $S$ and prediction set $B$ 

$$\mathcal{D} = \big < S, B\big >$$

For a support set that contains $k$ labeled examples for $N$ classes, we call this a $k$-shot-$N$-classification task. Thus, a 5-shot-2-classification task on images would consist of datasets that have a total of 10 example images divided into one or the other class equally, and our classifier needs to learn to classify images in an unseen dataset with unseen classes but having the same configuration.  We consider data  over which w  need to do the prediction as $\mathcal{D}_i = \{ (\bm{x}, \bm{y} )_j \}$ and the meta-data as a set of these datasets $\mathcal{D}_{meta-train} = \{\mathcal{D}_i \}$. Our training data contains $K$ input-output pairs and so can be written as 

$$\mathcal{D}^{tr} =  \{ (\bm{x}, \bm{y} )_{1:K}\} $$

This is a $K$-shot prediction problem on a test input $\bm{x}_{test}$ and thus our problem is to predict $\bm{y}_{test}$ using these inputs, training datasets, and parameters

$$\bm{y}_{test} = f (\mathcal{D}^{tr}, \bm{x}_{test}; \theta ) $$

This translates the problem of meta-learning as a design and optimization problem, similar to how we would go about 

Thus, the general way to go about solving the meta-learning problem is: 

1. Choose a function to output task-specific parameters $p(\phi_i | \mathcal{D}_i^{tr}, \theta)$
2. Use the training data $\mathcal{D}_{meta-train}$ to create a max-likelihood objective that can then be used to optimize $\theta$




<!-- %%% -->
# Evo: Algebra of Genetic Algorithms

## Formalizing genetic Algorithm Search

To formalize genetic algorithms we need to define inputs and outputs and formalize what this algorithm does through maps. So, let's consider $X$ to be the space of all solutions. From elements of this space of solutions, we can construct multisets by grouping them, so let's take $P_M(X)$ to be this maximal set of multi-sets. Thus, a population $P$ can  be thought of as one of these possible multisets and so 

$$P \in P_M(X)$$

Now, the process of mutation and cross-over can be formalized as a genetic operator that maps $P_M(X)$, which can also be thought of as mapping $X^k$ to $X$. Let $\Omega$  be a genetic operator. Thus, we can say: 

$$\Omega:P_M(X) \rightarrow P_M(X) \\
\Omega : X^k \rightarrow X  \,\,\,\, \,\,\,\,\forall k \in \N \\
\implies \Omega (x_1, ..., x_k \in X) = x \in X$$

Thus, we can now classify the operators as follows: 

- Recombination → $\Omega : X^2 \rightarrow X$
- Mutation → $\Omega : X \rightarrow X$
- Selection → $\Omega: A \subseteq P_M(X) \rightarrow B \subseteq A$

## Forma Analysis

### Schema Theorem

Forma analysis creates a rigorous framework to look at how genetic algorithms might work on different problems and comes from Schema Theorem. According to the original paper, the Schema Theorem can be essentially summarized as 

$$
\mathbb{E} \{N_\xi(t+1) \} \geq N_\xi(t) \frac{\hat{\mu}_\xi(t)}{\bar{\mu}(t)} \bigg[  1 - \sum_{\omega \in \Omega}p_\omega p_\omega ^\xi\bigg]
$$

The various elements in this equation are: 

- $\xi$  → Schema which is like a template of strings in which certain positions are fixed. For example, a schema for binary strings like 1XX0X1 means that the first, fourth, and sixth positions have to be 1, 0, and 1, respectively, and the other positions can be either 0 or 1
- $N_\xi(t)$  → Population size at time $t$ that belongs to a schema $\xi$
- $\hat{\mu}_\xi(t)$ → Average fitness of all the members of the population at time t that are instances (members) of the schema $\xi$
- $\bar{\mu}(t)$ → average fitness of the whole population at time t
- $\Omega$  → Set of genetic operators in Use
- $p_\omega p_\omega ^\xi$ → This signified the potentially disruptive effect when we apply the operators $\omega \in \Omega$  to members of schema $\xi$

The theorem is saying that the expectation of the next generation under a schema is proportional to the relative fitness of the schema and inversely to the disruptive potential of the genetic operators on this schema. This disruptive potential is directly proportional to the length of the schema, in addition to the probability of other operators like mutation and crossover. Thus, short and low-order schemata with above-average fitness increase exponentially in frequency in successive generations. The ability of the schema theorem, which governs the behavior of a simple genetic algorithm, to lead the search to interesting areas of the space is governed by the quality of the information it collects about the space through observed schema fitness averages in the population → If the schemata tend to collect together solutions with related performance, then the fitness-variance of schemata will be relatively low, and the information that the schema theorem utilizes will have predictive power for previously untested instances of schemata that the algorithm may generate. On the other hand, if the performances are not related in the schemata then the solutions generated cannot be assumed to bear any relation to the fitness of the parents i.e they might just be random. Thus, we need to incorporate domain-specific knowledge in the representations that we use for our algorithm since that signified the underlying distribution that might relate to similar performances in the future. Now, it was proven in the 1990s that this schema theorem applied to any subset of the schema $\xi$, and not just the whole schemata, under the constraint that we adjust the disruptive potential according to the subset. The generalized schema was termed Formae (Singular Forma) and this is how the theory around format came to be. Forma analysis allows us to develop genetic representations and operators that can maximize $\hat{\mu_\xi(t)}$ by selecting subsets of $\xi$ that are appropriate for the domain. This is done by constructing equivalence relations that partition the search space into appropriate equivalence classes which play the rôle of format. 

### Equivalence Relations → Basis

The first step to forma analysis is to define relations ($\sim$ ) on our search space $X$. This is simply saying that each element of our search space can have a property that is either true or false. For example, we can define a greater than relation $> : X \rightarrow \{0,1\}$ that compares our element to some integer. Now, these relations are called equivalence relations if they are 

- Reflexive → If each element of the domain is related to itself
- Symmetric → $a \sim b \implies b \sim a$
- Transitive → $a \sim b , b \sim c \implies a \sim c$

Equivalence relations are essentially partitions of $X$ since they partition it into equivalence classes. Going back to our example of a schema, if we are to consider our binary schema 1XX0X1 and generalize it to something like XXXXXX where X represents positions in a string that need to be specified and X represents the unspecified positions, then the equivalence relation here is that the 1, 4 and 6 positions need to be specified. Now, taking $\{0,1\}$  as our alphabet one of the equivalence classes that are induced by this equivalence relation is 1XX0X1, but there can be others like 0XX0X1, or 1XX0X0. Thus, our equivalence relation induces multiple equivalence classes that then form the schema. 

Let us denote the set of all equivalence relations on $X$ as $E(X)$ . So, if we have an equivalence relation $\psi \in E(X)$, then we can call $E_\psi$  to be the set of equivalence classes induced by $\psi$. This set of classes is called formae. Now, if we have a vector of relations, say $\bm{\Psi} \in E(X)$ , then we call $\Xi_\bm{\Psi}$  as the set of formae, given by: 

$$\Xi_\bm{\Psi} := \prod_{i=1}^\bm{\Psi} \Xi_{\psi_i}$$

And we can also denote the union of the formae as 

$$\Xi(\bm{\Psi}) := \bigcup_{\psi \in \bm{\Psi}} \Xi_\psi$$

Now, let's consider a relation that lies at the intersection of all the members of $\bm{\Psi}$ →  $\phi := \bigcap \bm{\Psi}$. This relation would induce equivalence classes that would be intersections of the classes induced by the elements of $\bm{\Psi}$, and this result can be mathematically written as: 

$$[x]_\phi = \bigcap \{ [x]_\psi \,\, |\,\, \psi \in \bm{\Psi} \}
$$

We can also define the span of $E(X)$  as a map from its power set onto itself

$$Span: \mathbb{P}(E(X)) \rightarrow \mathbb{P}(E(X))$$

If we have a condition where a set of relations $B \in E(X)$ has members that cannot be constructed by intersecting any other members of $B$, then $B$ is called an independent set of relations. Also, $B$ is said to be orthogonal to the order $k$ if given any $k$ equivalence classes induced by members of $B$, their intersection is non-empty. If $k = |B|$, then we call $B$ orthogonal. It has been shown that orthogonality implies independence, and so, we can use this concept to define a basis of $\bm{\Psi}$ → Any subset $B$ of $\bm{\Psi} \subseteq E(X)$ will constitute a basis iff: 

- $B$ in independent
- $B$ spans $\bm{\Psi}$

Thus, if $B$ is orthogonal then we have an orthogonal basis. Moreover, the number of elements in $B$ determines the dimensions of our basis. This notion of orthogonality of the set is important as it helps us ensure that our mapping from representations to solutions is fully defined.

### Representations through Basis

Once we have a basis, we can follow the vectorization procedure to vectorize $\bm{\Psi}$ in terms of the elements of $B$ → A general equivalence relation $\bm{\Psi}$ can be decomposed into component basic equivalence relations in $B$. Our first step would be to go from equivalence relations to representations, by defining a representation. We first define a partial representation function $\rho$  for an equivalence relation $b \in E(X)$: 

$$\rho_{b} : X \rightarrow \Xi_b$$

 Taking $[x]_b$ to be the equivalence class under the relation  $b,$ we can say 

$$\rho_b(x) := [x]_b$$

Thus, if we have a set $B = \{ b_1, b_2, ..., b_n\}$, we can define a genetic representation function as

$$\bm{\rho_B} := (\rho_{b_1}, ..., \rho_{b_2}) \,\,\,\,\, s.t \,\,\,\,\, \bm{\rho_B}: X \rightarrow \Xi_B \\
\implies \bm{\rho_B} (x) = ([x]_{b_1}, ..., [x]_{b_n})$$

Let $C$ be the space of chromosomes (Representations), we can call this set the image of $X$under $\bm{\rho_B}$ and if $\bm{\rho}_B$ is injective, we can define a growth function $g:C \rightarrow X$  as the inverse of the representation function: 

$$g: \Xi_b \rightarrow X \\
g(\bm{\xi})  := \bm{\rho}_B^{-1}(\bm{\xi})$$

We now have a vector space over which we have created a way to map representations to Chromosomes and back, which allow us to define genetic operations through these functions. 

### Coverage and Unique Basis

Our next step is to understand how these equivalence relations can generate representations, and how the Chromosomes relate to these equivalence relations. To go towards usefulness, we first have to define something called Coverage → A set of equivalence relations $\bm{\Psi} \subset E(X)$  is said to cover $X$ if, for each pair of solutions in $X$, there is at least one equivalence relation in $\bm{\Psi}$ under which the solutions in the pair are not equivalent. Formally, 

$$\forall x \in X, y \in X/\{x\} : \exists \psi  \in \bm{\Psi} : \psi(x,y) = 0$$

The significance of this notion is easy to understand → Coverage is important because if a set of equivalence relations covers $X$ then specifying to which equivalence class a solution belongs for each of the equivalence relations in the set suffices to
identify a solution uniquely. By this definition, we can also prove that any basis $B$ fo  $\bm{\Psi}$ would cover $X$ if it covers $\bm{\Psi}$ and extend it further to show that any orthogonal basis of $X$ that also covers it can be a faithful representation of $X$. This is the point that we have been trying to dig-into through formalism → The information that this orthogonal basis includes in its formulation is critical to the impact of genetic algorithm in search. 

### Genes and Alleles

We can define the Genes as the members of the basis $B$ of $\bm{\Psi}$ and the members of $\Xi_B$ will be called the formae, or the alleles. 

Using our basis we can track the information it transmits by checking the equivalence of the solutions generated under the relations in $B$. This is called the Dynastic Potential → Given a basis $B$ for a set of relations $\bm{\Psi} \subset E(X)$ that covers $X$, the dynastic potential $\Gamma$  of a subset $L \subseteq X$ is the set of all solutions in $X$ that are equivalent to at least one member of $L$ under the equivalence relations in $B$. 

$$\Gamma: P(X) \rightarrow P(X) \\
\Gamma(L) :=  \big \{ x \in X | \,\,\,  \forall b \in B : \exists l \subset L: b(l,x) = 1   \big \}$$

Thus, the dynastic potential of $L$ would be the set of all children that can be generated using only alleles available from the parent solutions in L. The solutions in $L$ belong to different equivalence classes or formae. Thus, by measuring how many formae include solutions in $L$. This is called the similarity set, formally defined as the intersection  of all the formae to which solutions in $L$ can belong: 

$$\Sigma(L) :=  \begin{cases} 
\bigcap \{ \xi \in \Xi \,\, | \,\, L \subset \xi \}, \,\,\,\, if\,\, \exists \xi \in \Xi: L \subset \xi \\
X, \,\,\,\, otherwise    
\end{cases}$$

Now, it has been proved that the dynastic potential is contained by the similarity set

$$\forall L\subset X: \Gamma(L) \subset \Sigma(L)$$

Thus, we now have a full  mathematical mechanism to se how the optimization process evolves: 

1. We have a representation of genes as our Basis of equivalence relations
2. These genes map to alleles through a vector of partial representations $\bm{\rho}_B$
3. The chromosomes then evolve to give a new set of genes through the growth function $g$ after applying genetic operators $\Omega$  to the representations 
4. The information that survives this process is quantified by the dynastic potential $\Gamma$  of the solution space hence generated.



<!-- %%% -->
# GT: Differentiable Structures

To get differentiable manifolds, we need to refine the notion of Compatibility. We do this by saying that an atlas $\mathcal{A}$ is $K$-compatible if $\forall (U,x), (V,y) \in \mathcal{A}$ we have either of the two following conditions: 

- $U \cap V = \Phi$
- $U \cap V \neq \Phi$  →  $U,V$ satisfy some condition $K$

Now, we can define $K$ to be usually: 

1. $C^0$ 
2. $C^k$ → Transition maps are $k$-times differentiable
3. $C^{\infty}$ → Smooth transitions 
4. $C^{\omega}$→ Analytic i.e can be expanded through a taylor expansion 
5. Complex → Transition functions satisfy the Cauchy-Riemann Equations

**Whitney Theorem** → The theorem says: 

- Any maximal $C^k$ Atlas, contains a $C^{\infty}$ Atlas
- Any two $C^k$ atlas that cotain the same $C^{\infty}$ atlas are identical

This implies that once we have a $C^1$ atlas, we can essentially construct a smooth manifold. Thus, in terms of differentiable structure, we don't need to worry about variable values of $k$. **The only thing we need to look for is whether the function is differentiable once or not.**

## Smooth Manifold

A $C^k$ Manifold is a triple $(M,O,\mathcal{A})$  where

- $(M,O)$ → Tological Space
- $\mathcal{A}$ → $C^k$ atlas

We can create this atlas by simply taking a $C^0$ atlas and removing the pairs that are not differentiable.

### Differentiable Maps

If we have two manifolds $(M, O_M, \mathcal{A}_M)$  and $(N, O_N, \mathcal{A}_N)$ , then we can say that a map

$$\phi: M \rightarrow N $$

is diffferentiable if for some points in some charts in the two atlases: 

$$p \in U : (U,x) \in \mathcal {A}_M \\
q \in V : (V,y) \in \mathcal {A}_N \\
q = \phi(p) $$

The transformation map that comes: 

$$y \,\, \circ \,\,\phi \,\, \circ \,\,  x^{-1}$$

is $C^k$  in $\R^{dim(M)} \rightarrow  \R ^ {dim(N)}$. Thus, the notion of differentiability, as expected, depends on the representation of $U,V$ in the atlases chosen i.e $x(u \in U), y(v \in V)$ and this begs the question → What if we had another chart? If this differentiability exists as a notion on all such charts, maybe we can also 'Lift' this concept from the chart to the Manifold level. To do this, let's consider another chart such that

$$p \in U : (U,x') \in \mathcal {A}_M \\
q \in V : (V,y') \in \mathcal {A}_N \\
q = \phi(p) $$

Given our original maps $(U,x), (V,y)$, we don't really have the guarantee that the transformation on the new map:

$$y' \,\, \circ \,\,\phi \,\, \circ \,\,  x'^{-1}$$

Is a differentiable structure.  However, if we see a transformation from $x,y$ to $x', y'$ we can definitely write it as

$$x(U)\rightarrow {x' \circ x^{-1}} \rightarrow x'(U) : \R^{dim(M) } \rightarrow \R^{dim(M)}\\
y(V)\rightarrow {y' \circ y^{-1}} \rightarrow y'(V) : \R^{dim(N) } \rightarrow \R^{dim(N)}$$

And because this transformation exists, we can say that the second chart is also differentiable since we can always transform it to the first chart and then impose differentiability. This is summarized in the figure below:

<img width=700 height=600 src="static/GT/DS.png">

This relation would, by extension, be true for any chart for which this relation holds → $C^k$-compatible chart. If the map $\phi : M \rightarrow N$ is $C^{\infty}$-compatible then this is called a **diffeomorphism,** and the two manifolds $(M, O_M, \mathcal{A}_M)$  and $(N, O_N, \mathcal{A}_N)$  are called diffeomorphic if there exists a diffeomorphism between them. 

$$M \cong_{diff} N $$

Usually, we consider diffeomorphic manifolds the same as smooth manifolds. 

### How many different Differentiable structures can we put on a manifold up to diffeomorphism ?

The answer depends on the dimension:

1. $dim = 1,2,3$  → **Radon-Moise Theorems** → For we can make a unique differential Manifold from topological manifolds since all the different ones are diffeomorphic, which allows us to work easily with differentiability 
2. $dim > 4$ → **Surgery Theory →** We can essentially understand a higher dimensional torus by using familiar structures like a sphere and cylinder, which can be 'intelligently' combined: If we take a sphere, make a hole, and insert a cylinder i.e. perform surgery, while controlling invariance, like fundamental group, homotopy group, homologies, etc., then we can essentially understand the torus since we understand the sphere and Cylinder. The assertion is that we can, similarly, understand all structures in higher dimensions by performing intelligent surgery. In the 1960s, it was shown that there are finitely many smooth manifolds one can make from a topological manifold in dimensions greater than 4. One practical application of this to physics, in principle, could be that if we are to assume that spacetime is a differential manifold of higher dimensions, then pure math tells us that there exist finitely many ways in which this higher dimensional structure could be projected to our 3D understanding, and we could conduct experiments to determine which one of these nature has chosen!
3. $dim = 4$  → For the case of compact spaces, there are non-countably many different smooth manifolds that can be created! For the case of compact spaces, we look at partial results based on Betti Numbers. Thus, when we look at Einstein's description of spacetime being $\R^4$, we can see that there are non-countably many smooth manifolds as far as we know from the analysis. Thus, if our theories fail, they could very well fail because of our choice of the structure. 

One of the key features of Differential Manifolds is tangent spaces. Since we are speaking of geometry intrinsically, we need to develop an intuition that is separate from the embedded space in which an object exists.




<!-- %%% -->
# GT: Topological Manifolds

We can define a Topological Manifold as a Paracompact and Hausdorff topological space $(M,O)$  in which every point $p \in M$  has a neighborhood $p \in U \in O$ and there exists a homomorphism:

$$x : U \rightarrow x(u) \in \R^d$$

In other words, we can say that a topological space is a Manifold if it behaves like $\R^d$ locally. This manifold, as obvious, is $d$-dimensional. For example, a circle and a square are homeomorphic and represent the $S^2$ manifold. 

### Sub-Manifold

We can also extend this notion to the subsets of our topological space → So, if we have a space $N \subseteq M$, then we call it a sub-manifold of $M$  if it is a manifold in its own right. All we need to check, thus, is the induced topology $(N, O|_N)$ and see if it is a manifold or not. So, a circle can be considered a sub-manifold of the $\R^2$ , but two circles touching each other at exactly one point might not satisfy this since the point at which they touch is locally behaving like $\R, \R$  which is not the same as $\R^2$. 

### Product Manifolds

We can also define a product manifold by taking two manifolds $(M, O_M), (N, O_N)$ and defining a topological manifold $(M \times N , O_{M \times N})$, which will have a dimension of $dim(M) + dim(N)$ . For example, a Toroid is a product of two circles and can be written as $T^2 = S^1 \times S^1$, while a cylinder is $C = \mathbb{S}^1 \times \R$

<img width=500 height=300 src="static/GT/TopMan/TopMan-1.png">

A Mobius strip is a curious case since it cannot be written as a product Manifold, even though locally it looks like a product Manifold. To describe this, we need to define something new, called a Bundle .

## Bundles

These are pretty central concepts to a lot of things. A bundle of topological Manifolds can formally be defined as a triplet $(E, \pi, M)$  where:

- $E$ → Total Space
- $M$ → Base Space
- $\pi$ → A continuous surjective map from the Base space to the total space a.k.a a projection

For a point $p \in M$ , the pre-image of the set only containing $p$ under the map $\pi$ is called a Fibre:

$$F := preim_\pi (\{ p\}) \,\,\,\,\, \exists  p \in M  $$

For example, let's take a product Manifold. For a Fibre Bundle $F$ and a base space $M$ , we can define the total space as: 

$$
\begin{aligned}
E = M \times F \\
\pi : M \times F \rightarrow M 
\end{aligned}
$$

So, a Mobius strip can be through of a Bundle constructed by taking a rectangle and identifying sides going in opposite direction and then projecting the points on to the center.

<img width=700 height=300 src="static/GT/TopMan/TopMan-2.png">

We find that even though it is not a product Manifold, we can say that the pre-image of every point maps to the interval $[-1, 1]$ and this makes it a bundle of $\mathbb{S}^1$. We can see that bundles are essentially a generalization of the idea of taking a product, by intuitively understanding that to make a bundle we basically take a base space and attach fibers in a certain way. However, the definition does not really mention any notion of the total space being built out of the base space, and this is the generalization bit. 

### Fiber Bundle

We can be a bit more restricted in our notion of a bundle and yet be more general than a simple product space, To better elucidate this, we can say that in a bundle the fiber for multiple points need not be the same for all points → We are only interested in the existence of some fiber as per its definition. So, if we restrict the points to having the same fiber 

$$
F := preim_\pi (\{ p\}) \,\,\,\,\, \forall  p \in M  
$$

Then we call $E \rightarrow^\pi M$ a Fiber Bundle with the Typical Fiber $F$. We  often write the map as 

$$
F \rightarrow E \rightarrow^{\pi} M 
$$

Thus, fiber bundles are between Product Manifolds and General Bundles.  

### Section

Once we have a fiber bundle, we can further define a section of the bundle as a map $\sigma : M \rightarrow E$ such that if we make a point $p \in M$  and map it to some point $q \in E$, and then use $\pi : E \rightarrow  M$ to map it $q$ back to $M$, then the projection of $q$ will be the same point:

$$\pi * \sigma = \bm{I}_M$$

<img width=450 height=300 src="static/GT/TopMan/TopMan-3.png">

A very good example of this is in quantum Mechanics → The wave function $\Psi$ is a section of the complex line $\mathbb{C}$-line bundle, over some physical space, such as $\R^3$. This goes as from the physical space to the complex space

### Sub-Bundles

We can use the same logic of Sub-manifolds to create sub-bundles. We take a bundle $E \rightarrow ^\pi M$  and then define another bundle  $E' \rightarrow^{\pi '} M'$ . Now, this new bundle will be a sub-bundle if it meets the following three conditions: 

1. $E' \subset E$
2. $M' \subset M$
3. $\pi |_{M'} = \pi$  → When we restrict the projection map of our parent bundle to the base space of the other bundle, then we should essentially get the projection map of the other bundle

### Isomorphism in Bundles

If we have two  bundles: 

$$E \rightarrow^{\pi_E} M \\
F \rightarrow^{\pi_F} N $$

And we have two maps:

$$\varphi:   E \rightarrow F \\
f : M \rightarrow N 
$$

Then, we call this a bundle morphism. This can essentially be seen as true if the map below commutes.

<img width=250 height=250 src="static/GT/TopMan/TopMan-4.png">

Now, if we also have $(\varphi^{-1}, f^{-1})$  as another bundle morphism such that

$$\varphi^{-1} : F \rightarrow E \\
f^{-1} : N \rightarrow M $$

Then, the above two bundles are called isomorphic, since they clearly have the same fiber, and these isomorphic bundles are the structure-preserving maps. The essence of bundles, thus, not only lies in the topology of the manifolds but also in the projection → We can have topological spaces that are homeomorphic to each other but if the projection does not create an inverse mapping, then they won't be isomorphic as bundles. Bundles can also be **locally isomorphic** if we restrict the mapping and they still maintain the relationship

### Common Terminology on Bundles

- **Trivial bundle** → A bundle that is isomorphic to a product bundle
- **Locally Trivial** → A bundle that is locally isomorphic to a product bundle. E.g Cylinder is a trivial and so, locally trivial bundle, while a Mobius strip is locally trivial, but not a trivial bundle. Locally, any section of a bundle can be represented as a map from the base space to a Fibre. Thus, in Quantum Mechanics, it is okay to talk about $\Psi$ locally as a function, but there might be spaces in the space where we cannot do so.
- **Pull-Back Bundle** → A fiber bundle that is induced by a map of its base-space. It allows us to create a sort of yellow-data from the white data. For example, if we have $M' \rightarrow^{f} M$  and $E \rightarrow^\pi M$ , then we can find the pullback-bundle $E' \rightarrow^{\pi '} M'$  as:

    $$E' := \big \{(m', e) \in M \times E \,\, \big | \pi(e) = f(m') \big \}$$

## Viewing Manifolds from Atlases

Let $(M,O)$  be a topological Manifold of dimension $d$. Then a pair $(U, x)$  where 

$$U  \in O \\
x: U \rightarrow \R^d$$

 is called a chart of the manifold. This is just a terminology formalizing the notion that the neighborhood of a point in a manifold that maps to some subset of $\R^d$ be called a chart. However, since $x$ maps to $\R^d = \R \times \R \times ...$ , we can now say that the components of $x$  are essentially coordinates of a point $p \in U$  w.r.t the chart  $(U,x)$. This is crucial to understand, since now we are realizing that on any topological manifold, we can only define coordinates based on a chart, and we can have different such charts.  Thus, there has to  exist a set of charts such that every point is covered i.e 

$$\cup_{(U,x) \in A} U = M $$

Thus, there will be many-empty charts that overlap, and the collection of such charts is called an **Atlas.** 

### Compatibility in Charts

Two chard $(U,x)$  and $(V,x)$  are called $C^0$-compatible if either fo the following conditions are met: 

1. $U \cap V = \Phi$
2. $U\cap V \neq \Phi$, but $y \circ x^{-1}$ exists

Now, this is essentially the case when we are looking at manifolds. However, as we can see, compatibility allows us to traverse between two charts without really worrying about the underlying manifold. For example, in physics, we are transforming between coordinate systems - which are the charts in this case - but we are working with the fundamental assumption that this transformation does not change the trajectory of the particle. In other words, we can see the trajectory as a curve on a manifold and the co-ordinate systems of measurement as the charts that are $C^0$-compatible. When these charts are pairwise compatible in an Atlas, then we get a $C^0$-Atlas.



<!-- %%% -->
# GT: Topology

Topology is the study of properties of spaces that do not change under smooth deformations. This is called invariance. The way I like to think of this is by imagining the surfaces to be made of clay. we can change the shapes of these surfaces, under certain rules, and the interest of the field of topology is in general properties shared by all these deformations. Thus, topologically all the shapes that belong to a certain class and can be interchanged through deformation are topologically equivalent. A standard joke is the topological equivalence of a coffee cup and a donut → Both have one hole! These kinds of relationships are called homeomorphisms in Topology

<img width=500 height=300 src="static/GT/Topology/Top-1.png">

The main idea starts by classifying sets and then figure out a way to stack structures on top of it and this is the essence of spaces. A recurring theme, then, is figuring out a way to preserve structures while transforming a set. In other words, we first need to define a way to give structure to a set. Let u define two things here:

1. **Space** → We take a set and add some kind of structure to it to form a space. This structure is defined by a topology on a set, or a group structure, etc. Thus, a set is basically space where the structure is Null
2. **Map →**  A map between two sets $\phi: A \rightarrow B$ is ar elation such that $\forall  a \in A$ there exists exactly one $b \in B$ such that $\phi(a,b)$, and we can say $b = \phi(a)$ 

Now, we can define this recurring theme as essentially the classification of spaces based on structure-preserving maps between those spaces.

## Topological Spaces

In vanilla calculus, we define a function as

$$
f:  \R^n \rightarrow \R^m \iff \forall x \rightarrow x' \in \R^m \implies f(x) \rightarrow f(x')
$$

We call $\R^n$ as the Domain of this function and $\R^m$ as the co-domain of this function. Now, if we are to generalize this notion, we are essentially generalizing the domain and the co-domain to arbitrary sets on which this function can be defined i.e 

$$
f: X \rightarrow Y \iff \forall x \in X \rightarrow y \in Y \implies f(x) \rightarrow f(y)
$$

However, we need to define what it means when we say $x \rightarrow y$. One way to do this is through the **Metric Space** where we say that the idea of $x$ approaching a value $y$ implies that a certain notion of distance between these points tends to reduce towards zero. This distance can be defined through any kind of metric and we basically have a way to define function over this space. However, there are certain issues with this: 

- The distance function contains extra path information that is probably not required if we are only interested in the notion of  $x \rightarrow y$ . In other words,  if I am on a contract or expand the geometry of my set $X$, given that this operation is continuous, the notion of $x \rightarrow y$ still holds. However, the change that I have applied changes all distances that can be measured. So, in a sense, the distance is extra information if we are interested in continuity
- Metric spaces are not general enough to express point-wise convergence → Suppose we have a sequence of functions - $f_n$ - that  share the same domain and co-domain, then the sequence converges point-wise to $f$ if these sequences converge to the function int eh limit of infinity. Now, if I am  to compare the $f_n$ sequence's individual members, there is a not a clear notion of distance that the Metric space can provide to do this.
    $$\lim_{n \rightarrow \infty} f_n(x) = f(x)$$


Thus, we export some notions from calculus to  get the topological notions to get this generalization: 

- We take the notion of $\epsilon$  enclosing region and extend it to define an $\epsilon$ -ball around $x \in \R^n$ as

    $$B_\epsilon(x) = \big\{ x' \big| \,\, ||x - x'|| < \epsilon \big \}$$

- We use this notion of the $\epsilon$-ball to define an Open set $S \subseteq \R^n$ as the set for which we can find an $\epsilon$-ball for all points inside this set, for $\epsilon > 0$. In other words, we can call $S$ a union of $\epsilon$-balls that can be contained inside $S$. Thus, an open set is defined as a subset containing a neighborhood for each of its points.
- We define the Neighborhood of some $x \in \R^n$ as the $\epsilon$-ball around $x$

To understand the topological spaces, we first need to understand what does it mean to endow a set with a topology. The core idea is to take a set and stack some extra information on it. This extra information on the set is called topology, and it helps us define notions of continuous deformation of subspaces, continuity, etc. The vanilla calculus that we learn on the Euclidean Space, and in general any kind of metric space, is essentially taking the set $\R^n$ and applying a metric onto it that helps us do calculus on this set. In the general sense, we define this topology through open sets, defined previously. Thus, we can now define a topological space as a set $X$ and a collection $O$ of open subsets, such that the following 3 criteria are met: 

1. $\Phi, X \in O$ 
2. $U,V \in O \implies  \cap \{ U, V \} \in O$  → elements, which are defined through open sets, are closed under finite intersection
3. $U,V \in O \implies  \cup \{ U, V \} \in O$ → elements are closed under arbitrary unions

This collection of subsets $O$ is called a **topology** on $X$, and the pair $(X, O)$ is called a topological space. We can define many different topologies on the same set, and each of these topologies helps us define the notions of continuity and convergence on this set, which helps us do calculus on this space. So, we can use the notion of the topological spaces to define continuity → For topological space $X, Y$ , we can say that $f: X \rightarrow Y$ is continuous if for any open set $V \subseteq Y$ we have  

$$
f^{-1} (v) = \{ x \in X | f(x) \in V\}
$$

which is open in $Y$. Thus, we are essentially defining continuity as open sets mapping to open sets by saying that any function that produces a value in an open set of $X$ will have an inverse image as also an open set in $X$; and we can now say that for a given $\epsilon > 0$  , we can find a $\delta > 0$ such that: 

$$
|x - x'| < \delta \implies |f(x) - f(x')| < \epsilon  
$$

Some common kinds of topology are: 

1. Chaotic Topology →  The topology defined by $O = \{M, \Phi\}$  i.e none of the elements in M are open sets
2. Discrete Topology → Defined by $O = \Rho(M)$ i.e each and every element of M is an open set 
3. Standard topology → Defined by taking the $\epsilon$-ball around each point and asserting that for all points in $M$, we can construct a ball with radius $\epsilon$ that is entirely inside M. This also highlights the importance of defining open Sets. If we were to take the boundaries of the set into its definition, thereby making it a closed set, then all the boundary points essentially violate the standard topology condition.
4. Metric Topology → The one applicable to metric spaces, where we define the $\epsilon$-ball as a distance measure i.e it has to obey the properties of being greater than zero, commutativity and triangle inequality

    $$B_\epsilon(x) = \big\{ x' \big| \,\, d(x, x) < \epsilon \big \}$$


## Constructing New topologies on a given Topology

Once we have a topological space, then we can create topologies on it. Let $(M,O)$ be a topological space. Then for a subset $N \subset M$, we define a new topology: 

$$O|_N := \big \{ U \cap N \,\,\, \big |  \,\,\, U \in O  \big \} \subseteq \Rho(N)$$

This new topology $O|_{N}$ is called an **induced (subset) topology** on $N$ which created by intersecting open sets that define the topology on the superset $M$ with the subset $N$, which leads to the new smaller collection of open sets being in the power set of $N$. we easily see why by testing the 3 criteria of $O|_{N}$ being a topology as defined above. A cool thing that induced topologies help us with is defining a topology on non-open subsets of $M$. For example, we can take the set $\R$ and add the standard topology on it to get $(R, O_{std})$ and take the subset $N = [-1, 1]$. Now if we are to consider the set $(0,1]$ and see that clearly, it does not belong to $O_{std}$ since it is not open. However, it can easily be written as $(0,1] = (0,2) \cap[-1,1]$  and this make it a set in the induced topology $O_{std}|_N$. Thus, 

### Convergence

To define convergence, we take a sequence and define it as a map from $\N$ to the set $M$  

$$q: \N \rightarrow M $$

which essentially means that we have a sequence of number in $\N$  that uniquely map to points in $M$. Now, let's take a point $a \in M$ . Let a belong to a subset  of the topology i.e $a \in U \in O$ , then we can say that the sequence $q$ is convergent to this limit point $a$ if :

$$\forall U \in O : \exists N \in \N : \forall n > N : q(n) \in U$$

In other words, we can say that $q$ is convergent to $a$ if, for any open set in the topology to which this point can belong, there exists a point $N$ in the set $\N$  beyond which the sequence always maps to this subset. If this were not happening, the sequence would never converge. For standard topology, these subsets would be the $\epsilon$-balls, and thus the notion of convergence would be that beyond $N$ the sequence should always map to a point within this $\epsilon$-ball. Hence, this is a generalized notion of our vanilla notion of convergence, extended to any kind of topology. 

### Continuity

For continuity, let's take two topological spaces  $(M, O_M)$ and $(N, O_N)$ and take a map between these spaces as $\phi: M \rightarrow N$ . Now, we can call this map continuous if

$$\forall U  \in O_N : \exists V \in O_m : \phi(v \in V) = u \in U$$

In other words we are saying that for all open subsets of $N$ -  $U \in O_N$ - the pre-image of U - $\big \{ m \in M \big | \phi(m) \in U \big \}$ exists in $O_M$

### Homeomorphism

Let $\phi: M \rightarrow N$  be a bijection. Now, if we equip these sets with topologies $(M, O_M) \,\,\,\, (N, O_N)$, then we say that $\phi$ is a homeomorphism if : 

1. $\phi: M \rightarrow N$  is continuous 
2. $\phi ^ {-1}: M \rightarrow N$  is continuous

Thus, we have essentially used the notion of continuity provided by topological spaces to define a map between two topologies that preserves the structure. This is why we can see that the non-geometrical essence of a toroid and a cup are the same since this homeomorphism exists. This homomorphism is providing a one-one pairing of the open sets of $M$ and $N$. And if such homeomorphisms exist, then we can say that  $M$ and $N$ are isomorphic in the topological sense

$$M \cong_{top} N$$


## Topological Properties

### Separation Properties

- **T1** → A topological space $(M,O)$ is called T1 if for any two distinct points $p \ne q$

    $$
    \exists U \in O : p \in U: q \notin U 
    $$

- **Hausdorff or T2** → A topological space $(M,O)$ is T2 if for $p \ne q$

    $$
    \begin{aligned}
    &\exists U \in O : p \in U \\
    &\exists V \in O : q \in V \\
    &U \cap V = \Phi
    \end{aligned}
    $$

T1 is a weaker argument than T2 since we are not applying the neighborhood condition on both points. Any topology that is Hausdorff will by extension be T1, but not the other way round. We can have multiple such properties for separation, depending on how we decide to separate our points, but the core idea remains the same. 

### Compactness

Compactness generalizes the notion of boundedness and closed sets in Euclidean space. Ideally, we can construct properties that are only valid for finite sets and become invalid for infinite sets. These essentially are: 

1. Boundedness of function → If $f: X \rightarrow \R$ we always have $f(x) \leq K$
2. All functions attain a maximum → There is some $x_0$ such that $f(x) < f( x_0) \,\,\,\,\, \forall  x \in X$ 

The first statement is essentially saying that if we can bound our function locally, then we can also boost this boundedness globally, while the second statement is asserting that the perturbations of our function are bounded some maximum value. it is easy to see why there is no guarantee that this might be the case if $X$ is an infinite set. Now, when we endow our domain with additional structure, to create the topological space $(M,O)$, then it turns out that some kinds of sets start exhibiting properties similar to finite sets, even though they may technically be infinite. We call these spaces compact. 

To understand this, we need to first generalize what it means to be closed or bounded in the topological sense, and we do this through is **Covers →** We can call a set $C$ a cover of  our topological space $(M, O)$ if it satisfied the following conditions: 

1. $C \subseteq O$  → $C$ is a collection of open sets
2. $\cup C = M$   → The union of the elements of $C$ give rise to $M$

Since $C$ solely comprises open sets, we also call it the **Open Cover** of $M$.  Now any such open cover $C'$ that can be formed and is a subset of $C$ will be called a subcover of $M$. Now, we can use these covers to define compactness → Any topological space $(M,O)$ is compact if every cover $C$ of $M$ has a finite subcover $C'$. 

- The need for including subcovers in this definition is because there is no finiteness guarantee for covers. For example, let's think of $\R$ as our topological space, and let's consider the set $(0,1)$. For every element of this set, we can create a partition $(0,\frac{1}{n})$ and $(\frac{1}{n}, 1)$ such that it will fall in any one of the two and thus, we can't say that there exists a subcover that is finite. However, if we include the endpoint to get the interval $[0,1]$, then we can see that $0$ and $1$ might not fall in these partitions and so there has to exist a finite sub-cover. Hence, the subcover is allowing us to essentially understand what it means to be small in a purely mathematical sense. Also, this process of including end-points is called compactification

This notion of compactness also extends to subspaces and homeomorphic spaces of a topological space $(M,O)$. The **Heine-Borel Theorem** says that for a metric space, every closed and bounded subset is compact. 

#### Paracompactness

Paracompactness is a weaker notion than compactness. To understand it, we will define a refinement on a cover $C$ as a subset $C' \subseteq C$  such that  

$$
\forall U \in C : \exists U' \in C' : U' \subseteq  U 
$$

Now, we can call our topological space $(M,O)$  paracompact if every open cover has an open refinement that is locally finite. We can call a refinement locally finite if every point of the space has a neighborhood that intersects only finitely many sets in the cover. Thus, we are saying that our space locally behaves in a certain bounded manner. This is important for defining manifolds. It can be seen that compactness implies paracompactness, and thus, the **Stone Theorem** says that every metrizable space is paracompact.

### Connectedness

The idea behind connectedness in topological spaces is to be able to define the notion of a 'whole'. Put simply, if we can express our topological space as the union of two or more disjoint non-empty open subsets, then essentially our space is a composite of those two spaces. Thus, define a topological space $(M,O)$  as connected if the following condition does not hold.

$$
\exists A,B \in O : M = \ A \cup B
$$

Here, we are not defining the notion of a 'how' $A$ and $B$  connect. All we are saying is that we can create $M$ through the union of $A$ and $B$. To explore these kinds of connections, we formalize the notion of a path from a point $p$ to a point $q$ as a continuous function $\gamma$ such that  : 

$$
\begin{aligned}
&\gamma : [0,1] \rightarrow M \\
&\gamma (0) = p \\
&\gamma (1) = q 
\end{aligned}
$$

Thus, if tis condition holds for every pair $p,q \in M$ , then we call our topology **Path-Connected**

## Homotopic Curves and the fundamental Group

The idea of Homotopy is to deform the paths between two points in a topological space into one another. We are essentially saying that if two points $p,q \in M$  are connected by two paths $\gamma, \delta$ such that: 

$$
\begin{aligned}
&\gamma (0) = \delta(0) = p \\
&\gamma (1) = \delta(1) = q
\end{aligned}
$$

Then we can talk about a function on all such paths between $p$  and $q$ : 

$$
\begin{aligned}
h: [0,1] \times [0,1] \rightarrow M  \\
h(0, \lambda) = \gamma(\lambda) \\
h(1, \lambda) = \delta(\lambda)
\end{aligned}
$$

And if this function exists, then $\gamma$ and $\delta$ are homotopic. Thus, we are essentially saying that all the paths between $p$ and $q$ that satisfy the requirements of $h$ are deformable into one-another. This is an interesting visualization, shown below:

<img width=300 height=300 src="static/GT/Topology/Top-2.png">

We can now define loops as essentially paths that start and end at the same point i.e

$$
L_p : \big \{ \gamma: [0,1] \rightarrow M \big| \gamma(0) = \gamma(1)      \big \}
$$


<!-- %%% -->
# GT: Intuitive Introduction to Geometry

Euclidean geometry rests on Euclid's 5 postulates, which are basically the set of rules for doing anything in Euclidean space: 

1. Draw a straight line from any point to any point.
2. Produce a finite straight line continuously in a straight line
3. Describe a circle with any center and distance
4. All right angles are equal to one another
5. if a straight line falling on two straight lines makes the interior angles on the same side less than two right angles, the two straight lines, if produced indefinitely, meet on that side on which are the angles less than the two right angles.

The fifth postulate, also known as the parallel postulate, was what was violated to create variants of geometry as follows: 

- **Spherical/Elliptical Geometry**  → This assumes that the parallel lines converge, for example, to the poles of a sphere. So, imagine two parallel lines start from somewhere in the equator and converging on the poles of a sphere. If we hold this sphere from the poles and pull it, we get a surface that is not equally curved at all points - an ellipse - and the similar idea of converging lines applies to it too. The rules that govern this 'world' would then need to be changed, like the interior angles of a triangle on this surface which would no longer be $\pi$ radians, but something like $\pi(1+4f)$ where $f$ is the fraction of the sphere's surface that is enclosed by the triangle
- **Hyperbolic Geometry** → Here we violate the fifth postulate by assuming a scenario where the parallel lines diverge. An example surface that obeys rules for this geometry could be the shape of the pringles chips - hyperbolic paraboloid - and we can easily produce 2 parallel lines and see that they diverge along the surface.

To go abstract, we want to understand how we can generalize to any kind of surface geometry → Abstract and define the ideas that are central to these geometries. So, we start with understanding some properties of the spherical and hyperbolic geometry.

## Nature of the Curvature

If we draw concentric latitudes around the pole, we would see that taking the radius of the circles as the length from the center of the sphere and then measuring the circumference, we would see that it would actually be the same as the standard circle formula  → it would be somewhat lesser for all points not at the equator: 

$$
\begin{aligned}
&C = 2 \pi R \sin(\frac{r}{R}) \\
\implies &C < 2 \pi R
\end{aligned}
$$

This is like saying that that the sphere is tending to curve towards the horizontal origin - the pole. For the saddle - the pringles - on the other hand, we would see that the curvature is actually not tending towards anything. To make this notion more precise, we can take the second derivative of the surface at each point by defining a vector that is normal to the surface at each point, and classify the curvature:

- **Positive Curvature** → Curvature has a tendency to curve in the same direction as the tangent i.e the second derivative is negative
- **Negative Curvature** → Curvature that tends away from the tangent i.e the second derivative is negative
- **Zero Curvature** → The notion of flat

This is somewhat similar to how we would define the points of maxima and minima for functions, and a 2D equivalent is shown below:

<img width=800 height=450 src="static/GT/Intuitive/Intuitive-1.png">

Thus, now we have a notion that allows us to say that the spherical curvature is positive, while the hyperbolic curvature is negative. The Euclidean curvature is, of course, flat. Another way to think of this would be that there is a notion of finiteness associated with the positive curvature of spherical geometry, which is what relates to the parallel lines focusing on one point when produced further, while there is a notion of infinity associated with negative curvature that makes the parallel lines diverge in the hyperbolic case. Since the Euclidean plane is flat, this means that the parallel lines would keep going-on till infinity without ever meeting. 

<img width=400 height=500 src="static/GT/Intuitive/Intuitive-2.png">

## Generalizing Geometry

- The first step to creating a general notion of geometry is to understand the point of view when we are talking about surfaces. The classical way of looking at geometry is through a higher dimensional space in which it is embedded. So, When I am looking at a place, I exist in $\R^3$ in which there is a surface in $\R^2$ that I can see and then comment on its properties like curvature, etc. This is an **Extrinsic View,** and so the curvature is the Extrinsic Curvature of the surface. However, this might not be the most ideal way to go about looking at curvature since we always need a higher dimensional space to be able to study any space. 
- Another view to studying geometry is the Intrinsic view, where we study the space from the perspective of the space itself. This is the same as saying we take a space, get some 'rulers' to measure something like a distance on this space, and 'protractors' to measure something like an angle. Using these tools, we create a system that allows us to understand the curvature of our space in and of itself. This curvature would be called the **Intrinsic Curvature**
- To demonstrate this, consider the figure shown below. One way to think of it would be to consider a Euclidean space that has been 'waved' a bit. The extrinsic picture from 3D is pretty clear. However, if we think from the point-of-view of a creature bound to this 2D space → To the creature this is still a flat surface.

<img width=800 height=400 src="static/GT/Intuitive/Intuitive-3.png">

To understand why we need to use vectors. Let's take the simple example of a sphere. At any point we can define two vectors: 

- **Normal Vector** → That protrudes outwards from the sphere and so is coming out into the 3D space
- **Tangential Vector** → This is tangent to the surface at every point, so remains in the tangent plane

We can use Normal vectors to define the extrinsic curvature → Consider a Normal Vector at a point $A$ on a sphere. If we parallel transport this vector to a point B i.e take this vector and put it at point B through some path while keeping its original orientation intact, we can then compare this vector with the normal vector at B and the difference between these vectors would define the extrinsic curvature of the surface

<img width=550 height=500 src="static/GT/Intuitive/Intuitive-4.png">

We can use Tangential vectors to study the intrinsic curvature of this surface → If we take a tangential vector at point A on this sphere and then make it go a loop around this sphere and then compare how it has changed, this should be proportional to the curvature of the region enclosed by the loop. For example, in the figure below, if we take the tangential vector, and then transport it through the upper hemisphere to the other end and then come back to the original point through the equator, we will actually get a $\pi$ radian shift.

<img width=550 height=500 src="static/GT/Intuitive/Intuitive-5.png">

This would not be the case if this vector was parallel transporting on a Euclidean space since in any loop we would get the same vector. If we use this procedure on the wavy surface, we can see that both the tangential vector would not change in a loop but the normal vector would. Hence, we say that the surface is extrinsically curved but intrinsically flat.

## Riemann's Geometry

We can use the ideas above to create some notions around any curved surface we want. to do this we first need to assume that the surfaces are smooth i.e the there are no abrupt changes. This idea if formalized further in Topology into the notion of a manifold. For now, let's go with the notion that if we are to zoom into this smooth surface, we would end up encountering an Euclidean space, similar to how the earth seems flat but in actuality it is curved (Flatearthers ?). Thus, if we zoom a good eough anoumt, we could get an infinitesimally small Euclidean space. O this space, we would not need to define the notion of a distance which comes from the L2 norm i.e the pythagoras theorem:

$$
ds^2 = dx_1 ^2 + dx_2^2 
$$

If we were to scale $x_1,x_2$ by som constants $a_1, a_2$ and then change the right angle to an angle $\theta$ between $a_1x_1$ and $a_2x_2$, then our equation would be modified to:

$$
ds^2 = a_1^2dx_1 ^2 + a_2^2dx_2^2 + 2a_1a_2dx_1dx_2\cos(\theta)
$$

This is called a **Metric Tensor.** We can express this in a general matrix form to make it extendable to more dimensions and more information that might be required for the surface oto characterize it

$$\begin{bmatrix}
   g_{11} & g_{12} \\
   g_{21} & g_{22} 
\end{bmatrix} = 
\begin{bmatrix}
   a_1^2 & a_1a_2\cos(\theta) \\
   a_1a_2\cos(\theta) & a_2^2 
\end{bmatrix} \\
$$

Thus, in general we can write: 

$$
ds^2 = g_{ij}dx^idx^j
$$

We can use this Metric tensor to measure distances between any two point by adding all the small distances $ds$ along the way : 

$$
S = \int_a^b \sqrt{g_{ij}dx^idx^j}  ds
$$

This distance can be along any path between the two points. If we consider the set of all paths that connect two points, we can then be interested in the shortest path out of this set. This is called a **Geodesic.** These points are given by the Euler-Lagrange formulations for an Energy function $E$ defined as: 

$$
E = \frac{1}{2} \int_a^b g_{ij}dx^idx^j ds \,\,\,\,\,\,\,\,\, s.t \,\,\,\,\,\,\,\,\, S^2 \leq 2(b-a)E 
$$

And the final equation for the geodesic comes out to be:

$$
dt^n + \Gamma_{mr}^nt^rdx^m = 0 
$$

Where $\Gamma_{mr}^n$ is called the Christoffel symbol and is defined as: 

$$
\Gamma_{mr}^n = \frac{g^{np}}{2}\bigg[ \frac{\partial g_{pm}}{\partial x^r} + \frac{\partial g_{pr}}{\partial x^m} - \frac{\partial g_{mr}}{\partial x^p}\bigg]
$$

Now, as we discussed with parallel transport previously, Riemann formalized that idea through the Riemann tensor → Take a vector $V_s$ and pass it through a loop on a curved surface back to its original point to get a vector $V_p$. This change is denoted by a vector $D_rV_s$ which characterizes the curvature and can be written as 

$$
D_rV_s = \partial_rV_s - \Gamma_{rs}^p V_p
$$

This characterizes the curvature of the surface and the general form of the curvature tensor is:

$$
R^t_{srn} = \partial_r \Gamma_{sn}^t - \partial_s \Gamma_{rn}^t + \Gamma_{sn}^p \Gamma_{pr}^t  - \Gamma_{rm}^p \Gamma_{ps}^t  
$$

Thus, we just need to specify a point and 2 basis vectors along which the loop needs to move ie a total of 3 vectors and we get the curvature at this point computed by Riemann Curvature Tensor. Since it exists for all points, we can also say that this is a field i.e the metric takes a value for each point and based on where the points are, we can have values for the metric. If we extend this idea further, then we see that for a collection of 2 points we will always have values pertaining to paths between these points. We can call this a connection field. This connection, as we saw before, comes from the Metric tensor. Thus, we can say that the Metric gives the connection between two points and the connection gives the curvature. Each path may have a different curvature depending on the nature of the surface.


<!-- %%% -->
# RTR: Conjuring Magic

An interesting thing that I recently read in [The Road to Reality](https://en.wikipedia.org/wiki/The_Road_to_Reality) was the power of abstraction in Mathematics. We traditionally learn numbers as being present and the set as being a natural extension of these numbers. Now, an interesting way to look at the power of the abstraction that the set theory allows us is to try and build the natural number system through empty sets! To do this, let us first consider the null set 

$$
\Phi = \{ \}
$$

Now, if we take the set of $\Phi$, we would get 

$$
\bm{1} = \{ \Phi \}
$$

Here, I have used the boldface to represent the number of elements in the set, and in this case it is 1. Now, let's take a collection of this new set and the empty set to make a second set with 2 elements: 

$$
\bm{2} = \{ \Phi, \{ \Phi \} \}
$$

If we again repeat this procedure to three and four elements, we get: 

$$
\begin{aligned}
\bm{3} = \{ \Phi, \{ \Phi \}, \{ \Phi, \{ \Phi \} \} \} \\
\bm{4} = \{ \Phi, \{ \Phi \}, \{ \Phi, \{ \Phi \} \}, \{ \Phi, \{ \Phi \}, \{ \Phi, \{ \Phi \} \} \} \}
\end{aligned}
$$

The interesting thing is, these are all valid sets! We have essentially used the null set to create 4 sets that are non-empty, by recursively using the null set. If we repeat this procedure infinitely, we see a clear isomorphism between the natural number line and this series of sets. Thus, we have somehow created numbers out of nothing!

## Broader Outlook 

We get an infinite sequence of abstract mathematical entities—sets containing, respectively, zero, one, two, three, etc., elements, one set for each of the natural numbers, quite independently of the actual physical nature of the universe. This just reminds me of Plato's belief in a world of abstract forms, independent of the world we live in. While we don't need to religiously believe in such a world, what seems interesting to me is that this 'abstraction' of mathematics allows us to somehow break the barriers of the physical world and develop some really cool and interesting stuff that might not even be discovered in our times. Moreover, as we have seen in the history of the sciences somehow this formal 'imagination' (If we must) goes hand-in-hand with our attempt to understand reality through physics. Roger Penrose, beautifully summarizes this through his adaptation of M.C Escher's Penrose stairs as shown in the diagram below 

<img width=500 height=450 src="static/RTR/Screenshot from 2021-02-16 00-57-02.png">


<!-- %%% -->
# MobMod: Vehicular Flow Modeling 

The key principle in flow modeling is to treat the vehicles as a liquid flowing through sections. We then put some numbers to this and get the car-following models. The major classification of the Vehicular Flow Models is shown below:

<img height=500 width=900 src="static/MobMod/Flow/flow-1.png">

The models are fundamentally differentiated on the basis of the level of precision that they go into, which is inversely related to the cost of modeling, and this gives us the 3 categories: 

- **Macroscopic:** Here, we model vehicle traffic as a flow as a segment of the road, and thus, our fluid is the cars entering/exiting that segment and the density of the number of cars in that segment. All the cars in this segment will have the same speed. Thus, we can have the relationship :

$$flow = speed \times density$$

- **Microscopic:** Here, we measure the position of each vehicle and have the speed of each vehicle as different. Thus, we have individual entities with aggregated models.
- **Mesoscopic:** This is a hybrid between the above two, where we get to keep the individual speeds of vehicles and yet analyze flows. Thus, we can say stuff like while all vehicles will have the same speed in a section, we can still control individual vehicles. This also allows us to look into mixed flows.

## Macroscopic Models

### Lighthill-Whitman-Richard (LWR) Model

In this model, we directly model the traffic as a fluid. Thus, we take the flow $m$, the density $\rho$  and the speed $v$ over a segment $[x,\Delta x]$ and apply the fluid lagrangian equation to it to get

$$m(x,t) = \rho(x,t)v(x,t) \\
\frac{\partial \rho(x,t)}{\partial t} + \frac{\partial m(x,t)}{\partial x} =  0
$$

These are the basic constraints that tell us that the flow in this section is conserved and any discrepancy is happening due to friction or heat, which can be modeled as abrupt changes and leaving cars respectively. However, we have two equations  and 3 unknowns, and so for the vehicular model we impose the third condition which says that the speed and density are related

$$v(x,t) = f(\rho(x,t))$$

This relation is intuitive since the more density we have, the slower is the speed. Now, the way we model this speed function determines how we solve the problem and this is what gets us to apply this model.

### Fundamental Diagram

The solution of macroscopic models results in the fundamental diagram, which is something that is common in macroscopic and mesoscopic models.

<img height=400 width=1000 src="static/MobMod/Flow/flow-2.png">

The diagram above is called the flow-density diagram. It accurately represents a highway phenomenon. Initially, we have a free highway and so, the more vehicles we add to this, the more is the speed of the vehicles until a certain point at which there are enough cars so that each car cannot move as fast as it wants. This is the region on bounded flow where the speeds of the cars are correlated. This happens till a maximum density that gives us a positive flow $\rho_{critical}$ after which we cannot add any more vehicles to the road without reducing the flow. Thus, our ideal is to stay at this density and try to play around this value. if w go beyond this flow, the speed goes down according to the strong correlation between vehicle speeds. This downward curve depends on the region in which we are. If we are on a highway, it is easier to have a straight line than in an urban environment. This result happens till a certain point after which we enter a traffic jam. This density is the $\rho_{jam}$ and is the starting of the region of traffic jam. In this region, the cars may be moving very slow till a density called $\rho_{max}$ at which the cars halt.

### Traffic Stream Model

This model is initially a macroscopic model since the whole model is characterized by macroscopic quantities, but to be able to implement this in a simulator, we need to discretize it. thus, we end up with individually modeled vehicles, but they all move according to macroscopic quantities i.e all vehicles traveling on the same road segment will have identical speed/acceleration. 

#### Fluid Traffic Model

This model exploits the inverse proportionality in the speed/density relationship → The speed is inversely related to the density of cars on a road :

$$v_i(t+ \Delta t) = max [v_{min}, v_{max}(1 - \frac{\rho(x,t)}{\rho_{jam}})]$$

As we can see, this is nothing but the velocity constraint needed to solve the lagrangian.

## Mesoscopic Models

### Queuing Model

In this model, each road is seen as a FIFO queue of length (size) $C$. Thus, the travel time is a function of $C$, but: 

1. At max, only $L$ cars can leave the queue
2. A car cannot leave until another queue accepts it
3. A queue can accept at mos  $C$  cars

<img height=350 width=900 src="static/MobMod/Flow/flow-3.png">

## Microscopic Models

Now, each vehicle has its own speed and we can have a lot of interesting things in this like leader-follower models, overtaking, etc., which is either not possible in macroscopic models, or is too complicated to model. All the three kinds of models are used for different kinds of applications: 

1. For **traffic safety**, we need fine granularity of vehicle representation since the communication range is short. Thus, we use Microscopic Models
2. For **Navigation,** we need to have a notikn of large scal  mobiltiy and Aggregated Metrics like average spped and time. Moreover, we need the capability of re-routing, and the precise position may not be that useful. Thus, while we can apply bot microscopic and mesoscopic models, the **mesoscopic models are preferred**
3. For Traffic monitoring, we just need a notion of aggregated metrics on a very large scale and need to see stuff like how the whole bunch of cars are populating a highwar. Thus, we need macroscopic modelling

### Single Lane Models

#### Stochastic Model

These are equivalent to Random models liek RWP, but tailored to a graph:

- We choose the location and speed, but can only move on a graph
- Each vehicle is completely independent of the other

The modelling is shown below:

<img height=400 width=900 src="static/MobMod/Flow/flow-4.png">

Here, we have the reference car in teh middle lane indexed by $i$ and has :

- Position → $x_i(t)$
- Speed → $v_i(t)$
- The cars in the lanes to the sides are indexed by $j, k$
- Distance between 2 cars - bumper to bumper distance - is calculated as

    $$\Delta x_i(t) = x_{i+1}(t) + x_i(t)$$


#### Constant Speed Model

This is an example of a stochastic model where we apply RWP on a graph. 

<img height=300 width=600 src="static/MobMod/Flow/flow-5.png">


The movement of each vehicle is structures in trips, and we choose the spped as

$$v_i = v_{min} + \eta(v_{max} - v_{min})$$

Where $v_{min}, v_{max}$ are the bounds on speed and $\eta \in [0,1]$ is a random variable. The key aspect is tuning this $\eta$ to get a good speed estimate. An example of this being applied to urban scenarios is called **Urban CSM** where we represent roads on a city as nodes and edges and we can set $v_{max}$ depending on the speed limits on the road. We can add optimizations to this, like uniformly sampling speed around max:

$$v_i \sim U(v_{max} + \epsilon , v_{max} - \epsilon )$$


#### Manhattan Model

This is basically the same as the Urban CSM, but we become more realistic in sampling speed :

- We update speed based on acceleration 

    $$v_i (t + \Delta t) = v_i(t) + \eta a \Delta t  $$

- Then we ensure that this speed is bounded between $[v_{min}, v_{max}]$ 

    $$v_i (t + \Delta t) = \min\{\max\{v_i (t + \Delta t), v_{min}, v_{max} \}$$

- We then add a speed reduction if the bumper-to-bumper distance is lesser than a safety metric to implement car-car interaction and collision avoidance

    $$v_i (t+ \Delta t) = v_{i+t} - \frac{a}{2}, \,\,\,\,\, if \Delta x_i(t) \leq \Delta x_{safe}$$

### Car Following Models

In these models we follow the leader-follower approach. Thus, the speed of a car depennds on the cars nearby and on an absolute relative speed of surrounding vehicles.. In orther words, the movemet of each vehicle is strongly correlated with other vehicles.

### Intelligent Driver Model

The objective of this model is to represent the **interaction** between two vehicles in the most realistic way. Interaction in microscopic models means either **acceleration** or **speed**. The idea is that our car should accelerate based on its current speed and its vicinity, and this is modeled as follows :

- The resulting acceleration depends on the maximum acceleration  $(a)$, the ratio of the current speed to the maximum speed $\frac{v_i(t)}{v_{max}}$, and the ratio of the desired inter-distance between the car ad the bumper-to-bumper distance $\frac{\Delta x_{dex}(t)}{\Delta x_i(t)}$ as follows:

    $$\frac{d v_i(t)}{dt} = a \bigg[1 - \bigg(\frac{v_i(t)}{v_{max}} \bigg)^4 - \bigg(\frac{\Delta x_{des}(t)}{\Delta x_i(t)}\bigg)^2 \bigg]$$

- The desired inter-distance between the cars, in turn, depends on a safe distance, a safety headway time, the current velocity and the maximum acceleration (a) and deceleration (b

    $$\Delta x_{des}(t) = \Delta x_{safe} + \big[v_i(t) \Delta t_{safe} - \frac{v_i(t) \Delta v_i(t) }{2 \sqrt{ab}} \big]$$

- The major idea here is that the desired distance between the car should at-least be a safety distance $\Delta x_{safe}$ and then to this safety distance we add a headway distance calculated by multiplying the headway time $\Delta t_{safe}$, which is the elapsed time between the front of the lead vehicle passing a point on the roadway and the front of the following vehicle passing the same point and is usually something like 2 seconds,  with the current velocity. This means that we are essentially adding a leeway to the minimal safety distance by seeing how fast we are moving and multiplying it by the headway time. Thus, if these two were the only factors in consideration, then we would have a model where when our car is at rest, $\Delta x_{safe}$ is the desired distance and in case our car is moving, then we add a velocity-based factor to it to take into account how fast the car is moving. We also subtract the minimal distance by $\frac{v_i(t) \Delta v_i(t) }{2 \sqrt{ab}}$ to account for the difference in the velocity of the car and the next car and normalize it by the breaking capability $2 \sqrt{ab}$. This factor takes into account the relative speeds w.r.t to the car model the driver adapting to the difference in speeds.

This model is used in SUMO, VANET MobiSIM etc.










<!-- %%% -->
# MobMod: Palm Calculus
- Source: [J-Y Le Boudec's Original Paper](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=650FABC44E04160877ED5C48308BFD62?doi=10.1.1.85.8803&rep=rep1&type=pdf) , [RWP Considered Harmful by Yoon et. al](http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=EE482699E40FC579D10F9B0D112599A9?doi=10.1.1.12.1241&rep=rep1&type=pdf)

## Issue with Random Models

To do this, we have to first realize that the performance measures of ad-hoc mobility models are directly impacted by the selection of the mobility model. So, to have any talk about the issues, we need to be clear about the model we are talking about first. Let us select the Random Waypoint Model. Now, one of the most important parameters of most mobility models is the speed, either of the individual particles as a constant value, or a distribution of speeds. Thus, if we have to compare the performance of routing protocols using the RWP model, then we have to be able to adjust this parameter in order to compare the performance of routing protocols under different levels of nodal speed. To be able to make this comparison, we need to assume that the model will reach a steady-state of speeds that can be used to compare average metrics, and in the RWP this average is often believed to be either half of the maximum speed $V_{max}$since the node speeds are chosen from a uniform distribution  $(0, V_{max}]$, or some other justified value in this range. Thus, to analyze the performance of a protocol and make comparisons, we assume that this average is reached at the start of the simulation. Hence, we would find simulations making comparisons by varying the value of $V_{max}$, and packaging results into averaged comparisons.

### Decay in Random Waypoint 
To understand the problem, let us take the average instantaneous speed which we define as 

$$
\bar{v}(t) = \frac{\sum_{i=1}^N v_i(t)}{N}
$$

Where, $N$ is the total number of nodes in the simulation, and $v_i(t)$ is the speed of a node at time $t$. For $V_{max} = 20$  and null pause times, an average of the instantaneous average speeds over 30 scenarios is shown in the following graph

<img width=500 height=350 src="static/MobMod/palm.png">

We see that this instantaneous average is consistently decreasing and it was proven that this can mathematically decay to 0 in the RWP model. An intuitive explanation is that the random waypoint model chooses a destination and a speed for a node at random and independently, and the node will keep moving at that speed until it reaches the destination. So, if we have a node with slow speed and a far-away destination, it is quite possible that this node might move so slow that it may take a very long time to finish the trip, or worse, never reach the destination. More and more nodes can be stuck with these slow speeds and thus, dominate the average value over time and this is what leads to the decay. This is a recurring issue with the majority of such model comparisons.

## Resolving through Palm Calculus

To understand Palm Calculus, we have to understand the difference between time and event averages. Time averages are obtained by sampling the system at arbitrary time instants, for example, the figure below is obtained by sampling the system every 10 sec of simulated time.

<img width=500 height=350 src="static/MobMod/palm-2.png">

Event averages are obtained by sampling the system when selected state transitions occur. For example, the figure of the same system as time averages, when sampled on the times at which the nodes reach end of Waypoints is shown below: 

<img width=500 height=350 src="static/MobMod/palm-3.png">

Palm calculus is a set of formulas that relate time averages versus event averages. Thus, the distribution obtained through event sampling is called a Palm distribution.

### Stationary Processes
A stationary process is one that is independent of a shift that can be applied to it. SO, for the joint distribution of stochastic variable $S_t$ from $t= t_1, .., t_n$, if we apply a shift of $u$, the new joint distribution $(S_{t_1+u}, S_{t_2+u}, ..., S_{t_n+u})$  should be independent of the shift u. 
- Thus,  the process does not change statistically as it gets older i.e it is Stationary. Palm calculus applied to any stationary process.

Now, If $X_t$ is jointly stationary with the simulation, the distribution of $X_t$ is, by definition, independent
of $t$ and this is called the time stationary distribution of $X$. If $X_t$ is ergodic i.e we can deduce its properties through taking a sufficiently long sample of the process, then we can assert that for a discrete set of times $t$, the expectation of $X_t$ being bounded by a function $\phi$  can be estimated by the time units for which the value of $\phi$ is valid, if these are of a sufficiently large number

$$\mathbb{E}(\phi(X_t)) = \frac{1}{T} \sum_{t=1}^T\phi(X_t)$$

In other words, the probability of$X_t$  being in a set of state $W$ is proportional to the fraction of time $X_t$ spends in the states of $W$. In other words, the time stationary distribution of $X_t$ can be estimated by a time average.

### Building up to Palm Formulas

Now, if we condition the expectation and probability on a point process, like reaching a waypoint, we can get palm intensity and probability. In general, a random sequence of time instants $T_n$ can be defined as the times at which the simulation $S_t$ reaches a certain subset of the state-space or does a transition from some state $s$ to some state $s'$, where both of these states belong to the same set (Like the set of waypoints). NOw, if we assume that the time at which we are observing our simulation is 0, then we can say that 

$$
T_0 \leq 0 < T_1
$$

Thus, $T_0$ is the last time a transition occurred before time 0, and $T_1$ the next transition
time starting from time 0. Now, we can define an intensity over this point process to be the probability of a transition to happen at a time point 

$$\lambda = \mathbb{P}(T_0 = 0)$$

In continuous-time, this would be the same as saying that the expectation over the number $N$ of transition in time-interval $[t, t+\tau]$ satisfied the above equation i.e 

$$\mathbb{E}(N(t, t+\tau)) = \lambda \tau$$

Thus, for our example of time points at which transitions occur, we could say that for a stationary process, the conditional expectation at time $t$ is independent of $t$ and is the same as the expectation at $t=0$, and so


$$\mathbb{E}^0(X_t) = \mathbb{E}(X_0|Transition \,)$$

This is called the palm expectation, and if our process is ergodic, we can apply the sam criteria to our palm distribution and say 

$$\mathbb{E}^0(X_0) = \frac{1}{N} \sum_{n=1}^N X_{T_n}$$

Thus, if $X_t$ is the speed and the transitions are departures from a waypoint, we are essentially saying 

$$\mathbb{P}^0(X_t \in W) \sim \mathbb{P}^0(X_0 \in W)  $$

and both of these probabilities are proportional to the fraction of transitions in which the speed of the node is in the set $W$.

### Major Palm Formulas

**Inversion Formula** : Gives the relation between the time-stationary expectations and palm expectations

$$
\begin{aligned}
&\mathbb{E}(X_t) = \mathbb{E}(X_0) = \lambda \mathbb{E}^0 (\sum_{s=1}^{T_1} X_s) = \lambda \mathbb{E}^0 (\sum_{s=0}^{T_1 - 1} X_s) \\
&\mathbb{E}(X_t) = \mathbb{E}(X_0) =  \lambda \mathbb{E}^0 (\int_0^{T_1} X_s ds)
\end{aligned}
$$

**Intensity Formula:**  The average number of selected transitions per time unit $\lambda$ satisfies

$$
\frac{1}{\lambda} = \mathbb{E}^0 (T_1 - T_0) = \mathbb{E}^0 (T_1)
$$


### Feller's Paradox

Assume at a bus stop there pass in average $\lambda$ buses per hour. If an inspector is measuring the time between the buses, then they would measure an estimate of  

$$
\mathbb{E}^0(T_1 - T_0) = \frac{1}{\lambda}
$$

Now, if someone arrives at time $t$ and measures the event as the difference between the time until the next bus and the time since the last bus, they would estimate $\mathbb{E}(X_0) = \mathbb{E}(T_1 - T_0)$ and the above two quantities can be related as: 

$$
\begin{aligned}
&\mathbb{E}(T_1 - T_0) =\lambda \mathbb{E}^0(\int_0^{T_1}X_t dt) \\
\implies &\mathbb{E}(T_1 - T_0) =\lambda \mathbb{E}^0(\int_0^{T_1}(+t - (-t))dt) \\ 
\implies &\mathbb{E}(T_1 - T_0) =\lambda \mathbb{E}^0(T^2) \\
\implies &\mathbb{E}(T_1 - T_0) =\frac{1}{\lambda} + \lambda var^0(T_1 - T_0) \\
\therefore \,\,\, &\mathbb{E}(T_1 - T_0) = \mathbb{E}^0(T_1 - T_0) + \lambda var^0(T_1 - T_0)
\end{aligned}
$$

This means that the estimate of the time that the person would have is  $\lambda var^0(T_1 - T_0)$ bigger than the inspector's estimate, even though both of them sample the same system. This is called the Feller's paradox. Intuitively, this occurs because a stationary observer (Joe) is more likely to fall in a large interval. Thus, Palm calculus gives us a way to relate distributions that have been sampled differently. So, in the case of RWP, this means that Palm calculus allows us to reconcile the issues we saw with the RWP models where we see decay in speed when sampled over the averages of particles in transitions, and the average speed when sampled over the averages

### Back to RWP

The $\frac{1}{\lambda}$ value that we get from the intensity formula can be related to the palm distribution of hte speed as follows:

$$
\frac{1}{\lambda} = \mathbb{E}^0(D1)\mathbb{E}^0(\frac{1}{V_0}) = \bar{\Delta}\int_0^{\infty} \frac{1}{v} f^0_V(v) dv
$$

Here, $\bar{\Delta}$ is the average distance between 2 points in teh Area selected and $f_V^0$ is the density function that represents the palm distribution of the speed. When we apply Palm distributions to RWP, we get:

1. **Selecting Speeds:** the time stationary distribution of the speed has a density proportional to $\frac{1}{v} f^0_V(v)$ . thus if we want our process to have a time distribution that is stationary, we need to choose this function very carefully. Some examples are: $f^0_V(v) = Cv1_{v_{min} < v < v_{max}}$, $f^0_V(v) = C\frac{v - v_{min}}{v_{max} - v_{min}}$, and $f^0_V(v) = C (v_{min} + v(v_{max} - v_{min}))$. The key is to get a function that when divided by $v$ would give a function that more-or-less is stable
2. **Selecting Mobile Positions:** Just like Feller's paradox, we do not have a closed-form solution for different mobile positions. Thus, we can't generate the directly from the distributions as in the case of velocity. However, we can apply heuristics to get around it. One heuristic is Choosing endpoints such that joint probability density proportional to the distance between them. So, if $Prev(t)$ is the previous waypoint before or at time $t$, and $Next(t)$ is the next waypoint after time $t$, then for the triplet $(Prev(t), M(t), Next(t))$ we need have the following conditions met: (1) $(Prev(t), Next(t))$ has a joint density over $A \times A$ given by $f_{Prev(t),Next(t)}(p,n) = K||p - n||$, and (2) The distribution $M(t)$ which we get from $Prev(t) = p, Next(t) = n$ should be uniform in the segment $[p,n]$

If we choose the initial speed and locations in to satisfy these conditions, then we can ensure a stationary distribution in time and palm spaces for RWP and eliminate the ergodic degreaing problem. This, would allow us to effectively compare the processes in different settings and thus, make the RWP model more useful.





<!-- %%% -->
# MobMod: Random Mobility
As with any analysis, the basics start from idealized scenarios. In terms of modeling, this would be random mobility. The historical viewpoint on this comes from Brownian Motion, which is the model of the movement of particles suspended in a liquid or gas caused by collisions with molecules of the surrounding medium. Of course, this is not the most realistic since vehicles don't just move randomly and hit each other, and so, our most basic models in random mobility come from a movement that is random to a certain extent but obeys tunable constraints. The two most basic models of random mobility are:

- **Random Walk:** For every new interval *t*, each node randomly and uniformly chooses its new direction $\theta(t)$ from an interval $[0, 2\pi]$. The new speed follows a Gaussian distribution in $[0, V_{max}]$. Therefore, during a time interval t, the node moves with the velocity vector $[v(t)\cos(θ(t)),v(t)\sin(θ(t))]$. If the node moves according to the above rules and reaches the boundary of a simulation field, the leaving node is bounced back to the simulation field with the angle of $\pi−\theta(t)$ or $\theta(t)$, respectively. This effect is called the border effect.
- **Random Waypoint:** We first choose a rectangular area of size $X_{max} \times Y_{max}$, and the total number of nodes N in this area. We then choose a random initial location $(x, y)$ for each node, where $x$ is uniformly distributed over $[X_{min}, X_{max}]$  and $y$ is uniformly distributed over $[Y_{min}, Y_{max}]$. Every node is then assigned a destination $(x', y')$ also uniformly distributed over the two-dimensional area, and a speed $v$, which is uniformly distributed over $[V_{min}, V_{max}]$, where $V_{max}$ is the user-assigned maximum allowed speed. A node will then start traveling toward the destination on a straight line at the chosen speed $v$. Upon reaching the destination $(x_0, y_0)$, the node stays there for some pause time, either constant or randomly chosen from a certain distribution. Upon expiration of the pause time, the next destination and speed are again chosen in the same way and the process repeats until the simulation ends.

<img width=300 height=200 src="static/MobMod/rwm-rwp.png">

#### Limitations of the Random Waypoint and Walk models
These models are not able to capture a lot of realistic scenarios, the major ones listed as follows:

1. **Temporal Dependency of Velocity:** In these models, the velocity of the mobile node is a memoryless random process since the values at each epoch are independent of the previous one. Thus, sudden mobility behaviors are possible in these models, including sharp turns, sudden acceleration, or sudden stop. However, in real situations, these values change smoothly.
2. **Spatial Dependency of Velocity:** In these models, each node moves independently of all the other nodes. However, in real scenarios, like battlefield communication or museum touring, these values may be correlated in different ways, which is not taken into account in these models
3. **Geographic Restriction of Movement:** In these models, the mobile nodes are allowed to move freely without any restrictions, but this might not be the case in real-life scenarios, like driving for instance, where the agents are contained in their movement by the roads, obstacles, etc.

## More Realistic Models
#### Manhattan Model
This model addresses the drawback of Geographic restriction on movement. The general idea is that initially, the nodes are placed randomly on the edges of the graph. Then for each node, a destination is randomly chosen and the node moves towards this destination through the shortest path along the edges. Upon arrival, the node pauses for T time and again chooses a new destination for the next movement. This procedure is repeated until the end of the simulation. In the Manhattan Model, these edges are in the form of a grid. Thus, this is just an extension of the Random Waypoint idea, but with added constraints on movement

<img width=300 height=200 src="static/MobMod/Mahattan-model.png">

#### Reference-point Group Mobility Model (RPGM)

This model addresses the drawback of spatial dependency of the velocity in the random models. Nodes are categorized into groups. Each group has a center, which is either a logical center or a group leader node. For the sake of simplicity, we assume that the center is the group leader. Thus, each group is composed of one leader and many members. The movement of the group leader determines the mobility behavior of the entire group. The movement of group leader at time t can be represented by motion vector V^t_{Group}. This motion vector can be either randomly chosen or carefully designed based on certain predefined paths. Each member of this group deviates from this general motion vector to some degree, for example, each mobile node could be randomly placed in the neighborhood of the leader. The velocity of each member can be expressed as V^t_{Group} + R_i , where R_i is the deviation of each member from the group leader's velocity.

<img width=300 height=200 src="static/MobMod/Ref-pt-model.png">

#### Gauss-Markov Model
This model **addresses the correlation of velocities**. In this model, the velocity of the mobile node is assumed to be correlated over time and modeled as a Gauss-Markov stochastic process:

$$R_t = \bar{\alpha}R(t-1) + (1-\bar{\alpha})R + \sqrt{1- \bar{\alpha}^2}\bar{X}_{t-1}$$

Here, $R(t)$  can be either speed or direction of movement, which is shown to be dependent on an autocorrelation function sampled from a Gaussian distribution $\bar{X} \thicksim \mathcal{N}(0, \sigma)$. The value of R at any time point is correlated to a certain extent with its value at time $t-1$ , and this is modeled through the parameter $\bar{\alpha} = e^{-\beta} \in [0,1]$. When $\bar{\alpha} = 0$, this equation decomposes to Brownian motion since the moton at any two points only depends on its current value and the Gaussian Noise $\bar{X}$,  while at $\bar{\alpha} = 1$ this represents a linear motion since the new variable perfectly depends on the previous variable, without any added noise. By tuning this $\bar{\alpha}$, this model is capable of duplicating different kinds of mobility behaviors lying on the spectrum of Linear and Brownian motion.

#### Smooth Motion Model
Another mobility model considering the temporal dependency of velocity over various time slots is the Smooth Random Mobility Model. Here, **instead of the sharp turns and accelerations as proposed in the Random Waypoint Model, these values are changed smoothly**. It is observed that mobile nodes in real life tend to move at certain preferred speeds, rather than at speeds purely uniformly distributed in the range. Therefore, in the Smooth Random Mobility model, the speed within the set of preferred speed values has a high probability, while a uniform distribution is assumed on the remaining part of the entire interval

$$V^{pref} \thicksim [V_1, V_2, ..., V_n] $$

The frequency of speed change is assumed to be a Poisson process. Upon an event of speed change, a new target speed is chosen according to the probability distribution function of speed above and the speed of the mobile node is changed incrementally from the current speed to the targeted new speed by acceleration or deceleration $a(t)$ . The probability distribution function of acceleration or deceleration a(t) is uniformly distributed among $[0, a_{max}]$ and $[a_{min}, 0]$, respectively. The new speed depends on the previous speed:

$$V_t = V_{t-1} + a(t)(V_t - V_{t-1})$$

A similar approach is followed for the direction update with angular accelerations.

## Problem of Unsteady Values
One of the most important factors that play a role in the analysis of the various parameters associated with these networks is the stability of values. In the evolution of network states, there is usually an initial phase where the process variables change over time, and this phase is called a **Transient Phase**. During this transient phase, analyzing these variables is not possible as any value that they predict for the system is not a good indicator. As this phase passes, these values start to stabilize and as they stop varying over time units on an average, a **Steady State** is reached. This is the point at which a feasible network analysis starts becoming possible. In case the network transitions into something else over time, another steady state is reached after an intermediate transient state. The problem comes when these transient states start lasting longer or start happening more frequently since then stable analysis of the system becomes increasingly difficult. Thus, one of the major areas of research had been in figuring out a way to effectively predict these steady-state values and use them in analyzing the models. The seminal work on this was done by J-Y Le Boudec at EPFL, where the team developed a method called **Palm Calculus** and used it to predict the steady-state distributions of all major random mobility models.

## Outlook on Random Mobility Models
1. Random Mobility Models are very powerful solutions to obtain the fast and controllable output. They are easily configurable and reproducible, and there are various models on different geometric data. Moreover, after the issues removed through palm calculus, they are able to offer decent steady-state characteristics
2. Random Models have to be well understood and analyzed for them to make sense. If not carefully modeled, as in the case of RWP with the initial selection of speed and mobility, they may never generate stable steady-state characteristics. Thus, we might not get expected mobility patterns, which is a bummer for routing protocols.
3. While Random models provide good statistics, they are not able to provide low scale mobility patterns like slow increase/decrease of speeds, the interaction between vehicles, etc.



<!-- %%% -->
# MobMod: Introduction to Mobility Modelling

The fundamental abstraction that is needed to understand inter-connected phenomena is a way to describe the different relationships between the various participating entities. For example, imagine a scenario where a disaster management Mobile Ad-hoc network of drones is deployed to triangulate sensitive points, the key to successful execution lies in how these drones interact and share knowledge. Thus, the **Network** that is shared between these drones - or, **Nodes** - would have a certain way of establishing communication and its performance can be analyzed through multiple metrics. One way to analyze this network would be by simulating how these nodes might move and what kind of impact they might have on the network. It is exactly for this kind of analysis and understanding that we create **Mobility Models**: They allow mimicking the behavior of mobile nodes when network performance is simulated. The simulation results are strongly correlated with the mobility model.

### Case of Connected Cars

The way this analysis fits into the driving scenario is in terms of modeling the uncertainty in the traffic scenario. In the case of autonomous driving, the limitation for an individual agent is more from the side of the sensors. Usually, we would use cameras and Radars to work stuff out, but Radars can only see around 10m. So, how would this agent work in a highly uncertain scenario? Imagine a car driving on an Indian road in a 2-tier city, with pedestrians walking on all sides and multiple vehicles going in all directions.

<img width=600 height=400 src="static/MobMod/Indian-road.png">

One way to approach this solution would be to shift a bit of the load from the individual agent to the network. This could be done if the cars are connected through a network based on 4G/5G technologies. Thus, in this case, we could view the cars as nodes communicating with each other and moving in some way. For each car, the regular functions - movement, mapping, obstacle detection, and avoidance, etc. - would be performed on the edge of this network while the coordinating function could be performed either centrally or in a distributed manner. Thus, the issue of modeling the Mobility and/or Modeling of the traffic flow, especially in the case of more uncertain scenarios, becomes extremely important.

#### Need for Modelling
1. The performance evaluation of such a large-scale network is bounded by simulation: Raw analytical analysis is too complex while conducting experiments is too expensive
2. Real-world traces are hard to find, either because they do not exist or are not publicly available (E.g City-Data).
3. Trivial representations of mobility might **bias** the simulation results. The available traces might not represent real-world situations very effectively and might have a lot of residual effects that can render them useful in very specific conditions, and thus, not generalizable.

#### The safe and sustainable mobility conundrum
Now if we were to go about this modeling, one of the central issues that would come up is that of optimality. The optimization here would be maximizing the usage of the road capacity while putting constraints in the form of driving rules and the safety of the driver and pedestrian. To better understand this, imagine we implement a model where a car is made to drive slow to increase safety. Now, this is fine for the individual driver, since the car always maintains a safe distance from another car in front of it. However, two problems come-up:

1. If the safe distance is too large, then a pedestrian might walk-in between leading to the car to halt
2. The speed of the car is slow and so, the general traffic speed is also slow. This might create problems on traffic bottlenecks, like a 7 lane highway leading to a 2 lane road, etc.

If we use the same policy on high-volume traffic conditions, then slow speeds and sudden halts can easily lead to shockwave propagation and Ghost Jams. Let us formalize this by looking at traffic in terms of flow

$$Flow = Density \times Speed$$

Thus, looking at this model we can see some safe directions to analyze this situation would be by controlling this flow through either reducing the speed or maybe keeping it and reducing the flow through density control. But more importantly, if we were to model the mobility by simulating this condition, we could develop interesting ways of cooperative navigation.

#### Vehicular Models
To analyze this network, we start looking at vehicles as nodes. Thus, the traffic becomes a MANET and our methods of simulation enable us to better understand this network.

<img width=1000 height=400 src="static/MobMod/car-node.png">

The impact of mobility is even more pronounced in the case of vehicular networks. The three factors that differentiate these networks from other networks are:

1. The speed of each node is not bounded in small intervals and is not smooth. It is highly variable
2. The movement is far from random. Thus, we cannot directly sample variables from standard distributions in realistic scenarios.
3. The nodes do not move independently, and in fact, have strong reciprocal dependencies

Consequently, the abstractions that this network viewpoint offers can be on three levels:

1. **Vehicular Traffic Model:** Abstraction of the large scale trajectories employed by the vehicles
2. **Vehicular Flow Model:** Abstractions of the physical inter-dependencies
3. **Vehicular Driver Model:** Abstractions of the actions of individual nodes, like breaks, turns, etc.

### History of Modeling 

- **Random Mobility Models:** These models come from the field of Computer Science and Telecommunications. The key idea is to think of Mobility as a perturbation in a system of particles and use stochastic analysis to create models. Thus, we would have models for how the particles move under constraints imposed by tunable hyperparameters, and we determine the stability and other related metrics of this system. A significant feature of these models is the need for the system to be at a steady-state, which is something that is not very realistic since if the system models something like traffic, it is very difficult to define what a steady-state would mean in this scenario. This will be discussed further below.
- **Flow Theory:** This model comes from the field of transport engineering and vehicle manufacturing where the view is to see mobility as flows between junctions i.e vehicles on a lane going from one street to another is a flow and thus, we can use metrics like speed and density to model this. one key distinguishing factor is that here we are interested in the exact opposite of Random Models - the transient phase - and we try to work with it. The issue with flow theory is the rules that it applies to the whole flow, which prevents it from being able to explain erratic behavior like accidents.
- **Behavioral Theory:** This comes from Traffic Telematics engineers, and the view here is to not have specific rules for the whole mass of movement as flow theory does but to look at mobility as a network of agents, with each agent behaving according to certain behavioral constraints. Thus, this view tries to bridge the gap between Flow theory and realism and is also able to explain stuff like accidents.


<!-- %%% -->
# MALIS: Gradients

## Gradient Descent

- **predicted value** $= intercept + slope$ 
- **metric** → $y = C + dy/dx x$

let $y'$ be the expected value:
- residual $= y' - y $ 
- sos = $(y' - y)^2$. 
- Thus, our optimization target becomes : $\frac{1}{2} \sum (y' - (C + Mx))^2$

How gradient Descent works is by taking steps towards the optimal target. This is different from least-squares since in the latter we numerically compute the optimal solution by differentiating the target w.r.t C and setting it to 0 to find the inflection, which will be the minimal point. Gradient descent, on the other hand, works by first selecting a random value of intercept, say  $C_1$ , and then moving a step in the direction of decrease in value. This step is determined by the learning rate $\alpha$ which is a hyperparameter. So, at $C_1$, we differentiate the SOS target w.r.t C and calculate the value by putting $C = C_1$ and then multiply this value by alpha to get the next intercept point → Thus, when we are far away from the inflection, we take larger steps and when we are closer to the inflection, we take smaller steps since the slope is saturating. evident in N-dimensional metrics → the same thing is happening on hyperplanes
- The learning rate $\alpha$ determines the size of the steps we take and tuning this is important, since if it is too small, our time to convergence is slower, while if it is too large, we overshoot the solution → Classic control phenomenon!
- One solution is to start with a large learning rate and make it smaller with each step! → **Schedule the Learning Rate**


## Stochastic Gradient Descent

The computations in the Gradient Descent step scale up pretty fast and thus, convergence becomes an issue. SGD resolves this by sampling points for the intercept residual calculation!  → Instead of using all points, we can randomly sample n points - **Mini-batch** - and use them → This is especially helpful when points are clustered in different clusters, since the points in one cluster wil more-or-less have similar residuals! 
- Again the sensitivity to $\alpha$ comes into picture and again we can adapt scheduling  to overcome this!



<!-- %%% -->
# MALIS: Generic ML Concepts

## Cross-Validation

Allows us to compare different ML methods! When working with a dataset for training a learner: 
- Naive Idea → Use all data 
- Better Idea →  Use x percent for training and y percent for testing

the issue is how do we know that the selection is good ? The lord answers through **K-Fold Cross Validation** → split data into K blocks → For each K-block train the data on the rest of the K-1 blocks and test it on K block and log the metric → Average out the performance and use this for comparison
- **Leave one out cv** → us each sample as a block 

## Confusion Matrix
Plot the Predicted Positives and Negatives vs, Ground truths : 

<img height=300 width=500 src="static/MALIS/Confusion.png">

NOTE: 
- The diagonal always shows the True values
- The non-diagonal elements ar  always false

The major Metrics are: 
- **Sensitivity →** Positive Labels correctly predicted : $\frac{TP}{TP + TN }$
- **Specificity →** Negative Labels Correctly predicted : $\frac{TN}{TN + FN}$

Let's say we test out logistic regression against Random Forests to classify patients with and without heart disease. Then the algorithm with the higher sensitivity should be chosen if our target is to classify patients with heart disease, while the algorithm with higher specificity should be chosen if we want to classify patients without heart disease

### What about Non-binary classification 
Calculate these values for each label by treating the values as Label and !label. For if we have three labels, we take the true positives as all the classifications done for label i and the TN as all the misclassifications done for label i → This means that if the data actually belonged to the other classes and was still classified as belonging to i, then it is a False Positive. Similarly, we take True Negative as all the classifications done on all other classes except out current label and the false negatives as the classifications. Let's take the following example:

<img height=300 width=500 src="static/MALIS/Confusion-2.png">

Here, for The class Cat, we get : 

- Sensitivity $= 5/(5 + 3 + 0) = 5/8 = 0.625$
- Specificity $= (3 + 2 + 1 + 11)/(3 + 2 + 1 + 11 + 2 ) = 17/19 = 0.894$

Other major metrics are: 
- **Accuracy** → $(TP+TN)/total = (100+50)/165 = 0.91$
- **Misclassification Rate** → $(FP+FN)/total = (10+5)/165 = 0.09$ = 1 - accuracy
- **Precision → $TP/predicted yes = 100/110 = 0.91$**
- **TP Rate** →  $TP/yes = 100/105 = 0.95$
- **FP Rate**  → $FP/No = 10/60 = 0.17$

The idea is to strike a balance between things and get hang of how our classifier is actually performing!

## Bias and Variance

- **Bias** → The inability for an algorithm to capture the true relationship in the data. Formally, it is the inherent error that we obtain from the model even with infinite training data due to the classifier being biased to a particular solution
- **Variance** → Difference in fits between the training and the testing data, i.e the error caused from sensitivity to fluctuations in the training set

High bias means the learned model is simpler and might not fit the training data very well and when it does not perform so well on the test set → **Underfitting**. High variance means that the learned model is has a better fit to the training set but does not perform so well on the test set → **Overfitting**

- What is happening is that the training set can be viewed essentially as a the true relationship curve plus some noise that scatters the data around the curve. This is the same for training and test sets. now, if our model fits so well to the training set that it is able to exactly pass through each data point, it has actually fitted to the the noise that scattered the data from the actual signal. Thus, is has so much variability that it won't perform good on other datasets which might inherently be sampled from the same curve with some random noise that scatters the data a bit differently. This is why the model has over-fitted to the training set by adapting to the noise.

In general, Error depends on the square of bias and the directly varies with variance and Noise:

$$
E= B^2 + V + N
$$

And this variation can be plotted as follows:

<img height=300 width=500 src="static/MALIS/BVT.png">

## ROC and AUC

The whole idea of ROC curve is adjusting our classification threshold for example in the case of logistic regression - to mess-around with the rates of TP and FP. We plot these values for each threshold against each other on a graph, as shown: 

<img height=200 width=350 src="static/MALIS/ROC.png">
- Here, we have plotted sensitivity a.k.a the True positive rate against FP Rate a.k.a 1 - specificity. At point (1,1) we see that our classifier is classifying all samples as TP and FP. Now, let's say our problem is to predict whether the patient has a certain disease or not, then this is not acceptable since the FP Rate is high, and we can't afford False Positive classification. So, we adjust our threshold and see the sensitivity remain the same through the next two points on the left, but the FP Rate decrease, which means our model is getting better. Then we see that both the rates fall and then, finally, our model reaches a level where the TP Rate is positive while the FP rate is negative. This is a desirable performance for our purposes. In case we are willing to accept some FPs for a better TP classification, we can select points on the right that increase the TP but also end up having some misclassifications

AUC -Area Under the Curve - is used to compare the performance of two classifiers, as shown below:

<img height=200 width=350 src="static/MALIS/AUC.png">

Since the AUC is greater for the red curve, and so the model that it represents is better since for the  same levels of FP, it delivers more TPs


<!-- %%% -->
# MALIS: Clustering

## Hierarchical Clustering
We take a group of points and cluster them according to similarity, or dissimilarity. Let's say we have N points, each parameterized by n dimensions. to measure the similarity, all we do is create a **meaning of distance between the points**,  which can be done in many  ways, two common ones being: 

1. **Euclidean Distance** → Take the difference of values across the n dimensions, square each one and add and then  take the square root 
2. **Manhattan Distance** → Sum the magnitudes of differences across n dimensions

So, we calculate the distances between all 2 combinations of samples and then see the one with the smallest distance → We combine them into a single cluster, for which we define the values across all dimensions based on the mean metrics of the point. These values are used for further selection, as we treat this cluster as a single point and repeat the procedure, this gives us a dendrogram diagram as shown below:

<img height=200 width=500 src="static/MALIS/cluster/hh.png">

The way to figure out or each point, to which cluster it belongs, we can use three metrics: 

1. Average values of all points in the cluster → **centroid linkage** 
2. Closest point fro each cluster  → **single linkage** 
3. Furthest point from each cluster → **Complete linkage** 

There are two kinds of problems that might arise here: 

1. **Chaining:** in the case of single linkage, we are taking the best similarity out of all the points in the clusters → What if the clusters are too far spread out and thus, overlap? 
2. **Crowding:** In the case of complete linkage, we take the worst-case scenario → What if the points in different clusters are closer together than the point in consideration i.e the clusters are too compact?

Average Linkage resolves this → centroids FTW

## K-means Clustering

Here, we try to fit our points into K clusters: 

1. Select the value of K 
2. Take K points out of data randomly
3. Calculate distances of all other points from these K points 
4. Classify each point into the cluster to which it has the smallest distance 
5. Repeat the process with the reference points now being the centroids of the points in the cluster 
6. Stop when the clusters do not change
7. Repeat all the steps fro 2-6 with different initial points 

Let's say we conduct N experiments, we now need to see which cluster is the best → We do this by selecting the one with the **least total variation across all clusters** → Variation is calculated just like in PCA i.e sum squares of distances of all points in the cluster from the center and then divide it by the number of points in the cluster.

<img height=200 width=500 src="static/MALIS/cluster/k-means.png">


<!-- %%% -->
# MALIS: PCA
- source: [StatQuest](https://www.youtube.com/watch?v=FgakZw6K1QQ&t=1081s)

The main idea is dimensionality reduction: We have data of say n dimensions, where each sample depends on all the n dimension. No we can't really visualize this data and we can't really work with storing all the data all the time since the curse of dimensionality messes us up. So, we start to analyze the importance of each dimension on each sample and try to project then according to that → This groups the samples with similar impacts from the n dimensions together and allows us to see them in clusters. Let's take the case of 2 gene, for multiple cells. We do PCA in the following steps:

<img height=300 width=500 src="static/MALIS/PCA/pca-1.png">

- On the two axes we take the mid-point of this data:  
    <img height=200 width=300 src="static/MALIS/PCA/pca-2.png">
- We now center the points around this mid-point
    <img height=200 width=300 src="static/MALIS/PCA/pca-3.png">
- We draw a random line that goes through the origin and then fit it through these points by projecting these points on to the line and then either minimizing the distances to this line or maximizing the distance of the projected points from the origin:
    <img height=200 width=300 src="static/MALIS/PCA/pca-4.png">
- The slope of this line to our original axes tells us the ratio of the importance of gene1 to gene2 i.e. if we were to make a call, we would need to mix 4 parts of gene1 and 1 part of  gene2 in this case
    <img height=200 width=300 src="static/MALIS/PCA/pca-5.png">
- The unit vector along this line → the **eigenvector** of this data → tells us these proportions through its span components : ( 0.97, 0.242 ) → **Loading scores**
- The square of the distances of the projected points on the eigen vector are the **eigenvalue of PC1: $\sum d_i^2 = EV_{PC_1}$**
    <img height=200 width=300 src="static/MALIS/PCA/pca-6.png">
- Similarly,  we can get another principal component to this data through the process which will be perpendicular to PC1 and it will also have its eigenvector and eigenvalue 
    <img height=200 width=300 src="static/MALIS/PCA/pca-7.png">
- Now we just remove the original axes and rotate the eigen vectors to see the points throgh the EVs, and the squared sum of the projected points on each PC gives us the original point
    <img height=200 width=300 src="static/MALIS/PCA/pca-8.png">
- We can convert the EVs into variations by dividing them by the sample size , and in this case, we V1 = 15 and V2 = 3 → Thus , PC1 contributes $\frac{15} {(15+3)} = 83%$ in importance, while  PC2 contributes $\frac{3} {(15+3)} = 17%$ in importance. These can be plotted on a **scree plot** , which tells us the importance of each PC 
    <img height=200 width=300 src="static/MALIS/PCA/pca-9.png">


In theory, for genes there should be n components → So, even if we can't visualize them we can just see the scree plots and analyze the data and the principal components decrease in order of importance. So, we can roughly take the two most important ones and use them for understanding


<!-- %%% -->
# MALIS: AdaBoost

Learners can be considered weak or strong as follows: 
- **Weak** → Error Rate is only slightly better than random
- **Strong** → Error Rate highly correlated with the actual classification

**Adaboost combines a lot of weak learners to create a strong classifier!**. This is characterized by 3 key points: 

1. It creates an RF of **stumps** → trees with only one question used for classification → which act as weak learners
2. All stumps in this forest don't have equal **say**, some have more and some have less and they are used as  weights for the classification that each stump makes
3. The errors made by the previous stumps are taken into account by the next stump to reduce misclassification i.e the stumps sequentially try to reduce misclassification in contrast to a vanilla RF where the stumps are all separate

The steps are a follows:
- Start with the dataset, but assign each data point a weight i.e create a new column with weights, which have to be normalized, and at the start, all have equal values
    <img height=300 width=300 src="static/MALIS/ADA/ada-1.png">
- Use a weighted impurity to classify nodes → We use the same formula, but for each label we use the associated weights in the gini calculation. Since all weights are the same, we ignore them for now and see that the gini for patient weight is the lowest, so we use it for our first stump:
- Now we see how many errors this stump made → in this case it is 1. We determine the say of this stump by summing the weights of the erroneously classified samples → $E = \sum W_i$ → and the total say as $S = \frac{1}{2} log(\frac{1 - E}{E})$ → we get the say as 0.47 for this stump
- Now we update the weight of the incorrectly classified sample using the formula: $w \leftarrow w * e^S$ and so, we get the new weight for the incorrect sample as 
- Now we will decrease the amount of weights for all the correctly classified samples by using the formula : $w \leftarrow w * e^{-S}$ which gives us the new weights of all other labels as 0.05
- Now we normalize the updated weight by dividing each weight by the sum all weights
    <img height=300 width=150 src="static/MALIS/ADA/ada-2.png">
- We repeat the procedure using a weighted gini index or just creating duplicates of the samples with the large weights.  

The main thing that characterizes this algorithm is Boosting, which is a fancy name for when we train multiple weak classifiers to create a stringer classifier by taking errors of the previous classifiers into account. AdaBoost is creating a forest of stumps, but each new stumps uses the normalized weights to determine which kind of mis-classifications to focus on and thus, in a sense, uses the errors of the previous stumps to improve. 



<!-- %%% -->
# MALIS: Random Forests
-source[StatQuest](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)

The issue with Decision Trees is that they are not flexible to achieve high accuracy. So, we use Random Forests, which alleviate this problem by creating multiple trees from different starting points. The steps are as follows: 

- Create a Bootstrapped Dataset of the same size by selecting random samples with replacement 
    <img height=250 width=300 src="static/MALIS/RF/rf-1.png">
    <img height=250 width=300 src="static/MALIS/RF/rf-2.png">
- Create a Decision tree by selecting questions at random from this bootstrapped dataset, and use the impurtiy function to segregate the metric to used
    <img height=250 width=300 src="static/MALIS/RF/rf-3.png">
- Wherever a question needs to be asked, select the new metric randomly out of the metrics except the one used i.e in this case Good blood Circulation
    <img height=250 width=300 src="static/MALIS/RF/rf-4.png">
- Go back to step 1 and repeat to create a new bootstrapped dataset and repeat everything to create another tree → Do this process a fixed number of times.

Thus, by creating a variety of trees → A forest → We are able to get trees with different performances that can predict the labels. For new data items, run it down all the trees and keep a track of the classification - Yes and No - and then choose the classification with the bigger number → **Bagging = Bootstrapping + Aggregating**. Thus, Bagging is an ensemble technique where we train multiple classifiers on subsets of our training data and then combine them to create a better classifier.

## Evaluating RF

the entries that didn't' end up in the bootstrap dataset - Out of bag Data - are run through the tree to get the classification from all the trees and again use Bagging to see what the final classification is → For all out of bag samples, we evaluate the confusion matrix and calculate the precision, accuracy, sensitivity, and specificity  

## Hyperparameters in RF
The hyperparameters in the RF are :

1. m → The number of variables we are using out of the subset in bootstrap to create the tree 
2. k → the number of trees we have in the forest 

We can do the Out-of-Bag stuff on different random forests and select the one with the best accuracy

<!-- %%% -->
# MALIS: Decision Trees
- Source: [StatQuest](https://www.youtube.com/watch?v=7VeUPuFGJHk)

Basically a way of codifying decisions as tree data structures. The tree has the same concepts as trees in basic data structures with: 

- Root Node→ starting node of the tree
- Intermediate nodes → All the nodes that come after the root node that need to be parsed in a depthwise search
- Leaf Nodes → The terminal nodes of the tree

<img height=200 width=300 src="static/MALIS/DT/dt-1.png">

The main idea is to start from the root and traverse along the intermediate nodes to reach decisions on observations and actions. 

## Creating a Tree

Based on the video from, Let us predict heart disease based on metrics :

- Chest Pain
- Good Blood Circulation
- Blocked Arteries

and we have the following table:

<img height=200 width=250 src="static/MALIS/DT/dt-2.png">

Let the observation nodes be represented as x/y, where x represents heart disease while y represents no heart disease. For chest pain, suppose we get the following observations from the labels:

<img height=150 width=220 src="static/MALIS/DT/dt-3.png">

For Good blood circulation:  

<img height=150 width=220 src="static/MALIS/DT/dt-4.png">

And for Blocked arteries:

<img height=150 width=220 src="static/MALIS/DT/dt-5.png">

Here, the total number of patients for each of these metrics is different, since some patients did not have observations for all metrics. Now, since none of the leaf nodes have 100% observation on whether the patient has heart disease, they are considered impure → We need a way to measure and compare this impurity to determine which metric is better and use that as a higher level node.

### Gini Impurity

The Gini impurity is mathematically written as :

$$
G(s) = 1 - \sum_{i=1}^Kp_i(1-p_i)
$$

Here, $p_i$  are the probabilities of the sub-observations → take one metric and divide it by total observations in that leaf node. In our example, for the case of chest pain, the gini for the leaf node corresponding to the observations that come when chest pain is detected, can be calculated as: 

$$
1 - (\frac{105}{105+39})^2 - (\frac{39}{105+39})^2 = 0.395
$$

Similarly, the gini of the other leaf node for chest pain is: 

$$
1 - (\frac{34}{34+125})^2 - (\frac{125}{34+125})^2 = 0.336
$$

Since the total number of heart patients in the leaf nodes for chest pain is not the same, we take a weighted average of Gini impurities as the Gini impurity for chest pain : 

$$
G_{cp} = (\frac{144}{144+159})0.395 + (\frac{159}{144+159})0.336 = 0.364
$$

Similary the coefficient blood circulation is →  $G_{GBC} = 0.360$ and the coefficient for blocked arteries is → $G_{BA} = 0.381$

Since good blood circulation has the lowest Gini value, it has the lowest impurity → it separates the heart disease the best! Thus, we will use is as the root node, and so our tree is:

<img height=150 width=220 src="static/MALIS/DT/dt-6.png">

So, in our decision we start with looking at good blood circulation. If the patient has good blood circulation then there are 37 such patients wth heart disease and 127 without, and if they don't have good blood circulation, then 100 such patients with heart disease and 33 without. Now, in the left node we again compute the sub-trees for Chest pain and blocked arteries, out of these 37 patients **from the table** to get the following possible branches:

<img height=150 width=400 src="static/MALIS/DT/dt-7.png">

The Gini values are:

$$
\begin{aligned}
&G_{CP} = 0.3 \\
&G_{BA} = 0.290
\end{aligned}
$$

Thus, blocked arteries is a better metric after Good blood circulation, and so we update it in the tree:

<img height=220 width=220 src="static/MALIS/DT/dt-8.png">

Now, we repeat this procedure for the left and right nodes of blocked arteries. For the left child of blocked arteries to get the chest pain values as:

<img height=150 width=200 src="static/MALIS/DT/dt-9.png">

This will be added to the left child of blocked arteries. Ideally, we would repeat this procedure for the right child, but there is one important factor that comes into play here, which is that the Gini impurity of the right child of Blocked arteries is: 

$$1- (\frac{13}{13 + 102}) + (\frac{102}{13 + 102}) = 0.2$$

While the Gini for the chest pain in this case is:

<img height=150 width=200 src="static/MALIS/DT/dt-10.png">

$$
G_{CP} = 0.29
$$

Thus, the right child of blocked arteries is by itself a better separator than chest pain, and so, we let it be! Hence, we can summarize the steps followed as: 

1. Calculate the Gini scores 
2. If the Gini of the node is lower, then there is no point separating patients and it becomes a leaf node.
3. If separating the data improves the gini, then separate it with the separation of the lowest gini value

We repeat these steps for the right child of the root node, and the final tree comes out to be:

<img height=400 width=600 src="static/MALIS/DT/dt-11.png">

## Other Impurities

Gini is just one of the ways to measure the impurities. It is used in the CART algorithm. Another measure for quantifying impurity is Entropy: 

$$
H(S) = - \sum_i p_i log(p_i)
$$

It is primarily used to define information, which is defined in terms of the change in entropy. It used in the ID3 algorithm, which does similar stuff as described above. The basic idea of working with trees remains the same → Use an impurity function to determine if the node needs further improvement, and then improve it by asking the question that would lead to the best separation.

## Feature Selection

If we are to follow the procedure described previously for trees, then the issue that comes is that of over-fitting and to deal with this, we can do feature selection to simplify the tree. For example, if chest pain in the previous example never gave an improved separation as compared to the leaf nodes, then we can just remove this feature and thus, the tree would only have good blood circulation adn blocked arteries. Similarly, we could also specify a threshold for separation saying if te gini is low than this threshold, then we consider the separation good and thus, if any feature is unable to separate below this threshold, we discard it! 






<!-- %%% -->
# MALIS: Kernels

The basic idea of Kernels is the problem of separating randomly distributed data, as shown below:

<img height=200 width=300 src="static/MALIS/Kernels.png">

The separating hyperplane is not exactly a line in this 2D space. So, how can we learn this separation? One trick is to Transform this data by mapping each data point into a new space for computation, learn the separator in that space, and then transform this data back to our original space. This is a similar technique to what we do in the time-frequency transformations of signals using Fourier transforms, to better understand some characteristics in time and frequency domains. The naive way to do this transformation is through a kernel is to the variable x from our initial space to a variable $\phi(x)$ in the new space. Thus, if this problem is transformed into a linear classification problem in this new space, then we are essentially applying a linear classifier to a non-linear classification problem. The only issue with this is the issue of computation that comes with higher dimensional problems! To alleviate this, we use Kernels

## Kernel Trick

Let's say we have the base features $x_i$ and we apply a transformation on them $\phi(x_i)$ which  transforms each point according to some set rule. Now, if we were to apply any technique, say linear regression, on the original points, our problem would be to calculate the weights: 

$$
w = (X^TX)^{-1} X^TY
$$

And so, with the transformations, it becomes: 

$$
w = (\phi^T\phi)^{-1} \phi^TY
$$

If we add regularization to it, we just add the ridge regression term to it:

$$
w = (\phi^T\phi + \lambda I)^{-1} \phi^TY
$$

The problem is, calculating $\phi(x)$ is hard! And when we look at high volume data and the central role that these transformations might play in the case of SVMs, calculating this transformation for every point just adds complexity. Let's try to simplify this a bit, by looking at our ridge-regressor:

$$
\begin{aligned}
&J(w) = \sum_{i=1}^N(y_n - w^T\phi(x_n))^2 + \frac{\lambda}{2} \sum_{i=1}^N ||w||^2 \\
\implies &\bm{w}^* = \frac{1}{\lambda} \sum_{i=1}^N (y_n - \bm{w}^T\phi(\bm{x}_n))\phi(\bm{x}_n)
\end{aligned}
$$

Let first term be:                    
$$
\alpha_n = \frac{1}{\lambda} \sum_{i=1}^N (y_n - \bm{w}^T\phi(\bm{x}_n))
$$

Thus, we can re-write the weights as:

$$
\bm{w}^* = \bm{\phi}^T\bm{\alpha}
$$

Thus, if we substitute this value in the expression for $J(\bm{w})$ , we get a dual form that depends on the $\bm{\alpha}$  and $\bm{\phi}^T \bm{\phi}$ , very similar to what we get in the dual form of SVMs. This dot product transformation can be written as a **Kernel Matrix**:

$$
\bm{\phi} \bm{\phi}^T = K = [\phi(x_i)\phi(x_j)] \,\,\,\,\,\, \forall \,\,\,\,\,\, i,j = 1, ..., N 
$$

This matrix has 2 properties: 

1. It is Symmetric → $K = K^T$
2. It is Positive Semi-Definite → When we multiply it by some other matrix and its transpose, we get a result greater tha or equal to 0 i.e  $\alpha^T K \alpha \geq 0$

When we put this in our simplification term, we get the following result:

$$
\bm{\alpha}^* = (\bm{K} + \lambda ' \bm{I})^{-1} Y
$$

The difference between the original problem to this problem is simple → Before we had to compute $\bm{\phi}^T \bm{\phi}$  but now we have to compute $K = \bm{\phi} \bm{\phi}^T$ which seems similar but there is a catch: 

According to Mercer's Theorem, any Symmetric and PSD Matrix can be expressed as an inner product 

$$
K(x,y) = \langle \phi(x), \phi(y) \rangle
$$

In other words, we can write K as an inner product of the original features:

$$
\bm{\phi} \bm{\phi}^T = K = [\phi(x_i)\phi(x_j)] = [K(x_i,x_j)]
$$

This is the Kernel Trick!  To better elaborate, let's take a transformation: 

$$
\phi: x \rightarrow \phi(x) = [x_1^2 \,\,\,\, \sqrt(2)x_1x_2 \,\,\,\, x_2^2]^T
$$

A dot product of this transformation can be re-written as

$$
\begin{aligned}
&\phi(x_m)^T\phi(x_n) = x_{m1}^2x_{n1}^2 + 2x_{m1}x_{n1}x_{m2}x_{n2} + x_{m1}^2x_{n1}^2 \\
\implies &\phi(x_m)^T\phi(x_n) = (x_{m1}x_{n1} + x_{m2}x_{m2})^2 = (\bm{x^T_m}\bm{x_n})^2
\end{aligned}
$$

Thus, all we need is this final form of the kernel to work since the whole optimization relies on this dot-product and we don't really need to transform every feature through the original expression! For predictions, originally we needed to calculate the weights since :

$$
y = \bm{w}^T \phi(x)
$$

But, as we have shown these weights depend on the dot-product which can be expressed as a Kernel

$$
\bm{w}^T \phi(x) = y(\bm{K} + \lambda ' \bm{I})^{-1} k_x
$$

And to compute this Kernel, we don't need to know the true nature of $\phi(x)$ !  This is what allows the SVMs to work in higher dimensions.


<!-- %%% -->
# MALIS: State-Vector Machines

## Intuition
Classification is essentially finding a boundary that separates the data points into two segments so that when we get a new data point, the segment that it falls on determines the label it belongs to! The real challenge is positioning this boundary → which is the focus of every classifier.  Let's take the figure shown below as an example

<img height=50 width=400 src="static/MALIS/SVM/svm-1.png">

Here, the data is 1D - Mass(g) - and we need to classify the person as Obese (Green) or Not Obese (Red). If we put the decision boundary at the edge of the red points, as shown below, our classifier would work for points inside the respective clusters, but for a point at the boundary, as shown, it would classify it as not obese even though this point is closer to the red cluster. 

<img height=70 width=400 src="static/MALIS/SVM/svm-2.png">

This is clearly not desirable, and this would also be the case if we move this boundary very close to the green points. Let's define the shortest distance between the observations and the boundary as the Margin. One way to get a decent classifier would be to place this boundary in such a way so that the margins are equal → This is possible in the case where we put the boundary at exactly half the shortest distance between the data-sets, as shown below 

<img height=70 width=400 src="static/MALIS/SVM/svm-3.png">

This is called the Maximum Margin Classifier since the margins, in this case, have the largest value than any other possible case. But the shortfall of this is that it is sensitive to outliers:

<img height=70 width=400 src="static/MALIS/SVM/svm-4.png">

Here, the outliers push the Max Margins very close to the green cluster and thus, overfit the boundary to the training set as clearly a more equally distributed test set would have to make some misclassifications! Thus, to make a boundary that is more robust to these outliers, we must allow misclassifications by introducing soft margins around our boundary that signify regions in which misclassifications are allowed:

<img height=120 width=400 src="static/MALIS/SVM/svm-5.png">

Thus, any red data point within this soft margin that falls in the green zone would be classified as green while the data in the red zone would be classified as red. This soft margin, in contrast to the margin above, need not be the shortest distance from the data to the boundary but can be tuned through cross-validation. This classifier is called a **Support Vector Classifier →** The observations within the soft margins are called support vectors which  can be tuned. In 2D, the boundary  would be a line, in 3D a plane → In general, it is called a hyperplane. 

### Non-linearly seperable data
Now, let us look at new case of drug dosage classification, where the dosage is only safe within a certain range of values and unsafe outside it:

<img height=70 width=400 src="static/MALIS/SVM/svm-6.png">

Here, the data is not linearly separable since no single boundary can separate it. Thus, support vector classifiers fail here! One way to tackle this is by transforming this data into a higher dimensional plane. If we square each data point and look at the 2D data, then we see that it is now linearly separable!  

<img height=150 width=400 src="static/MALIS/SVM/svm-7.png">

Now, we can create a boundary in this new higher dimensional plane and just map that back to our original plane. This is called a **Support Vector Machine.** The transformation we performed on the data is called a **Kernel,**  and in this case, the Kernel is a $d^n$ kernel with n having the value of 2 → Polynomial kernel.Technically, the support vector classifier is also an SVM in is dimension.


## Math of SVM

### Hard Margin SVM

For a set of training data $\{x_i, y_i\}$, with $x \in \R^M$  and $y \in \{-1,+1\}$ with the classifications being -1 and + 1 and $i = 1, ...., N$  , let us apply a transformation first

$$
\phi: \R^M \rightarrow \R^D \,\,\, s.t. \,\,\, \phi(x) \in \R^D
$$

Our objective is to fit a line through this data, and so our model is 

$$ 
\hat{y}(\bm{x}) = \bm{w}^T \phi(\bm{x}) + w_0
$$

which implies that classifications are:

- $y_i(\bm{w}^T\phi(x_i) + w_0 ) > 0$ if the classification  is correct
- $y_i(\bm{w}^T\phi(x_i) + w_0 ) < 0$ if the classification is incorrect.

The distance of any training sample from the seperation line line is → $d(\phi(x_i), L) = \frac{y_i}{||\bm{w}||}(\bm{w}^T\phi(x_i) + w_0 )$ 

Now, our optimization problem is to **maximize the minimum distance for each class,** which can be formalized as  

$$
\argmax_{\bm{w}, w_0} M = \argmax_{\bm{w}, w_0} \{\min_i d(\phi(\bm{x_i}),L) = \argmax_{\bm{w}, w_0} \frac{1}{||\bm{w}||} \{ \min_i y_i (\bm{w}^T\phi(\bm{x_i}) + w_0 ) \}
$$

Now, we add some constraints to our Margin → Set $M = 1/||\bm{w}||$ , which essentially means that the minimum distance for our problem has been set to 1 so that we only have to worry about $\frac{1}{||w||}$. In other words, we need to have: 

$$
|\bm{w}^T \phi(\bm{x_i})* + w_0| = 1 
$$

This is only possible if data points satisfy: 

$$
y_i(\bm{w}^T \phi(\bm{x_i}) + w_0 ) \geq 1
$$

If we look at the problem of maximizing $1/||w||$, it is the same as minimizing $||w||$  → Let's add two transformations to it :

1. $||w|| \rightarrow ||w||^2$
2. $||w||^2 \rightarrow \frac{1}{2}||w||^2$

We  see our minimization of $||w||$  is still satisfied since minimizing half of its square will still give us the same result and so minimizing $\frac{1}{2}||w||^2$  should be the same as maximizing $||w||^{-1}$, which was out the original problem. Thus, we now have to find:

$$
\min_{\bm{w},w_0} \frac{1}{2}||\bm{w}||^2  \,\,\, s.t. \,\,\, y_i(\bm{w}^T\phi(\bm{x_i}) + w_0 ) \geq 1 \,\,\,\, \forall \,\,\,\, i = 1, ...., N
$$

This is called the **primal form** → the form we reached using our intuition. We had a maximization type optimization and we converted it to a minimization problem without messing with our original goal. We now have to solve a constrained minimization problem. We solve this by creating a Langrangian Formulation →  a function of the form: 

$$
L(x,y,\lambda) = f(x,y) - \sum \lambda_i g_i(x,y) 
$$

where f is our optimization target and we put constraints on it in the form of $g_i$  which are $\lambda_i$ is the Lagrange multipliers. Our approach to solving this is to differentiate the Lagrangian w.r.t the variables to get critical points

$$\nabla L(x,y,\lambda) = 0 \longrightarrow \nabla f(x,y) = \lambda \nabla g(x,y) $$

and substitute it back into $L$ to get a **Dual Formulation**. So, formalizing our SVM minimization as a Lagrangian with $\alpha$ as our multiplier, we get: 

$$
L(\bm{w}, w_0, \bm{\alpha}) = \frac{1}{2}||\bm{w}||^2  - \sum_{i=1}^{N}\alpha_i [y_i(\bm{w}^T \phi(\bm{x_i}) + w_0 ) - 1]
$$

To minimize this  function, we first differentiate L w.r.t $\bm{w}$ : 

$$
\begin{aligned}
&\partial L/\partial \bm{w} = \bm{w} - \sum\alpha_iy_i \phi(\bm{x_i}) = 0 \\
\implies &\bm{w} = \sum\alpha_iy_i \phi(\bm{x_i}) \\
\end{aligned}
$$

Then, we differentiate L w.r.t $w_0$ as follows:

$$
\begin{aligned}
& \partial L/\partial w_0 = - \sum\alpha_iy_i = 0\\
\implies &\sum \alpha_iy_i = 0 \\
\end{aligned}
$$

Then we differentiate w.r.t w :

$$
\begin{aligned}
& \partial L/\partial \bm{w} = \bm{w} - \sum\alpha_iy_i\phi(x_i) = 0 \\
\implies & \bm{w} = \sum \alpha_iy_i\phi(x_i) \\
\end{aligned}
$$

We now plug the values into our lagrangian: 

$$ 
\begin{aligned}
&\frac{1}{2}||\bm{w}||^2  - \sum_{i=1}^{N}\alpha_i [y_i(\bm{w}^T \phi(\bm{x_i}) + w_0 ) - 1] \\
 &= \frac{1}{2}(\sum\alpha_iy_i \phi(\bm{x_i})  \sum\alpha_jy_j\phi(\bm{x_j})) - (\sum\alpha_iy_i\phi(\bm{x_i}) \sum\alpha_jy_j \phi(\bm{x_j})) - (\sum \alpha_iy_iw_0) + \sum \alpha_i \\
\therefore \,\,\, & L(\bm{\alpha}) = \sum \alpha_i - \frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j  [\phi(\bm{x_i}) . \phi(\bm{x_j})] \\
\end{aligned}
$$

The new Expression → $L(\bm{\alpha}) = \sum \alpha_i - \frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j  [\phi(\bm{x_i}) . \phi(\bm{x_j})]$ is the **Dual Form** of the Maximum Margin Problem. An important thing to notice is that our dual form only depends on the dot products of our data points and thus, we can take advantage of Kernelization to make this independent of the nature of $\phi$ :

$$L(\bm{\alpha}) = \sum \alpha_i - \frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j K(\bm{x_i}.\bm{x_j})$$

And for classifying any new point, all we have to do it replace it in the place of $x_j$ as shown: 

$$\hat{y}_{test} = \sum_{i=1}^{N} \hat{\alpha_i} [y_i(\phi(\bm{x_i}).\phi(\bm{x_{test}}) + w_0 )]$$

and this should satisfy the Karush-Kuhn-tucker conditions: 

1. $\alpha_i \geq 0$ 
2. $y_i(\bm{w}^T\phi(\bm{x_i}) + w_0 ) - 1 > 0$ 
3. $\alpha_i [y_i(\bm{w}^T\phi(\bm{x_i}) + w_0 ) - 1] = 0$

This is the key to SVMs → If $\alpha = 0$ then the point is not contributing to the margin and is not a support vector and for all $\alpha > 0$ , the point $\bm{x}$  is a support vector and contributes to the margin.


<img height=400 width=450 src="static/MALIS/SVM/svm-8.png">

Thus, in summary, the dual expression gives us a set of $\alpha_i$ that represent points that are either support vectors or not. If we want to find the best boundary for these hard margins, all we have to do is compute   

$$
\bm{\hat{w}} = \sum \hat{\alpha_i} y_i \phi(\bm{x_i})
$$

Which we then substitute into the third KKT condition, t get $w_0$ as follows: 

$$
\begin{aligned}
&\alpha_i [y_i(\bm{\hat{w}}^T\phi(\bm{x_i}) + w_0 ) - 1] = 0 \\
\implies &y_i(\bm{\hat{w}}^T\phi(\bm{x_i}) + w_0 ) = 1 \\
\end{aligned}
$$

Now, the y labels are either +1 or -1, and so, $w_0 = 1 - \bm{\hat{w}}^T\phi(\bm{x_i})$  or $w_0 = -1 - \bm{\hat{w}}^T\phi(\bm{x_i})$ , which we can also write as:   

$$
w_0 = y_i - \bm{\hat{w}}^T\phi(\bm{x_i}) 
$$

In practice, we compute $w_0$  by summing multiple such expressions and dividing by $\alpha_i$. A new point will be classified to 

- Class 1 if $\bm{\hat{w}}^T\phi(\bm{x_i}) + \hat{w}_0 > 0$
- Class 2 if $\bm{\hat{w}}^T\phi(\bm{x_i}) + \hat{w}_0 < 0$

### Soft-Margin SVM

The hard-margin constraint is very strong but does not work very well for overlapping classes or spread out data. Thus, we relax the constraints by allowing points to violate the margins → this is quantified by the slack variable $\xi_i \geq 0$ for each point i, which measures the extent of violation.

<img height=400 width=450 src="static/MALIS/SVM/svm-9.png">

Thus, for point outside the margins $\xi_i = 0$ and for points inside the margin $\xi_i = |y_i - \hat(y(\bm{x_i})|$, and now, our constraint becomes 

$$
y_i(\bm{w}^T\bm{x_i} + w_0 ) \geq 1 - \xi_i
$$

And so, our problem becomes:

$$
\min_{\bm{w},w_0, \bm{\xi}} C \sum \xi_i + \frac{1}{2} ||w||^2 \,\,\,\, s.t \,\,\,\, y_i(\bm{w}^T\phi(\bm{x_i}) + w_0 ) \geq 1 - \xi_i \,\,\,\, \xi_i \geq 0 \,\,\, \forall n
$$

We again formulate the Lagrangian:

$$
L(\bm{w}, w_0, \bm{\xi}, \bm{\alpha}, \bm{\lambda}) =  = \{\frac{1}{2}||\bm{w}||^2 + C\sum_i \xi_i \}- \sum_{i=1}^{N}\alpha_i [1 - \xi_i - y_i(\bm{w}^T \phi(\bm{x_i}) + w_0 )] + \sum_i \lambda_i(-\xi_i)
$$

Which, when differentiated gives the follownig expresions:

$$
\begin{aligned}
&\bm{w} = \sum\alpha_iy_i \phi(\bm{x_i})\\
&\sum \alpha_iy_i = 0 \\
&C - \alpha_i - \lambda_i = 0
\end{aligned}
$$

The interesting thing is that the third expression helps us eliminate $\xi_i$ from the Dual form to get the expression:

$$\sum \alpha_i - \frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j \phi(\bm(x_i)).\phi(\bm(x_j))$$

Which is the same as before. Thus, we can see that the only impact $\xi$ has is in shifting translating the constraint to include misclassifications. Thus, soft-margins allow us to perform the same optimization with an added impact on the classification constraint.

<!-- %%% -->
# MALIS: Naive Bayes Classifier

source: StatQuest

## Multinomial Naive Bayes

Let's take the case of spam classification → we have emails with the combination of words tha  we want to use to classify future emails as spam or not. Let's say they have the combination of the four words : **Dear, Friend, Lunch, Money**. Now, we just count the frequency of these 4 four words in the normal emails and then assign **Likelihoods** to each as follows. Say D = 8, F = 5, L =3, M=1:

$$
\begin{aligned}
&P(D|N) = 8/17 = 0.47 \\ 
&P(F|N) = 5/17 = 0.29 \\
&P(L|N) = 3/17 = 0.18 \\
&P(M|N) = 1/17 = 0.06 \\
\end{aligned}
$$

Now , we do the  same for 7  spam emails:
$$
\begin{aligned}
&P(D|S) = 2/6 = 0.29 \\
&P(F|S) = 1/7 = 0.17 \\
&P(L|S) = 0/7 = 0 \\
&P(M|S) = 4/7 = 0.57 \\
\end{aligned}
$$

Then, we define the ratios for the Normal (N) and Spam (S) : 

$$
\begin{aligned}
&P(N) = 0.67 \\ 
&P(S) = 0.33 \\
\end{aligned}
$$

Thus, now every word combination we get, we just multiply the priors with the likelihoods and compare. For example, If we get **Dear Friend :**

$$
\begin{aligned}
&P(N) * P(D|N) * P(F|N) = 0.09 \\
&P(S) * P(D|S) * P(F|S) = 0.01 \\
\end{aligned}
$$

The key realization is that the product of the priors and likelihood, according to the Bayes theorem,  should be proportional to the Likelihood of email being normal given the letters seen i.e P(N) and the same for spam. Thus, directly comparing the two values above tells us that the email has more chance of being normal → We classify it as normal. But, what if the a word not previously encountered in spam is seen in the email? → Take the example of **Lunch Money Money Money Money  :** 

$$
\begin{aligned}
&P(N) * P(L|N) * P(M|N) ^4 = 2e-5 \\
&P(S) * P(L|S) * P(M|S) ^ 4 = 0  \\
\end{aligned}
$$

This is a problem since it limits our ability to classify → We alleviate this by introducing **placeholder observations** into the spam group. These observations are $\alpha$ in number and can be included in the counting process for frequentist likelihoods to eradicate this problem → so, for the value of 1 additional observation, we get 

$$
\begin{aligned}
&P(D|N) = 9/(17 + 4) = 0.43 \\ 
&P(F|N) = 6/(17 + 4) = 0.29 \\
&P(L|N) = 4/(17 + 4) = 0.19 \\
&P(M|N) = 2/(17 + 4) = 0.10 \\ 
&\\
&P(D|S) = 3/(7+ 4)  = 0.27 \\
&P(F|S) = 2/(7+ 4) = 0.18 \\
&P(L|S) = 1/(7+ 4) = 0.09 \\
&P(M|S) = 5/(7+ 4) = 0.45 \\
\end{aligned}
$$

Using hte  same prior values,  we get :

$$
\begin{aligned}
&P(N) * P(L|N) * P(M|N) ^4 = 1e-5 \\
&P(S) * P(L|S) * P(M|S) ^ 4 = 1.22e-5  
\end{aligned}
$$

Now, we see that the email is more likely to be spam!

### why Naive ?  
Naive bayes treats language as just bag of words and so the score for **Dear Friend** would be the same as **Friend Dear →** In the general sense, for any such problem, the Naiive bayes does not exploit inter-dependencies in value, as seen from the probability segregation into disjoint sets while calculation

## Gaussian Variant

We do the sam process, but this time we create gaussian distributions for the variables to represent likelihoods and thus, we take the points on these gaussian curves as the likelihood values that need to be multiplied by the prior to generate the score

- To manage underflow, we take the log of all probabilities and add them to calculate the score
- The score with the higher log value is the class into which the new observation should be classified

One way to study the impact of different variables is to weight the different classes → weighted log-loss. Cross validation helps in determining which class has more impact


<!-- %%% -->
# MALIS: Logistic Regression




<!-- %%% -->
# MALIS: Maximum Likelihood Estimation

**Main Idea:** 

1. Make an explicit assumption about what distribution the data was modeled from 
2. Set the parameters of this distribution so that the data we observe is most likely i.e **maximize the likelihood of our data**

For  a simple example of a coin toss, we can see this as maximizing the probability of observing heads from a binomial distribution: 

$$p(z_1, ..., z_n) = p(z_1 ...., z_n|\theta)$$

we assume I.I.D condition and so we should be able to break this down into :

$$p(z_1|\theta)p(z_2|\theta)....p(z_n|\theta) $$

Formally, let us deinfe a likelihood function as:

$$L(\theta) = \prod_{i=1}^N p(z_i|\theta)$$

Now, our task it to find the $\hat{\theta}$ that maximizes this likelihood:

$$\hat{\theta}_{MLE} = \argmax_{\theta}\prod_{i=1}^N p(z_i|\theta)$$

instead of maximizing a product, we can also view this problem as minimizing a sum if we take the log of all values:

$$\hat{\theta}_{MLE} = \argmax_{\theta}\sum_{i=1}^N Log(p(z_i|\theta))$$

Let us use this idea for the regression problem. We assume that our outputs are distributed in a Gaussian manner around the line w  have to find out. This basically means that our $\epsilon$  is a Gaussian Noise that is messing up our outputs from the fundamental distribution

<img height=500 width=800 src="static/MALIS/MLE.png">

Thus, our equation for getting this probability of our output would be :

$$p(y_i|x_i; w_i, \sigma^2) = \frac{1} {\sigma \sqrt {2\pi } } exp\{ \frac{ - (y_i - x_iw)^2 }{2 \sigma^2} \}$$

Our task is to estimate $\hat{w}$ such that the likelihood of $p$ is maximized. This, in other words, means we need to find the value of $w$ that maximizes the above expression:

$$\begin{aligned}
& \hat{w} = \argmax_{w}\prod_{i=1}^N p(y_i|x_i; w_i, \sigma^2) \\
\implies & \hat{w} = \argmax_{w}\prod_{i=1}^N \frac{1} {\sigma \sqrt {2\pi } } exp\{ \frac{ - (y_i - x_iw)^2 }{2 \sigma^2} \}
\end{aligned}$$

we can again do the log trick to make this a sum maximization :

$$\begin{aligned}
&\hat{w} = \argmax_{w} \sum_{i=1}^N Log(\frac{1} {\sigma \sqrt {2\pi } } exp\{ \frac{ - (y_i - x_iw)^2 }{2 \sigma^2} \}) \\
\implies &\hat{w} = \argmax_{w} \{ \sum_{i=1}^N Log(\frac{1}{\sigma \sqrt {2\pi}}) + \sum_{i=1}^N  Log(exp\{ \frac{ - (y_i - x_iw)^2 }{2 \sigma^2} \}) \} \\
\implies &\hat{w} = \argmax_w -\sum_{i=1}^N \frac{(y_i - x_iw)^2}{2 \sigma^2}
\end{aligned}$$

This is basically the same as the normal expression we had, the only difference being the normalizing factor $\sigma$. If we change the negative maximization to minimization:

$$\hat{w} = \frac{1}{2 \sigma^2} \argmin_w \sum_{i=1}^N (y_i - x_iw)^2$$

Which is the same as minimizing :

$$\hat{w} = \argmin_w \sum_{i=1}^N (y_i - x_iw)^2$$

and the solution is: 

$$\bm{w} = (\bm{X}^T\bm{X})^{-1} \bm{X}^T\bm{y}$$

Surprise, Surprise!!


<!-- %%% -->
# MALIS: Linear Regression

**Main ideas :**

- Use Least squares to fit a line to Data
- Use R square
- Use p value

**Fitting the Line →** Try to minimize a metric that represents the fit

- Let the Line be $y(x) = w_0 + w_1x$
- Now, our optimization goal is to find the values of $w_0, w_1$ so that the variation around this line is minimal → We do this by minimizing the squared Errors

To know if taking into the samples actually improves anything or not, all we have to do is calculate the variance around the fit and compare it with variance around the mean of the y values of the point, and give an answer in percentages! This is called the $R^2$ value: 

$$
R^2  = \frac {Var(mean) - Var(fit)}{Var(mean)}
$$

Thus, if this value is 0.6 , we get a 60% improvement in the variance by taking the x features into account. 

Let's go to the interesting stuff → The Math of this all 

## Math of Regression 

Let's take the  case of a set of multidimensional features $\bm{X} \in \R^D \,\,\,$where, $i= 1,...,N$. For each of these set of D dimensional inputs, we have one output $\bm{y} \in \R$. Thus, we have our data as $\{(X_i,y_i)\}$ to which we have to fit a D dimensional hyperplane so that the variance around this hyperplane is minimal. Let's start by defining our model:

$$
\begin{aligned}
&y_i(X_i) = f(X_i) + \epsilon \\
&\hat{y_i} = \hat{f}(X_i) \\
\end{aligned}
$$

Here, the actual data is a function $f: \R^D \rightarrow  \R$ and our hyperplane is function $\hat{f}: \R^D \rightarrow \R$ which produces the targets $y$  and prediction $\hat{y}$, respectively. So, we can check the error between our actual data and the predicted data, which  we call the Sum-of-square error:

$$
\begin{aligned}
&\bm{e} = (\bm{y} - \bm{\hat{y}})^2 \\
&\bm{e} = (\bm{y} - \bm{\hat{y}})^T(\bm{y} - \bm{\hat{y}})\\
\end{aligned}
$$

Here, I have used bold to represent vector notation. since our model is linear, we can define it as:

$$
\hat{f}(\bm{X}) = \bm{X}\bm{w} 
$$

- **Note:** to make this work by taking bias into account we let $\bm{w}  \in \R^{D+1}$ where the D weights are corresponding to D features and the extra weight is the bias. Thus, $\bm{X} \in \R^{NX(D+1)}$ which basically means that our N observations are stacked vertically and each observation is of D dimensions, but to make the notation work, we add a 1 at the start, which will be the multiplier for our bias term, and thus, have D+1 as the dimension of the row.

Thus, our error now becomes

$$
\begin{aligned}
&\bm{e} = (\bm{y} -\bm{X}\bm{w}  )^T(\bm{y} - \bm{X}\bm{w}) \\
\implies &\bm{e} = (\bm{y} -\bm{w}^T\bm{X}^T  )(\bm{y} - \bm{X}\bm{w}) \\
\implies &\bm{e} = \bm{y}^T\bm{y} -  \bm{y}^T\bm{X}\bm{w} - \bm{w}^T\bm{X}^T\bm{y} + \bm{w}^T\bm{X}^T\bm{X}\bm{w} \\
\end{aligned}
$$

Now, to get our optimal weights we follow the method to get the minima of e i.e differentiate e w.r.t $\bm{w}$ and then set it to 0:

$$
\begin{aligned}
&\nabla_w\bm{e} = 0 \\
\implies &\nabla_w(\bm{y}^T\bm{y} -  \bm{y}^T\bm{X}\bm{w} - \bm{w}^T\bm{X}^T\bm{y} + \bm{w}^T\bm{X}^T\bm{X}\bm{w} ) = 0 \\
\implies &\nabla_w(\bm{y}^T\bm{y}) -  \nabla_w(\bm{y}^T\bm{X}\bm{w}) - \nabla_w(\bm{w}^T\bm{X}^T\bm{y}) + \nabla_w(\bm{w}^T\bm{X}^T\bm{X}\bm{w}) = 0 \\
\implies &-2\bm{y}^T\bm{X} - 2\bm{w}^T\bm{X}^T\bm{X} = 0 \\
\implies &(\bm{X}^T\bm{X})\bm{w}^T = \bm{y}^T\bm{X} \\
\therefore \,\, &\bm{w} = (\bm{X}^T\bm{X})^{-1} \bm{X}^T\bm{y}\\
\end{aligned}
$$

Hence, all we need to do is plug-in $\bm{w} = (\bm{X}^T\bm{X})^{-1} \bm{X}^T\bm{y}$ into our original equation and we get the solution. Of course, this is the optimization variant of our regression problem and gradient descent goes around this by computing this solution iteratively by taking an initial guess of \bm{w} and then moving towards the direction of decrease, and moving proportionally to the rate of decrease. However, the solution to which it should end up converging is the same! We can also do all sorts of gymnastics around this solution to make the variance go down even further. For example, we could transform our input $\bm{X}$ to a new space by $\bm{\phi(\bm{X})}$, in which subject to 1-1 mapping, our solution would simply become

$$
\bm{w} = (\bm{\phi(\bm{X})}^T\bm{\phi(\bm{X})})^{-1} \bm{\phi(\bm{X})}^T\bm{y}
$$

The essence of regression remains the same. in the case where $D= 2$, we use this same technique on 2D matrices and those simplistic equation for the starting points of regression that we see in most places.



<!-- %%% -->
# Clouds: Agreement in Distributed Systems

Agreement is the crown problems in Distributed systems → How to make different nodes work together even while they have different view. The Agreement Problem is essentially the scenario that **some nodes propose value v and some nodes propose value v' and now all nodes need to have a way to decide which value to accept!**. The values that these nodes agree on can be something  like 

- whether or not to commit transaction to DB
- Who has a lock in a distributed lock service, when multiple clients are requesting it
- Whether to move to a new stage etc.

The fundamental requirements here are:

1. **Safety →** All nodes agree on the same value, which is  proposed by some node 
2. **Liveness** → If less than some fraction of nodes crash, the rest should still reach an agreement


Failure Models help us understand classes of failures, ad in this case we use 2 models: 

- **Synchronous Systems** → We set a timer and compare the activity of each machine to that. Thus, if the machine is inactive after the timeout, we know there is a failure because either hte machine crashed or the network is slow
- **Asynchronous Systems →** systems can arbitrarily be delayed and so there is no proper way to tell if the machine is just slow or if it has crashed.

Agreement problem comes in two flavors: 

1. **Atomic Commitment Problem** → Participants need to agree on a value but each has its own specific constraint on what makes a value acceptable. Thus, the issue is whether a participant can individually commit or not. For example, people agreeing on a time to meet together is a commitment problem since everyone might not be comfortable at all times.
2. **Consensus Problem** → Participants need to agree on a value and they are willing to accept any value. For example, people have decided when to meet and now the issue is where to meet and everyone is 'fine with any place' as long as they get to meet


## Atomic Commitment

The origin of this problem comes from partitioning DB to store and access them in a distributed manner. For example, in the way of 3 copies of chunks in GFS, and this creates Semantic Challenges → We usually segregate the DB into shards based on certain criteria. For example, a Banking application getting end-client requests from users might segregate it into 3 shards based on User-ID . However, what happens when transactions span multiple shards ? For example, a transaction involving a user in one node and another user in another shard → How do we manage agreement and the related issues of atomicity and sharing ?   

### Single Phase Commit

This is a simple way of managing atomicity in which a transaction coordinator is assigned with the following responsibilities: 

1. It begins the transaction and assigns unique IDs
2. It is responsible for commits and aborts

Many systems allow any client to be the co-ordinator and thus, the servers with the data would be the participants. When a commit is validated by the coordinator, it broadcasts **'commit'** and waits for all participants to acknowledge

- **Issue** → What if one participant fails? The other participants cannot undo what they have already committed, as instructed by the coordinator

### 2-Phase Commit 

Here we break down the comi into 2 phases: 

1. **Voting** → Each participant prepares to commit and votes on whether it can commit or not commit
2. **Commit →** Each participant commits or aborts 

This can be realized through the following operations : 

- `canCommit(V)` → Coordinator asks the participants whether they can commit the value V
- `doCommit(V)` → the Coordinator asks the participant to actually commit the value V
- `doAbort(V)` →  The Coordinator asks te participant to abort the commit process
- `haveCommitted(participant, V)` → Participant tells the coordinator if it actually committed value V
- `getDecision(V)` → Participant can ask the coordinator if V can be committed or not

In the voting process, the coordinator asks the participants whether they can commit or not through `canCommit`. The participants prepare to commit using **permanent storage** and write `prepare-yes` to the log. Once the participant replies yes to the `canCommit`, they are not allowed to abort. However, the outcome of event V is uncertain till the doCommit or doAbort happens. The coordinator, then, collects all the votes. If there is a unanimous yes from the participants, then the commit is a go. However, even if one participant votes No, then the commit is aborted. After Voting the coordinator records the fate in permanent storage and then issues the `doCommit` or `doAbort` to the participants. Following is a timeline of this process:

<img height=500 width=800 src="static/Clouds/Agreement/2PC.png">

The following figure shows the state transitions for participant and coordinator:

<img height=500 width=800 src="static/Clouds/Agreement/2PC-1.png">

Now, we can see that that recovery in this scenario is easy → since the participants have been logging their state changes, we can track them. However, these can be differentiated on the basis of timeouts and failures. In timeouts, the issue is that the `ACK` has not been sent by the participant while in failures, there will be an `ACK`.

#### Handling Timeouts
From the coordinator's perspective, two timeout scenarios are relevant: 

1. **Timeout happens in the wait state** → In this case the coordinator cannot unilaterally commit or abort
2. **Timeout in Commit or Abort States** → Here, since the commit has already begun, there is nothing that the coordinator can do but wait for `ACK`.

From the perspective of the participants, following scenarios are possible : 

1. **Timeout in Initial State** → The only way this is possible is if the coordinator has failed the change. Thus, unilaterally Abort
2. **Timeout in Ready** →  Since the participant is in an uncertain state where the decision to commit or abort hasn't been made, a **termination protocol** needs to be activated, which can be blocking or co-operative:

In **Blocking** protocols, we the node waits until communication can be re-established. In **cooperative** protocol, 
the node asks the other participants about the information of the commit and proceeds as follows:
    - Q in commit → P can move to commit
    - Q in Initial → P must Abort
    - Q in Abort → P must abort
    - Q in Ready → Contact another proces
    - The protocol blocks if everyone is in READY 

#### Handling Failures

From the coordinator,  the following scenarios are possibe : 

1. **Failure in Initial state** → Start the commit upon recovery 
2. **Failure in Wait state** → Restart the commit after recovery 
3. **Failure in Abort or Commit States** → If all ACKs have been received, do nothing. Otherwise, a **termination protocol** is needed

From the participant , the following scenarios are possible : 

1. **Failure in  Initial** → Unilaterally Abort 
2. **Failure in Ready State** → The Coordinator has been informed already, follow the termination protocol, blocking or cooperative 
3. **Failure in Commit or Abort** → Nothing special needs to be done


**The 2 phase commit is called a blocking protocol because it cannot make progress during failures**. For example, if the TC sends a message to a node and it commits, but then both crash, then all other nodes need to keep waiting for them to recover since they can't be sure of what to do → A could have sent either yes or no to canCommit and so, they can't move forward! → T**his also makes  2PC vulnerable to the failures of the TC Node.** **Thus, 2PC is safe, but not live!**

### 3-Phase Commit

The idea here is to alleviate the blocking nature of 2PC by splitting commit into 2 phases: 

1. Communicate te outcome to everyone 
2. Let them commit only after everyone knows the outcome === ACK


<img height=500 width=800 src="static/Clouds/Agreement/3PC.png">

It is clear in the timeline, we now have a preCommit message and an associate ACK, after which the commit can happen. The relevant state transitions now become :

<img height=400 width=800 src="static/Clouds/Agreement/3PC-1.png">

The blocking issue of 2PC can now solved as follows : If both TC and a node fail :-  

- If even one of the nodes has received a pre-commit → They all commit
- If no one received a preCommit → Unilateral Abort

Thus, we now have a live protocol. However, while it is safe in the case of the failure of TC and node,  it is not safe in the cases where TC or node are just offline and not actually crashed. The following Scenario is a good example:

- A receives prepareCommit from TC
- Then, A gets partitioned from B/C/D and TC crashes
- None of B/C/D have received prepareCommit, hence they all abort upon timeout
- A is prepared to commit, hence, according to protocol, after it times out, it unilaterally decides to commit


This is a classic example of Asynchronocity! 3PC guarantees safety and liveness for synchronous machines since in that case we know the upper bound for message delays and can see if timeout exceeds that to determine if a system has crashed or not!

### Fischer-Lynch-Paterson Impossibility Result (FLP)

- Thus, we see that **3PC trades safety for liveness, while 2PC trades liveness for safety.** The obvious question is "**Can we design a system that is both safe and live in the general scenario ? "**
- The answer is **NO!** 
- According to the FLP Result,  It is impossible for a set of processors in an asynchronous system to agree on a binary value, even if only a single process is subject to an unannounced failure.
- However, the FLP result does not talk about asymptotic sense with safety and liveness i.e. it never specifies how close a system can get to Safety and Liveness, without actually ideally reaching it!! This leads us to Consensus protocols like Paxos, Raft that actually get close in practice


## Consensus

The consensus problem for a collection processes $P_i$ can be described as follows:

- The  processes propose values $V_i$ and send messages to others to exchange proposals
- Different processes propose different values, but they all need to accept a single value, which can be any one of the proposed value
- Only one of the proposed values can be chosen and all nodes need to know this one value


This puts the following constraints: 

1. **Consistency** → Once a value is chosen, the chosen value for all working processes is the same 
2. **Validity →** The chosen value was proposed by one of the processes
3. **Termination** → Eventually all processes agree on the same value


### Process Reliance
The core idea behind process resilience is that even if some processes fail, we should be able to deliver to the client's request. to achieve this, we develop methods to **replicate the process** so that we have ways of getting a backup process running

- **K-Fault Tolerant Group** → A group of processes that can mask any K-concurrent failures of their member processes. Here, **k is called the degree of fault tolerance**. 

In this group, we make assumptions about the members:

1. They are all working identically
2. They process commands in the same order

Thus, now we basically need a way to ensure that **Non-faulty group members reach a consensus on which command to execute next**

## Flooding-Based Consensus

### Model

1. The process group is $\bm{P} = \{P_i\}$
2. Every process either fails or runs, there is no mid-way. Thus, the process has a **Fail-Stop model** in which on failure it stops and this allows us to reliably detect failures
3. Each process maintains a list of proposed commands 
4. A client contacts process $P_i$ to request it to execute a command

The fail-stop model basically allows us to detect errors reliably. Thus, we need at least (k + 1) processes running so that if k processes go wrong, we know at least 1 process will go right

### Algorithm

For each round r :

1. At the start of the round, each process $P_i$  multicasts its list of commands $C^r$ to all other processes 
2. At the end of the round, all the commands in the lists of all processes in $\bm{P}$ are merged together into a new set $C$
3. the next command is selected through a **globally shared deterministic function** from the this new set of merged commands : $cmd  \leftarrow select(C_j^{r+1})$

### Example

<img height=400 width=700 src="static/Clouds/Agreement/FBC.png">

Here, there are 4 processes → $\bm{P} = \{P1, P2, P3, P4\}$ → and at the start of our observation, they all try to share their commands with each other process.  However, in the process of sharing its commands, P1 fails, and thus, only P2 receives the commands from P1. Now, P2 has received the commands from all processes and thus, proceeds to make a decision, but P3 and P4 have not received the command from P1. Since they are maintaining a timer with them, they will be able to reliably detect that P1 has failed, but are not sure if it was able to share any command with P2 or not. Thus, here P3 and P4 do not proceed forward and wait for the next round, in which they move with the knowledge that P1 has failed and so they don't need to wait. Thus, in the next round, when they receive the command from P2, and they make their decision. This works because P2 has factored-in P1's proposed command and since P3 and P4 only wait since they can reliably detect failure, they are able to rely on the next command of P2 since it made its decision based on having received P1's command. 

- In the worst case, we would have only process moving forward since it is the only non-faulty one!

While this model works, its **Fail-Stop assumption makes it non-realistic**. Moreover, the reliability assumption is also not the most realistic one!

## Building up to Paxos 

### Assumptions 
1. Partially synchronous system (Can be even asynchronous)
2. Communication between processes may be unreliable since the messages can be lost, duplicated or  re-ordered
3. Corrupted messages can be detected
4. Deterministic operations → Once an execution starts, we know exactly what it will do 
5. Processes can exhibit Crash failures, but not arbitraty failures
6. Processes do not collude

Here, we need at least ( 2k + 1 ) processes, where if k processes can arbitrarily go wrong, we need k +1 other processes to be non-arbitrary so that we can reliably detect the failure and in the end, at least 1 process will run.

### Starting Point 

We assume a client-server architecture with initially one client. To this, we add a backup server, so that one server is the primary server and the other is the secondary server. To ensure that commands are executed in order, we assign sequence numbers to them, and so, each server executes commands in the same order - whatever that may be

#### 2 Server situation 

<img height=400 width=700 src="static/Clouds/Agreement/2SS.png">

- In this scenario we have servers S1 and S2, where S1 is the primary server - also called the Leader - and S2 is the secondary server. There are two clients C1 and C2, which are requesting operations $o^n$  from the servers. The servers respond to these commands as $\sigma^i_j$ where i is the number of the server and j is the sequence number of the operation.
- In Paxos, a leader sends an accept message  - `ACC(o,t)` - to the backups when it assigns a timestamp t to operation o, and the backup server responds by sending a learned message - `LRN(o,t)`. If the leader notices that it has no received `LRN` from the backups, it re-transmits the `ACC`.
- In the first case, we think of a situation where the primary sends the ACC and then decides to move forward with accepting operation o1, in which case it assigns it sequence number 1 and sends $\sigma^1_1$ to C1. However, S2 never received the ACC and so, looks at its timeout counter for failure and notices that it has been exceeded. It assumes the leadership role and decides to go for o2 first, thus, assigning it sequence 2 and sending $\sigma^2_2$ to C2. **This is a consensus violation.**

<img height=400 width=700 src="static/Clouds/Agreement/2SS-1.png">

**Solution → Never execute an operation before it is clear that it has been learned**. Thus, if  S1 does not move forward without receiving an LRN from S2, the above situation is rectified since now S1 re-transmits the message till it receives `LRN`, and then both S1 and S2 have a consensus on which operation to perform.

<img height=400 width=700 src="static/Clouds/Agreement/2SS-2.png">

#### 3 Servers with 2 crashes

<img height=400 width=700 src="static/Clouds/Agreement/3SS.png">

- In this case, we see the same issue and we realize that we need to extend the requirement of not moving forward before receiving LRN to S2 and S3 both. Thus, S1 should not execute unless it gets a LRN from both, S2 and S2. 
- But what if LRN from S2 never reaches S1 ? →  even if the LRN from S2 to S1 is lost, it should wait till it gets LRN(o1) from S3 before proceeding. 

Thus, the **Paxos Fundamental Rule → A server S cannot execute an operation o until it has received `LRN(o)` from all other non-faulty servers.**


### Removing the Failure Detection Assumption

**Let's remove the assumption that the processes can reliably detect crashes.** Thus, in an asynchronous system the only solution is **Heartbeat →** Each server periodically sends an `ALIVE` signal to all the servers, and tracks for this signal using a timer from each server. On timeout, it tries to ping the server to determine if the server is still alive.
- But what if the Heartbeat is delayed ? → Say the heartbeat of S2 is delayed, S1 will assume the leadership and execute and S2 will assume the leadership and execute if S1 is delayed!!

<img height=400 width=700 src="static/Clouds/Agreement/HBT.png">

Thus, in this scenario, we need at least 3 servers so that for each server it needs 2 heartbeats to get a majority and then execute consensus. **Extending this to k faults, we need (2k+1) servers to get a majority!**

<img height=400 width=700 src="static/Clouds/Agreement/HBT-2.png">

- **Adapted Fundamental Rule** →  In Paxos with three servers, a server S cannot execute an operation o until it has received at least one (other) LRN(o) message so that it knows that a majority of servers will execute o.

Now in this 3 server 1 failure scenario let's look at another possibility. Let's say the Leader crashes after executing o1. In this case, let's say S1 executes o1 and dies, then 2 things can happen:
1. **S3 has no idea of the activity of S1** → S3 never received the `ACC` from S1 so it waits. However, S2 received the `ACC`, after which it detected the crash and became the leader. S2 now sends the `ACC(o2, 2)` to S3, at which point S3 sees the unexpected timestamp 2 and sends a negative back to S2 that it missed o1. Thus, S2 re-transmits `ACC(o1,1)`, and S3 is able to catch-up
2. **S2 missed ACC(o1,1)** → S2 detects the crash and becomes the leader and either sends `ACC(01,1)` to S3, which then either transmits `LRN(o1)`, or it sends `ACC(o2,1)` in which case S3 notices that it was expecting o1 and sends a negative allowing S2 to catch-up

**Thus,  Paxos (with three servers) behaves correctly when a single server crashes, regardless of when that crash took place.**

### False Crash Detection

What if the ACC by S1 is highly delayed? → In this case, S2 detects a failure and becomes a leader, while S3 receives `ACC(o1,1)` after `ACC(o2,2)`. This can be solved by adding the identity of the current leader in messages. 

- However, Paxos can still come to grinding halt when LRN form S3 is lost, which blocks S1 and S2 from doing anything → It is not Live

### Liveness in Paxos
To deal with Liveness, we add an explicit Takeover of leadership where before a takeover, the server  has to deal with any outstanding tasks by the former leader → This takeover needs to be communicated explicitly to all the servers


## PAXOS Actual Protocol

### General Rules on Protocols 

Each proposal has a unique number, so that:
- Higher numbers take priority over lower numbers
- The proposer should be able to choose this number to be higher than anything it has ever received or seen.This can be implemented by setting Proposal Number to be a concatenation of Round Number and Server ID, so that each server stores the maxRound - the Largest round number it has seen so far and a new proposal number can easily be generated by incrementing the maxRound and concatenating it with server ID. 

Each Node maintains four variable  : 
- `my_n` → my proposal number in the current Paxos
- `n_a` → higher proposal number accepted
- `v_a` → value corresponding to the highest proposal number
- `n_h` → highest proposal number seen

### Propose Phase
- A node decides to be the leader and propose
- proposer chooses `my_n > n_h`
- leader sends `<preapre, n>` to all nodes 
- Upon receiving `<preapre, my_n>`: if `n < n_h` → reply `<prepare-reject>`, else → reply `<prepare-ok, n_a, v_a>` and update `n_h = n`.

### Accept Phase
- If the leader gets a majority of `<prepare-accept>` → It sends `<accept, my_n, V>` to all nodes, where V is `n_a` if it is not null, or a random value
- If majority is not there, then restart Paxos
- Upon getting `<accept, n, V>`, the nodes: reply with `<accept-reject>` if `n < n_h`,  else → update `n_a = n,  v_a = V, n_h = n` and send `<accept-ok>`

### Decide Phase
- If the leader gets a majority of `<accept-ok>` then it sends a `DONE`  to the client
- It keeps sending `<Decide, v_a>` to all nodes until it gets `<Decide-ok>` to ensure no nodes are left behind







<!-- %%% -->
# Clouds: Concurrency and Consistency 

The major issue here is solving the scenario of multiple people trying to access the same database. For this, let's assume we have a pointer to the DB or its attribute, or whatever  → X. We can read and write to X. Now, if we want to do two operations on X, say subtracting 20 and adding 10, we need to do these operations in isolation. If we interleave these operations, we can get a situation where subtracting 20 and adding 10 to 100 might result in a final value of 110. The clearest solution is Do not perform simultaneous R/W to the Db. However, this does not exploit the fact that most DBs are on multiple cores and thus, can actually benefit from parallel execution of queries. Moreover, if this value 100 is, say, the value in our bank account and while executing the query there comes a failure, then how do we handle that? → We need to build failure tolerance into it.

## Transactions
Transactions are a sequence of one or more SQL operations treated as a unit. These transactions appear to run in isolation and changes are only registered if they are complete. Thus, if our system fails, then we restart the incomplete transaction which would be the ones that were not registered. The correctness of transactions is determined by the **ACID Properties**:
- **A**tomicity →  Either all actions in the transaction happen, or none happen. In other words, transactions can't be done partially → They are atomic.
- **C**onsistency → If the DB starts consistent, and all transactions are consistent, then the DB ends up consistent
- **I**solation → Execution of one transaction is isolated form another transaction.
- **D**urability → If a transaction commits, its effects persist.

### Atomicity
A transaction might **commit** after completing all of its actions, or **abort** after completing none or partial action. The key point is that from the user's point of view, a transaction always executes all, or none, of its actions → There is no sense of partial completion. This can be implemented in 2 ways :

- **Logging →** DBMS logs all the actions so that it can undo actions for non-atomic transactions in case of failures (Think Github )
- **Shadow Paging →** While executing a new transaction, the execution is done on a shadow of the original DB units, so that any intermediate failures will not hamper the concurrency of the original unit (Think caching). Once the  transaction is complete, all the units that referred to the original page are updated to refer to the shadow page

### Durability
It has 3 phases when a crash occurs

1. **Analysis →** Scan the log from the most recent checkpoint to see for all the actions that were active when the crash happened
2. **Redo →** Redo the updates as needed so that all the logged updates are carried-out and written to the disk.
3. **Undo →** Undo the write of all the transactions that were active at the crash.


Thus, at the end only the commited updates are reflected in the database

### Consistency
We need to enforce the integrity constraints in the database so that the input and the output of consistent transactions are consistent Databases

### Isolation
Each transaction operates as if it were the only transaction running.  This is done by the database in a fashion where the interleaving of operations do not result in simultaneous updates. For example, transaction $T_1, T_2$ are shown below which operate in isolation

<img height=100 width=500 src="static/Clouds/CC/Isolation.png">

## Anomalies in Concurrency 

There are 3 kinds of anomalies that usually occur: 

- **Reading Uncommitted Data** → When the data has not been committed and we read it in the middle of interleaving, it creates **dirty reads**. So, in the example shown below, T1 did not commit after Reading and Writing to A and the un-committed value was read by T2 from A and was committed, which is wrong. 

<img height=50 width=500 src="static/Clouds/CC/RUD.png">

- **Unrepeatable Reads** → The reads do not produce the same value. In the below example, T1 reads A and then again reads it to verify, but T2 comes in the middle and Writes something leading to the 2 reads of A by T1 not producing the same result.

<img height=60 width=500 src="static/Clouds/CC/UR.png">

- **Overwriting Uncommitted Dat**a → Data is written over before being committed, as seen in the example below where T1 writes to A and B and commits, but before it committed, T2 has already written values to A and B, and thus, the data is  not the same.

**We solve this through serializability!** But for that we need to define **Conflicting Operations** → Two operations conflict if they are performed on the same data by different  transactions and one of them is a write


## Schedules
Scheduling is creating a schema Interleaved actions from different transactions. These can be done in three manners: 
1. **Serial** → does not interleave data
2. **Equivalent** → 2 schedules creating equivalent effect. Schedules are conflict equivalent  if they involve the same actions on the same data and the conflicting actions are ordered the same way
3. **Serializable**→  A schedule that is equivalent to some serial execution of the transactions. Schedules are conflict serializable iff a schedule is conflict equivalent to a serializable schedule

Let's take an Example:

<img height=300 width=700 src="static/Clouds/CC/Sch-ex.png">

Here, we have schedules $S_1, S_2, S_3$  working on DBs A and B, and the conflicting operations are: 

$$
\begin{aligned}
&R_1(A) \leftrightarrow W_2(A) \\
&R_2(A) \leftrightarrow W_1(A) \\
&W_1(A) \leftrightarrow W_2(A) \\
&R_1(B) \leftrightarrow W_2(B) \\
&R_2(B) \leftrightarrow W_1(B) \\
&W_1(B) \leftrightarrow W_2(B) \\
\end{aligned}
$$

Now, in $S_1$ and $S_2$ we see that for the case of both A and B, $R_1$ and $W_1$ precede $W_2$ and $W_1$ precedes $W_2$ in the same manner. Thus, $S_1 \equiv S_2$ . Moreover, we see that $S_2$  is a serial schedule, thus, $S_1, S_2$  are serializable, but this is not the case with $S_1, S_3$ since the actions of the T2 come before T1.

### Precedence Graphs

We can formalize the check for conflict serializability in schedules by the simple process of swapping adjacent non-conflicting schedules to see if we get a serial schedule or not, as shown below 

<img height=300 width=650 src="static/Clouds/CC/pg.png">

But an even better way is to see it in terms of a precedence graph. So, we just go along the order of operations in a schedule and for each conflict, we see if which transaction's operation precedes, and we make a connection on the nodes name after transactions in that order. For example, in the schedule below, we see that in the cases of A and B both, the operations of the first transaction precede that of the second, and so we have a single like from the first to second in the graph.

<img height=350 width=700 src="static/Clouds/CC/pg-2.png">

If we have a graph with cycles, then we know that the  schedule is not conflict serializable, as shown in the example below

<img height=350 width=700 src="static/Clouds/CC/pg-3.png">

Voila! now we have a framework that tells us that if our schedule of interleaved operations is conflict serializable, then it is a valid schedule, and thus, the transactions would crate valid databases if acting on a valid Database! Voila! now we have a framework that tells us that if our schedule of interleaved operations is conflict serializable, then it is a valid schedule, and thus, the transactions would crate valid databases if acting on a valid Database! 

## 2PL Locking 

There is a better way to ensure that our schedule is conflict serializable → Locking. We create two kinds of locks

- **Shared Lock** → Acquired by the Transaction for reading the  object and can be acquired by multiple transactions at the same tie - hence, the name shared
- **Exclusive Lock** → Acquired b the transaction while writing to an object and can only be acquired by one transaction at a time

The rule is → once a transaction releases a lock, then it can't acquire any more locks. Thus,  we can have a graph showing the slow growth phase of acquiring locks nad a release phase of  releasing locks

<img height=300 width=600 src="static/Clouds/CC/2pl.png">

Thus, if our system sticks to this schedule, then it will be conflict serializable. However, this leads to a problem of cascaded Aborts, where the issue is that in case a data  object is written to before the first transaction aborts the operation, then it might lead to issues, as shown below: 

<img height=60 width=500 src="static/Clouds/CC/2pl-1.png">

To alleviate this we create a stricter 2PL where the release happens all at once for each lock and thus, after acquiring alll locks the transactions wait to complete everything and then release the locks, resulting in the graph shown below:

<img height=300 width=600 src="static/Clouds/CC/2pl-2.png">

## Networked File System (NFS)

It allows remote hosts to mount file systems over a network and interact with those file systems as though they are mounted locally. This enables system administrators to consolidate resources onto centralized servers on the network. The schematic is shown below: 

<img height=300 width=600 src="static/Clouds/CC/NFS.png">

- The communication is over TCP/IP and Remote Procedure Calls (RPC) encapsulates the low-level data handling on the network into a set of procedures that can be used by the code, by creating procedures for API calls on client-side and executing these procedures on the server-side. Some common RPC frameworks are → **SOAP, gRPC, Thrift** 
- So, the naive way to design the FS would be to forward every Fs operation over RPC to the server and thus, make the system operate as if they are working on the same filesystem. However, **the volume of RPC calls adds latency** → So we add client-side caching to this!

### What do we cache?
- Read-only files
- Data written by client machine → **write-back caching →** Issue of failure tolerance
- Data written by other machines → Issues with consistency

### What about Consistency?
- NFS Caches Data and File Attributes. The data never expires, but the file attributes expire after 60 seconds → so if a file is modified, the new time is reflected in its attribute in the server, and thus, it can be checked and updated on each client.
- Dirty data are buffered in the client machine until the file closes or till 30 sec → If the machine crashes between that, everything is lost
- Thus, **NFS sacrifices consistency for less traffic**
- **Close-to-open consistency →** We can write a way to ask the server for the latest file everytime before opening it!
- NFS does not provide any guarantee for multiple writes.

### What about Failures?
- NFS uses a stateless server → The NFS server does not track anything but instead checks for permission for each operation
- No pending R/W operations across crash
- Read request needs to get an exact positin of hte file →
- Operations ar  Idempotent → operations use unique ID of files and so, cannot be confused
- Write-Through Caching

## Andrew File System (AFS)

In this, the files share the same namespace across machines but work with the assumption that the client-side machines cannot be trusted → thus, they must prove that they have the rights to perform certain operations → this is implemented through modifying RPC to Secure-RPC. TThe client Machines have disks and these can be used for caching. Thus, they  realized the following characteristics, which were then incorporated: 
- It's very rare for simultaneous R/W → they found this through analysis. Thus, they started aggressively caching on local disks to reduce the traffic load → **Close-to-open consistency is fine!**
- **Prefetching** → Large reads are faster than a lot of small reads on local disks and so, they fetched the whole data of the file
- **Invalidation Callback**  → Clients registers with the servers when they have a copy of the file  and when this file changes, the server tells them to invalidate this copy → If the server caches, then we reconstruct callback  information by asking every client what file they have cached

## Google File System (GFS)

Here, the following desing constraints are taken into account 

1. Machine Failures are normal 
2. Designed for Big-Data workloads
3. Many files are written once and they are read sequentially 
4. High bandwidth is more important than latency 

Thus, the GFS is geared towards these characteristics and the google applications are designed to work with this. A file is divided into chunks and labeled with 64-bit global IDs - called handles - which are then stored on **chunk servers.** Each chunk is stored 3 times on 3 different chunk-servers, and the master keeps a track of the metadata → which chunks belong to which files

## Theory of Consistency
- Consistency concerns arise when we are replicating or caching files.

**Replication** → Maintain data in multiple computers. It is necessary for
- Improving performance → Closer data is faster
- Increasing availability of services → To handle server and client crashes
- Enhancing the scalability of systems → E.g. CDNs  that store the data locally and then
- Securing against malicious attack

In a Distributed system, we store data in distributed shared memory, distributed databases, or distributed file systems → **referred to as data-store** 

- Multiple processes can access shared data by accessing any replica on the data-store

## Distributed Shared Memory

Communication in Distributed system happens through either sharing the memory or message passing. Shared-memory is more intuitive for consistency and so is more popular. The goal, thus, is to create a distributed system of memory shared by multiple systems, but each system thinks that it is accessing the same memory from the large memory pool! 

<img height=300 width=600 src="static/Clouds/CC/dsm.png">

The naive way to do this would be through local copies of the whole memory with all the machine so that
- **Read** → Machine reads from local memory → Fast
- **Write** → The updates are sent to all the memory copies, while the machine does not wait for this to complete

So, in a way, this approach is basically message-passing applied to shared memory. This is fast, but has the following problems:
1. Since we  are not waiting for ACK after a write operation, what if the message gets lost and the order of delivering messages gets messed up → Since we have no control, we will see weird behavior
2. Since we have no control over the order of updates, what if there are disagreements in udpates ?


## Models of Consistency

In brief, consistency can be summarized as a contract between nodes that the last write of the data is shared. Let's take an example, that we will use again and again, to explain consistency models. Here, we have P1, P2, P3, and P4 as processes that are trying to read and write to the same variable x in the shared memory. The question is that if P1 writes a and P2 writes b, then what do P3 and P4 read?

<img height=200 width=400 src="static/Clouds/CC/moc-1.png">


### Strict Consistency

Each operation has a global timestamp and the order of execution is determined by sorting this. The rules are 

1. Each operation gets the latest value of the variable → Reads are never stale  
2. All operations are executed in the order of their timestamps 

In this case, P3 and P4 will always observe b due to the time stamping of strict consistency → It is like running the process on one processor, where we use semaphores to work with x for all the processes → Target Achieved?

<img height=150 width=400 src="static/Clouds/CC/strict.png">

The issue is the implementation → We need to make the processes wait for write operations to complete before read → this take  time and so we need exact clock synchronization. Computer clocks experience drift in their quartz crystals → leads to change in the rate of the timer interrupts used for maintaining time → Thus, we need **Universal Coordinated Time (UTC)**. The Cs-133 atom-based time is broadcast → the computers can receive this signal and synchronize their clocks. However, even nanoseconds might create issues! **Thus, strict consistency is hard to implement**

### Sequential Consistency

We let go of the assumption of synchronizing in real-time, and instead focus on preserving the order of events so that logical outputs are not affected. Thus, we now have the following rules: 
1. Each machine has an order on its own operation
2. Results appear according to **some total order**

Thus, Reads may be stale in terms of real-time, but not in logical time but the writes are strictly ordered! Hence, the output of this on our example would be . For example, in the picture below the second case, which was not possible in the case of strict ordering, we now observe a dirty read. However, if we look at the order of the events, the read of b always happens after the read of a in both the machines. Hence, we still have the same logical order of operations and thus, our program  will be sequentially consistent

<img height=150 width=600 src="static/Clouds/CC/sequential.png">

This is easier to implement than strict consistency since now we can interleave the operations and again if the operations are concurrent serially through the mechanism discussed before, we can have the same execution of programs. 

- Requests to an individual memory location (storage object) are served from a single FIFO queue → Writes occur in a single order and the read happens only after the writes have occurred, thus maintaining consistency

**Thus, we can say that not that all processes agree on exactly what time it is, but that they agree on the order in which events occur** → This is the difference between timed and ordered processes. However, this is still expensive due to communication and wait times!

## Lamport Logical Clocks

The basic idea is understanding which event happens before, which is represented by the '→ ' symbol. This is a partial order since there might be instances in a set of orders where the exact order cannot be determined, but the final order is clear.

Now, we need to use this relation to establish Logical clocks and we do this through **Lamport's Logical Clocks,**. We, first attach a counter to events so that, events (e) satisfy the following properties: 
- **P1** → If a and b are two events in the same process, then they would be preserved in the time in which they take place $a \rightarrow b \implies C(a) < C(b)$
- **P2 →**  If  a corresponds to sending a message and b corresponds to receiving that message, then also  $C(a) < C(b)$

Using these properties we have a counter $C_i$ attached to each process $P_i$, such that
- For each new event in $P_i$ , we increment $C_i$ by 1 
- Each time a message is sent from $P_i$, it receives a timestamp of the value of $C_i$ so that → $ts(m) = C_i$
- Whenever $P_j$ receives a message from $P_i$, it adjusts its local counter as → $C_j = \max \{C_j, ts(m)\}$


This lamport ordering is based on the events and not the other way round. Thus, $C(e) < C(e')$  does not imply $e \rightarrow e'$, meaning  it does not encode causal relationship. So, if I have $C(a) < C(b)$ then this does not mean that a necessarily preceded b! This is an issue for concurrency 

### Vector Clocks 

We increase the counter to a vector of counter for k processes, so that : 

- Each process $P_i$ has its counter vector that has its counter at $C[i]$ while the count of all other k-1 processes at the other places
- Whenever a process happens, the process increments the value of $C[i]$ by 1, while all other values remain the same
- When it has to send a message, the process now shares the whole vector to the other process
- The process $P_j$ which receives the vector updates its vector to the new value and increments the count of its own by 1 and then follows whatever it has to do

An example of this is shown below: 

<img height=250 width=600 src="static/Clouds/CC/vc.png">

Here, we can see that in part a, all the processes start with values (0,0,0) and this then goes on as follows:

- P2 sends does performs an operation incrementing its counter by 1 in the vector while all other remain 0
- P2 sends a message to P1, which copies the vector and increments its counter by 1
- P1 performs an operation and increments its counter by 2
- P1 sens a message m2 P3, which receives m2 and updates its own counter b 1 after copying the vector
- P1 performs 2 more operations and then sens m3 to P2, which copies the value and increments its counter by 1
- P2 performs and operation j and increments its own counter by 1 and then sends the message m4 to P3, which updates its vector and adds 1 to its counter, which was previously at 1 and so the resulting vector is (4,3,2) instead of (4,3,1)

We now define a causal relationship based on the property that **if any message has all its vector value  < or = the values of the vector of another message, and at least one of the values is strictly less, then it causally precedes the other**. So, in this case, ts(m2) = (2,1,0) while the ts(m4)  = (4,3,0), which implies that m2 may causally precede m4. In case (b) we see that ts(m2) =  (4,1,0) while the ts(m4) = (2,3,0) → Thus, m2 and m4 may conflict.

## Causal Consistency

If all causal operations are executed in an order that reflects their casual relationship, then the executions are causally consistent! So if two operations are concurrent, then they can be read in different orders by different machines, till the time the causal order is followed. For the same process example below as other consistencies:

<img height=200 width=500 src="static/Clouds/CC/causal.png">

The issue here is that we see that that P1 writes a, and then P2 reads a and writes b. Since write of b happened after P2 reading a, there might be a causal relationship between P1s write of a and P2s write of b. Thus, when P3 reads b, then it cannot read a again since the writes are not concurrent → We can also reason in terms of messages.  Assume the processes start with a null value for x. Now, after P1 writes a to x, the only way P2 can read a from x is if P1 has sent a message. Thus, when P2 writes b to x, which is clearly happening after it reading a, we can say that the  write of b is causally related to the write of a by P1. Thus, the only way  P3 can read b form the variable is whe P2 sends a message of update to P3. Since no other write has happened after P3 reading b, it is not possible for it to read a again. However, if we modify this as follows:

<img height=200 width=500 src="static/Clouds/CC/causal-2.png">

Now we see that the write of b happens after the write of a, and since there is no global time tracking the writes, these operations might not be causally related and thus, concurrent. Hence, if P first reads b and then reads a, it is acceptable since the writes are concurrent, and the same reasoning allows P4 reading in a different manner! this is very much possible if P1 writes a and sends the message to P4 → In terms of messages, we can see that if P1 writes a and P2 writes b, then essentially they are free to write to the variable since there is no read happening before the write i.e this is possible even if no message is exchanged between the processes. Now, P3 reading b is possible if a message is exchanged between P2 and then it reaching a is possible if it receives another message from P1 on the update. Similarly, P4 reading a is possible if it receives a message from P1 after its update, and then reading b is also possible since it can easily receive a message from P2 after it writes b to x. Hence, this is causally plausible and is thus, consistent
- **Causal consistency is strictly weaker than sequential and strict consistency, but one can get better performance with it since parallel operations can be executed in different orders by different machines.**


<!-- %%% -->
# Clouds: Apache Spark 

The main issue with MapReduce is the read/write from and to the disk. For example, in the case of K-means, the main steps are: 

- **HDFS Read** → Map → Network Shuffle → Reduce → **HDFS Write**

Now, the issue with the R/W operations is the issue of accessing data from the memory and disk → **Random Access from disk is slower, but it offers a larger volume of data**. So, the question is should I store my data on disk or in Memory? 

The answer → if the data is accessed more than once in 5 minutes, cache it in memory; otherwise, store it on the disk → [The 5-minute rule](https://www.hpl.hp.com/techreports/tandem/TR-86.1.pdf)

### Economics of Data Access 
- If I have 2000 euros per access from disk and 5 euros for KB of data in memory, then for each kb of data we save 2000 euros for every 5 euros we spend on memory each second. If our rate of access of 1 access per 10 secs, we save 200 euros, and this trend continues. The break-even point is 400 secs, which is roughly 5 minutes. Hence, the 5-minute rule.
- Now, HDFS stores all data in the disk and so, nothing is cached in memory → misaligned with the 5 minutes rule. Plus, Map and Reduce are too simple computationally.

## Apache Spark 
- **Let's keep the good stuff from Hadoop, but also add the touch of memory and added functions**.

The main workflow is shown below:

<img height=350 width=400 src="static/Clouds/spark/wf.png">

- While in MapReduce we had only 1 master to assign worker nodes as Map and Reduce, in spark we have a separation where there is a driver program scheduling the applications and the executions, while the cluster manager allocates the resources. The primary advantage is that the driver program can be initiated with a spark-context that holds the main configuration and is flexible to different kinds of configurations - single-threaded, multi-threaded, local, distributed, etc. - and thus, allows managing different operations. 
- The cluster manager sends app-codes and tasks for executors to run. The workers have a cache that they can use to run their bits composed of locally schedules tasks, in an isolated manner .
- There is no data shared between workers,  but the executors within workers share the same virtualization. Thus, if we have 2 applications, they can be run on 2 different workers and can have multiple tasks that can be scheduled in executors that share the data, and execute each task as a thread. In MapReduce, this would be executed in a purely distributed manner through mapper nodes with overheads for each. **Thus, overheads are reduced in spark.**

## RDD

Resilient Distributed Datasets extend the concepts of functions to data-structures. Thus, they are immutable objects that either point directly to a data source (HDFS), or apply filter transformations to parent RDDs. Thus the functions are of two types: 

1. **Transformations** → Apply to RDD and return an RDD. E.g., map, filter, groupBy, sortBy
    - They are lazily evaluated → Only triggered through actions
2. **Actions** → Use an RDD to return values

Thus, we can write applications as transformations on RDD and need to only execute them based on actions determined on the time on which they need to be executed. This is similar to the lambda functions in python → They work Lazily. 

The execution steps are:

1. Create DAG of computation
2. Create a Logical execution plan with as much pipelining as possible 
3. Partition the tasks into nodes
4. Determine Dependency and Split DAG into “stages” based on the **need to shuffle data** → determined by the kind of function in the stage
5. Submit Each stage and its task as ready
6. Launch task via Master
7. Retry failed and straggler tasks 
8. Execute tasks
9. Store and serve blocks

The dependency mentioned in step-4 can be of two types: 
- **Narrow Dependency** → The mapping from parent to Children RDD is on a 1-1 basis i.e each parent RDD will share data with at-most 1 child RDD. Thus, there is no shuffling step in the middle which reduces overhead. e.g `map`. `filterMap`, `filter`, `sample`.
- **Wide Dependency** → Multiple child partitions may depend on one partition of the parent RDD. E.g. `sortByKey`, `reduceByKey`, `groupByKey`, `cogroupByKey`, `join`, `cartesian`.

The key idea in creating a pipeline is to look for shuffling → If we have a group of tasks that do not require shuffling, we can group them together as a stage, and then shuffle. Thus, the stuff in one stage can be executed iteratively. For example, in the DAG shown below: 

<img height=400 width=200 src="static/Clouds/spark/pipeline-1.png">

We know that `groupBy()` requires shuffling but the `map()` does not. Thus, we break a stage here. Then, we see that   `mapvalues()` function does not require shuffling, so we have two stages as shown below:

<img height=400 width=300 src="static/Clouds/spark/pipeline-2.png">


## RDD and Spark
- **We can cache outpus in the memory improve performance!**

Let's take the example of Log Mining, where we want to see the error log to look for certain kinds of error like MySQL and php. The following code template would be used for reference:

```python
lines = spark.textfile("hdfs://...)
errors = lines.filter(lambda s: s.startswith("Error"))
messages = errors.map(lambda s: s.split("\t")[2])
messages.cache()

messages.filer(lambda s: "mysql" in s).count()
```

Here, we are taking an HDFS file and filtering for the word "Error" in it. Then, we split it around tabs "\t" and look at the second element in the split array. This output is cached. To this cached output, we apply the filter for "MySQL", to search for SQL errors. If we were to close the code till the cache, the output would still be there when we apply MySQL query to it. Now, if we add a PHP query later to this 

```python
lines = spark.textfile("hdfs://...)
errors = lines.filter(lambda s: s.startswith("Error"))
messages = errors.map(lambda s: s.split("\t")[2])
messages.cache()

messages.filer(lambda s: "mysql" in s).count()
messages.filer(lambda s: "php" in s).count()
```
It would only work on the cached output, and not repeat the process before it. Thus, the new computation would only happen on local machines, and the data would be fetched from the memory instead of the disk. This is the key feature that makes spark aligned with the 5-minute rule: **We can cache the computations that are being accessed regularly while keeping the rest in disk!!**

### Failure Tolerance 
The RDD abstraction is immutable, and simultaneous updates are not allowed. Thus, it can be cached and shared across processes and tasks! This allows failures to be taken into account easily.



<!-- %%% -->
# Clouds: MapReduce

MapReduce is a programming model and an associated implementation for processing and generating big data sets with a parallel, distributed algorithm on a cluster.

## Key Ideas behind MapReduce 

### Scaling out instead of scaling up
If we have workloads that are data-intensive, it is preferable to do it on a large number of commodity low-end servers (Scaling out) instead of a small number of high-end servers (Scaling up). This is because the scale-up approach is costly since the **costs of machines do not scale linearly** and the costs associated with the operational issues like energy required for cooling etc. are additional overheads that turn out to be less flexible for the latter. Thus, most MapReduce applications are built for low-end servers. Scaling-out leads to the following implications: 

- Processing Data is quick, but I/O is really slow due to the network bottleneck imposed by low bandwidth
- There is flexibility in what the computers end up sharing → In a shared-nothing architecture, all the systems are performing individual computations and only haring the relevant data as managed by a distributed file system

### Failure is the Norm, not the Exception

In clusters, failures are not only inevitable but commonplace. Mature implementations of the MapReduce programming model are able to robustly cope with failures through a number of mechanisms such as automatic task restarts on different cluster nodes.

### Data Locality Principle

In traditional HPC applications, the servers are segregated into **storage** and **compute nodes** linked together by a high-capacity inter-connect. However, many data-intensive workloads do not require a high processor capability and so, this segregation creates a bottleneck. It is more efficient to move the processing around instead of the data by co-locating the processor and the data storage and running the job on the processor directly attached to the data, **managing synchronization through a distributed file system.**

### Sequentially Process Data

Data-intensive applications mean that the datasets are large and thus, must be held on disks. However, the seek times for random data access on disks are fundamentally limited. Thus, it is more efficient to avoid this and process the data sequentially in batches, which is what the MapReduce architecture is based upon.

### Hide the System Level Details from Developers

MapReduce abstracts the system-level details and provides a framework that the developer can use, thus, separating the lower-level details of the computations from the commands to do them. Thus, the execution framework needs to be designed only once.

### Scalability

We can define scalability along two dimensions: 

- Given twice the amount of Data, the same algorithm should take at-most twice as long to run, if everything else is the same
- Given twice the number of processors, the same algorithm should take at-most half the time to run

These settings should, ideally, work for a high range of data → MB to PB → and all kinds of clusters. Moreover, the ideal algorithm should not require further tuning. WHile MapReduce does not achieve all of it, it is a step in this direction


## MapReduce and Functional Programming

The key feature of functional languages is the concept of **higher-order functions** that can accept other functions as arguments. Two functions that are common are: 

1. **Map →** Takes a function $f$  as an input and applies it to all elements in a list.
2. **Fold →** Takes a function $g$ and a first data as inputs and applies it to the first item in the list. the result of this computation is stored as an intermediate variable and then applied as an input to the second item, and so on.

This is summarized in the figure below: 

<img height=350 width=300 src="static/Clouds/MR/functional.png">

So, Map is a **transformation** on the input functions that can be parallelized in a straightforward manner since it happens on all the items in a list, while Fold is an aggregation operation that needs to happen on individual elements, that must be brought together before applying it. This is the essence of MapReduce, which can be translated to the following steps: 

1. Apply a user-defined computation in a parallel manner on all the elements in a list
2. Aggregate intermediate results by another user-specified computation

## MapReduce Working 

The input to a job is data stored on the underlying distributed file system. To this data, a mapper and a reducer are defined a  follows :

- Mapper → $(k_!, v_1) \rightarrow [(k',v')]$
- Reducer → $[(k',[v'])] \rightarrow [(k_2,v_2)]$

The mapper generates an arbitrary number of intermediate key-value pairs for every input key-value pair, distributed across multiple files. The reducer is applied to all the values associated with an intermediate key - which is sort-of an inherent grouby operation - and generates an output key-value pair. These output key-value pairs from each reducer are written persistently back onto the distributed file system. The files in the file system are of the same number as the number of reducers, and these output files can further serve as an input to another Mapper.

A classic example of  MapReduce is a program to count the words n files, and this would go as follows; 

1. The input to the mapper is of the form - `(document_id, document_text)` - where each document has a unique ID
2. The mapper tokenizes the document and emits an intermediate key-value pair for every word in a document, where the key is the word and the value is the 1 in the vanilla version, or count if we apply a combiner. So, for this implementation let's say it is the count of the word in that document.
3. The shuffler guarantees that all the values associated with the same key are brought together to one reducer, and ensure this happens for all the keys. Thus, every pair corresponding to the word **the** would come to the same reducer.
4. The reducer emits the word and its count as the output, which is in the form of an individual document.

### Partitioners

These determine which reducer is responsible for which data key. The mapper writes the key-value pairs to a partitioning block and the partitioner maps each key to an integer i \in [0,R], which is then used to send the pairs to R reducers. We can also use URLs, hex hashes, etc. to create the identity of the reducers

<img height=400 width=200 src="static/Clouds/MR/part.png">

### Combiners
These are just mini-reducers. Their input is the same as reducers and the output is the same as mappers. So, in the counting example, the mapper would generate the key-value pairs in the form of  [word, 1] for each word. To this, we can add combiners that aggregate the counts for each file while still maintaining the output format for the mappers, as shown below. This kind of pre-aggregation saves network time. 

<img height=500 width=850 src="static/Clouds/MR/comb.png">

However, there are certain cautionary notes on combiners: 

1.  The correctness of the algorithm cannot depend on computation (or even execution) of the combiners
2. They don't work for all problems. e.g. Mean of letters

### K-means in MapReduce 
A classic example of Distributed algorithms would be K-means, which is an iterative task. In this case of an iterative task, we need a driver to run the MapReduce multiple times and check for convergence. Each map-reduce iteration would be as follows: 
1. We would need a file containing the co-ordinates of each centroid 
2. The input to the mapper would be the data points, and each mapper would compute the distance of the point from each cluster. the output of the mapper would be (cluster, point)
3. the reduce would be receive the data points grouped by a cluster ids, and it ou compute the centroid, thus, producing the output (cluster, centroid)

## Architecture of MapReduce

The Architecture is based on the **Google File-System (GFS),** and the run is as follows:

- Master breaks work into tasks and schedules them on workers dynamically
- Workers implement the MapReduce Functionality on the GFS server daemons

<img height=500 width=850 src="static/Clouds/MR/GFS.png">

The following diagram shows the flow of MapReduce from the paper: 

<img height=150 width=850 src="static/Clouds/MR/GFSPap.png">

It can be explained as follows : 

1. Library splits files into 16-64MB pieces
2.  Master picks workers and assigns map or reduce task (M map, R reduce tasks)
3.  Map worker reads input split, calls map function, buffers map output in memory
4.  Periodically, in-memory data flushed to disk & master is informed of disk location (Partitioning)
5.  Master notifies reduce worker of location, reduce worker reads map output files, sorts data
6.  Reduce worker iterates over sorted data, passes each unique key, list of values to reduce function. The output of reduced function written out to files.

### Fault Tolerance 
- The mapper spreads tasks over GFS Replicas of inputs so that even if a mapper crashes, a copy of the output is available to reducers through re-runs, and the reducers are notified of the re-run
- If the reducer crashes, the tasks that were completed are stored in GFS with replicas and the ones still remaining are re-run.

### Load-Balancing 
1. Scales linearly with data → as required
2. For stragglers i.e the tasks that take a lot of time, the workers who have already finished a task are assigned new tasks  → the  no. of tasks are always greater than the number of workers


<!-- %%% -->
# Clouds: Parallelism and Distributed Computing


## Parallelism in CPUs
A CPU executes instructions in stages, the major stages being Fetch, Decode, Execute, Memory, and Write. Parallelism in CPUs can be achieved in many ways, the most basic being through pipelining instructions, where independent instructions are executed together to improve efficiency. This is represented in the waterfall model shown below:

<img width=650 height=300 src="static/Clouds/waterfall.png">

A measure of how many of the instructions in a computer program can be executed simultaneously is called **Instruction-level parallelism** and a processor that executes this kind of parallelism is called a **Superscalar Processor**. The problem with the above parallelization is the possibility of conflicts that increases with an increase in clock cycles i.e fitting increasingly more instructions together as the pipeline stage continues. Moreover, automatic search for independent instructions requires additional resources.

### Vectorization: Automatic and Explicit
One way to overcome the roadblocks of deeper cycles in CPUs is by exploiting parallelism in data. For example, if the same operation - say addition - needs to be performed on two arrays then this operation can be replaced by a single operation on the whole array. This is called **vectorization**.

<img width=350 height=300 src="static/Clouds/vectorization.png">

Vectorization can be **Automatic** when the scalar operation is automatically converted by the processor into a parallel one, and **Explicit** when the user manually implements vectorization. While the obvious benefit of automatic vectorization is the ease of implementation, it does not always work. For example, in the following code auto-vectorization will not work because for each element the addition depends on the previous element and so, the operation cannot be split into chunks.  

```cpp

    for(int i=1; i < n; i++){
        a[i] += a[i-1]
    }
```

The subtraction might work if the loop is not checking the previous, but an element that is one more than the length of the vector

```cpp
    for(int i=1; i < n; i++){
        a[i] += a[i - N] ;
    }

```

However, if we just replace the 'i-1' with an 'i+1' as shown in the code below, vectorization works since now all the processor needs to do is take a snapshot of the element that the for loop has not reached yet and add that to the current element.

```cpp

    for(int i=1; i < n; i++){
        a[i] += a[i+1] ;
    }
```

Another case in which Automatic Vectorization does not work is when there is assumed dependence as shown below, where the code would only work if a and b are not aliased ( a == b - 1) and b > a

```cpp
    for(int i=1; i < n; i++){
        a[i] += b[i] ;
    }
```
Thus, the limitations of auto vectorization are:

1. Works on only innermost loops
2. No Vector dependence
3. Number of iterations must be known

However, we can guide auto-vectorization by using the SIMD directives. An example is shown below:

```cpp
    #pragma omp declare simd double func(double x); 

    const double dx = a / (double)n ;
    double integral = 0.0 ; 

    #pragma omp simd reduction(+, integral)for (int i=0; i<n; i++){
        const double xip2 = dx * ( (double)i + 0.5) ;
        const double dI = func(xip2) * dx ; 
        integral += dI
    }
```
The pragma directive signals the **SIMD**(Single Instruction Multiple Data) processors to parallelize 'func()' while the reduction on the addition of the integral signals that the sum needs to be calculated in a reductive manner, where different parts of the parallel sums are combined to get a result which is then finally added to the integral variable instead of using an integral variable every time.

## Parallelism in Multi-Core CPUs

A multi-core CPU has multiple CPU units sharing the same memory, as shown below:

<img width=700 height=250 src="static/Clouds/multi-core-cpu.png">

Thus, all the stuff explained above is happening on one vector unit. Since the memory is shared, all the CPU units have the ability to access and modify the contents of the same memory. Thus, they don't need external communication as it is implemented implicitly. However, the relevant issue now becomes the synchronization of these processes since the code written is Multi-Threaded.

### OpenMP

Open Multi-Processing (OpenMP) is a framework for shared-memory programming that allows the distribution of threads across the CPU cores for parallel speedup. It can be included in the CPP programs and easily used through the pragma keyword. For example, in the above Reimann sum OpenMP can be applied as follows:

```cpp
    #pragma omp declare simd 
    double func(double x); 

    const double dx = a / (double)n ;
    double integral = 0.0 ; 

    #pragma omp parallel for reduction(+: integral)
    for (int i=0; i<n; i++){
        const double xip2 = dx * ( (double)i + 0.5) ;
        const double dI = func(xip2) * dx ; 
        integral += dI
    }
```
Here, the reduction is applied to use the reduction sum instead of a normal sum as integral is a shared variable that is incremented in each iteration. If we were to use a normal parallel sum without reduction, the performance would not speed up since the mutex between threads would prevent the loops from parallel operation as each loop would wait for one operation to complete and release the variable. Thus, the addition of the reduction sum allows parallelization, and the performance improves dramatically as shown in the figure below: 

<img width=600 height=300 src="static/Clouds/openMP.png">

### Adding More Cores : MIMD

The next step in the trend was to add more cores and make each core perform the same function Thus, more transistors performing specialized tasks allowed splitting independent work over multiple processors, for example in pixel analysis of images. This is called **Task Parallelism**, and this leads to **Multiple Instructions Multiple Data (MIMD)** cores. When the work being done by each core is identical, but the data is different, then it is called **Single Program Multiple Data (SPMD)**, a subcategory of MIMD. The most obvious addition that can be done to SPMDs is sharing the fetch and decode parts of processing amongst multiple processes as shown below:

<img width=600 height=300 src="static/Clouds/simt.png">

This is called **Single Instruction Multiple Thread (SIMT)** approach, and this is the foundation for GPUs and CUDA.

## GPUs
The SIMT approach forms the core of the Graphical Processing Units (GPUs) where each unit does identical work. Many SIMT threads grouped together make up a GPU core. A GPU has many such cores and a hierarchy can be created as follows:

<img width=400 height=600 src="static/Clouds/gpu-cuda.png">

### Explaining CUDA

As explained before, the core idea in a GPU is to make multiple smaller cores perform the same function, thus, maximizing throughput. This differentiates GPUs from CPUs, which are designed to minimize latency by implementing advanced control logic and caching. Thus, the focus of GPUs is on the cores that have threads executing the same task in large numbers in a parallel fashion, and these cores occupy the major area of the Silicon Wafer. These Cores are called **Kernels**. When we run a function on these kernels - called launching a kernel - each function is mapped to a thread of execution on a core. Thus, these programs are massively multi-threaded. Now, the basic way of going about parallel programs is to make the CPU run the normal execution, but make it share its DRAM with the GPU through a PCI Bus, which allows it to  parallelize computations that are massive and can be broken down to be done by the GPU

<img width=600 height=300 src="static/Clouds/PDC/CUDA.png">


Thus, the CPU is called the **Host** and the GPU is the **Device,** and this way of sharing computations is called **Heterogeneous Parallel Programming.** This is implemented in the NVIDIA GPUs through the CUDA language → which is essentially C with added instructions. The threads execute Kernel instructions in a SIMT manner, and are organized into 3 classes: 

1. **Threads** → A set of threads is executed by a Kernel.
2. **Blocks →** Threads are grouped into blocks executed on a set of cores.
3. **Grid** → Sets of Blocks → Each kernel Launch is executed as a grid mapped to the entire GPU.

The threads and blocks can be 1D, 2D, and 3D structures. The identifiers are as follows: 

- Grid Dimension → blockDim
- BLock ID → blockIdx
- Thread index → threadIdx. 

The Thread Identity depends on the block identities as follows : 
- 1D → Thread ID == Thread Index
- 2D → Thread ID (x,y) = $x + D_xy$
- 3D → Thead ID (x, y, z) = $x + D_xy + D_xD_yz$

1. Decalre pointers to memory 
2. Allocate memory to the Cuda device → `cudaMalloc (pointer, size of variable type)`
3. Transfer memory to the device → `cudaMemcpy( dst, src, size of variable type, direction )`
4. Configure the grid and block parameters → `dim3(x,y,z)`
5. Launch Kernel → `<<<grid, block >>>(...)`
6. Copy the results back to the main execution after completion → `cudaMemcpy( dst, src, size of variable type, direction )`
7. De-allocate the memory → `cudaFree(pointer)`

An example of this is shown below: 

```cpp
    void main {
        
        // Declare vairables
        int *h_c ;// Host 
        int *d_c ;// Device 
        
        //Allocate the memory to device 
        cudaMalloc( (void**)&d_c, sizeof(int) ) ;

        //Set-up the Data transfer
        cudaMemcpy (d_C, h_c, sizeof(int), cudaMemcpyHostToDevice ) ;

        //Define the Grid and Block cofigs
        dim3 grid_size(3,2) ;
        dim3 block_size(4,3) ;

        //Launch the kernel 
        kernel<<grid_size, block_size>>>(...) ;

        //Copy the data after completion
        cudaMemcpy (h_c, d_C, sizeof(int), cudaMemcpyDeviceToHost ) ;

        //De-allocate the memory 
        cudaFree(d_c); 
        cudaFree(h_c);

    }
```

The kernel is defined using `__global__` keyword and always returns void.  The function defined inside the Kernel will always be executed by all the threads. 

### Parallelizing For Loop

In the CPU code, the for loop is written as : 

```cpp
    void increment_cpu(int *a, int N) {

        for ( int i=0; i<N; i++) {
            a[i] = a[i] + 1 ; 
        }

    }
```

Since each step fo the loop performs the same  operation, we can parallelize it :

```cpp
    __global__ void Kernel( int* a, int N) {
        int i = threadIdx.x ; 
        
        if ( i < N ){
            a[i] = a[i] + 1 ; 	
        } 
    }


    void main {
        
        // Declare vairables
        int *h_c[N] = ... ;// Host 
        int *d_c ;// Device 
        
        //Allocate the memory to device 
        cudaMalloc( (void**)&d_c, sizeof(int) ) ;

        //Set-up the Data transfer
        cudaMemcpy (d_C, h_c, sizeof(int), cudaMemcpyHostToDevice ) ;

        //Define the Grid and Block cofigs
        dim3 grid_size(1) ;
        dim3 block_size(N) ;

        //Launch the kernel 
        kernel<<grid_size, block_size>>>(d_c, N) ;

        //Copy the data after completion
        cudaMemcpy (h_c, d_C, sizeof(int), cudaMemcpyDeviceToHost ) ;

        //De-allocate the memory 
        cudaFree(d_c); 
        cudaFree(h_c);

    }
```
Another Example is to do Matrix Multiplication on GPU, shown below: 
```cpp
    __global__ void MatrixMultiplyKernel(
                                const float* devM, 
                                const float* devN,
                                float* devP, 
                                const int width ){
            
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            
            // Initialize accumulator to 0
            float pValue = 0;

            // Multiply and add
            for (int k = 0; k < width; k++) {
                float m = devM[ty * width + k];
                float n = devN[k * width + tx];
                pValue += m * n;
            }
            
            // Write value to device memory - 
            // each thread has unique index to write to
            devP[ty * width + tx] = pValue;
    }

    void MatrixMultiplyOnDevice(float* hostP, 
                                                            const float* hostM, 
                                                            const float* hostN, 
                                                            const int width
                                                        )
    {
        int sizeInBytes = width * width * sizeof(float);
        float *devM, *devN, *devP;
        
        // Allocate M and N on device
        cudaMalloc((void**)&devM, sizeInBytes);
        cudaMalloc((void**)&devN, sizeInBytes);
        
        // Allocate P
        cudaMalloc((void**)&devP, sizeInBytes);

        // Allocate the dimensions
        dim3 threads(width, width);
        dim3 blocks(1, 1);
        
        // Copy M and N from host to device
        cudaMemcpy(devM, hostM, sizeInBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(devN, hostN, sizeInBytes, cudaMemcpyHostToDevice);
        
        // Launch the kernel
        MatrixMultiplyKernel<<<blocks, threads>>>(devM, devN, devP, width)
        
        // Copy P matrix from device to host
        cudaMemcpy(hostP, devP, sizeInBytes, cudaMemcpyDeviceToHost);
        // Free allocated memory
        cudaFree(devM);
        cudaFree(devN);
        cudaFree(devP);
    }
```

## Inter-node parallelism 

All that has been discussed previously is specific to parallelism implemented within a node on a cluster and so, is called i**ntra-node parallelism**. Since the memory is shared between the CPUs and each of them can have their own caches, synchronized using mutexes and executed in a multi-threaded manner, the previous approach is also called **Shared-Memory Parallelism**. However, when we speak of computing on several nodes in a cluster, the intra-node sync vanishes since now each node has its own memory which is separate from the other nodes, and so this is called **Inter-Node Parallelism**. Moreover, now the synchronization cannot happen through the shared memory approach and is implemented by passing messages between nodes → **Message Passing Parallelism →** and so, the thing that is central here is **a deadlock.**

<img width=500 height=250 src="static/Clouds/msg-psng.png">


### Message Passing Interface (MPI)

MPI is a library standard defined by a committee of vendors, implementers, and parallel programmers that is used to create parallel programs based on message passing. It is Portable and the De-facto standard platform for the High-Performance Computing (HPC) community. The 6 basic routines in MPI are :

1. `MPI_Init` : Initialize 
2. `MPI_Finalize` : Terminate : 
3. `MPI_Comm_size` : Determines the number of processes 
4. `MPI_Comm_rank` : Determines the label of calling process
5. `MPI_Send` : Sends an unbuffered/blocking message
6. `MPI_Recv` : Receives an unbuffered/blocking message.

**MPI Communicators** define the communication interface over MPI and are used by the message passing functions. The prototypes of each of the above functions are shown below:

```cpp
1. int MPI_Init(int *argc, char ***argv)
2. int MPI_Finalize()
3. int MPI_Comm_size(MPI_Comm comm, int *size)
4. int MPI_Comm_rank(MPI_Comm comm, int *rank
5. int MPI_Send(void *buf, int count, MPI_Datatype datatype,int dest, int tag, MPI_Comm comm)
6. int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)  
```

MPI also provides a Send and Receive function that helps in avoiding deadlock through handshakes, and other functions for scattering and broadcast.

1. `MPI_Sendrecv` : Send and Recieve in one-shot 
2. `MPI_Bcast` : Broadcast same data to all processes in a group
3. `MPI_Scatter` : Send different pieces of an array to different  processes through partitioning
4. `MPI_Gather` : Take elements from many processes and gather them to one single process

There are two other important functions that help in reduction sums: 

1. `MPI_Reduce` : Takes an array of input elements on each process and returns an array of output elements to the root process given a specified operation 
2. `MPI_Allreduce`: Like `MPI_Reduce` but distribute results to all processes

As an example, we can use the `MPI_Allreduce` for numerical Integration and it gives a massive advantage, even compared to the vanilla Multi-threading:

<img width=600 height=300 src="static/Clouds/PDC/MPI-AR.png">

The advantages offered by MPI, clearly, are in the allocation of resources that it allows through explicit and direct communication with the resources. Thus, it is very useful in the scientific domain for HPC applications. However, it has certain weaknesses:
1. it requires careful tuning of applications 
2. It is not tolerant to variability
3. Dealing with failures is hard → No way to save information for later use


<!-- %%% -->
# Clouds: Basics of Cloud Technologies

The best way to look at the development of the cloud is to look at the lifecycle of major utilities throughout history. Take the case of water, initially, the people procured water themselves which was very intensive in terms of effort and time. However, models were developed to separate the process of procurement of water and its usage. Thus, the market moved towards some players procuring water and delivering it to the populace who could use it. However, this also went ahead and developed into a system where water was delivered through pipelines and a user would be charged on a pro-rata basis, depending on their usage. The same thing happened with electricity. This trend can be generalized to the lifecycle shown in the figure below:

<img width=500 height=200 src="static/Clouds/general-cycle.png">

If we look at IT from the lens of this cycle, then the innovation phase would be the phase where new kinds of products and services were introduced into the market, the product phase would be when companies maintained these on a growing user base, and the service phase would be when companies started addressing the growing demand and user-base by trying to achieve economies of scale through the cloud.

## Defining Cloud Computing
Cloud Computing can be defined in the following three ways:
1. It is the delivery of computing as a service rather than a product
2. It is a method to offer shared resources, software, and information to computers and other devices
3. It is a metered service over the network

### IT as a Service
There are 3 primary ways in which IT as a service can be offered:
1. **Software-as-a-Service (SaaS):** These are applications running on a browser
2. **Platform-as-a-Service (Paas):** These are software platforms made available to developers through APIs, to build applications
3. **Infrastructure-as-a-Service (Iaas):** These are basic computing resources like CPU, Memory, Disk, etc. made  available to users in the form of Virtual Machine Instances

Some other models that are also possible are: 
 - **Hardware-as-a-Service (Haas):** where users can get access to barebones hardware machines, do whatever they want with them (E.g clusters)
 - **X-as-a-service (Xaas):** which might extend to Backend, Desktop, etc.

### Cloud Infrastructure
Servers are computers that provide service to user machines - the client - and the main idea behind these is that they can be designed for reliability and to service a high number of requests. Dual socket servers are the fundamental building blocks of cloud infrastructure. Organizations usually require many physical servers, like a web server or database server, to provide various services. These servers are grouped, organized, and placed into racks. For standardization, 1 Rack Unit (RU) is defined as 4.45 cm.

<img width=500 height=200 src="static/Clouds/RU.png">

A data center is a facility that is used to house a large number of servers. It needs to provide Air-Conditioning to cool the servers, Power supply to all the servers and needs to implement monitoring, network, and security mechanisms for these servers. Now the companies all have the option of privately owned data centers, but these are certain problems associated with this:
- These are expensive to set-up with a high CAPEX for real-estate, servers, and peripherals
- They have high OPEX in energy and administration costs
- It is difficult to grow or shrink applications → If the company initially budgets a small number of servers, and then a demand surge happens, sometimes even overnight for companies like FaceApp, they would have to expand the area abruptly, which is very difficult. Now, let us say they are able to expand the area and resource pool, they would not be able to shrink these if they demand tapers off. These things are simply not possible for smaller companies, as much as they are for bigger companies like Dropbox.
- Servers can also suffer from the problem of low utilization. This can be caused by uneven usage of applications, where one application might be exhausting one resource while leaving the others stranded-off. Another reason for this is sudden demand spikes, which taper off even more suddenly

Thus, the idea behind cloud infrastructure is to alleviate these problems by separating the server infrastructure from the end-users. The servers can be grouped into a large resource pool and then access can be given to applications based on their demand and the pricing can be set-up based on the usage of this resource pool. Hence, the applications don't need to worry about the usage statistics as far as to look into load balancing. Moreover, the sudden demand spikes and shrinks can be easily adjusted by changing the user requests. However, to offer such a service two requirements need to be met:
- A Means for rapidly and dynamically satisfying fluctuating resource need of the application → provided by **Virtualization**
- A Means for servers to Quickly and reliably access shared and persistent data → done by programming models and distributed file/storage/database systems

This resource pool can also be defined based on its location:
- **Single-Site Cloud :** This would be the collection of hardware and software that the vendors use to offer computing resources and services to users.
- **Geographically Distributed Cloud :** This is a resource pool that is spread across multiple locations and has a composition of different structures and services.

### Cloud Hardware and Software stack
The full stack for clouds has 9 components, as shown in the figure below:

<img width=200 height=300 src="static/Clouds/cloud-stack.png">

- **Applcations :** These are applications like Web-apps or Scientific Computation Jobs etc.
- **Data :** These are the database systems like Old SQL (Oracle, SQLServer), No SQL (MongoDB, Cassandra), and New SQL (TimesTen, Impala, Hekaton) systems.
- **Runtime Environment :** These are runtime platforms like Hadoop, Spark, etc. to support cloud programming models.
- **Middleware :** These are platforms for Resource Management, Monitoring, Provisioning, Identity Management, and Security.
- **Operating Systems :** These are operating systems like Linux used on a personal machine, but they can also be packed with libraries and software for quick deployment. For example, Amazon Machine Images (AMI) contain OS as well as required software packages as a “snapshot” for instant deployment.
- **Virtualization :** This layer is the key enabler of the cloud services. It creates a mapping between the lower hardware layers and the upper applications and OS layers and contributes towards multi latency. For example, the Amazon EC2 is based on the Xen virtualization platform, and Microsoft Azure is based on HyperV.

The stuff below virtualization has already been discussed. However, one thing that can now be understood is how does this stack help in differentiating between the offered services. As shown in the figure below, in the case of Saas the user has only access to the applications offered by the cloud. In the case of Paas, the user manages the application and Data layer of the stack. In the case of Iaas, the user has access to all the layers above the virtualization layer, so that they can build their own application on the offered resources.

<img width=800 height=300 src="static/Clouds/stack-resources.png">

### Types of Cloud
There are three basic types of clouds:
1. **Public (external) Cloud :**  This is a resource pool that serves as an open market for on-demand computing and IT resources. However, the availability, reliability, security, trust, and SLAs can have limitations.
2. **Private (Internal) Cloud :** This is the same set of services of cloud, but devoted to the functions of a large enterprise with the budget of large-scale IT.
3. **Hybrid Cloud :** This is the best of both worlds. The private cloud is extended by connecting it to other public cloud vendors to make use of their available cloud services. So, a company can use their private cloud, and when the resources surge they can also extend usage to the public cloud, of course paying pro-rata.

### Applications Enabled by the Cloud
The applications that can be enabled by the cloud are of 4 types
1. **High-Growth Applications:** This the same case as FaceApp that was discussed previously. Imagine a startup that is growing. They would need a dynamic resource usage mechanism, that as discussed previously, is comfortably offered by the cloud. The risk of not setting up a distributed resource management method is losing on customer experience. This was the case with Friendster(2001), which had a similar offering as Facebook but could not keep up with the user growth.
2. **Aperiodic Applications:** These are applications that face sudden demand peaks and need a way to handle this. The cloud enables them comfortably, and again the risk is user experience. For example, Flipkart offered the 'Big-Billion Day' sale in a similar manner to Amazon's Prime Day, but initially, they could not handle the load and the customer experience was ruined. However, they did fix it over time.
3. **On-off Applications:** These are one-off applications for which extending private resources makes no sense. for example, scientific simulations requiring 1000s of computers.
4. **Periodic Applications:** These are applications that will have a periodic demand surge, like stock market analysis tools or HFT tools, and thus, dynamic, flexible infrastructure can reduce costs, improve performance.

### Advantages Offered by Cloud Computing
1. Pay-as-you-go economic model
2. Simplified IT Management
3. Quick adn Effortless scalability
4. Flexible options
5. Improved Resource Utilization
6. Decrease in Carbon Footpriint


## Cloudonomics
In 2012, Joe Weinman came up with the economic theory to estimate the business value of cloud computing, calling it **Cloudonomics**. The major benefits of the cloud that come out are the following:
1. Common Infrastructure
2. Location Independence
3. Online connectivity
4. Utility pricing

### Utility Pricing Calculation
To understand how utility pricing allows cloud services to be advantageous, we look at the load and the related quantities as follows:
- **L(t) →** Load demand as a function of time, with T being the total time
- **P →** maximum load or peak load
- **A  →** Average load
- **B  →** Baseline cost i.e the cost associated with owning the infrastructure
- **C →** Cloud unit cost i.e cost per second incurred when using a cloud service
- **U →** Utility Premium = C / B

Now, when we measure the costs for a time period of T, then we get:
$$
\begin{alignedat}{2}
&B_T = P.B.T \\
&C_T = \int L(t)dt = A.U.B.T \\
\end{alignedat}
$$

The condition for the cloud services to be cheaper is that the aggregated cost of using the cloud is less than the cost of owning the service i.e 

$$
C_T < B_T
$$

When combined with the above equations, we get the condition as :
$$
U < \frac {P}{A}
$$

Thus, by checking if the utility premium is less than the peak-to-average ratio it can be determined whether the cloud is beneficial or not. 

## Value Created by Cloud

The value that the cloud provides is through the following two methods:

1. Resource Pooling: When resources are shared between multiple services, the profit can be made in reducing the overhead of setting up the infrastructure (like cooling facility, etc.) and economies of scale that come with exploiting synergies.
2. Multiplexing: By multiplexing services over time, the benefit comes from building the infrastructure for handling peak and average loads.

### Measuring the benefit of Multiplexing: Smoothness

The figure below shows the activity profile of a sample of 5,000 Google Servers over a period of 6 months:

<img width=500 height=300 src="static/Clouds/Google-sample.png">

The way multiplexing helps here is twofold:

1. For the part that is built to handle peak load, it yields higher utilization and lowers costs per resource
2. For the part build to handle less than peak load, it reduces the unserved requests and penalties associated with them on the off chance that service level agreements are violated.

To understand how multiplexing does this, the metric used is the **smoothness** of the load. This is measured by a load variation coefficient defined as follows:

$$
C_v = \sigma / | \mu |
$$

Here, $\sigma$ is the standard deviation of the load variation and μ is the mean of this standard deviation. This coefficient is always non-negative since we are taking the modulus of the mean, and so when its value is closer to 1 the load is smoother since this either happens with a lower standard deviation or with a higher mean. Now, let's take the case of n independent jobs $X_1, X_2, ..., X_n$ running with the same values for the mean and standard deviation. Thus, when we multiplex them, we get: 

$$
\begin{aligned}
&X = X_1 + X_2 + ... + X_n \\
&\mu(X) = n*\mu \\ 
&Var(X) = n*Var(X_i) \implies \sigma(X) = \sqrt{n} \sigma \\
\end{aligned}
$$

Thus, the  coefficient for the multiplexed variable comes out to be
$$
C_v(X) = \frac {1} {\sqrt{n}} C_v(X_i)
$$

Hence, by multiplexing the load variation scales down proportional to the number of jobs that are multiplexed! The ideal scenario is when two jobs are negatively correlated, in which case $ X_2 = 1 - X_1$ and we get a deviation of 0, which leads to a flat curve.

## Virtualization

The key idea behind virtualization is sharing computing resources among multiple applications. This translates to mapping the key components to abstract counterparts i.e CPU to a virtual CPU, Disk to a virtual disk, NIC to virtual NIC, etc., to create a **Virtual Machine** that can be used in the place of a real machine. Through this, each tenant can be provided with a virtual machine that they can use to access the compute resources, and thus, multiple tenants can be hosted, as shown below:

<img width=400 height=300 src="static/Clouds/VM-Arch.png">

This mapping is created through a **Virtual Machine Monitor (VMM)** , also called a **Hypervisor**, which can be of two types :
- **Type 1:** VMM runs directly on the hardware, and performs scheduling and allocation of resources. E.g. VMWare ESX Server.
- **Type 2:** VMM is built completely on top of an OS where the host OS provides the resource allocation and standard execution environment. E.g User-mode Linux (UML), QEMU.

### How it works 

The CPU has the **Instruction Set Architecture (ISA)** which defines the registers and the memory available to the user and the operations that can be used to modify the contents of these. The ISA has 2 parts:

1. **User ISA:** This is used for computation and has the fetch, decode, etc. instruction that can modify the user virtual memory, but it cannot modify the kernel
2. **System ISA:** This is controlled through **privilege** and used for resource management of the kernel. It can modify the actual registers, can set traps, and interrupts and modify the Memory Management Unit.

Virtualization creates an isomorphism between the ISA on the machine and the virtual system provided to the user by emulating the commands entered on the VM on the ISA on the host machine. This decoupling allows controlling what multiple users can modify by abstracting that bit out into the VM that is provided to them. This emulation is done by encapsulating the instruction set on the host machine into a set of commands that can be executed on the guest machine and creating a schema that maps these commands from the guest machine to the host machine. There are three ways to do this, each one addressing a problem in the previous approach:

1. **Exact Mapping:** The most basic way is to create a 1-1 mapping between each command. This is exhaustive and easy to implement but can be extremely slow due to the interpretation overhead that comes with it.
2. **Trap and Emulate:** The key realization in this approach is only the instructions written to the system ISA need to be interpreted and 'worked around'. Thus, we let the user ISA instructions run as they are and every time the command to the kernel is accessed, the system will generate an interrupt which can be caught (trap) and handled by rewriting them by an interpreter in the privileged mode (emulate). The issue with this approach is that not all architectures (For example, x86) trap the attempts to write to the privileged mode from unprivileged access.
3. **Binary Translation:** Here we translate each guest instruction to the minimal binary set of host instructions required to emulate it, thus avoiding the function-call overhead of an interpreter. We can also re-use translations by using a translator cache. However, this is still slower than direct execution.

In the [DISCO Approach](https://dl.acm.org/doi/10.1145/268998.266672) :

- trap-and-emulate for the non-privileged part of the guest instruction set
- binary translation for the privileged part.

### Containers

Containers raise the abstraction to another level by virtualizing over the OS as shown: 

<img width=400 height=300 src="static/Clouds/Containers.png">

The key benefits come to the hosting providers:

1. It is now possible to host multiple applications/tenants on a single server as containers work on an OS abstraction level as compared to the hypervisors that work on the hardware abstraction level
2. They offer high density as multiple containers can be packed in a server
3. They are easy to scale-up (Everything in google is containerized)
4. There is no virtualization overhead
5. They reduce multitenancy and license fee that comes with providing the OS and libraries for every application
6. They dramatically improve the SDLC

The key point here is to find a way to extend OS to securely isolate multiple applications by observing and controlling the resource allocation and limiting visibility and communication across and between multiple processes. This was first done in Linux through Control Groups (CGroups) and Namespaces, which allowed multiple Linux distributions to share the same kernel (LXC). Thus, apart from the Linux kernel, multiple applications running on RHEL, Debian, Ubuntu, etc. could be isolated.

**Docker** was the obvious next step that has primarily two functions:

1. **Package System :** Can pack an application and all dependencies as a container image after development
2. **Transport System:** Ensures that the application image runs exactly similar on test and production systems

Thus, with Docker one can package everything from libraries to applications, and till the time the kernel is shared, it can be run on multiple devices, servers, etc.

### Serverless Computing

The idea here is to abstract even above OS and allow multiple applications to share the server and runtime.

<img width=350 height=200 src="static/Clouds/Serverless.png">

The model is primarily event-driven and can be described as follows:

1. The developer develops business logic and provides it to a provider (like amazon) which encapsulates this in the form of functions (FaaS)
2. Whenever a client requests a function through the application, a notification is triggered by a listener
3. The server tries to locate the code that is responsible for answering the request
4. Only the relevant bit of code is loaded into a container which then executes the code
5. The result of the execution is used to build a response which is then sent to the client

The way the listener works is through using the backend for authentication as a separate service. The advantages of the serverless approach are:

- Less server-side work
- Reduced Cost that comes through being able to use a pay-as-you-go model and economies of scale
- Reduced risk and increased efficiency through specialization
- Scalability
- Shorter lead time

The limitations of this approach are:

- Managing the state is relatively complex
- Higher latency due to increased calls
- Vendor lock-in due to control shifted to the providers, but this might change as more providers enter the market.


<!-- %%% -->
# RL: Policy Gradients

The core idea of Reinforcement learning is to learn some kind of behavior through optimizing for rewards. The behavior learned by an agent i.e. the schema it follows while going through this process is the learned policy that it uses to decide which action to take and thus, the transition from one state to another. One way to close the loop for the agent to learn is by evaluating the states and actions through value functions and thus, our way to measure the learned policy is seen through these value functions, approximated by lookup tables, linear combinations, Neural Networks e.t.c. Policy Gradient methods take a different approach where they bypass the need for a value function by parameterizing the policy directly. While the agent can still use a value function to learn, it need not use it for selecting actions. The advantages that policy gradient methods offer are 3 fold: 

1. Approximating the policy might be simpler than approximating action values and a policy-based method might typically learn faster and yield a superior asymptotic policy. One very good example that illustrates this is the work by [Simsek et. al](http://proceedings.mlr.press/v48/simsek16.pdf) on the game of Tetris where they showed that it is possible to choose amongst actions without really evaluating them.
2. Policy gradient methods can handle stochastic policies. The case of card games with imperfect information, like poker, is a direct example where the optimal play might be to do 2 different things with certain probabilities. If we are maximizing the actions based on value approximations, we don't really have a natural way of finding stochastic policies. Policy Gradient methods can do this.
3. Policy gradient methods offer stronger convergence guarantees since with continuous policies the action probabilities change smoothly. This is not the case with the fixed $\epsilon$ -greedy evaluation since there is always a probability to do something random.  
4. The choice of policy parameterization is sometimes a good way of injecting prior knowledge about a desired form of the policy into the system. This is especially helpful when we look at introducing Meta-Learning strategies into Reinforcement Learning.

In the following sections, I first use the theoretical treatment done in Sutton and Barto's book since I was better able to understand policy gradients' essence through this. However, I again do the derivation by looking at the whole thing from the viewpoint of trajectories, since I find it more intuitive.

## Policy Gradient Theorem

The issue with the parameterization of the policy is that the policy affects both the action selections and the distribution of states in which those selections are made. While going from state to action is straightforward, going the other way round involves the environment and thus, the parameterization is typically unknown. Thus, with this unknown effect of policy changes on the state distributions, the issue is evaluating the gradients of the performance. This is where the policy gradient theorem comes into the picture, as it shows that the gradient of the policy w.r.t its parameters does not involve the derivative of the state distribution. For episodic tasks, if we assume that every episode starts in some particular state $s_0$, then we can write a performance measure as the value of the start state of the episode

$$J(\bm{\theta}) = v_{\pi_\theta} (s_0)$$

For simplicity, we remove the $\bm{\theta}$  from the subscript of $\pi$ . Now, to get a derivative of this measure, we start by differentiating this value function w.r.t $\bm{\theta}$:

 

$$\begin{aligned}
\nabla_{\bm{\theta}} J(\bm{\theta}) & =  \nabla v_\pi (s) \\
& = \nabla \bigg [ \sum_{a \in \mathcal{A}}\pi(a|s) q_\pi(s,a)   \bigg ] \\
& = \sum_a \bigg[ \nabla\pi(a|s) q_\pi (s, a)  + \pi(a|s) \nabla q_\pi(s,a)    \bigg ] \\
& = \sum_a \bigg[ \nabla\pi(a|s) q_\pi (s, a)  + \pi(a|s) \,\, \nabla \sum _{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a) ( r + v_\pi (s') )     \bigg ]
\end{aligned}$$

we now extend $q_\pi (s,a)$  in the second term on the right to the rollout for the new state $s'$

$$
\nabla v_\pi (s) = \sum_a \bigg[ \nabla\pi(a|s) q_\pi (s, a)  + \pi(a|s) \,\, \nabla \sum _{s' \in \mathcal{S}, r \in \mathcal{R}} p(s', r | s, a) ( r + v_\pi (s') ) \bigg ] $$

The reward is independent of the parameters $\theta$, so we can set that derivative inside the sum to 0, and so, we get:

$$
\nabla v_\pi (s) = \sum_a \bigg[ \nabla\pi(a|s) q_\pi (s, a)  + \pi(a|s) \,\, \nabla \sum _{s' \in \mathcal{S} } p(s' | s, a) v_\pi (s') \bigg ] $$

Thus, we now have a recursive formulation of $\nabla v_\pi (s)$  in terms of $\nabla v_\pi (s')$ . To calculate this derivative in the infinite horizon episodic case we just need to unroll this infinitely many times, which can be written as 

$$\sum_{x \in \mathcal{s}} \sum _{k=0}^\infty P(s \rightarrow x, k , \pi )  \sum_{a \in \mathcal{A}} \nabla \pi (a|s)  q_\pi (x, a) $$

Here, $P(s \rightarrow x, k, \pi )$  is the probability of transitioning from state $s$ to state $x$ in $k$  steps under policy $\pi$. To estimate this probability we use something called the stationary distribution of the Markov chains. This term comes from the [Fundamental Theorem of Markov Chains](http://www.math.uchicago.edu/~may/VIGRE/VIGRE2008/REUPapers/Plavnick.pdf) which intuitively says that in very long random walks the probability of ending up at some state is independent of where you started. When we club all these probabilities into a distribution over the states, then we have a stationary distribution, denoted by $\mu(s)$ . In on-policy training, we usually estimate this distribution by the fraction of time spent in a state. In our case of episodic tasks, if we let $\eta(s)$  denote the total time spent in a state in an episode, then we can calculate $\mu(s)$  as 

$$\mu(s) = \frac{\eta(s)}{\sum_s \eta(s)}$$

In our derivation, $P(s \rightarrow x, k, \pi )$  for very long walks can be estimated by the total time spent in the state $s$. Thus, we can inject $\eta(s)$  into our equation as follows:

$$\begin{aligned}
\nabla_{\bm{\theta}}J(\bm{\theta})  & = \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla\pi(a|s)  q_\pi(s,a) \\
& = \sum_{s \in \mathcal{S}} \eta(s) \sum_{s \in \mathcal{S}} \frac{\eta(s)}{\sum_{s \in \mathcal{S}} \eta(s)} \sum_{a \in \mathcal{A}} \nabla\pi(a|s)  q_\pi(s,a) \\
& \propto \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}} \nabla\pi(a|s)  q_\pi(s,a) 
\end{aligned}$$

Thus, we get the form of the theorem as 

$$\nabla_{\bm{\theta}}J(\bm{\theta}) \propto \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}}  q_\pi(s,a) \nabla\pi(a|s) $$

The proportionality is the average length of an episode. In the case of continuous tasks, this is $1$. Thus, now we see that we have a gradient over a parameterized policy, which allows us to move in the direction of maximizing this gradient i.e gradient ascent. We can estimate this gradient through different means.

## REINFORCE: Monte-Carlo Sampling

For Monte-Carlo sampling of the policy, our essential requirement is sampling from a distribution that allows us to get an estimate of the policy. From the equation of the policy gradient theorem, we can write this again as an expectation over a sample of states $S_t \sim s$ in the direction of the policy gradient

$$\nabla_{\bm{\theta}}J(\bm{\theta})  = \mathbb{E}_\pi \bigg [\sum_{a \in \mathcal{A}}  q_\pi(S_t,a) \nabla_{\bm{\theta}} \pi(a|S_t, \bm{\theta}) \bigg ]$$

The expectation above would be an expectation over the actions if we were to include the probability of selecting the actions as the weight. Thus, we can do that to remove the sum over actions too:

$$\begin{aligned}
\nabla_{\bm{\theta}}J(\bm{\theta})  & = \mathbb{E}_\pi \bigg [\sum_{a \in \mathcal{A}}  q_\pi(S_t,a) \nabla_{\bm{\theta}} \pi(a|S_t, \bm{\theta}) \frac{\pi(a|S_t, \bm{\theta})}{\pi(a|S_t, \bm{\theta})} \bigg ] \\
& = \mathbb{E}_\pi \bigg [\sum_{a \in \mathcal{A}}   \pi(a|S_t, \bm{\theta}) q_\pi(S_t,a) \frac{\nabla_{\bm{\theta}} \pi(a|S_t, \bm{\theta})}{\pi(a|S_t, \bm{\theta})} \bigg ] \\
& =\mathbb{E}_\pi \bigg [ q_\pi(S_t,A_t) \frac{\nabla_{\bm{\theta}} \pi(A_t|S_t, \bm{\theta})}{\pi(A_t|S_t, \bm{\theta})} \bigg ] 
\end{aligned}$$

The expectation over $q_\pi (S_t, A_t)$  is essentially the return $G_t$. Thus, we can replace that in the above equation to get: 

$$\nabla_{\bm{\theta}}J(\bm{\theta})  = \mathbb{E}_\pi \bigg [ G_t\frac{\nabla_{\bm{\theta}} \pi(A_t|S_t, \bm{\theta})}{\pi(A_t|S_t, \bm{\theta})} \bigg ] $$

We now have a full sampling of the states and actions conditioned on our parameters in the gradients. This can be considered a sample from the policy and we can update our parameters using this quantity to get our update rule as: 

$$\begin{aligned}
\bm{\theta}_{t+1} &= \bm{\theta}_t + \alpha \nabla_{\bm{\theta}}J(\bm{\theta}) \\ & =\bm{\theta}_t + \alpha \bigg ( \, G_t\frac{\nabla_{\bm{\theta}} \pi(A_t|S_t, \bm{\theta})}{\pi(A_t|S_t, \bm{\theta})} \bigg )
\end{aligned}$$

This is the REINFORCE Algorithm! We have each update which is simply the learning rate $\alpha$ multiplied by a quantity that is proportional to the return and a vector of gradients of the probability of taking a certain action in a state. From the gradient ascent logic, we can see that this vector is the direction of maximizing the probability of taking action $A_t$ again, whenever we visit $S_t$. Moreover, The update is increasing the parameter vector in this direction proportional to the return, and inversely proportional to the action probability. Since the return is evaluated till the end of the episode, this is a Monte-Carlo Algorithm. We can further refine this using the identity of log differentiation and adding the discount factor to get the update as:

$$\bm{\theta}_{t+1} = \bm{\theta} + \alpha \gamma^t G_t \nabla_{\bm{\theta}} \ln \pi(A_t| S_t , \bm{\theta}) $$

## Looking at Trajectories

Another way to look at the above formulation is through sampled trajectories, as done in [Sergey Levine's slides](http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-5.pdf). While theoretically, this treatment is similar to the one done above, I just find the notation more intuitive. Recall, a trajectory is a sequence of states and actions over time, and the rewards accumulated in this sequence qualify this trajectory. Thus, we can say that the utility objective $J(\bm{\theta})$  is the sum of the accumulated rewards of some number of trajectories sampled from a policy $\pi_\theta$ 

$$J(\bm{\theta}) = \mathbb{E}_{\tau \sim \pi_{\bm{\theta}} (\tau)} \bigg [  \sum_t r(s_t, a_t)  \bigg ] \approx \frac{1}{N} \sum_i \sum_t r(s_{i:t}, a_{i:t}) $$

Let this sum of reward be denoted by $G(\tau)$  for a trajectory $\tau$ . Thus, we can re-write the above equation as

$$J(\bm{\theta}) =  \mathbb{E}_{\tau \sim \pi_{\bm{\theta}}(\tau) } \big [ G(\tau)  \big ] $$

This expectation is essentially sampling a trajectory from a policy and weighing it with the accumulated rewards. Thus, we can write it as 

$$J(\bm{\theta}) =  \int \pi_{\bm{\theta}} (\tau) G(\tau) d\tau   $$

Now, we just differentiate this objective, and add a convenient policy term to make it an expectation:

$$\begin{aligned}
\nabla_{\bm{\theta}} J(\bm{\theta})  & = \nabla_{\bm{\theta}} \int \pi_{\bm{\theta}} (\tau) G(\tau) d\tau   \\
& = \int \nabla_{\bm{\theta}}  \pi_{\bm{\theta}} (\tau) G(\tau) d\tau \\
& = \int \pi_{\bm{\theta}}(\tau) \frac{\nabla_{\bm{\theta}}  \pi_{\bm{\theta}}(\tau) }{\pi_{\bm{\theta}}(\tau)} G(\tau) d\tau

\end{aligned}$$

Now, we just use an identity $\frac{dx}{x} = d \log x$ and get

$$\nabla_{\bm{\theta}} J(\bm{\theta})   = \int \pi_{\bm{\theta}} (\tau) \nabla_{\bm{\theta}}  \log \pi_{\bm{\theta}} (\tau ) G(\tau) d\tau   $$

Thus, we can write this as:

$$\nabla_{\bm{\theta}} J(\bm{\theta}) = \mathbb{E}_{\tau \sim \pi_{\bm{\theta}} (\tau)} \big [ \nabla_{\bm{\theta}}  \log \pi_{\bm{\theta}} (\tau ) G(\tau)   \big ] $$

Hence, we get the same final result as before: the gradient depends on the gradient of the log policy weighted by the rewards accumulated over the trajectory. Now, we can translate this to states and actions over the trajectory by simply considering what the policy of the trajectory represents: 

$$\begin{aligned}
& \pi_{\bm{\theta}} (s_1, a_1, ..., s_T, a_T) = \mu(s) \prod_{t=1}^{T} \pi_{\bm{\theta}}(a_t|s_t)p(s_{t+1}|s_t, a_t) \\
\implies & \log \pi_{\bm{\theta}} (s_1, a_1, ..., s_T, a_T) = \log \mu(s) + \sum_{t=1}^{T} \log \pi_{\bm{\theta}}(a_t|s_t) + \log p(s_{t+1}|s_t, a_t)
\end{aligned}$$

When we differentiate this log policy w.r.t $\bm{\theta}$, we realize that $\mu(s)$  and $p(s_{t+1}|s_t, a_t)$ do not depend on this parameter, and would be set to 0. Thus, our utility expression would end up looking like

$$\nabla_{\bm{\theta}} J(\bm{\theta}) = \mathbb{E}_{\tau \sim \pi_{\bm{\theta}} (\tau)} \big [  \sum_{t=1}^{T} \nabla_{\bm{\theta}} \log \pi_{\bm{\theta}}(a_t|s_t) \sum_{t=1}^{T}r(s_t, a_t)   \big ] $$

Then we take average of the samples as the expectation, we get 

$$\nabla_{\bm{\theta}} J(\bm{\theta}) \approx \frac{1}{N} \sum _{i=1}^N \bigg [  \sum_{t=1}^{T} \nabla_{\bm{\theta}} \log \pi_{\bm{\theta}}(a_t|s_t) \sum_{t=1}^{T}r(s_t, a_t)   \bigg ] $$

And this is the key formula behind REINFORCE again and the update can thus, be written as

$$\bm{\theta} \leftarrow \bm{\theta} + \alpha \nabla_{\bm{\theta}} J(\bm{\theta}) $$

This is where I find it more intuitive to just use $\tau$ for trajectories since now we can just write our REINFORCE algorithm as : 

1. Sample a set of trajectories $\{ \tau ^i\}$ from the policy $\pi_{\bm{\theta}}(a_t|s_t)$ 
2. Estimate $\nabla_{\bm{\theta}} J(\bm{\theta})$ 
3. Update the parameters $\bm{\theta} \leftarrow \bm{\theta} + \alpha \nabla_{\bm{\theta}} J(\bm{\theta})$


## Reducing Variance

The idea of parameterizing the policy and working directly with the sampled trajectories has an intuitive appeal due to its clarity. However, this approach suffers from high variance. This is because when we compute the expectation over trajectories, we are essentially sampling $N$ different trajectories and then taking the average of the accumulated rewards. If we take this sampling to be uniform, we can easily imagine scenarios, where the trajectories sampled, have wildly different accumulated rewards. Thus, the chances of getting a high variance in the values that we are averaging over increase. If we were to scale $N$  to $\infty$  then our average becomes closer to the true expectation. However, this is computationally expensive. There are multiple ways to reduce this variance.

### Rewards to go

In the trajectory formulation, we are accumulating all of the rewards from $t=0$ to $t = N$. However, one way to make this online would be to just consider the rewards from the timestep at which we take the policy log value till the end of the horizon. This can be written as

$$\nabla_{\bm{\theta}} J(\bm{\theta}) \approx \frac{1}{N} \sum _{i=1}^N \bigg [  \sum_{t=1}^{T} \nabla_{\bm{\theta}} \log \pi_{\bm{\theta}}(a_t|s_t) \sum_{t'=t}^{T}r(s_{t'}, a_{t'})   \bigg ] $$

Thus, by reducing the number of  rewards we consider for each policy evaluation, we are essentially better able to control the variance up to a certain extent

### Baselining

Another way to control the variance is to realize that the actions in a state are the quantities that create a variance for each state in trajectory. However, we see that our return $G(\tau)$  is dependent on both states and actions. Thus, if we could compare this value to a baseline value $b(s)$  , we eliminate the variance resulting from fixed state selection. In the policy gradient theorem, this would look like

$$\nabla_{\bm{\theta}}J(\bm{\theta}) \propto \sum_{s \in \mathcal{S}} \mu(s) \sum_{a \in \mathcal{A}}  \big ( q_\pi(s,a) - b(s)  \big )  \nabla\pi(a|s) $$

since $b(s)$  does not vary with actiosn, it would not have an effect on our summation since its sum over all actions would be 0:

$$\begin{aligned}
\sum_a b(s) \nabla_{\bm{\theta}}\pi(a|s, \bm{\theta}) & = b(s) \nabla_{\bm{\theta}} \sum_a \pi(a|s, \bm{\theta}) \\
& = b(s) \nabla_{\bm{\theta}} 1 \\
& = 0
\end{aligned}$$

Thus, we can effectively update our REINFORCE update with this baseline to get

$$\bm{\theta}_{t+1} =   =\bm{\theta}_t + \alpha \big ( \, G_t - b(s) \big ) \frac{\nabla_{\bm{\theta}} \pi(A_t|S_t, \bm{\theta})}{\pi(A_t|S_t, \bm{\theta})}$$

Or, in the trajectory formulation 

$$\bm{\theta}_{t+1} =   =\bm{\theta}_t + \alpha \nabla_{\bm{\theta}}  \log \pi_{\bm{\theta}} (\tau ) \big (G(\tau)  - b(s) \big ) $$

One good function for $b(s)$ could be the estimate of the state value $\hat{v}(s_t, \bm{w})$ . Doing something like this may seem like going to the realm of actor-critic methods, where we are parameterizing the policy and using the value function, but this is not the case here since we are not using the value function to bootstrap. We are stabilizing the variance by using the estimate of the value function as a baseline. Baselines are not just limited to this. We can inject all kinds of things into the baseline to try scaling up our policy gradient. For example, an [interesting paper](https://arxiv.org/abs/2102.10362) published recently talks about using functions that take into account causal dependency as a Baseline. There are many other extensions like Deterministic Policy Gradients, Deep Deterministic Policy Gradients, Proximal Optimization e.t.c that look deeper into this problem.



<!-- %%% -->
# RL: Model-Free Control

While prediction is all about estimating the value function in an environment for which the underlying MDP is not known, Model-Free control deals with optimizing the value function. While many problems can be modelled as MDPs, in a lot of problems we don't really have that liberty in some sense. The reasons why using an MDP to model the problem might not make sense are: 

- MDP is unknown → In this case we have to sample experiences and somehow work with samples.
- MDP is known, but too complicated in terms of space and so, we again have to rely on experience

We can classify the policy learning process into two kinds based on the policy we learn and the policy we evaluate upon: 

- **On-Policy Learning** → If we learn about policy $\pi$ from the experiences sampled from $\pi$ , then we are essentially learning on the job.
- **Off-Policy Learning** → If we use a policy $\mu$  to sample the experiences, but our target is to learn about policy $\pi$ , then we are essentially seeing someone else do something and learning how to do something else through that


## Generalized Policy Iteration

As explained before our process of learning can also be broken down into 2 stages: 

1. **Policy Evaluation** → Iteratively estimating $v_\pi$ throught the samled experiences a.k.a iterative policy evaluation 
2. **Policy Improvement →**  Generating a policy $\pi' \geq \pi$  

The process of learning oscillates between these two states in sense → We evaluate a policy, then improve it, then evaluate it again and so on until we get the optimal policy $\pi^*$ and the corresponding optimal value $v^*$. Thus, we could see this as state transitions, as shown below: 

<img width=350 height=250 src="static/Reinforcement Learning/Model-Free-Control/mfc-1.png">

A really good way to look at convergence is shown below:

<img scale=1 src="static/Reinforcement Learning/Model-Free-Control/mfc-2.png">

Each process drives the value function or policy toward one of the lines representing a solution to one of the two goals. The goals interact because the two lines are not orthogonal. Driving directly toward one goal causes some movement away from the other goal. Inevitably, however, the joint process is brought closer to the overall goal of optimality. The arrows in this diagram correspond to the behavior of policy iteration in that each takes the system all the way to achieving one of the two goals completely. In GPI one could also take smaller, incomplete steps toward each goal. In either case, the two processes together achieve the overall goal of optimality even though neither is attempting to achieve it directly. Thus, almost all the stuff in RL can be described as a GPI, since this oscillation forms the core of it, and as mentioned before the ideal solution is usually out of our reach due to computational limitations, and DP has its set of disadvantages.

## On-Policy Monte-Carlo Control

Our iteration and evaluation steps are: 

- We use Monte-Carlo to estimate the value
- We use greedy policy improvement to get a better policy

Ideally, we can easily use the value function $v_\pi$ for evaluation and then improve upon it. However, if we look at the improvement step using $v_\pi$ , w have the equation

$$\pi'(s) = \argmax_{a \in \mathcal{A}} R^a_s + P_{ss'}^a V(s')$$

Here, to get the $P_{ss'}^a$ we need to have a transition model, which goes against our target of staying model-free. The action-value $q_\pi$, on the other hand, does not require this transition probability: 

$$\pi'(s) = \argmax_{a \in A} Q(s,a)$$

Thus, using $q_\pi$ allows us to close the loop in a model-free way. Hence, we now have our 2-step process that needs to be repeated until convergence as: 

- Iteratively Evaluate $q_\pi$  using Monte-Carlo methods
- Improve to $\pi' \geq \pi$  greedily

## Maintaining Exploration and Stochastic Strategies

Greedy improvement is essentially asking us to select the policy that leads to the best value based on the immediate value that we see. This suffers from the problem of maintaining exploration since many relevant state-action pairs may never be visited → If $\pi$  is a deterministic policy, then in following $\pi$ one will observe returns only for one of the actions from each state. With no returns to average, the Monte Carlo estimates of the other actions will not improve with experience. Hence,  we need to ensure continuous exploration. One way to do this is by specifying that the first step of each episode starts at a state-action pair and that every such pair has a nonzero probability of being selected as the start. This guarantees that all state-action pairs will be visited an infinite number of times within the limit of an infinite number of episodes. This is called the assumption of **exploring starts**. However, it cannot be relied upon in general, particularly when learning directly from real interactions with an environment. Thus, we need to look at policies that are stochastic in nature, with a nonzero probability of selecting all actions.

### $\epsilon$-greedy Strategy

One way to make deterministic greedy policies stochastic is to follow an $\epsilon$-**greedy strategy**  → We try all $m$ actions with non-zero probability, and choose random actions with a probability $\epsilon$, while maintaining a probability of $1- \epsilon$  for choosing actions based on the greedy evaluation. Thus, by controlling $\epsilon$ as a hyperparameter, we tune how much randomness our agent is willing to accept in its decision: 

$$\pi(a|s)  = \begin{cases}
\frac{\epsilon}{m} + 1 - \epsilon  \,\,\,\,\,\,\,\, if \,\,\, a^* = \argmax_{a \in A} Q(s,a) \\
\frac{\epsilon}{m}   \,\,\,\,\,\,\,\, otherwise

\end{cases}$$

The good thing is that we can prove that the new policy that we get with the $\epsilon$-greedy strategy actually does lead to a better policy: 

 

$$\begin{aligned}
q_\pi(s, \pi'(s)) & = \sum_{a \in \mathcal{A}} \pi'(a|s) q_\pi(s,a) \\
& = \frac{\epsilon}{m} \sum_{a \in \mathcal{A}} \pi'(a|s) q_\pi(s,a) +  ( 1- \epsilon) \max_{a \in \mathcal{A} } q_\pi(s,a) \\
& \geq \frac{\epsilon}{m} \sum_{a \in \mathcal{A}} \pi'(a|s) q_\pi(s,a) +  ( 1- \epsilon) \sum_{a \in \mathcal{A}} \frac{\pi(a|s) - \frac{\epsilon}{m}}{1 - \epsilon} q_\pi(s,a) \\
& = v_\pi (s) \\
\therefore v_\pi(s') & \geq v_\pi (s) 
\end{aligned} $$

### GLIE

Greedy in the Limit with Infinite Exploration, as the name suggests, is a strategy in which we are essentially trying to explore infinitely, but reducing the magnitude of exploration over time so that in the limit the strategy remains greedy. Thus, a GLIE strategy has to satisfy 2 conditions: 

- If a state is visited infinitely often, then each action in that state is chosen infinitely often 

    $$\lim_{k \rightarrow \infty } N_k(s,a) = \infty $$

- In the limit, the learning policy is greedy with respect to the learned Q-function 

    $$\lim_{k \rightarrow  \infty} \pi_k(a|s) = \bm{1} (a = \argmax_{a \in \mathcal{A}} Q_k(s, a'))$$

So, to convert our $\epsilon$-greedy strategy to a GLIE strategy, for example, we need to ensure that the magnitude of $\epsilon$  decays overtime to $0$. Two variants of exploration strategies that have been shown to be GLIE are: 

- $\epsilon$-greedy with exploration with $\epsilon_t = c/ N(t)$ where $N(t)$ → number of visits to state $s_t=s$
- Boltzmann Exploration with 

    $$
    \begin{aligned}
    & \beta_t(s) = \log (\frac{n_t(S)}{C_t(s)} ) \\
    & C_t(s) \geq \max_{a,a'}|Q_t(s,a) - Q_t(s,a')| \\
    & P(a_t = a | s_t = s ) = \frac{exp(\beta_t(s) Q_t(s,a))}{\sum_{b\in \mathcal{A}} exp(\beta_t(s) Q_t(s,a))}
    \end{aligned}
    $$

### GLIE Monte-Carlo Control

We use Monte-Carlo estimation to get an estimate of the policy and then improve it in a GLIE manner as follows: 

- Sample k$^{th}$ episode using policy $\pi$ so that $\{S_1, A_1, R_1, ....., S_T \} \sim \pi$
- For each state and action, update the Number of visitation

    $$\begin{aligned}
    & N(S_t, A_t) \leftarrow N(S_t, A_t) + 1 \\
    & Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)}  (G_t - Q(S_t, A_t) )  
    \end{aligned}$$

- Improve the policy based on action value as

    $$\begin{aligned}
    & \epsilon \leftarrow \frac{\epsilon}{k} \\ 
    & \pi \leftarrow \epsilon-greedy(Q)
    \end{aligned}$$

Thus, now we update the quality of each state-action pair by running an episode and update all the states that were encountered. Then we lip a coin with $1- \epsilon$  probability of selecting the state-action pair with the highest $Q(s,a)$  from all the possible choices and $\epsilon$  probability of taking a random pair. Hence, we are able to ensure exploration happens with an $\epsilon$  probability, and this probability changes proportional to $\frac{1}{k}$ which essentially allows us to get more greedy as $k$ increases, with the hope that we converge to the optimal policy. In fact, it has 

## Off-Policy Monte-Carlo Control

To be able to use experiences sampled from a policy $\pi'$ to estimate $v_\pi$ or $q_\pi$, we need to understand how the policies might relate to each other. To be even able to make a comparison, we first need to ensure that every action taken under $\pi$ is also taken, at least occasionally, under $\pi'$ → This allows us to guarantee representation of actions being common between the policies and thus, we need to ensure 

$$\pi(s,a) > 0  \implies \pi'(s,a) > 0$$

Now, let's consider we have the $i^{th}$ visit to a state $s$ in the episodes generated from $\pi'$ and the sequence of states and actions following this visit and let $P_i(s), P_i'(s)$ denote the probabilities of that complete sequence happening given policies $\pi, \pi'$ and let $R_i(s)$ be the return of this state. Thus to estimate $v_\pi(s)$ we only need to weigh the relative probabilities of $s$ happening in both policies. Thus, the desired MC estimate after $n_s$ returns from $s$ is: 

$$V(s) = \frac{\sum_{i=1}^{n_s} \frac{P_i(s)}{P_i(s')} R_i(s)}{{\sum_{i=1}^{n_s} \frac{P_i(s)}{P_i(s')}}}$$

We know that the probabilities are proportiona to the transition probabilities in each policy and so

$$P_i(s_t) = \prod_{k=t}^{T_i(s) - 1} \pi(s_k, a_k) \mathcal{P}^{a_k}_{s_k s_{k+1}}$$

Thus, when we take the ratio, we get

$$\frac{P_i(s)}{P_i'(s)} = \prod_{k=t}^{T_i(s) - 1} \frac{\pi(s_k, a_k)}{\pi'(s_k, a_k)} $$

Thus, we see that the weights needed to estimate $V(s)$  only depend on policies, and not on the dynamics of the environment. This is what allows off-policy learning to be possible. The advantage this gives us is that the policy used to generate behavior, called the behavior policy, may in fact be unrelated to the policy that is evaluated and improved, called the estimation policy. Thus, the estimation policy may be deterministic (e.g., greedy), while the behavior policy can continue to sample all possible actions!!

### Importance Sampling

In MC off-policy control, we can use the returns generated from policy $\mu$ to estimate policy $\pi$ by weighing the target $G_t$ based on the similarity between the policies. This is the essence of importance sampling, where we estimate the expectation of a different distribution based on a given distribution: 

$$\begin{aligned}
\mathbb{E}_{X \sim P}[f(X)] & = \sum P(X) f(X)  \\
& = \sum Q(X) \frac{P(X)}{Q(X)}f(X) \\
& = \mathbb{E}_{X \sim Q} \bigg[ \frac{P(X)}{Q(X} f(X) \bigg]
\end{aligned}$$

Thus, we just multiple te importance sampling correlations along the episodes and get:

$$G^{\pi/\mu}_t = \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)} \frac{\pi(a_{t+1}|s_{t+1})}{\mu(a_{t+1}|s_{t+1})} ... \frac{\pi(a_T|s_T)}{\mu(a_T|s_T)} G_t$$

And now, we can use $G^{\pi/\mu}_t$ to compute our value update for MC-control. 

## TD-Policy Control

The advantages of TD-Learning over MC methods are clear: 

1. Lower variance
2. Online
3. Incomplete sequences 

We again follow the pattern of GPI strategy, but this time using the TD estimate of the target and then again encounter the same issue of maintaining exploration, which leads us to on-policy ad off-policy control. As was the case with MC control, we need to remain model-free and so we shift the TD from estimating state-values to action-values. We know that formally, they both are equivalent and essentially Markov chains. 

### On-Policy Control : SARSA

We can use the same TD-target as state values to get the update for state-action pairs: 

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \big[ R_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \big]$$

This is essentially operating over the set of current states and action, one step look-ahead of the same values and the reward of the next pair → $(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$ → and the order in which this is written is **S**tate → **A**ction → **R**ewards → **S**tate → **A**ction. Thus, this algorithm is called SARSA.  We can use SARSA for evaluating our policies and then improve the policies, again, in an $\epsilon$-greedy manner. SARSA converges to $q^*(s,a)$ under the following conditions: 

- GLIE sequences of policies $\pi_t(a|s)$
- Robbins-Monro sequence of step-sizes $\alpha_t$

    $$\begin{aligned}
    & \sum_{t=1}^\infty \alpha_t = 0 \\
    & \sum_{t=1}^\infty \alpha^2_t < \infty 
    \end{aligned}$$

We can perform a similar modification on SARSA to to extend it to n-steps by defining a target based on n-step returns: 

$$\begin{aligned}
& q_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + ... +  \gamma^{n-1} r_{t+n} + \gamma^n Q(s_{t+n}) \\
\therefore \,\,\,& Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \big[ q_t^{(n)} - Q(s_t, a_t) \big]
\end{aligned}$$

Additionally, we can also formulate a forward-view SARSA($\lambda$)  by combining n-step returns: 

$$q_t^\lambda  = (1 - \lambda) \sum _{n=1}^\infty \lambda ^{n-1} q_t^{(n)}$$

and just like TD($\lambda$), we can implement Eligibility traces in online algorithms, in which case there will be one eligibility trace for each state-action pair: 

$$\begin{aligned}
& E_0(s,a) = 0 \\
& E_t(s,a) = \gamma \lambda E_{t-1}(s,a) + \bm{1}(s,a) \\
\therefore \,\,\, & Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \, E_t(s,a) \big[ q_t^{(n)} - Q(s_t, a_t) \big]
\end{aligned}$$

### TD Off-Policy Learning : Q-Learning

For off-policy learning in TD, we can again look at the relative weights and use importance sampling. However, since the lookahead is only one-step and not n-step sequence sampling, we only need a single importance sampling correction to get

$$V(s_t) \leftarrow V(s_t) + \alpha \, \bigg( \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)} (R_{t+1} + \gamma \, V(s_{t+1})) - V(s_t) \bigg)$$

The obvious advantage is the requirement of only a one-step correlation, which leads to much lower variance. One of the most important breakthroughs in reinforcement learning was the development of Q-learning, which does not require importance sampling. The simple idea that makes the difference is:

- we choose our next action from the behavior policy i.e $a_{t+1} \sim \mu$  BUT we use the alternative action sampled from the target policy $a' \sim \pi$  to update towards. Thus, our equation becomes:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \big[ R_{t+1} + \gamma Q(s_{t+1}, a'_{t+1}) - Q(s_t, a_t) \big]$$

- Then, we allow both behavior and target policies to improve → This means that $\pi(s_{t+1} | a_{t+1} ) =  \argmax_{a'} Q(s_{t+1}, a')$  while $\mu(s_{t+1} | a_{t+1} ) =  \argmax_{a} Q(s_{t+1}, a)$ .  Thus, our final equation simplifies to:

    $$Q(s, a) \leftarrow Q(s, a) + \alpha \big[ R +  \gamma \max_{a'} Q(s', a') - Q(s, a) \big]$$

This dramatically simplifies the analysis of the algorithm. We can see that all that is required for correct convergence is that all pairs continue to be updated.

<!-- %%% -->
# RL: Model-Free Prediction
One of the problems with DP is that it assumes full knowledge of the MDP, and consequently, the environment. While this holds true for a lot of applications, it might not hold true for all cases. In fact, the upper limit does turn out to be the ability to be accurate about the underlying MDP. Thus, if we don't know the true MDP behind a process, the next best thing would be to try to approximate them. One of the ways to go about this is **Model-Free RL**.

## Monte-Carlo Methods
The core idea behind the Monte-Carlo approach, which has its root in gambling, is to use probability to approximate quantities. Suppose we have to approximate the area of a circle relative to a rectangle inside which it is inscribed (This is a classic example and an easy experiment), the experiment-based approach would be to make a board and randomly throw some balls on it. In a true random throw, let's call the creation of a spot on the board a simulation ( experiment, whatever!). After each simulation, we record the number of spots inside the circular area and the total number of spots including the circular area and the rectangular area. A ratio of these two quantities would give us an estimate of the relative area of the circle and the rectangle. Now as we conduct more such experiments, this estimate would actually get better since the underlying probability of a spot appearing inside the circle is proportional to the amount of area that the circle occupies inside the rectangle. Hence, if we keep doing this our approximation gets increasingly closer to the true value.

### Applying MC idea to RL
Another way to put the Monte-Carlo approach would be to say that Monte-Carlo methods only require experience. In the case of RL, this would translate to sampling sequences of states, actions, and rewards from actual or simulated interaction with an environment. An experiment in this sense would be a full rollout of an episode which will create a sequence of states and rewards. When multiple such experiments are conducted, we get better approximations of our MDP. To be specific, our goal here would be to learn $v_{\pi}$ from the episodes under policy $\pi$. The value function is the expected reward:

$$v_{\pi}(s)= \mathbb{E}_{\pi}[G_t∣S_t=s]$$

So, all we have to do is estimate this expectation using the empirical mean of the returns for the experiments

### First-Visit MC Evaluation
To evaluate state s, at the first time-step t at which s is visited:

1. Increment counter:  $N(s) \leftarrow N(s) + 1$
2. Increment total return: $S(s) \leftarrow S(s) + 1$
3. Estimate value by the mean return: $V(s) = \frac{S(s)}{N(s)}$ 

As we repeat more experiments and update the values at the first visit, we get convergence to optimal values i.e $V(s) \rightarrow v_{\pi}$ as $N(s) \rightarrow \infty$

### Every-Visit MC Evaluation
This is same as first visit evaluation, except we update at every visit:

1. $N(s) \gets N(s) + 1$ 
2. $S(s) \gets S(s) + 1$
3. $V(s) = \frac{S(s)}{N(s)}$

### Incremental MC updates
The empirical mean can be expressed as an incremental update as follows:

$$
\begin{aligned}
\mu_k &= \frac{1}{k} \sum_{j=1}^n x_j \\
&= \frac{1}{k} (x_k + \sum_{j=1}^{k-1} x_j) \\
&= \frac{1}{k} (x_k + (k-1)\mu_{k-1})
\end{aligned}
$$

Thus, for the state updates, we can follow a similar pattern, and for each state $S_t$ and return $G_t$, express it as:

1. $N(S_t) \gets N(S_t) + 1$
2. $(S_t) = V(S_t) + \frac{1}{N(S_t)}(G_t - V(S_t))$

Another useful way would be to track a running mean:

$$V(S_t) \gets V(S_t) + \alpha (G_t - V(S_t)$$



## Temporal-Difference (TD) Learning
The MC method of learning needs an episode to terminate in order to work its way backward. In TD, the idea is to work the way forward by replacing the remainder of the states with an estimate. This is one method that is considered central and novel to RL (According to Sutton and Barto). Like MC methods, TD methods can learn directly from raw experience, and like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome, something which is called Bootstrapping - updating a guess towards a guess (meta-guess-update?).

### Concept of Target and Error
If we look at the previous equation of Incremental MC, the general form that can be extrapolated is

$$V(S_t) \gets V(S_t) + \alpha (T - V(S_t)$$

Here, the quantity $T$ is called the **Target** and the quantity $T - V(S_t)$ is called the **error**. In the MC version, the target is the return $G_t$, which means that the MC method has to wait for this return to be propagated backward to see the error of its current value function from this return, and improve. This is where TD methods show their magic; At time $t+1$, they can immediately form a target and make a useful update using the observed reward and the current estimate of the value function. The simplest TD method, $TD(0)$, thus has the following form:

$$V(S_t) \gets V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$$

This is why bootstrapping is a guess of guess, since the TD method bases its update in part on an existing estimate.

## Comparing TD, MC and DP
Another way to look at these algorithms would be through the Bellman optimality equation:

$$v_{\pi}(s) = E [ R_{t+1} + \gamma v_{\pi}(S_{t+1})| S_t = s]$$

which allows us to see why certain methods are estimates:

- The MC method is an estimate because it does not have a model of the environment and thus, needs to sample in order to get an estimate of the mean.
- The DP method is an estimate because it does not know the future values of states and thus, uses the current state value estimate in its place.
- The TD method is an estimate because it does both of these things. Hence, it is a combination of both. However, unlike DP, MC and TD do not require a model of the environment. Moreover, the online nature of these algorithms is something that allows them to work with samples of backups, whereas DP requires full backup.

TD and MC can further be differentiated based on the nature of the samples that they work with: 

- TD requires shallow backups since it is inherently online in nature
- MC requires deep backups due to the nature of its search.

Another way to look at the inherent difference is to realize that DP inherently does a breadth-first search, while MC does a depth-first search. TD(0) only looks one step ahead and forms a guess. These differences can be summarized on the following spectrum by David Silver, which I find really helpful:

<img width=600 height=500 src="static/Reinforcement Learning/TD-MC-DP.png">

## Extending TD to n-steps 

The next natural step for something like TD would be to extend it to further steps. For this, we generalize the target and define it as follows:

$$G^{(n)}_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+ n} + \gamma^n V(S_{t+n}))$$

And so, the equation again follows the same format:

$$V(S_t) \gets V(S_t) + \alpha (G^{(n)}_t - V(S_t)$$

One interesting thing to note here is that if the value of $n$ is increased all the way to the terminal state, then we essentially get the same equation as MC methods!

### Averaging over n returns 
To get the best out of all the $n$ steps, one improvement could be to average the returns over a certain number of states. For example, we could combine 2-step and 4-step returns and take the average :

$$G_{avg} = \frac{1}{2} [ G^{(2)} + G^{(4)} ]$$

This has been shown to work better in many cases, but only incrementally.

### $\lambda$-Return → Forward View

This is a method to combine the returns from all the n-steps:

$$G^{(\lambda)}_t = (1 - \lambda ) \displaystyle\sum_{n=1}^{\infin} \lambda^{n-1} G^{(n)}_t$$

And this, is also called **Forward-view $TD(\lambda)$.**

### Backward View 

To understand the backward view, we need a way to see how we are going to judge the causal relationships between events and outcomes (Returns). There are two heuristics:

1. **Frequency Heuristic →** Assign credit to the most frequent states
2. **Recency Heuristic →** Assign credit to the most recent states.

The way we keep track of how each states fares on these two heuristics is through **Eligibility Traces**:

$$E_t(s) = \gamma \lambda E_{t-1}(s) + \bold{1}(S_t = s)$$

These traces accumulate as the frequency increases and are higher for more recent states. If the frequency drops, they also drop. This is evident in the figure below:

<img width=300 height=100 src="static/Reinforcement Learning/ET.png">

So, all we need to do it scale the TD-error $\delta_t$ according to the trace function:

$$V(S) \gets V(S) + \alpha \delta_t E_t(s)$$

Thus, when $\lambda = 0$, we get the equation for $TD(0)$ , and when $\lambda =1$, the credit is deferred to the end of the episode and we get MC equation.


<!-- %%% -->
# RL: Planning and Dynamic Programming

Dynamic programming (DP) is a method that solves a problem by breaking it down into sub-problems and then solving each sub-problem individually, after which it combining them into a solution. A good example is the standard Fibonacci sequence calculation problem, where traditionally the way to solve it would be through recursion

```cpp
int fib(int *x*) {
	
	if (x < 2) {
		return 1;
	}
	return fib(x-1) + fib(x-2);	

}
```

However, the way DP would go about this would be to cache the variables after the first call, so that the same call is not made again, making the program more efficient:

```cpp
int fib(int *x*) {

	static vector<int> cache(N, -1);
	int& result = cache[x];
	
	if (result == -1) {
	
		if (x < 2) result = 1;
	
		else result = fib(x-1) + fib(x-2);
	}
	
	return result;
}
```

The 2 characteristics that a problems need to have fo DP to solve it are:

1. ****Optimal Substructure :**** Any problem has optimal substructure property if its overall optimal solution can be constructed from the optimal solutions of its subproblems i.e the property $Fib(n) = Fib(n-1) + Fib(n-2)$ in fibonacci numbers
2. ****Overlapping Sub-problems:**** The problem involves sub-problems that need to be solved recursively many times

Now, in the case of an MDP, we have already seen that these properties are fulfilled:

1. The Bellman equation gives a recursive relation that satisfies the overlapping sub-problems requirement
2. The value function is able to store and re-use the solutions from each state-visit, and thus, we can exploit it as an optimal substructure

Hence, DP can be used for making solutions to MDPs more tractable, and thus, is a good tool to solve the planning problem in an MDP. The planning problem, as discussed before, is of two types:

1. ****Prediction Problem:**** ****How do we evaluate a policy ?**** or, Using the MDP tuple as an input, the output is a value function $v_{\pi}$ and/or a policy $\pi$
2. ****Control Problem:**** ****How do we optimize the policy ?**** Using the MDP tuple as an input, the output is an optimal value function $v_*$ and/or a policy $\pi_*$

## Iterative Policy Evaluation

The most basic way is to iteratively apply the Bellman equation, using the old values to calculate a new estimate, and then using this new estimate to calculate new values. In the Bellman equation for the state-value function

$$v_{\pi}(s) = \sum_{a \in A} \pi (a|s) \big[ R^{a}s  +  \gamma \sum{s' \in S} P^{a}{ss'} v{\pi}(s) \big]$$

As long as either $\gamma < 1$ or the eventual termination is guaranteed from all states under the policy $\pi$, the uniqueness of the value function is guaranteed. Thus, we can consider a sequence of approximation functions $v_0, v_1, v_2, ...$  each mapping states to Real numbers, start with an arbitrary estimate of $v_0$, and obtain successive approximations using Bellman equation, as follows:

$$v_{k+1}(s) = \sum_{a \in A} \pi (a|s) \big[ R^{a}_s  +  \gamma \sum_{s' \in S} P^{a}_{ss'} v_{k} (s') \big]$$

The sequence $v_k$ can be shown to converge as $k \rightarrow \infty$. The process is basically a propagation towards the root of the decision tree from the roots.

<img width=300 height=200 src="static/Reinforcement Learning/It-pol-eval.png">

This update operation is applied to each state in the MDP at each step, and so, is called ****Full-Backup**.** Thus, in a computer program, we would have two cached arrays - one for $v_k(s)$ and one for $v_{k+1}(s)$

## Policy Improvement
Once we have a policy, the next question is do we follow this policy or shift to a new improved policy? one way to answer this problem is to take any action that this policy does not suggest and then evaluate the same policy after that action. If the returns are higher then we can say that taking that action is better than following the current policy. The way we evaluate the action is through the action-value function:

$$q_{\pi}(s,a) = R^{a}_s  +  \gamma \sum_{\substack{s' \in S}} P^{a}_{ss'} v_{\pi} (s')$$

If this value is greater than the value function of a state S, then that essentially means that it is better to select this action than follow the policy $\pi$ , and by extension, it would mean that anytime we encounter state $S$, we would like to take this action. So, let's call the schema of taking action $a$ every time we encounter $S$ as a new policy $\pi'$, and so, we can now say

$$q_{\pi}(s,{\pi}'(s)) \geq v_{\pi}(s)$$

This implies that the policy $\pi'$ must be **at-least** as good as the policy $\pi$

$$v_{{\pi}'} \geq v_{\pi}$$

Thus, if we extend this idea to multiple possible actions at any state $S$,  the net incentive is to go full greedy on it and select the best out of all those possible actions:

$${\pi}'(s) = \argmax_a q_{\pi}(s,a)$$

The greedy policy, thus, takes the action that looks best in the short term i.e after one step of lookahead. The point at which the new policy stops becoming better than the old one is the convergence point, and we can conclude that optimality has been reached. This idea also applies in the general case of stochastic policies, with the addition that in the case of multiple actions with the maximum value, a portion of the stochastic probability can be given to each.

## Policy Iteration

Following the greedy policy improvement process, we can obtain a sequence of policies:

$${\pi}_0 \rightarrow v_{\pi_0} \rightarrow {\pi_1} \rightarrow v_{\pi_1} .... \rightarrow {\pi}_* \rightarrow v_{{\pi}_*}$$

Since a finite MDP has a finite number of policies, this process must converge at some point to an optimal value. This process is called ****Policy Iteration****. The algorithm, thus, follows the process:

1. ****Evaluate**** the policy using the Bellman equation
2. ****Improve**** the policy using greedy policy improvement.

A natural question that comes up at this point is that do we actually need to follow this optimization procedure all the way to the end? It does sound like a lot of work, and in a lot of cases, a workably optimal policy is actually reached much before the final iteration step, where the steps after achieving this policy add minimal improvement and thus, are somewhat redundant. Thus, we can include stopping conditions to tackle this, as follows:

1. $\epsilon$-convergence
2. Stop after $k$ iteratiokns
3. Value Iteration


## Value Iteration

In this algorithm, the evaluation is truncated to one sweep → one backup of each state. To understand this, the first step is to understand something called the **Principle of Optimality**. The idea is that an optimal policy can be subdivided into two parts:

- An optimal first action $A_*$
- An optimal policy from the successor state $S'$

So, if we know the solution to $v_*(s')$ for all $s'$ succeeding the state $s$, then the solution can be found with just a one-step lookahead

$$v_*(s) \gets \max_{a \isin A} R^a_s + \gamma \sum_{\substack{s' \in S}} P^{a}_{ss'} v_{*} (s')$$

The intuition is to start from the final reward and work your way backward. There is no explicit update of policy, only values. This also opens up the possibility that the intermediate values might not correspond to any policy, and so interpreting anything midway will have some residue in addition to the greedy policy.  In practice, we stop once the value function changes by only a small amount in a sweep. A summary of synchronous methods for DP is given by David Silverman:

<img width=500 height=200 src="static/Reinforcement Learning/sync-DP-summary.png">

<!-- %%% -->
# RL: Markov Processes
These are random processes indexed by time and are used to model systems that have limited memory of the past. The fundamental intuition behind Markov processes is the property that the future is independent of the past, given the present. In a general scenario, we might say that to determine the state of an agent at any time instant, we only have to condition it on a limited number of previous states, and not the whole history of its states or actions. The size of this window determines the order of the Markov process.

To better explain this, one primary point that needs to be addressed is that the complexity of a Markov process greatly depends on whether the time axis is discrete or topological. When this space is discrete, then the Markov process is a Markov Chain. A basic level understanding of how these processes play out in the domain of reinforcement learning is very clear when analyzing these chains. Moreover, the starting point of analysis can be further simplified by limiting the order of Markov Processes to first-order. This means that at any time instant, the agent only needs to see its previous state to determine its current state, or its current state to determine its future state. This is called the ****Markov Property****

$$\mathbb{P}(S_{t+1}|S_t) = \mathbb{P}(S_{t+1}|S_1, ..., S_t) )$$

## Markov Process
The simplest process is a tuple $<S,P>$ of states and Transitions. The transitions can be represented as a Matrix $P = [P_{ij}]$, mapping the states - i - from which the transition originates,  to the states - j - to which the transition goes.

$$\begin{bmatrix}
P_{11} & . & . & . & P_{1n}\\
. & . & . & . & . \\
. & . & . & . & . \\
. & . & . & . & . \\
P_{n1} & . & . & . & P_{nn}
\end{bmatrix}$$

Another way to visualize this would be in the form of a graph, as shown below, courtesy of David Silver.

<img width=800 height=500 src="static/Reinforcement Learning/MP.png">

This is a basic chain that represents the actions a student can take in the class, with associated probabilities of taking those actions. Thus, in the state - Class 1 - the student has an equal chance of going to the next class or browsing Facebook. Once they start browsing Facebook, then they have a 90% chance of continuing to browse since it is addictive. Similarly, other states can be seen too. 

## Markov Reward Process
Now if we add another parameter of rewards to the Markov processes, then the scenario changes to the one in which entering each state has an associated expected immediate reward. This, now, becomes a Markov Reward Process. 

<img width=800 height=500 src="static/Reinforcement Learning/MRP.png">

To fully formalize this, one more thing that needs to be added is the discounting factor $\gamma$. This is a hyperparameter that represents the amount of importance we give to future rewards, something like a 'shadow of the future'. The use of Gamma can be seen in computing the return $G_t$ on a state:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$$

The reasons for adding this discounting are:

- To account for uncertainty in the future, and thus, better balance our current decisions → The larger the value, the more weightage we give to the 'shadow of the future'
- To make the math more convenient → only when we discount the successive terms, we can get convergence on an infinite GP
- To avoid Infinite returns, which might be possible in loops within the reward chain
- This is similar to how biological systems behave, and so in a certain sense, we are emulating nature.

Thus, the reward process can now be characterized by the tuple $<S, P, R, \gamma >$ . To better analyze the Markov chain, we will also define a way to estimate the value of a state - ****Value function**** - as an expectation of the Return on that state. Thus,

$$V(S) = \mathbb{E} [ G_t| S_t = s ]$$

An intuitive way to think about this is in terms of betting. Each state is basically a bet that our agent needs to make. Thus, the process of accumulating the rewards represents the agent's understanding of each of these bets, and to qualify them, the agent has to think in terms of the potential returns that these bets can give. This is what we qualify here as the expectation. But the magic comes when we apply it recursively, and this is called the ****Bellman Equation****

$$V(S) = \mathbb{E} [R_{t+1} + \gamma V(S_{t+1}| S_t = s]$$

This equation signifies that the value of the current state can be seen in terms of the value of the next state and so on, and thus, we can have a correlated relationship between states. To better see how this translates to the whole chain, we can also express this as a Matrix Operation:

$$\begin{bmatrix}
V_1 \\
.  \\
.  \\
V_n
\end{bmatrix}
=
\begin{bmatrix}
R_1 \\
.  \\
.  \\
R_n
\end{bmatrix}

\begin{bmatrix}
P_{11} & . & . & P_{1n}\\
. & . & . & . \\
. & . & . & . \\
P_{n1} & . & . & P_{nn}
\end{bmatrix}
\begin{bmatrix}
V_1 \\
.  \\
.  \\
V_n
\end{bmatrix}$$

And so, the numerical way to solve this would be to invert the matrix (assuming it is invertible) and the solution, then would be:

$$\bm{\bar{V}} = (1 - \gamma \bm{\bar{P}})^{-1} \bm{\bar{R}}$$

However, as anyone familiar with large dimensions knows, this becomes intractable pretty easily. Hence, the whole of RL is based on figuring out ways to make this tractable, using majorly three kinds of methods:

1. Dynamic Programming
2. Monte-Carlo Methods
3. Temporal Difference Learning

## Markov Decision Process

If we add actions to the Markov Reward Process, then there can multiple states that the agent can reach by taking action to each state. Thus, the agent now has to decide which action to take. This is called a Markov Decision Process.

<img width=800 height=500 src="static/Reinforcement Learning/MDP.png">

Thus, the MDP can be summarized by the tuple $<S, A, P, R, \gamma >$. Here, we can also define the transitions and rewards in terms of the actions:

$$
\begin{aligned}
R^{a}_s &= \mathbb{E}[ R{t+1} | S_t=s, A_t=a ] \\
P^{a}_{ss'} &= \mathbb{P}[ S{t+1}=s'| S_t=s, A_t=a ]
\end{aligned}
$$

Now, the important thing is how the agent makes these decisions. The schema that the agent follows for this is called a ****Policy****, which can be seen as the probability of taking an action, given the state:

$$
\pi (a|s) = \mathbb{P} [ A_t = a | S_t = s ]
$$

Under a particular policy $\pi$, the Markov chain that results is nothing but an MRP, since we don't consider the actions that the agent did not take. This can be characterized by $<S, P^{\pi}, R^{\pi}, \gamma>$, and the respective transitions and rewards can be described as:

$$
\begin{aligned}
R^{\pi}_{s} &= \sum_{\substack{a \in A}} \pi (a|s) R^{a}_{s}\\
P^{\pi}_{ss'} &= \sum_{a \in A} \pi (a|s) P^{a}_{ss'}
\end{aligned}
$$

Another important thing that needs to be distinguished here is the value function, which can be defined for both states and actions:

- **State-Value Function ($v_{\pi}$):** Values for states when policy $\pi$ is followed

    $$v_{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]$$

- **Action-Value Function ($q_{\pi}$):** Expected return on starting from state $s$, following policy $\pi$, and taking action $a$

    $$q_{\pi}(s, a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a]$$


### Bellman Equation for MDPs
We can extend the bellman formulation to recursively define the qualities of state and actions:

$$\begin{aligned}
&v_{\pi}(s) = \mathbb{E}_{\pi} \big[R_{t+1} + \gamma v_{\pi}(s')| S_t = s, S_{t+1} = s'\big] \\
&q_{\pi}(s, a) = \mathbb{E}_{\pi} \big[R_{t+1} + \gamma q_{\pi}(s', a')| S_t = s, S_{t+1} = s', A_t = a, A_{t+1} = a' \big]
\end{aligned}$$

However, a better way is to look at the inter-dependencies of these two value functions. The value of the state can be viewed as the sum of the value of the actions that can be taken from this state, which can, in turn, be viewed as the weighted sum of values of the states that can result from each action.

#### Bellman Expectation in second recursive form

The expectation for the value of the states is the sum of the values of the actions that can result from that state

<img width=400 height=200 src="static/Reinforcement Learning/sve.png">

Thus, under the policy $\pi$ this value is the sum of the q-values of the actions: 

$$
v_{\pi}(s) = \sum_{a \in A} \pi (a|s) q_{\pi} (s,a)
$$

Now, the action can be viewed in a similar manner as a sum over the value fo the states that can result from it

<img width=300 height=200 src="static/Reinforcement Learning/ave.png">

and written in the same manner

$$
q_{\pi}(s,a) = R^{a}_s  +  \gamma \sum_{s' \in S} P^{a}_{ss'} v_{\pi} (s)
$$

And, if we put these equations together, we can get a self-recursive formulation of the bellman expectation equation. Thus, for the state this would be

$$
v_{\pi}(s) = \sum_{a \in A} \pi (a|s) [ R^{a}_s  +  \gamma \sum_{s' \in S} P^{a}_{ss'} v_{\pi} (s) ]
$$

A Visualization for this would basically be a combination of the above two trees

<img width=300 height=200 src="static/Reinforcement Learning/sveave.png">

A similar process can be done for the action value function, and the result comes out to be

$$
q_{\pi}(s,a) = R^{a}_s  +  \gamma \sum_{s' \in S} P^{a}_{ss'} \sum_{a' \in A} \pi (a'|s') q_{\pi} (s',a')
$$

<img width=300 height=200 src="static/Reinforcement Learning/avesve.png">


#### Bellman Optimality Equation

With the recursive forms, the question really comes on how do we go about creating a closed-loop optimality criterion. Here, the key point that needs to be taken into account is **The agent is free to choose the action that it can take in each state, but it can't choose the state that results from that action**. This means, we start from a state, and maximize the result by choosing the action with the maximum action value. This is the first step of lookahead. Now, each of those actions has the associated action value that needs to be determined. In the case where the action can only lead to one state, it's all well and good. However, in the case where multiple states can result out of the action, the value of the action can be determined by basically rolling a dice and seeing which state the action leads to. Thus, the value of the state that the action leads to determines the value of the action. This happens for all the possible actions from our first state, and thus, the value of the state is determined. Hence, with this ****Two-step lookahead****, we can formulate the decision as maximizing the action values.

$$v_{\pi}(s) = \max_{a} \{ R^{a}s + \gamma \sum_{s' \in S} P^{a}_{ss'} v{} (s) \}$$

Now, the question arises as to how can this equation be solved. The thing to note here is the fact that it is not linear. Thus, in general, there exists no closed-form solution. However, a lot of work has been done in developing iterative solutions to this, and the primary methods are:

- **Value Iteration:** Here methods solve the equation by iterating on the value function, going through episodes, and recursively working backward on value updates
- **Policy Iteration:** Here, the big idea is that the agent randomly selects a policy and finds a value function corresponding to it. Then it finds a new and improved policy based on the previous value function, and so on.
- **Q Learning:** This is a model-free way in which the agent is guided through the quality of actions that it takes, wit the aim of selecting the best ones
- **SARSA:** Here the idea is to iteratively try to close the loop by selecting a **S**tate, **A**ction, and **R**eward and then seeing the **S**tate and **A**ction that follows.

## Extensions to MDP

MDPS, as a concept, has been extended to make them applicable to multiple other kinds of problems that could be tackled. Some of these extensions are:

1. **Infinite and Continuous MDPs:** In this extension, the MDP concept is applied to infinite sets, mainly countably infinite state or action spaces, Continuous Spaces (LQR), continuous-time et. al
2. **Partially Observable MDPs (POMDP):** A lot of scenarios exist where there are limits on the agent's ability to fully observe the world. These are called Partially-Observable cases. Here, the state is formalized in terms of the belief distribution over the possible observations and encoded through the history of the states. The computations become intractable in theory, but many interesting methods have been devised to get them working. Eg. DESPOT
3. **Undiscounted and Average Reward MDP:** These are used to tackle ergodic MDPs - where there is a possibility that each state can be visited an infinite number of times ( Recurrence), or there is no particular pattern in which the agent visits the states (Aperiodicity) - and to tackle this, the rewards are looked at as moving averages that can be worked with on instants of time.


<!-- %%% -->
# RL: Introduction to Reinforcement learning
One way to look at the behavior of an organism is by looking at how it interacts with its environment, and how this interaction allows it to behave differently over time through the process of learning. In this view, the behavior of the organism can be modeled in a closed-loop manner through a fundamental description of the action and sensation loop. It receives input - sensation - from the environment through sensors, acts on the environment through actuators, and observes the impact of its action on its understanding of the environment through a mechanism of quantification in the form of the rewards it receives for its action. A similar thing happens in RL. The thing that we need to train is called an Agent. The language of communication with this agent is through numbers, encoded in processes that we create for it to understand and interact with the world around it. The way this agent interacts with the world around it is through Actions (A) and the way it understands the world is through Observations (O). Now, our task is to define these actions and observations and train this agent to achieve a certain task by creating a closed-loop control of feedback for the actions it takes. This feedback is the Reward (R) that agent receives for each of its actions. So, the key is to devise a method to guide the agent in such a way that it 'learns' to reach the goal by selecting actions with the highest Expected Rewards (G), updating these values by observing the environment after taking that action. Thus, the agent first takes random actions and updates its reward values, and slowly, it starts to favor actions with higher rewards, which eventually lead to the goal.

<img width=800 height=500 src="static/Reinforcement Learning/agent-env.svg">

The way we define observations is through formalizing it as a **State (S)** in which this agent exists, or can exist. This state can either be the same as the observation, in case the agent can see everything about its environment, for example, in an extreme case imagine if you were able to see all the atoms that constitute your surroundings, or the state can be defined in terms of **Beliefs (b)** that agent the might have based on its observation. this distinction is important for problems of partial observability, a topic for the future. A standard testbed in RL is the Mountain Car scenario. As shown in the figure below, the car exists in a valley and the goal is at the top. The car needs to reach this goal by accelerating, but it is unable to reach the top by simply accelerating from the bottom. Thus, it must learn to leverage potential energy by driving up the opposite hill before the car is able to make it to the goal at the top of the rightmost hill.

<img width=700 height=400 src="static/Reinforcement Learning/mountain-car.jpg">

One way to define the values for the agent - the car - would be to define the state as the (position, velocity) of the car, the actions as (Do nothing, Push the car left, Push the car right), and rewards as -1 for each step that leads to a position that is not the goal and 0 for reaching the goal. To characterize the agent, the following components are used in the RL vocabulary:

- **Policy ($\pi: S \rightarrow A$):** This is the behavior of the agent that i.e the schema it follows while navigating in the environment it observes by taking actions. Thus, it is a mapping from state to action
- **Value Function (V):** This is the agent's prediction of future rewards. The way this fits into the picture is that at each step the agent predicts the rewards that it can get in the future by following a certain set of actions under a policy. This expectation of reward is what determines which actions the agent should select.
- **Model:** The agent might make a model of the world that it observes around itself. Then it can use this model to extract information that it can use to better decide the actions that it can take. There are two types of models that are used, Reward Model and Transition
- **Reward Model:** Model to predict the next immediate reward. This is defined in terms of Expectation fo reward conditioned on a state and action :

    $$R^{a}_{s} = \mathbb{E}[ R | S=s, A=a ]$$

- **Transition Model:** Model to predict the next state using the dynamics of the environment. This is defined in terms of probability of a next state, conditioned on the current state and actions :

    $$P^{a}_{ss'} = \mathbb{P}[ S'=s'| S=s, A=a ]$$

Thus, using the above components learning can be classified into three kinds:

1. **Value-Based RL:**  In this type, the agent uses a value function to track the quality of states and thus, follows trends in the value functions. For example, in a maze with discretized boxes as steps, the agent might assign values to each step and keep updating them as it learns, and thus, end up creating a pattern where a trend of following an increase in the value would inevitably lead to the way out of the maze
2. **Policy-Based RL:** In this case, the agent would directly work with the policy. So, in the case of the maze example, each step might be characterized by four directions in which the agent can traverse (up, down, left, right) and for each box, the agent might assign a direction it will follow once it reaches that, and as it learns it can update these directions t create a clear path to the end of the maze
3. **Actor-Critic:** If two ideas are well-established in the scientific community, in this case, the value-based, and policy-based approach, then the next best step could be to try and merge them to get the best of both worlds. This is what the actor-critic does; it tries to merge both these ideas by splitting the model into two parts. The actor takes the state as an input and outputs the best actions by following a learned optimal policy (policy-based learning). The critic generates the value for this action by evaluating the value function ( value-based learning). These both compete in a game to improve their methods and overall the agent learns to perform better.

The learning can also be distinguished based on whether the agent has a model of the world, in which case the learning is ****Model-Based RL****, or whether the agent operates without a model of the world i.e ****Model-Free RL****. This will be explored in more detail in the next sections. Finally, certain paradigms are common in RL which recurs regularly, and thus, it might be good to list them down:

- **Learning and Planning:** In learning the rules of the game are unknown and are learned by putting the agent in the environment. For example, I remember some people once told me how some coaches teach the basics of swimming by asking the learner to directly jump into the semi-deep water and try to move their hands and legs in a way so that they can float. Irrespective of whether this actually happens or not, if someone learned this way I could think of it as a decent enough analogy. Planning, on the other hand, is driven by a model of the rules that need to be followed, which can be used by the agent to perform a look-ahead search on the actions that it can take.
- **Exploration and Exploitation:** This is the central choice the agent needs to make every time it takes an action. At any step, it has certain information about the world and it can go on exploiting it to eventually reach a goal (maybe), but the problem is it might not know about the most optimal way to reach this goal if it just acts on the information it already has. Thus, to discover better ways of doing things, the agent can also decide to forego the path it 'knows' will get the best reward according to its current knowledge and take a random action to see what kind of reward it gets. Thus, in doing so the agent might end up exploring other ways of solving a problem that it might not have known, which might lead to higher rewards than the path it already knows. Personally, the most tangible way I can visualize it is by thinking of a tree of decisions, and then imagining that the agent knows one way to reach the leaf nodes with the maximum reward. However, there might exist another portion of the tree that has higher rewards, but the agent might not ever go to if it greedily acts on its current rewards.
- **Prediction and Control:** Prediction is just finding a path to the goal, while control is optimizing this path to the goal. Most of the algorithms in RL can be distinguished based on this.

<!-- %%% -->
# LIS: Setting up RAI on HPC

## List of RPMs:
```
- ann-devel-1.1.2-3.el7.x86_64.rpm        
- gflags-2.1.1-6.el7.x86_64.rpm          
- jsoncpp-0.10.5-2.el7.x86_64.rpm      
- poly2tri-0.0-10.20130501hg26242d0aa7b8.el7.x86_64.rpm
- assimp-devel-3.1.1-2.el7.x86_64.rpm             
- gflags-devel-2.1.1-6.el7.x86_64.rpm    
- lapack-3.4.2-8.el7.x86_64.rpm        
- proj-4.8.0-4.el7.x86_64.rpm
- atlas-3.10.1-12.el7.x86_64.rpm                  
- glfw-3.2.1-2.el7.x86_64.rpm            
- libann-1.1.2-alt5.x86_64.rpm         
- pybind11-devel-2.4.3-2.el8.aarch64.rpm
- atlas-devel-3.10.1-12.el7.x86_64.rpm            
- glibc-2.17-307.el7.1.x86_64.rpm        
- libassimp3-3.3.1-alt1_5.x86_64.rpm   
- qhull-2003.1-20.el7.x86_64.rpm
- ceres-solver-1.12.0-5.el7.x86_64.rpm            
- glibc-devel-2.17-307.el7.1.x86_64.rpm  
- libgcc-4.8.5-39.el7.x86_64.rpm       
- suitesparse-4.0.2-10.el7.x86_64.rpm
- ceres-solver-devel-1.12.0-5.el7.x86_64.rpm      
- glibc-static-2.17-307.el7.1.i686.rpm   
- libgeotiff-1.2.5-14.el7.x86_64.rpm   
- suitesparse-devel-4.0.2-10.el7.x86_64.rpm
- ceres-solver-devel-1.13.0-12.el8.x86_64.rpm     
- glog-0.3.3-8.el7.x86_64.rpm            
- libGLEW-1.10.0-5.el7.x86_64.rpm      
- tbb-4.1-9.20130314.el7.x86_64.rpm
- eigen3-devel-3.3.4-6.el7.noarch.rpm             
- glog-devel-0.3.3-8.el7.x86_64.rpm      
- libstdc++-4.8.5-39.el7.x86_64.rpm    
- zlib-1.2.7-18.el7.x86_64.rpm
- f2c-20160102-1.el7.x86_64.rpm                   
- gnuplot-4.6.2-3.el7.x86_64.rpm         
- libX11-1.6.7-2.el7.x86_64.rpm
- freeglut-3.0.0-8.el7.x86_64.rpm                 
- graphviz-2.30.1-21.el7.x86_64.rpm      
- libX11-devel-1.6.7-2.el7.x86_64.rpm
- gcc-x86_64-linux-gnu-4.8.5-16.el7.1.x86_64.rpm  
- irrXML-1.8.1-3.el7.2.x86_64.rpm        
- minizip-1.2.7-18.el7.x86_64.rpm
```
## Initial Set-up:
```
- module load comp/gcc/7.2.0  
- module load python/3.6.8_tf-cpu
- module load nvidia/cuda/9.2.88
- export C_INCLUDE_PATH=$C_INCLUDE_PATH:$HOME/usr/include:$HOME/usr/local/bin/include/python3.6m
- export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$HOME/usr/include:$HOME/usr/local/bin/include/python3.6m
- export PATH=$PATH:$HOME/usr/local/bin/bin
- alias python3=python3.6
```
## Problems: 
These were the list of problems faced during the set-up

### Eigen and Assimp Issue (Type = Not Linked )
  - ln -sf eigen3/Eigen Eigen
  - ln -sf assimp/ Assimp 
### Cannot find -ljsoncpp and cannot find -llapack ( Type = .so file not present)
```
- ld -ljsoncpp --verbose
- ld -llapack --verbose
- Re-installed the rpm, the normal binary, and not the devel one, and set the symbolic link to what was being searched for 

- Added to generic.mk the following:
    CXXFLAGS += -L/home/users/a/amsks1996/usr/lib64
    CXXFLAGS += -Wl,-rpath=/home/users/a/amsks1996/usr/lib64

- Symbolic Links $HOME/usr/lib64
- ln -sf libjsoncpp.so.0 libjsoncpp.so
- ln -sf liblapack.so.3 liblapack.so
```
-  Assimp issue ( Type = include ) : download the devel rpm and place the files in the include folder

- Ceres not found ( Type = include ) : download the devel rpm adn place the files in the include folder

-  glog/logging.h not found ( Type = include ): Downloaded the rpm for 

- gflags/gflags.h ( Type = include ) : same 

-  qhull/qhull_a.h ( Type = include ) : Same for qhull-devel-2003.1-20.el7.x86_64.rpm 

-  **-lglew**, **-lqhull**, **-lGeo**, **-lglfw**, **-lcgraph**, **-lgvc**     **-ljsoncpp** **-llapack**  -lOptim  **-lann** : Compiled from the code

### libspqr.so libtbbmalloc.so libtbb.so libcholmod.so libccolamd.so libcamd.so libcolamd.so libamd.so liblapack.so libf77blas.so libatlas.so libsuitesparseconfig.so librt.so libcxsparse.so liblapack.so libf77blas.so libatlas.so libsuitesparseconfig.so librt.so libcxsparse.so libgflags.so.2.2.1 libglog.so 

These are related to the Ceres solver, and are defined in rai/buil/defines.ml

- Change the location to the usr/home

```
-lceres -lglog -lcholmod -llapack -lblas -lpthread  $(HOME)/usr/lib/libspqr.so $(HOME)/usr/lib/libtbbmalloc.so $(HOME)/usr/lib/libtbb.so $(HOME)/usr/lib/libcholmod.so $(HOME)/usr/lib/libccolamd.so $(HOME)/usr/lib/libcamd.so $(HOME)/usr/lib/libcolamd.so $(HOME)/usr/lib/libamd.so $(HOME)/usr/lib/liblapack.so $(HOME)/usr/lib/libf77blas.so $(HOME)/usr/lib/libatlas.so $(HOME)/usr/lib/libsuitesparseconfig.so $(HOME)/usr/lib/librt.so $(HOME)/usr/lib/libcxsparse.so $(HOME)/usr/lib/liblapack.so $(HOME)/usr/lib/libf77blas.so $(HOME)/usr/lib/libatlas.so $(HOME)/usr/lib/libsuitesparseconfig.so $(HOME)/usr/lib/librt.so $(HOME)/usr/lib/libcxsparse.so $(HOME)/usr/lib/libgflags.so.2.2.1 -lpthread $(HOME)/usr/lib/libglog.so"
```
- Recursively the dependencies


### Random Shared Issue 

g++ -L/home/users/a/amsks1996/git/ceres-solver/build/lib -L/home/users/a/amsks1996/opt/physx3.4/lib -L/beegfs/home/users/a/amsks1996/git/rai-python/rai/lib -L/home/users/a/amsks1996/opt/lib -L/usr/local/lib -L/home/users/a/amsks1996/usr/lib64 -L/home/users/a/amsks1996/usr/include -L/home/users/a/amsks1996/usr/bin  -o libOptim.so ./BayesOpt.o ./GlobalIterativeNewton.o ./GraphOptim.o ./Graph_Problem.o ./KOMO_Problem.o ./RidgeRegression.o ./benchmarks.o ./constrained.o ./convert.o ./gradient.o ./lagrangian.o ./newOptim.o ./newton.o ./opt-ceres.o ./optimization.o ./primalDual.o -lCore -lceres -lglog -lcholmod -llapack -lblas -lpthread  /usr/lib/x86_64-linux-gnu/libspqr.so /usr/lib/x86_64-linux-gnu/libtbbmalloc.so /usr/lib/x86_64-linux-gnu/libtbb.so /usr/lib/x86_64-linux-gnu/libcholmod.so /usr/lib/x86_64-linux-gnu/libccolamd.so /usr/lib/x86_64-linux-gnu/libcamd.so /usr/lib/x86_64-linux-gnu/libcolamd.so /usr/lib/x86_64-linux-gnu/libamd.so /usr/lib/x86_64-linux-gnu/liblapack.so /usr/lib/x86_64-linux-gnu/libf77blas.so /usr/lib/x86_64-linux-gnu/libatlas.so /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so /usr/lib/x86_64-linux-gnu/librt.so /usr/lib/x86_64-linux-gnu/libcxsparse.so /usr/lib/x86_64-linux-gnu/liblapack.so /usr/lib/x86_64-linux-gnu/libf77blas.so /usr/lib/x86_64-linux-gnu/libatlas.so /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so /usr/lib/x86_64-linux-gnu/librt.so /usr/lib/x86_64-linux-gnu/libcxsparse.so /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1 -lpthread /usr/lib/x86_64-linux-gnu/libglog.so -lpthread -lrt -lPhysX3Extensions -lPhysX3_x64 -lPhysX3Cooking_x64 -lPhysX3Common_x64 -lPxFoundation_x64 -lBulletSoftBody -lBulletDynamics -lBulletCollision  -lLinearMath -lrt -shared


### Path Variable Backup 

```
/beegfs/home/cluster/python/3.6.8/bin:/cluster/comp/binutils/2.29/bin:/cluster/comp/gcc/7.2.0/bin:/home/users/a/amsks1996/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/home/users/a/amsks1996/.local/bin:/home/users/a/amsks1996/bin:/cluster/cuda/cuda-9.2.88//bin/:/cluster/cuda/cuda-9.2.88//open64/bin/:/cluster/cuda/cuda-9.2.88//nvvm/:/cluster/cuda/cuda-9.2.88//samples/bin/x86_64/linux/release/:/home/users/a/amsks1996/usr/include
```

### Linking step error

AAA -lceres -lglog -lcholmod -llapack -lblas -lpthread  $(HOME)/usr/lib/libspqr.so $(HOME)/usr/lib/libtbbmalloc.so $(HOME)/usr/lib/libtbb.so $(HOME)/usr/lib/libcholmod.so $(HOME)/usr/lib/libccolamd.so $(HOME)/usr/lib/libcamd.so $(HOME)/usr/lib/libcolamd.so $(HOME)/usr/lib/libamd.so $(HOME)/usr/lib/liblapack.so $(HOME)/usr/lib/libf77blas.so $(HOME)/usr/lib/libatlas.so $(HOME)/usr/lib/libsuitesparseconfig.so $(HOME)/usr/lib/librt.so $(HOME)/usr/lib/libcxsparse.so $(HOME)/usr/lib/liblapack.so $(HOME)/usr/lib/libf77blas.so $(HOME)/usr/lib/libatlas.so $(HOME)/usr/lib/libsuitesparseconfig.so $(HOME)/usr/lib/librt.so $(HOME)/usr/lib/libcxsparse.so $(HOME)/usr/lib/libgflags.so.2.2.1 -lpthread $(HOME)/usr/lib/libglog.so -lceres -lglog -cholmod ld: final link failed: Nonrepresentable section on output

### PYTHON not shareable issue 

python is not built with share option

```
Re-install python 3.6.8 from source locally and build the project with that. 

Commands: 
    - curl -O https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz
    -  tar -xzf Python-3.6.8.tgz
    -  cd Python-3.6.8/
    -  ./configure --enable-optimizations --enable-shared --prefix=$HOME/usr/local/bin/
    -  make altinstall

```
### Python Linking Issue
Have to specify the path variables and  link the stuff

```
g++ -g -O3 -Wall -DRAI_PYBIND `python3-config --cflags` -DRAI_PHYSX -D_DEBUG -DPX_DISABLE_FLUIDS -DCORELIB -DPX32 -DLINUX -DRAI_BULLET -DBT_USE_DOUBLE_PRECISION -Wno-terminate -fPIC -static -L /home/users/a/amsks1996/usr/lib64/ -std=c++14 -L/home/users/a/amsks1996/usr/lib64 -o ry-Feature.o -c ry-Feature.cpp
/bin/sh: python3-config: command not found
        - Solved via adding python to the C_INCLUDE_PATH, and CPLUS_INCLUDE_PATH , and creating alias
```

## libassimp.so issue (Total Bitch!)

```
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `std::__cxx11::basic_stringstream<char, std::char_traits<char>, std::allocator<char> >::basic_stringstream()@GLIBCXX_3.4.26'
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `p2t::CDT::CDT(std::vector<p2t::Point*, std::allocator<p2t::Point*> > const&)'
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `std::__cxx11::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::basic_ostringstream()@GLIBCXX_3.4.26'
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `log@GLIBC_2.29'
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `powf@GLIBC_2.27'
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `pow@GLIBC_2.29'
/home/users/a/amsks1996/usr/lib64/libassimp.so.3: undefined reference to `p2t::CDT::AddHole(std::vector<p2t::Point*, std::allocator<p2t::Point*> > const&)'

libIrrXML.so.1()(64bit) -
libc.so.6(GLIBC_2.14)(64bit) ------ found libc.so.6 at /home/users/a/amsks1996/usr/lib64//libc.so.6
libgcc_s.so.1()(64bit) ------  found libgcc_s.so.1 at /cluster/comp/gcc/7.2.0/lib64/libgcc_s.so.1
libgcc_s.so.1(GCC_3.0)(64bit) ------  ound libgcc_s.so.1 at /cluster/comp/gcc/7.2.0/lib64/libgcc_s.so.1
libm.so.6()(64bit)  ------   found libm.so.6 at /home/users/a/amsks1996/usr/lib64//libm.so.6
libm.so.6(GLIBC_2.2.5)(64bit)   - 
libminizip.so.1()(64bit) ------  found libminizip.so.1 at /home/users/a/amsks1996/usr/lib64//libminizip.so.1
libpoly2tri.so.1.0()(64bit) ------- found libpoly2tri.so.1.0 at /home/users/a/amsks1996/usr/lib64//libpoly2tri.so.1.0
libpthread.so.0()(64bit)    ------- found libpthread.so.0 at /home/users/a/amsks1996/usr/lib64//libpthread.so.0
libpthread.so.0(GLIBC_2.2.5)(64bit) -
libstdc++.so.6()(64bit) ------ found libstdc++.so.6 at /cluster/comp/gcc/7.2.0/lib64/libstdc++.so.6
libstdc++.so.6(CXXABI_1.3)(64bit)   -
libstdc++.so.6(GLIBCXX_3.4)(64bit)  -
libstdc++.so.6(GLIBCXX_3.4.11)(64bit)   -
libstdc++.so.6(GLIBCXX_3.4.15)(64bit)   -
libstdc++.so.6(GLIBCXX_3.4.9)(64bit)    -
libz.so.1()(64bit)  -
rtld(GNU_HASH)



attempt to open ./libassimp.so succeeded
-lassimp (./libassimp.so)
libz.so.1 needed by ./libassimp.so --->  found libz.so.1 at /home/users/a/amsks1996/usr/lib64//libz.so.1
libminizip.so.1 needed by ./libassimp.so ---> found libminizip.so.1 at /home/users/a/amsks1996/usr/lib64//libminizip.so.1
libpoly2tri.so.1.0 needed by ./libassimp.so ---> found libpoly2tri.so.1.0 at /home/users/a/amsks1996/usr/lib64//libpoly2tri.so.1.0
libstdc++.so.6 needed by ./libassimp.so ---> found libstdc++.so.6 at /home/users/a/amsks1996/usr/lib64//libstdc++.so.6
libm.so.6 needed by ./libassimp.so ---> found libm.so.6 at /home/users/a/amsks1996/usr/lib64//libm.so.6
libgcc_s.so.1 needed by ./libassimp.so ---> found libgcc_s.so.1 at /home/users/a/amsks1996/usr/lib64//libgcc_s.so.1
libc.so.6 needed by ./libassimp.so ---> found libc.so.6 at /home/users/a/amsks1996/usr/lib64//libc.so.6
libGL.so.1 needed by /home/users/a/amsks1996/usr/lib64//libpoly2tri.so.1.0 ---> found libGL.so.1 at /usr/lib64/libGL.so.1
ld-linux-x86-64.so.2 needed by /home/users/a/amsks1996/usr/lib64//libstdc++.so.6 ---> found ld-linux-x86-64.so.2 at /home/users/a/amsks1996/usr/lib64//ld-linux-x86-64.so.2


libexpat.so.1 needed by /usr/lib64/libGL.so.1 ---> found libexpat.so.1 at /usr/lib64/libexpat.so.1
libxcb-dri3.so.0 needed by /usr/lib64/libGL.so.1 ---> found libxcb-dri3.so.0 at /usr/lib64/libxcb-dri3.so.0
libxcb-present.so.0 needed by /usr/lib64/libGL.so.1 ---> found libxcb-present.so.0 at /usr/lib64/libxcb-present.so.0
libxcb-sync.so.1 needed by /usr/lib64/libGL.so.1 ---> found libxcb-sync.so.1 at /usr/lib64/libxcb-sync.so.1
libxshmfence.so.1 needed by /usr/lib64/libGL.so.1 ---> found libxshmfence.so.1 at /usr/lib64/libxshmfence.so.1
libglapi.so.0 needed by /usr/lib64/libGL.so.1 ---> found libglapi.so.0 at /usr/lib64/libglapi.so.0
libselinux.so.1 needed by /usr/lib64/libGL.so.1 ---> found libselinux.so.1 at /usr/lib64/libselinux.so.1
libXext.so.6 needed by /usr/lib64/libGL.so.1 ---> found libXext.so.6 at /usr/lib64/libXext.so.6
libXdamage.so.1 needed by /usr/lib64/libGL.so.1 ---> found libXdamage.so.1 at /usr/lib64/libXdamage.so.1
libXfixes.so.3 needed by /usr/lib64/libGL.so.1 ---> found libXfixes.so.3 at /usr/lib64/libXfixes.so.3
libX11-xcb.so.1 needed by /usr/lib64/libGL.so.1 ---> found libX11-xcb.so.1 at /home/users/a/amsks1996/usr/lib64//libX11-xcb.so.1
libX11.so.6 needed by /usr/lib64/libGL.so.1 ---> found libX11.so.6 at /home/users/a/amsks1996/usr/lib64//libX11.so.6
libxcb.so.1 needed by /usr/lib64/libGL.so.1 ---> found libxcb.so.1 at /usr/lib64/libxcb.so.1
libxcb-glx.so.0 needed by /usr/lib64/libGL.so.1 ---> found libxcb-glx.so.0 at /usr/lib64/libxcb-glx.so.0
libxcb-dri2.so.0 needed by /usr/lib64/libGL.so.1 ---> found libxcb-dri2.so.0 at /usr/lib64/libxcb-dri2.so.0
libXxf86vm.so.1 needed by /usr/lib64/libGL.so.1 ---> found libXxf86vm.so.1 at /usr/lib64/libXxf86vm.so.1
libdrm.so.2 needed by /usr/lib64/libGL.so.1 ---> found libdrm.so.2 at /usr/lib64/libdrm.so.2
libpthread.so.0 needed by /usr/lib64/libGL.so.1 ---> found libpthread.so.0 at /home/users/a/amsks1996/usr/lib64//libpthread.so.0
libdl.so.2 needed by /usr/lib64/libGL.so.1 ---> found libdl.so.2 at /home/users/a/amsks1996/usr/lib64//libdl.so.2
libXau.so.6 needed by /usr/lib64/libxcb-dri3.so.0 ---> found libXau.so.6 at /usr/lib64/libXau.so.6
libpcre.so.1 needed by /usr/lib64/libselinux.so.1 ---> found libpcre.so.1 at /usr/lib64/libpcre.so.1


ld: warning: cannot find entry symbol _start; not setting start address
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `p2t::CDT::AddHole(std::vector<p2t::Point*, std::allocator<p2t::Point*> > const&)'
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `log@GLIBC_2.29'
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `std::__cxx11::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >::basic_ostringstream()@GLIBCXX_3.4.26'
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `p2t::CDT::CDT(std::vector<p2t::Point*, std::allocator<p2t::Point*> > const&)'
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `std::__cxx11::basic_stringstream<char, std::char_traits<char>, std::allocator<char> >::basic_stringstream()@GLIBCXX_3.4.26'
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `powf@GLIBC_2.27'
/home/users/a/amsks1996/usr/lib64//libassimp.so: undefined reference to `pow@GLIBC_2.29'
```

### Relocation Error : Third Party Software

```
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/usr/lib64


readlink -f /path/file


/beegfs/home/cluster/comp/gcc/7.2.0/lib64/libstdc++.so.6.0.24

```



<!-- %%% -->
# Misc: Setting up Envrironment

#### Installing VS Code 

- Install the .deb link from 

```
https://code.visualstudio.com/Download
```
- Navigate to the Downloads folder

```
cd ~/Downloads
```
- Install using dpkg 

```
sudo dpkg -i Name_of_file
```

#### Installing Sublime Text

- Open Terminal and install the key

```
wget -qO - https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add -
```

- Add the apt repository

```
echo "deb https://download.sublimetext.com/ apt/stable/" | sudo tee /etc/apt/sources.list.d/sublime-text.list
```

- Finally, check updates and install sublime-text via apt

```
sudo apt update

sudo apt install sublime-text
```

#### Installing CUDA

- Remove all NVIDIA traces in the system

```
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*
```

- Setup the correct CUDA PPA on the system

```
sudo apt update
sudo add-apt-repository ppa:graphics-drivers

sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'

sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
```

- Install CUDA 10.1 packages

```
sudo apt update
sudo apt install cuda-10-1
sudo apt install libcudnn7
```

- To specify PATH to CUDA in ‘.profile’ file, open it :

```
sudo gedit ~/.profile
```

- Then add this to the end of the file

```
# set PATH for cuda 10.1 installation
if [ -d "/usr/local/cuda-10.1/bin/" ]; then
    export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```

- Restart and check the versions for the installation. For CUDA, NVIDIA and libcudnn

```
nvcc  – version

nvidia-smi

/sbin/ldconfig -N -v $(sed ‘s/:/ /’ <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn
```

#### Setting up Virtual Environment for DL

- Install virtual env using apt

```
sudo apt update

sudo apt install virtualenv
```

- Install virtualenvwrapper

```
sudo apt install virtualenvwrapper
```

- Check the installation paths ( Should be in /usr/bin/ )

```
which virtualenv

which virtualenvwrapper
```

- Create the new environment for keras and tensorflow

```
mkvirtualenv keras_tf -p python3
```

- Check if the global commands work

```
workon keras_tf 

deactivate
```

- Install and Check tf

```
pip install --upgrade tensorflow

python
>>> import tensorflow as tf
>>> tf.__version__

```

- Install Keras related dependencies

```
pip install numpy scipy
pip install scikit-learn
pip install pillow
pip install h5py
```

- Install keras

```
pip install keras
```

#### Setting up Jekyll for local website
- First check if ruby and gem are already installed on your system

```
ruby -v
gem -v 
```

- if no, then install ruby

```
sudo apt-get install ruby-full
```

- Install other dependencies

```
build-essential zlib1g-dev
```

- Configure gem installation related stuff in the bashrc

```
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

- Install jeyll and bundler

```
gem install jekyll bundler
```

- Clone the github pages repo and navigate to it. Then make an orphan branch

```
git checkout --orphan gh-pages
```

- To create a new Jekyll site, use the jekyll new command, replacing VERSION with the current dependency version for Jekyll

```
bundle exec jekyll VERSION new .
```

- Update the gemfile with the sources

```
gem "github-pages", "~> VERSION", group: :jekyll_plugins
```

- Check if any other related dependencies are missing

```
bundle install
```

- Run the localhost:4000 website

```
bundle exec jekyll serve
```