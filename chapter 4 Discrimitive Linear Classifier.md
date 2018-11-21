<!-- $theme: default -->

# Chapter 4 
# Discriminative Linear Classifiers
---
## 4.1 
## Log-Linear Models

---
### Log-linear Models are a type of *discriminative linear models* that give probabilistivally interpretable scores to outputs.
#
The form of Discriminative Linear Classifiers:


${ score } ( x , y ) = \vec { \theta } \cdot \vec { \phi } ( x , y )$
where $\vec { \theta } \cdot \vec { \phi }(x,y)$ is a feature vector representation of $(x,y)$ and $\vec { \theta}$ is is the model(parameter vector). 

Different instances of discriminative linear models:
* SVMs
* Perceptrons
* Log-linear models

---
### Define a probabilistic linear discriminative model
#
Inspiration from an observation of the Naive Bayes classifier:

P ( c | d ) $\propto$ $\prod _ { i = 1 } ^ { n } P \left( w _ { i } | c \right) P ( c )$

The log form of P(c|d) is a linear model:

log P ( c | d ) $\propto$ $\sum _ { i = 1 } ^ { n } \log P \left( w _ { i } | c \right) + \log P ( c )$

which is similar to a discriminative linear model.

---
### Log-linear model for multi-class classification
#
Make  $P ( y | x )$ $\propto$ $e ^ { \vec { \theta } \cdot \vec { \phi } ( x , y ) }$ so that $logP(y|x)$ $\propto$ $\vec { \theta } \cdot \vec { \phi } ( x , y )$

Derive $P(y|x)$ of log-linear models by normalisation over C(the set of all possible outputs):

$P ( y | x ) = \frac { e ^ { \vec { \theta } \cdot \vec { \phi } ( x , y ) } } { \sum _ { y ^ { \prime } \in C } e ^ { \vec { \theta } \cdot \vec { \phi } \left( x , y ^ { \prime } \right) } }$

Which can also be described as:

$P ( y | x ) = \ { softmax } _ { C } ( \vec { \theta } \cdot \vec { \phi } ( x , y ) )$

**Softmax function**(an exponential function) maps an input in [-$\infty$,$\infty$] to [0,1].


---
### Log-linear model form for binary classification
#
**Logit function** or **sigmoid function** is an exponential function that maps a number in  [-$\infty$,$\infty$] to [0,1].

$Logit(x)=$ $\frac { e ^ { x } } { 1 + e ^ { x } }$

Using the logit function, a binary classifier $score(y=+1)=$$\vec { \theta } \cdot \vec { \phi } ( x )\in[-\infty,+\infty]$ can be mapped into a probabilistic classifier

$P(y=+1|x)=logit(\vec { \theta } \cdot \vec { \phi } (x))$ $= \frac { e ^ { \vec { \theta } \cdot \vec { \phi } ( x ) } } { 1 + e ^ { \vec { \theta } \cdot \vec { \phi } ( x ) } }$
$P(y=-1|x)=logit(\vec { \theta } \cdot \vec { \phi } (x))$ $= \frac { e ^ { \vec { \theta } \cdot \vec { \phi } ( x ) } } { 1 + e ^ { \vec { \theta } \cdot \vec { \phi } ( x ) } }$


---
### Training log-linear models
#
We want to train the parameters $\vec { \theta }$ so that the scores $P$($\cdot$) truly represent probabilities.

Given a set of training examples $D = \left. \left\{ \left( x _ { i } , y _ { i } \right) \right\} \right| _ { i = 1 } ^ { N }$ ,

we use maximum likelihood estimation (MLE) to train a log-linear model.
The training objective is 

$P ( Y | X ) = \prod _ { i } P \left( y _ { i } | x _ { i } \right)$

which is maximising the conditional likelihood of training data.

---
## 4.1.1
## Training Binary log-linear models

---
Given $P ( y = + 1 | x ) = \frac { e ^ { \vec { \theta } \cdot \vec { \phi } ( x ) } } { 1 + e ^ { \vec { \theta } \cdot \vec { \phi } ( x ) } }$, our MLE training objective is 

$P ( Y | X ) = \prod _ { i } P \left( y _ { i } | x _ { i } \right)$
$= \prod _ { i ^ { + } } P ( y = + 1 | x _ { i } ) \prod _ { i ^ { - } } ( 1 - P ( y = + 1 | x _ { i } ) )$

Maximising $P(Y|X)$ can be achieved by maximising

$logP(X|Y)$
$= \sum _ { i }log P$ $\left( y _ { i } | x _ { i } \right)$
$= \sum _ { i ^ { + } } \log P ( y = + 1 | x _ { i } ) + \sum _ { i ^ { - } } \log ( 1 - P ( y = + 1 | x _ { i } ) )$
$= \sum _ { i ^ { + } } \log \frac { e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) } } { 1 + e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) } } + \sum _ { i ^ { - } } \log \frac { 1 } { 1 + e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) } }$
$= \sum _ { i ^ { + } } \left( \vec { \theta } \cdot \vec { \phi } \left( x _ { i } ^ { + } \right) - \log \left( 1 + e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } ^ { + } \right) } \right) \right) - \sum _ { i ^ { - } } \log \left( 1 + e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } ^ { - } \right) } \right)$

---
For linear model, the grandient of the objective is :

$\vec { g } = \frac { \partial \log P ( Y | X ) } { \vec { \theta } }$
$= \sum _ { i ^ { + } } \left( \vec { \phi } \left( x _ { i } \right) - \frac { e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) } } { 1 + e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) } } \vec { \phi } \left( x _ { i } \right) \right) - \sum _ { i ^ { - } } \left( \frac { e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) } } { 1 + e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) } } \vec { \phi } \left( x _ { i } \right) \right)$
$= \sum _ { i ^ { + } } \left( 1 - \frac { e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) } } { 1 + e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) } } \right) \vec { \phi } \left( x _ { i } \right) - \sum _ { i ^ { - } } \left( \frac { e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) } } { 1 + e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) } } \right) \vec { \phi } \left( x _ { i } \right)$
$= \sum _ { i ^ { + } } ( 1 - P ( y = + 1 | x _ { i } ) ) \vec { \phi } \left( x _ { i } \right) - \sum _ { i ^ { - } } P ( y = + 1 | x _ { i } ) \vec { \phi } \left( x _ { i } \right)$

---
### Gradient descent
#
 A simple numerical solution to the minimization of convex functions.
 
 
 *Gradient Descent*
 **Inputs**: An objective function F;
 **Initialization**:$\vec\theta_{0}$=random(), t=0;
 **repeat**:
 |$\vec g_{t}$=$\nabla \vec \theta_{t}$$F(\vec \theta_{t})$;
 |$\vec \theta _ {t+1}$=$\vec \theta_{t}-\alpha \vec g_{t}$;
 |$t$=$t+1$;
 **until** $||\vec \theta_{t}-\vec\theta_{t-1}||<\epsilon$
 **outputs**:$\vec\theta_{t}$;
 Here $\alpha$ is the **learning rate**(hyper-parameter), the value of $\alpha$ influences both the accuracy and the efficiency of gradient descent.
 
---
### Stochastic gradient descent
#
(batch) Gradient descent can be used to minimize the negative log-likelihood of log-linear models.
But finding $\vec g$ at each iteration can be computationally inefficient.

A common solution is a *online* learning algorithm called **stochastic gradient descent (SGD)**
SGD updates model parameters more frequently, converge much faster, while does not always converge to the same otimal point as gradient descent.
The algorithm of SGD training is highly similar in structure to the perceptron algorithm but more fine-grained and results in the probabilistic interpretation of the model.

---
### Mini-batch SGD
#
A compromise between gradient descent and SGD training.

Split the set of training examples $D$ into several equal-sized subsets $D_{1},D_{2},...,D_{M}$, each containing $N/M$ training examples.

The mini-batch size $N/M$ controls the tradeoff between efficiency and accuracy of approximation.

Make a **random shuffle** to the training set before each training iteration, which can improve the accuracies for some NLP tasks and datasets. 

---
## 4.1.2
## Training multi-class log-linear models

___
For training pairs $\vec \phi(x_{i},y_{i})$, where $y_{i}\in C,|C|>=2.$
The probability of $y_{i}=\mathbf{c},\mathbf{c}\in C$ is:

$P(y_{i}=c|x_{i})=$$\frac { e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , \mathbf { c } \right) } } { \sum _ { \mathbf { c } ^ { \prime } \in C } e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , \mathbf { c } ^ { \prime } \right) } }$

The log-likelihood of $D$ is

$\log P ( Y | X ) = \sum _ { i } \log P \left( y _ { i } | x _ { i } \right)$ $= \sum _ { i } \left( \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , y _ { i } \right) - \log \left( \sum _ { \mathbf { c } \in C } e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , \mathbf { c } \right) } \right) \right)$
For each training example,
$logP(y_{i}|x_{i})=$ $\vec { \theta } \cdot \vec { \phi } \left( x _ { i } , y _ { i } \right) - \log \left( \sum _ { \mathbf { c } \in C } e ^ { \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , \mathbf { c } \right) } \right)$
The local gradient is:

$\vec { g } = \frac { \partial \log P \left( y _ { i } | x _ { i } \right) } { \partial \vec { \theta } }$
$= \sum _ { \mathbf { c } \in C } \left( \vec { \phi } \left( x _ { i } , y _ { i } \right) - \vec { \phi } \left( x _ { i } , \mathbf { c } \right) \right) P ( y = \mathbf { c } | x _ { i } )$

---
## 4.2
## SGD training of SVMs

--- 
### 4.2.1 Binary classification
#
The training objective of binary classification SVM can be modified into minimising $\frac { 1 } { 2 } | | \vec { \theta } | | ^ { 2 } + C \sum _ { i } \max \left( 0,1 - y _ { i } \left( \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) \right) \right)$ given $D = \left. \left\{ \left( x _ { i } , y _ { i } \right) \right\} \right| _ { i = 1 } ^ { N }$
which is equivalent to minimizing
$\sum _ { i } \max \left( 0,1 - y _ { i } \left( \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) \right) \right) + \frac { 1 } { 2 } \lambda \| \vec { \theta } \| ^ { 2 }$
where $\lambda$ is a hyper-parameter of the model,just as the role of $C$.

This equation can be optimised using sub-gradient descent.
For each training example, the derivative of the local training objective is
$\left\{ \begin{array} { l l } { \lambda \vec { \theta } } & { \text { if } 1 - y _ { i } \vec { \theta } \cdot \vec { \phi } \left( x _ { i } \right) \leq 0 } \\ { \lambda \vec { \theta } - y _ { i } \vec { \phi } \left( x _ { i } \right) } & { \text { otherwise } } \end{array} \right.$

---
### Multi-class SVM.
#
The training objective of multi-class SVM is to minimise
$\frac { 1 } { 2 } \left. | \vec { \theta } | \right| ^ { 2 } + C \sum _ { i } \max \left( 0,1 - \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , y _ { i } \right) + \max _ { \mathbf { c } \neq y _ { i } } \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , \mathbf { c } \right) \right)$
which is equivalent to minimising
$\sum _ { i } \max \left( 0,1 - \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , y _ { i } \right) + \max _ { \mathbf { c } \neq y _ { i } } \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , \mathbf { c } \right) \right) + \frac { 1 } { 2 } \lambda \| \vec { \theta } \| ^ { 2 }$
where $(x_{i},y_{i})\in D,\lambda =\frac{1}{C}.$

Using SGD, the derivative for each training example is:
$\left\{ \begin{array} { l l } { \lambda \vec { \theta } } & { \text { if } 1 - \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , y _ { i } \right) + \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , z _ { i } \right) \leq 0 } \\ { \lambda \vec { \theta } - \left( \vec { \phi } \left( x _ { i } , y _ { i } \right) - \vec { \phi } \left( x _ { i } , z _ { i } \right) \right) } & { \text { otherwise } } \end{array} \right.$
where $z_{i}=argmax_{\mathbf{c} \neq y_{i}}\vec \theta \cdot \vec \phi(x_{i},\mathbf{c)}.$

---
### 4.2.2 A perceptron training objective function
#
Perceptron updates can also be viewed as SGD training of a certain objective function.
The training objective is to minimize 
$\sum _ { i = 1 } ^ { N } \max \left( 0 , - \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , y _ { i } \right) + \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , \arg \max _ { \mathbf { c } } \vec { \theta } \cdot \vec { \phi } \left( x _ { i } , \mathbf { c } \right) \right) \right)$

---
## 4.3
## A Generalised Linear Model for classification
---
### Generalised linear classification model
![](/Users/liuhanmeng/Desktop/figure4.png)

* parameter vector $\vec \theta$
* feature vector $\vec \phi$
* output class label $y$ using the dot product $\vec \theta \cdot \vec \phi$

---
### 4.3.1 Unified Online Training
#
![](/Users/liuhanmeng/Desktop/algorithm.png)
Given a set of training data $D = \left. \left\{ \left( x _ { i } , y _ { i } \right) \right\} \right| _ { i = 1 } ^ { N }$, the algorithm goes over $D$ for $T$ iterations, processing each training example $(x_{i},y_{i}),$and update model parameters when neccessary.

---
### $ParameterUpdate(x_{i},y_{i})$ for perceptrons,SVMs and log-linear models
![](/Users/liuhanmeng/Desktop/Screenshot%202018-11-15%20at%204.27.43%20PM.png)

---
### 4.3.2 Loss Functions
The training objectives for linear models can be regarded as to minimise different **loss functions** of a model over a training set.
![](/Users/liuhanmeng/Desktop/Screenshot%202018-11-15%20at%205.11.09%20PM.png)

---
### Different types of loss functions
#
* **Hinge loss**: the loss functions of SVMs and perceptrons
* **Log-likelihood loss**: the loss functions for log-linear models
* **$0/1$ loss**:loss is 1 for an incorrect output and 0 for a correct output.

![](/Users/liuhanmeng/Desktop/Screenshot%202018-11-15%20at%205.24.11%20PM.png)

---
### Risks
#
The true **expected risk** of a linear model with parameter can be formulated as

$risk(\vec { \theta } ) = \sum _ { x , y }   loss ( \vec { \theta } \cdot \phi ( x , y ) ) P ( x , y ),$

which cannot be calculated, we use **empirical risk** as a proxy

$\tilde{risk} (\vec { \theta } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } loss \left( \vec { \theta } \cdot \phi \left( x _ { i } , y _ { i } \right) \right) , \left( x _ { i } , y _ { i } \right) \in D$

---
### 4.3.3 Regularization
#
A large element in the parameter vector $\vec \theta$ implies higher reliance of the model to its corresponding feature, sometimes unnecessarily much.

**L2 regularisation** and **L1 regularisation** directly minimise a polynomial of $\vec \theta$ in loss functions, help reduce over-fitting of models on given training data.

---
## 4.4
## Working with Multiple Models

---
### 4.4.1 Comparing model performances
#
* different training objectives (large margin or log-likelihood)
* different feature definitions
* different hyperparameters (number of training iterations, learning rate)

A useful way to make choice between alternative models is to make *Empirical comparisons*

We can calculate the probability of obtaining the observed test results, we take small values of such a probability as significance levels, using the significance levels the degree of generalizability.

---
### 4.4.2 Ensemble models
#
**Ensemble approach**: a combination of multiple models for better accuracies.

**Voting**: a simple method to ensemble different models.

Given a set of models $M=(m_{1},m_{2},...m_{|M|})$ and output classes $C=(c_{1}, c_{2},...,c_{|C|}),$ the output class $y$ for a given input $x$ can be decided by counting the vote (*hard $0/1$ votes*):

$v _ { i } = \sum _ { j = 1 } ^ { | M | } \mathbf { 1 } \left( y \left( m _ { j } \right) , c _ { i } \right)$
*majority voting* chooses the class label that receives more than half the total votes, 
*plurality voting* chooses the class with the most votes.

More fine-grained voting methods are soft voting and weighted voting.

---
**Stacking**: 
use the outputs of one model as features to inform another model.

**Training for stacking**
the stacking method trains $A$ after $B$ is trained.

We use **K-fold jackknifing** to make model $B$ output accuracies on the training data as close to the test scenario as possible.

**Bagging** use different subsets of $D$ to obtain different models and then emsemble them. Voting is then performed between models given a test case. Bagging can outperform a single model for many tasks.

---
### 4.4.3 Tri-training and co-training
#
*Semi-supervised learning* use different models trained on $D$ to predict the labels on a set of unlabelled data $U$, augmenting $D$ with the outputs that most models agree on.
* Tri-training
* Co-training
* Self-training

the more accurate the baseline models are on $U$, the more likely that the new data form $U$ can be correct and useful.

---
## 4.5 Summary
* Log-linear models for binary and multi-class classification
* Stochastic Gradient Descent (SGD) training of log-linear models and SVMs
* A generallised linear discriminative model for text classification
* The correlation between SVMs, perceptrons and log-linear models in terms of training of training objective (loss) functions and regularisation terms
* Significance testing
* Ensemble methods for integrating different models