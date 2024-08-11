---
layout: post
title: "One-Step Generative ODE Flow"
date: 2024-08-10
categories: None
excerpt: My viewpoint of generative ODE flow.
permalink: /post/generative-ode-flow
---

## Aim

This post intends to introduce you my interpretation of the diffusion models (ODE-based, specifically), and some insights of my recent work, which develops a one-step generation scheme for diffusion models, exploiting the deterministic nature of the ODE flows.
Throughout the post, I'll not dive into the mathematical details and theorems, but focus on the intuitive ideas only.


## Task

In generative learning, we aim to generate samples from a target distribution. We probably know nothing about the density of the target distribtion, but have plenty random samples.
A most common example is the the distribution of images of dog faces. We certainily don't know what the distribution looks like, but we can find tons of such images in real life.

## Model

In plain word, the diffusion models bridges between a source distribution and a target distribution. 
The source distribution is usually easy to sample from, e.g. the standard normal distribution.
It is natural to imagine the diffusion model as a river that flows through the source distribution at start, and reaches the target distribution in the end.
Now what's left is how the river flows? 
As you can imagine, there are infinitely many ways to build the process!

## Interpolant Viewpoint

Recall the elementary math, two fixed point in 2D Euclidean space determines a line that passes throught them simultaneously.
Now let's consider the distributional counterpart of this idea.
Given the source distribution $\mu_0$ and the target distribution $\mu_1$, we want to design a process such that its marginal distribution is exactly $\mu_0$ at time 0, and $\mu_1$ at time 1.
A natural design is by convolution. Let $X_0 \sim \mu_0$ and $X_1 \sim \mu_1$, define a series of random variable $X_t$ by (rescaled) convolution:

$$X_t = \alpha_t X_0 + \beta_t X_1,$$

where $\alpha_t$ and $\beta_t$ are interpolant coefficients such tha, when $t=0$, $X_t$ is exactly $X_0$ obeying $\mu_0$, and when $t=1$, $X_t$ is exactly $X_1$ obeying $\mu_1$. 
As for $t \in (0, 1)$, $X_t$ is a convoltion of $X_0$ rescaled by $1-t$ and $X_1$ rescaled by $\alpha_t$ and $\beta_t$ respectively.
We can design infinitely many combinations of $\alpha_t$ and $beta_t$ (satisfying some mild conditions omitted here), such as $1-t$ and $t$; or $\sqrt{1-t^2}$ and $t$, ...

Look into the series of density functions of $X_t$, denoted by $(p_t(X_t))_{t \in [0, 1]}$. I'll show you the result directly and omit the mathematical details here. There is a linear continuity equation corresponding to this process:

$$\partial_t p_t + \nabla_x \cdot (p_t v(t, x)) = 0, (t, x) \in [0, 1] \times \mathbb{R}^d,$$

where $v(t, x) = \mathbb{E} [\dot{\alpha}_t X_0 + \dot{\beta}_t X_1 \| X_t = x]$.


By the method of characteristics, there is a corresponding ODE system

$$\dot{x}_t = v(t, x_t).$$

Now, assume that we can compute or approximate $v(t, x)$ by some means, then simply sampling $x_0$ from $\mu_0$ (which is easy as mentioned before), and then solve the ODE system with some numerical solver (such as Euler's method) from time 0 to 1, we will obtain a sample $x_1$ randomly drawn from $\mu_1$.

## Velocity Estimation

What remains now is how can we find a good way to compute/approximate the velocity field?
Thanks to the great approximation power of the neural networks, and the fact that $v(t, x = \mathbb{E} [X_1 - X_0 \| X_t = x]$, we are able to train a neural network by minimizing the velocity matching loss:

$$L(v_\theta) = \int_0^1 \mathbb{E}_{X_0, X_1} \| v_\theta(t, X_t) - \dot{\alpha}_t X_0 -  \dot{\beta}_t X_1\|_2^2 \, \mathrm{d}t, $$

which can be easily interpreted to its empirical counterpart.

## Connection with Denoising Score Matching

The velocity matching technique is mathematically equivalent to the well-known denoising score matching technique introduced in score-based diffusion models.

The Stein score is the gradient of log density function of the distribution. Denote the score function of the marginal density at time $t$ by $s(t, x)$, then some calculation yields

$$v(t, x) = \frac{\dot{\beta}_t}{\beta_t} x + \alpha_t^2 (\frac{\dot{\beta}_t}{\beta_t} - \frac{\dot{\alpha}_t}{\alpha_t}) s(t, x).$$

By this equation, the denoising score matching would be just equivalent to the velocity matching mathematically.

In recent works, people realize that the score function itself may changes rapidly near time $1$, which makes it harder to train neural networks.
A good practice is to adopt the denoiser setting defined by 

$$d(t, x) = \mathbb{E} [X_1 | X_t = t].$$

Ideally, the denoiser should be of the same range as $X_1$, which is good as it's finite.
Just as before, there is an equivant expression between $v(t, x)$ and $d(t, x)$:

$$v(t, x) = \frac{\dot{\alpha}_t}{\alpha_t} x + \beta_t (\frac{\dot{\beta}_t}{\beta_t} - \frac{\dot{\alpha}_t}{\alpha_t}) d(t, x).$$


In practice, to align with mainstream works, we would adopt the denoiser matching setting.

## Numerical Solver

Assume that we have trained a velocity field, what remains is how to numerically solve the ODE system.

In the early days of diffusion models, people refer to the forward Euler method, which was sufficient to generate samples back then.
However, as time goes by, people realized that the Euler method is too slow compared to other refiend method, in convergence rate.
Many acceleration techniques have been developed over that past years. 
Among them I'd like to single out the exponential integrator, which I find most useful in practice.

The exponential integrator has been developed to solve differential equations in the past century. 
It exploits the semi-linearity of the ODE system. By the "variantion of constants", we can solve the linear part analytically, rather than passing it to the numerical solver.
In literatures, this solver has been re-discovered or re-invented many times. 
The first order solver of several famous literatures are exactly the exponential integrator (maybe under another name). 
In our setting, the exponential integrator yileds solution:

$$x_s = \frac{\alpha_s}{\alpha_t}x_t + \alpha_s (\frac{\beta_s}{\alpha_s} - \frac{\beta_t}{\alpha_t}) d(t, x_t).$$

It's worth noting that the exponential integrator is a first order solver, just like the forward Euler method, but it eases the error propagation of linear terms, and have been widely chosen as the effective sampler for ODE-based diffusion models.

## Characteristic Reviewed

A good property of well defined ODE system is that: for any given initial condition, the solution trajectory is determined. 
That is to say, for each noise $x_1$, there is one and only one corresponding $x_1$ solved by the ODE.
This deterministic property holds only on ODE-based models. For SDE models, the solution trajectory is stochastic due to the diffusion part.
A natural idea is to train a new network which mimics the solution of the ODE system over any time interval. 
The new network, denoted by $g(t, s, x)$ should take in three inputs.
Once trained, to generate a sample, we simply tell the $g$: the initial point $x$, the starting time $x_t$ and the end time $s$, then ideally, it would output the solution $x_s$.

The question is how to design and train this new network?
There're 3 key insights during our researches:

1. A good parameterization and initialization

	We absolutely do not want to train the new network from scratch, but reuse the pretrained denoiser network.
	Inspired by the exponential integrator, we parameterize $g$ by 

	$$g(t, s, x) = \frac{\alpha_s}{\alpha_t}x_t + \alpha_s (\frac{\beta_s}{\alpha_s} - \frac{\beta_t}{\alpha_t}) \bar{d}(t, s, x_t),$$

	where $\bar{d}$ is initialized from a pretrained $d_\theta$ with an extra temporal input $s$.

2. Local consistency

	Since the new network is an integral network, its time derivative should be identical to the velocity field.
	Thus we can reuse the velocity (denoiser) matching technique again to ensure the local consistency: when $s = t$, $\bar{d}(t, s, x) = d(t, x)$.

3. Glocal consistency

	The integral operator admits a semi-group property: 

	$$g(u, s, g(t, u, x)) = g(t, s, x).$$
