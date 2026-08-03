---
title: 'ProximalAlgorithms.jl: A Julia framework for proximal algorithms in nonsmooth optimization'
tags:
  - Julia
  - nonsmooth optimization
  - nonconvex optimization
  - convex optimization
  - proximal algorithms
  - splitting methods
authors:
  - name: Lorenzo Stella
    orcid: ...
    affiliation: 1
  - name: Niccoló Antonello
    orcid: 0000-0002-0803-5385
    affiliation: 2
  - name: Alberto De Marchi^[corresponding author]
    orcid: 0000-0002-3545-6898
    affiliation: 3
  - name: Puya Latafat
    orcid: 0000-0002-7969-8565
    affiliation: 4
affiliations:
 - name: AWS AI Labs, Berlin, Germany (work done prior to joining Amazon)
   index: 1
 - name: ...
   index: 2
 - name: University of the Bundeswehr Munich, Germany
   index: 3
 - name: IMT School for Advanced Studies Lucca, Italy
   index: 4
date: 3 August 2026
bibliography: paper.bib
---

# Summary

[ProximalAlgorithms.jl](https://github.com/JuliaFirstOrder/ProximalAlgorithms.jl) is a Julia package that implements a broad collection of first-order splitting methods, commonly known as *proximal algorithms*, for solving structured nonsmooth optimization problems of the form
\begin{equation}\label{eq:generic}
    \underset{x \in \mathbb{R}^n}{\text{minimize}} \quad \sum_{i=1}^{N} f_i(x),
\end{equation}
where $N$ depends on the algorithm of choice, and each term $f_i$ may be smooth or nonsmooth, convex or, for some algorithms, nonconvex.
Terms can also be composed with a linear mapping, as in $h(Lx)$, which arises frequently in imaging, signal processing, and control applications.
Rather than assuming access to explicit derivatives or closed-form solutions, proximal algorithms rely only on first-order information about each term: its gradient $\nabla f_i$, when it is differentiable, or its proximal mapping
\begin{equation}\label{eq:prox}
    \operatorname{prox}_{\gamma f_i}(v) = \underset{x \in \mathbb{R}^n}{\arg\min} \left\{ f_i(x) + \frac{1}{2\gamma}\|x - v\|^2 \right\},
\end{equation}
when it is not.
This makes the methods well suited to large-scale problems in which nonsmooth terms (indicator functions of simple sets, sparsity-inducing penalties, norms, etc.) admit an inexpensive proximal mapping, even though the objective as a whole is not directly minimizable in closed form.

ProximalAlgorithms.jl organizes its solvers according to the structure of the problem they address:

- **Two-term splitting** ($f + g$): (Fast) proximal gradient / forward-backward splitting [@lions-mercier-1979; @tseng-2008; @beck-teboulle-2009], Douglas-Rachford splitting [@eckstein-bertsekas-1992], and the line-search-based methods PANOC [@stella-themelis-sopasakis-patrinos-2017], PANOC+ [@demarchi-themelis-2022], ZeroFPR [@themelis-stella-patrinos-2018], and Douglas-Rachford line search (DRLS) [@themelis-stella-patrinos-2022].
- **Three-term splitting** ($f + g + h$): the Davis-Yin splitting scheme [@davis-yin-2017].
- **Primal-dual splitting** ($f + g + h \circ L$): Chambolle-Pock [@chambolle-pock-2011], Vũ-Condat [@vu-2013; @condat-2013], and the asymmetric forward-backward-adjoint (AFBA) method [@latafat-patrinos-2017].

All solvers are implemented as Julia iterators, so that users can either call a solver directly to obtain a solution, or loop over the underlying iterator to gain fine-grained control over logging, stopping criteria, or warm-starting.
Gradients are obtained, by default, through automatic differentiation via [DifferentiationInterface.jl](https://github.com/gdalle/DifferentiationInterface.jl), which gives access to essentially every Julia automatic-differentiation backend, while proximal mappings follow the lightweight [ProximalCore.jl](https://github.com/JuliaFirstOrder/ProximalCore.jl) interface implemented, among others, by [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl).
Users can just as easily supply hand-written gradients or proximal mappings for custom objective terms.

# Statement of need

Nonsmooth composite optimization problems of the form \eqref{eq:generic} arise pervasively in statistics, machine learning, signal and image processing, and optimal control, wherever nonsmooth regularizers or constraints are combined with a differentiable data-fidelity term.
Because no single algorithm applies uniformly well to every instance of this problem class, practitioners benefit from a library that offers many interchangeable methods behind a common interface, rather than a single monolithic solver. Moreover, methods differ in the number and structure of terms they support, in the convexity assumptions they require, and in their empirical convergence behavior.

ProximalAlgorithms.jl deliberately does not provide a modeling language that automatically decomposes an arbitrary objective into terms and picks a matching algorithm, in the way that, e.g., disciplined convex programming frameworks do for problems expressible in their grammar.
Instead, the user formulates the problem directly in terms of the objective terms ($f$, $g$, $h$, and, where relevant, a linear operator $L$) that a chosen algorithm expects, and the package supplies the iteration logic, termination handling, and (where applicable) adaptive step-size or line-search machinery.
This design keeps the package lightweight and composable: new algorithms can be added by implementing a small, self-contained iterator interface, and existing algorithms can be reused as subproblem solvers inside other packages.

This complements, rather than duplicates, other optimization software in the Julia ecosystem.
For instance, [RegularizedOptimization.jl](https://github.com/JuliaSmoothOptimizers/RegularizedOptimization.jl) [@gollier-habiboullah-leconte-baraldi-demarchi-orban-diouane-2026] implements model-based trust-region and quadratic-regularization methods for a similar class of nonsmooth problems. Such methods typically require fewer evaluations of the smooth term and its gradient than the line-search-based methods in ProximalAlgorithms.jl, at the cost of more proximal-operator evaluations.
ProximalAlgorithms.jl gives users of the JuliaFirstOrder and broader Julia optimization ecosystems direct access to a wide range of well-established, first-order splitting algorithms through a single, consistent, allocation-conscious interface, backed by [ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl)'s extensive library of proximable functions.

# Example

The following condensed example minimizes the two-dimensional Rosenbrock function added to a l1-norm regularizer, using the accelerated (fast) forward-backward splitting method:

```julia
using ProximalOperators, ProximalAlgorithms

struct Rosenbrock{T}
  a::T
  b::T
end
(f::Rosenbrock)(x) = (f.a-x[1])^2 + f.b*(x[2]-x[1]^2)^2
function ProximalAlgorithms.value_and_gradient(f::Rosenbrock, x)
  fx = f(x)
  gx = similar(x)
  gx[1] = 2*(x[1]-f.a) + 4*f.b*(x[1]^2-x[2])*x[1]
  gx[2] = 2*f.b*(x[2]-x[1]^2)
  return fx, gx
end

f = Rosenbrock(1.0, 100.0)
g = NormL1(2.0)

solver = ProximalAlgorithms.FastForwardBackward(tol = 1e-5, verbose = true)
x, iters = solver(x0 = ones(2), f = f, g = g)
```

Here `f` wraps the smooth Rosenbrock term and `g` is the scaled l1-norm, whose proximal mapping is the coordinate-wise soft-thresholding operator.
The `FastForwardBackward` algorithm is instantiated with a termination tolerance and verbosity option, and then called with the initial point and the two objective terms to produce the solution and the iteration count. One can wrap the smooth cost for automatic differentiation, as with `f_auto`:

```julia
using Zygote
using DifferentiationInterface: AutoZygote

f_auto = ProximalAlgorithms.AutoDifferentiable(
    x -> (f.a-x[1])^2 + f.b*(x[2]-x[1]^2)^2,
    AutoZygote(),
)
x, iters = solver(x0 = ones(2), f = f_auto, g = g)
```

## Numerical illustration

Add comparison tables or plots here, comparing a few algorithms such as `ForwardBackward`, `FastForwardBackward`, `PANOC`, and `ZeroFPR`, on a representative test problem.

# Acknowledgements

Add acknowledgements of funding and contributions here.

# References
