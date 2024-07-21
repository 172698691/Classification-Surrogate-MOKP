# Classification-Based Surrogate Evolutionary Algorithm for Expensive Multi-Objective Knapsack Problems

## Background

The multi-objective knapsack problems (MOKPs) pose a significant challenge in combinatorial optimization, finding applications in various real-world scenarios. However, it encounters limitations when confronted with computationally expensive objective functions. To address this issue, surrogate-assisted evolutionary algorithms (SAEAs) are emerged, aiming to approximate objective functions through surrogate models. While regression-based SAEAs have proven effective for continuous problems, they may exhibit suboptimal performance when applied to high-dimensional and discrete problems like MOKPs. Conversely, classification-based SAEAs offer promise in handling discrete solutions.

## Introduction

Here a novel classification-based evolutionary algorithm (CBEA) designed specifically for expensive MOKPs, harnessing the capabilities of surrogate models to enhance both efficiency and effectiveness. The algorithm uses two classifiers to predict the dominance relation and crowding distance between offspring solutions instead of directly approximating objective function values. By reducing the computational resources and time required for evaluations, CBEA can efficiently solve expensive MOKPs without compromising solution accuracy.

The results substantiate its capability to achieve desirable convergence and diversity while significantly reducing the number of function evaluations. Notably, the algorithm surpasses existing methods in terms of computational efficiency, scalability, and solution quality.

## Requirements

The multi-objective evolutionary algorithm in this project depends on [pymoo: Multi-objective Optimization in Python](https://pymoo.org/) library. Please make sure that the library is in the environment before running.
