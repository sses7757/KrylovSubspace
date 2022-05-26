# Krylov Subspace Methods For Solving Large Matrices
This repository contains my introduction note to various Krylov subspace methods( including Lanczos, Arnoldi, Krylov-Schur, GMRES, MinRes and conjugate gradient methods) for solving and eigen-solving large matrices which are suitable for large-scale parallel computation.

If you want to reference some runnable codes, I highly recommend my repository `Althea` (which is currently being developed and will be published soon). You can find most of the implementations of main text in `Althea.Backend.CSharp.Solver.Krylov.KrylovBased.cs` and the ones of Chapter 0 in `Althea.Backend.CSharp.LinearAlgebra.MatrixSolvers.cs`, whose comments contains a number of TeX equations which can be shown by extension tools like TeX Comments.

> [中文简介](README_CN.md)

# Preface

I have applied Finite Element Method, Gradient Flow, Density Matrix Renormalization Group, etc. to solve problems in ordinary/partial differential equations, optimization of functionals, ground states of quantum tensor networks, etc. in my limited experiences. The problems are actually abstracts in different disciplines such as engineering, physics, chemistry and biology, which depend on algorithms like Lanczos, Krylov-Schur and GMRES to solve sparse or other kinds of special large matrices that cannot be stored in memory in dense format. And these algorithms are highly connected: they are all Krylov subspace methods. However, I have not found any textbook or note that introduce them completely and orderly. Yet, I think that it would be beneficial for beginners to build their own framework of knowledge if I could write note in this way. By reading this note, I hope that you can use the appropriate algorithm when writing codes, or locating problems of these algorithms when they malfunctioned, or develop readable and efficient algorithms that are specifically modified according to demands. And I hope this note will help you avoiding some detours.

This note are mainly consisted by three parts: Preliminaries and Introduction, Eigen-Problems, Linear System Problems. Due to space limitations, the Preliminaries are isolated to Chapter 0 and the Eigen-Problems are divided into two parts to discuss Hermitian and non-Hermitian ones separately.

Chapter 0 mainly introduces the QR algorithm and the eigen decompositions and Schur decompositions implemented via it. The algorithms in this chapter does not involve Krylov subspace, and does not expand various proof processes. However, I still think this chapter is necessary: the Krylov subspace methods require one or more of these algorithms, and their complexity is so small relative to the Krylov subspace methods, such that it can be implemented in any language that you are accustomed to while the effect to the overall performance is negligible and avoids the dependence of some basic libraries such as MKL. For example, if you want to use GPU to accelerate the calculation, you can implement the time-consuming part of the Krylov subspace method on the GPU (large matrix and vector operations), and write the naive Chapter 0 algorithms in CPU without considering whether the MKL supports the CPU you are using or not.

The first part introduces the concept of Krylov subspace from the simplest Krylov subspace method -- power iteration -- and obtains the most basic Arnoldi algorithm suitable for non-Hermitian matrices, which is introduced next. The following Lanczos algorithm depends on the conclusions of this part.

The second part begins with the simplification of Arnoldi algorithm by Hermitian matrix condition, which leads to the basic Lanczos algorithm. Secondly, this part mainly focuses on the convergence proof of the Lanczos algorithm since most of the subsequent convergence proofs are very similar to it and they are often ignored in order to reduce the burden and save some space. This part also discusses a vital trick -- restart which avoids some accumulated numerical errors and reduces memory consumption. It is also quite important in the following eigen-problem algorithms.

The third part goes back to the basic Arnoldi algorithm and introduces its restart method. As already introduced in Chapter 0, the eigen decomposition of small non-Hermitian matrices often requires intermediate steps of Schur decomposition to enhance numerical stability. From this perspective, I then introduce the Krylov-Schur algorithm which is generally better than Arnoldi in convergence.

The fourth section focuses on linear system problems instead. Because of the previous foreshadowing, I choose to begin with the most complex and most closely related algorithm -- the generalized minimal residual method (GMRES) -- which aims for solving non-Hermitian linear systems. By considering more and more special matrices, I then introduce the minimal residual method (MinRes), the conjugate gradient method (CG), and their preprocessing algorithms.

# Table of Content
- Part O
	- [Chapter 0 Preliminaries](en/ch0/main.md)
- Part I Introduction
	- [Chapter 1 Beginning With Power Iteration](en/ch1/main.md)
	- [Chapter 2 Krylov Subspace](en/ch2/main.md)
	- [Chapter 3 Arnoldi Algorithm](en/ch3/main.md)
- Part II Hermitian Matrix Eigen-Problem
	- [Chapter 4 Basic Lanczos Algorithm](en/ch4/main.md)
	- [Chapter 5 Advanced Lanczos Algorithms](en/ch5/main.md)
	- [Chapter 6 Restart Method of Lanczos Algorithm](en/ch6/main.md)
	- [Chapter 7 Lanczos Algorithm With All Techniques](en/ch7/main.md)
- Part III Non-Hermitian Matrix Eigen-Problem
	- [Chapter 8 Arnoldi Algorithm](en/ch8/main.md)
	- [Chapter 9 Krylov-Schur Algorithm](en/ch9/main.md)
- Part IV Linear System Problems
	- [Chapter 10 Non-Hermitian Matrix's Linear System](en/ch10/main.md)
	- [Chapter 11 Hermitian Matrix's Linear System](en/ch11/main.md)
	- [Chapter 12 Convergence and Preconditioning](en/ch12/main.md)
- Part V Performance Analysis
	- [Chapter 13 Hermitian Matrix's Algorithms](en/ch13/main.md)
	- [Chapter 14 Non-Hermitian Matrix's Algorithms](en/ch14/main.md)