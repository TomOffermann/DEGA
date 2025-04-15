
## Diversity-Preserving Crossover Exploitation

**Authors:** *Tom Offermann*, *Prof. Dr. Johannes Lengler*  
**Institution:** ETH Zürich, Switzerland

This repository includes all the code for running the simulations in the following paper:  
*Diversity-Preserving Crossover Exploitation*, "TODO: Add link to paper".

It may also be useful for simulating & plotting the performance of other optimization algorithms.  
This template is very minimal but offers simple scheduling and plotting for the performance of different optimization algorithms.  
These implementations focus on Genetic Algorithms (GAs) for optimizing pseudo-boolean functions  
$f: \{0,1\}^n \mapsto \mathbb R$.

---

### How to get started

1. Clone this repository & make sure Python (at least `3.9`, check with `python --version`) is correctly installed.
2. Make sure all necessary packages are installed (if not, use `pip install ...`).

---

### General Structure

* The `./data` folder is used as a database for all the runs and can also be seen as a cache for runs that were already completed.
* Modify the `run.py` script to start new simulations. It won't rerun any simulations that were previously started with the same parameters — you can delete the corresponding file in `./data` to force a rerun.
* Modify the `plot.py` script to create plots from the simulations. The plots will appear in the `./plots` folder.

---

### Implementation Details & Provided Functionality

Implementations for the DEGA variants and other GAs are located in `./src/algorithms`.  
The code leaves room for natural optimizations — if you improve any of the implementations, feel free to submit a pull request :)

The paper (TODO: Add link to paper) mentions three different DEGA implementations:

1. **DEGA** – Simplest version used in the theoretical proofs (Section 2 of the paper).
2. **DEGA_A** – A more robust version discussed in Section 5.1 (includes a flowchart).
3. **DEGA_B** – An illustrative version utilizing algorithmic ideas from [1], more tailored toward *LeadingOnes*.

Other implemented algorithms include:
* `$(2+1)$-GA`
* `$(1+1)$-GA`
* `$(1+(\lambda, \lambda))$-GA`
* `UMDA`

These algorithms can be tested on a subset of the PBO suite from the IOHanalyzer [2], including:
* **LeadingOnes**
* **OneMax**
* **JUMP**
* **LFWH** (Linear Function, Harmonic Weights)
* **MIVS** (Maximum Independent Vertex Set)

---

### References

[1] Benjamin Doerr, Daniel Johannsen, Timo Kötzing, Per Kristian Lehre, Markus Wagner, and Carola Winzen.  
*Faster black-box algorithms through higher arity operators*.  
In Foundations of Genetic Algorithms (FOGA 2011). ACM, New York, 163–172.

[2] Hao Wang, Diederick Vermetten, Furong Ye, Carola Doerr, and Thomas Bäck.  
*IOHanalyzer: Detailed Performance Analyses for Iterative Optimization Heuristics*.  
ACM Transactions on Evolutionary Learning and Optimization 2, 1 (Apr 2022), 29 pages.  
[https://doi.org/10.1145/3510426](https://doi.org/10.1145/3510426)
