## Diversity-Preserving Crossover Exploitation

**Authors:** Johannes Lengler, Tom Offermann

**Institution:** ETH Zürich, Switzerland

---
This repository includes all the code for running the simulations in the following paper:
https://dl.acm.org/doi/10.1145/3729878.3746613.

It may also be useful for simulating & plotting the performance of other optimization algorithms.  
This template is very minimal but offers simple scheduling and plotting for the performance of different optimization algorithms.  
These implementations focus on Genetic Algorithms (GAs) for optimizing pseudo-boolean functions  
$f: \{0,1\}^n \mapsto \mathbb R$.

---

### How to get started

1. Clone this repository & make sure Python (at least `3.9`, check with `python --version`) is correctly installed.
2. Make sure all necessary packages are installed (if not, use `pip install ...`). These include common packages like `matplotlib` and `numpy`

---

### General Structure

- The `./data` folder is used as a database for all the runs and can also be seen as a cache for runs that were already completed.
- Create runs by using the `JobSuiteBuilder`. Example uses are in the `create_runs*.py` scripts
- Run the `run.py` script to start the simulations. If you want to override the cached results use the `-f, --force` options. Other options include: `-n RUN_NAME`, to name your simulation result, `-w WORKERS` to specify the number of workers, `-c CONFIG` to specify the JSON file of `create_runs*.py` and `-x --clean` to remove all cached data.
- To plot the results use the `Plotter` class or just use plain matplotlib. The `Plotter.plot_any` function may be useful. Have a look at the `plot*.py` scripts for example uses.

---

### Implementation Details & Provided Functionality

Implementations for the DEGA variants and other GAs are located in `./src/algorithms`.  
The code leaves room for natural optimizations — if you improve any of the implementations, feel free to submit a pull request :)

The paper mentions three different DEGA implementations:

1. **DEGA** – Simplest version used in the theoretical proofs (Section 2 of the paper).
2. **DEGA_Limit** – **DEGA** constrained to $u(n)$ exploitation phases. We set a default $u(n) = \lambda \log n$.
3. **DEGA_A** – A more robust version discussed in Section 5.1 (includes a flowchart).
4. **DEGA_B** – An illustrative version utilizing algorithmic ideas from [1], more tailored toward _LeadingOnes_. Also discussed in Section 5.1.

Other implemented algorithms include:

- **$(2+1)$-GA**
- **$(1+1)$-GA**
- **$(1+(\lambda, \lambda))$-GA**
- **UMDA**

These algorithms can be tested on a subset of the PBO suite from the IOHanalyzer [2], including:

- **LeadingOnes**
- **OneMax**
- **LFWH** (Linear Function, Harmonic Weights)
- **MIVS** (Maximum Independent Vertex Set)

---

### References

[1] Benjamin Doerr, Daniel Johannsen, Timo Kötzing, Per Kristian Lehre, Markus Wagner, and Carola Winzen.  
_Faster black-box algorithms through higher arity operators_.  
In Foundations of Genetic Algorithms (FOGA 2011). ACM, New York, 163–172.

[2] Hao Wang, Diederick Vermetten, Furong Ye, Carola Doerr, and Thomas Bäck.  
_IOHanalyzer: Detailed Performance Analyses for Iterative Optimization Heuristics_.  
ACM Transactions on Evolutionary Learning and Optimization 2, 1 (Apr 2022), 29 pages.  
[https://doi.org/10.1145/3510426](https://doi.org/10.1145/3510426)
