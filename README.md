# [Numerical-Methods-for-Optimal-Control](https://mariozanon.wordpress.com/teaching/numerical-methods-for-optimal-control/)

Solutions of the assignments of the course *Numerical Methods for Optimal Control* (M. Zanon), as Jupyter notebooks.

## Installation

Requires Python 3.10 or newer.

1. Clone the repository and enter the folder:
   ```bash
   git clone https://github.com/GiacomoCorradini/Numerical-Methods-for-Optimal-Control.git
   cd Numerical-Methods-for-Optimal-Control
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start Jupyter **from the repository folder** (the notebooks import the solvers from `solvers/`):
   ```bash
   jupyter notebook
   ```

## Notebooks

| Notebook | Topic |
| --- | --- |
| [01_newton.ipynb](01_newton.ipynb) | Newton's method for equality-constrained problems |
| [02_sqp.ipynb](02_sqp.ipynb) | Sequential Quadratic Programming (SQP) |
| [03_interior_point.ipynb](03_interior_point.ipynb) | Interior-point method |

The solvers used in the notebooks are implemented in the [solvers/](solvers/) folder.
