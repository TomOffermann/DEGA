from simulation import JobSuiteBuilder
from simulation.config_builder import AlgorithmInspector

from math import *

# print(AlgorithmInspector.list_algorithms())
# print(AlgorithmInspector.get_signature('DEGA_A'))
# print(AlgorithmInspector.get_defaults('DEGA_B'))

n_start = 100
n_end_om = 20000
num = 10
reps = 10
budget = lambda n: 10 * (n**2)
budget_desc = "10*n^2"
range_type = "log"

builder = JobSuiteBuilder()

# DEGA:
builder.add_range_sweep(
    algorithm="DEGA",
    benchmark_key="OM",
    algo_args={"lamb": lambda n: n ** (2 / 3)},
    n_start=n_start,
    n_end=n_end_om,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "n^(2/3)"},
)
builder.add_range_sweep(
    algorithm="OPLLGA",
    benchmark_key="OM",
    algo_args={"lamb": lambda n: sqrt(log(n)), "chi": lambda n: sqrt(log(n))},
    n_start=n_start,
    n_end=n_end_om,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "sqrt(log(n))", "chi": "sqrt(log(n))"},
)

builder.write("jobs_fig_4.json")
