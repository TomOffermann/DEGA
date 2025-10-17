from simulation import JobSuiteBuilder
from simulation.config_builder import AlgorithmInspector

from math import *

# print(AlgorithmInspector.list_algorithms())
# print(AlgorithmInspector.get_signature('DEGA_A'))
# print(AlgorithmInspector.get_defaults('DEGA_B'))

n_start = 100
n_end_lo = 1000
n_end_rest = 5000
num = 10
reps = 50
budget = lambda n: 10 * (n**2)
budget_desc = "10*n^2"
range_type = "log"

builder = JobSuiteBuilder()

# DEGA:
builder.add_range_sweep(
    algorithm="DEGA",
    benchmark_key="LO",
    algo_args={"lamb": lambda n: n**(2/3)},
    n_start=n_start,
    n_end=n_end_lo,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "n^(2/3)"},
)
builder.add_range_sweep(
    algorithm="DEGA",
    benchmark_key="LO",
    algo_args={"lamb": lambda n: log(n)},
    n_start=n_start,
    n_end=n_end_lo,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "log(n)"},
)
builder.add_range_sweep(
    algorithm="DEGA",
    benchmark_key="OM",
    algo_args={"lamb": lambda n: n**(2/3)},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "n^(2/3)"},
)
builder.add_range_sweep(
    algorithm="DEGA",
    benchmark_key="OM",
    algo_args={"lamb": lambda n: log(n)},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "log(n)"},
)
builder.add_range_sweep(
    algorithm="DEGA",
    benchmark_key="LFHW",
    algo_args={"lamb": lambda n: n**(2/3)},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "n^(2/3)"},
)
builder.add_range_sweep(
    algorithm="DEGA",
    benchmark_key="LFHW",
    algo_args={"lamb": lambda n: log(n)},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "log(n)"},
)

# DEGA A:
builder.add_range_sweep(
    algorithm="DEGA_A",
    benchmark_key="LO",
    algo_args={},
    n_start=n_start,
    n_end=n_end_lo,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={},
)
builder.add_range_sweep(
    algorithm="DEGA_A",
    benchmark_key="OM",
    algo_args={},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={},
)
builder.add_range_sweep(
    algorithm="DEGA_A",
    benchmark_key="LFHW",
    algo_args={},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={},
)

# DEGA B
builder.add_range_sweep(
    algorithm="DEGA_B",
    benchmark_key="LO",
    algo_args={},
    n_start=n_start,
    n_end=n_end_lo,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={},
)
builder.add_range_sweep(
    algorithm="DEGA_B",
    benchmark_key="OM",
    algo_args={},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={},
)
builder.add_range_sweep(
    algorithm="DEGA_B",
    benchmark_key="LFHW",
    algo_args={},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={},
)

# (2+1):
builder.add_range_sweep(
    algorithm="TPOGA",
    benchmark_key="LO",
    algo_args={},
    n_start=n_start,
    n_end=n_end_lo,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={},
)
builder.add_range_sweep(
    algorithm="TPOGA",
    benchmark_key="OM",
    algo_args={},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={},
)
builder.add_range_sweep(
    algorithm="TPOGA",
    benchmark_key="LFHW",
    algo_args={},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={},
)

# UMDA:
builder.add_range_sweep(
    algorithm="UMDA",
    benchmark_key="LO",
    algo_args={"lamb": lambda n: sqrt(n)*log(n), "mu": lambda n: log(n)},
    n_start=n_start,
    n_end=n_end_lo,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "sqrt(n)*log(n)", "mu": "log(n)"},
)
builder.add_range_sweep(
    algorithm="UMDA",
    benchmark_key="OM",
    algo_args={"lamb": lambda n: sqrt(n)*log(n), "mu": lambda n: log(n)},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "sqrt(n)*log(n)", "mu": "log(n)"},
)
builder.add_range_sweep(
    algorithm="UMDA",
    benchmark_key="LFHW",
    algo_args={"lamb": lambda n: sqrt(n)*log(n), "mu": lambda n: log(n)},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "sqrt(n)*log(n)", "mu": "log(n)"},
)

# (1+(l,l)):
builder.add_range_sweep(
    algorithm="OPLLGA",
    benchmark_key="LO",
    algo_args={"lamb": lambda n: sqrt(log(n)), "chi": lambda n: sqrt(log(n))},
    n_start=n_start,
    n_end=n_end_lo,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "sqrt(log(n))", "chi": "sqrt(log(n))"},
)
builder.add_range_sweep(
    algorithm="OPLLGA",
    benchmark_key="OM",
    algo_args={"lamb": lambda n: sqrt(log(n)), "chi": lambda n: sqrt(log(n))},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "sqrt(log(n))", "chi": "sqrt(log(n))"},
)
builder.add_range_sweep(
    algorithm="OPLLGA",
    benchmark_key="LFHW",
    algo_args={"lamb": lambda n: sqrt(log(n)), "chi": lambda n: sqrt(log(n))},
    n_start=n_start,
    n_end=n_end_rest,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "sqrt(log(n))", "chi": "sqrt(log(n))"},
)
builder.write("jobs_fig_3.json")
