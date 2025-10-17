from simulation import JobSuiteBuilder
from simulation.config_builder import AlgorithmInspector

from math import *

# print(AlgorithmInspector.list_algorithms())
# print(AlgorithmInspector.get_signature('DEGA_A'))
# print(AlgorithmInspector.get_defaults('DEGA_B'))

n_start = 7500
n_end = 7500
num = 1
reps = 50
budget = lambda n: 10 * (n**2)
budget_desc = "10*n^2"
range_type = "log"

builder = JobSuiteBuilder()
# builder.add_range_sweep(
#     algorithm="DEGA_A",
#     benchmark_key="LO",
#     algo_args={},
#     n_start=n_start,
#     n_end=n_end,
#     num=num,
#     reps=reps,
#     range_type=range_type,
#     budget=budget,
#     budget_description=budget_desc,
#     param_descriptions={},
# )
builder.add_range_sweep(
    algorithm="DEGA",
    benchmark_key="LO",
    algo_args={"lamb": lambda n: (n * log(n)) ** (2 / 3)},
    n_start=n_start,
    n_end=n_end,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "(n*log(n))**(2/3)"},
)
builder.add_range_sweep(
    algorithm="DEGA",
    benchmark_key="LO",
    algo_args={"lamb": lambda n: sqrt(n)},
    n_start=n_start,
    n_end=n_end,
    num=num,
    reps=reps,
    range_type=range_type,
    budget=budget,
    budget_description=budget_desc,
    param_descriptions={"lamb": "sqrt(n)"},
)
# builder.add_range_sweep(
#     algorithm="DEGA",
#     benchmark_key="LO",
#     algo_args={"lamb": lambda n: n**(2/3)},
#     n_start=n_start,
#     n_end=n_end,
#     num=num,
#     reps=reps,
#     range_type=range_type,
#     budget=budget,
#     budget_description=budget_desc,
#     param_descriptions={"lamb": "n^(2/3)"},
# )
# builder.add_range_sweep(
#     algorithm="DEGA",
#     benchmark_key="LO",
#     algo_args={"lamb": lambda n: sqrt(n*log(n))},
#     n_start=n_start,
#     n_end=n_end,
#     num=num,
#     reps=reps,
#     range_type=range_type,
#     budget=budget,
#     budget_description=budget_desc,
#     param_descriptions={"lamb": "sqrt(n*log(n))"},
# )
# builder.add_range_sweep(
#     algorithm="DEGA",
#     benchmark_key="LO",
#     algo_args={"lamb": lambda n: n**(1/3)},
#     n_start=n_start,
#     n_end=n_end,
#     num=num,
#     reps=reps,
#     range_type=range_type,
#     budget=budget,
#     budget_description=budget_desc,
#     param_descriptions={"lamb": "n^(1/3)"},
# )
# builder.add_range_sweep(
#     algorithm="DEGA",
#     benchmark_key="LO",
#     algo_args={"lamb": lambda n: 2},
#     n_start=n_start,
#     n_end=n_end,
#     num=num,
#     reps=reps,
#     range_type=range_type,
#     budget=budget,
#     budget_description=budget_desc,
#     param_descriptions={"lamb": "2"},
# )
# builder.add_range_sweep(
#     algorithm="DEGA_B",
#     benchmark_key="LO",
#     algo_args={},
#     n_start=n_start,
#     n_end=n_end,
#     num=num,
#     reps=reps,
#     range_type=range_type,
#     budget=budget,
#     budget_description=budget_desc,
#     param_descriptions={},
# )
# builder.add_range_sweep(
#     algorithm="DEGA_Limit",
#     benchmark_key="LO",
#     algo_args={"lamb": lambda n: (n * log(n)) ** (2 / 3), "u": lambda n: 10 * log(n)},
#     n_start=n_start,
#     n_end=n_end,
#     num=num,
#     reps=reps,
#     range_type=range_type,
#     budget=budget,
#     budget_description=budget_desc,
#     param_descriptions={
#       "lamb": "(n*log(n))**(2/3)",
#       "chi": "10*log(n)",
#     },
# )
# builder.add_range_sweep(
#     algorithm="TPOGA",
#     benchmark_key="LO",
#     algo_args={},
#     n_start=n_start,
#     n_end=n_end,
#     num=num,
#     reps=reps,
#     range_type=range_type,
#     budget=budget,
#     budget_description=budget_desc,
#     param_descriptions={},
# )
# builder.add_range_sweep(
#     algorithm="OPLLGA",
#     benchmark_key="LO",
#     algo_args={"lamb": lambda n: sqrt(log(n)), "chi": lambda n: sqrt(log(n))},
#     n_start=n_start,
#     n_end=n_end,
#     num=num,
#     reps=reps,
#     range_type=range_type,
#     budget=budget,
#     budget_description=budget_desc,
#     param_descriptions={
#       "lamb": "sqrt(log(n))",
#       "chi": "sqrt(log(n))",
#     },
# )
# builder.add_range_sweep(
#     algorithm="OPOGA",
#     benchmark_key="LO",
#     algo_args={},
#     n_start=n_start,
#     n_end=n_end,
#     num=num,
#     reps=reps,
#     range_type=range_type,
#     budget=budget,
#     budget_description=budget_desc,
#     param_descriptions={},
# )
# builder.add_range_sweep(
#     algorithm="UMDA",
#     benchmark_key="LO",
#     algo_args={"lamb": lambda n: n**(1/2)*log(n), "mu": lambda n: log(n)},
#     n_start=n_start,
#     n_end=n_end,
#     num=num,
#     reps=reps,
#     range_type=range_type,
#     budget=budget,
#     budget_description=budget_desc,
#     param_descriptions={
#       "lamb": "sqrt(n)*log(n)",
#       "mu": "log(n)",
#     },
# )
builder.write("jobs.json")
