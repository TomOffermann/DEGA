from simulation import JobSuiteBuilder
from simulation.config_builder import AlgorithmInspector

# print(AlgorithmInspector.list_algorithms())
# print(AlgorithmInspector.get_signature('DEGA_A'))
# print(AlgorithmInspector.get_defaults('DEGA_B'))

builder = JobSuiteBuilder()
builder.add_range_sweep(
    algorithm="DEGA_A",
    benchmark_key="LO",
    algo_args={},
    n_start=50,
    n_end=100,
    num=10,
    reps=30,
    range_type="log",
    budget=lambda n: 100*n**2,
    budget_description="100n^2",
    param_descriptions={}
)
builder.add_range_sweep(
    algorithm="DEGA_B",
    benchmark_key="LO",
    algo_args={},
    n_start=50,
    n_end=100,
    num=10,
    reps=30,
    range_type="log",
    budget=lambda n: 100*n**2,
    budget_description="100n^2",
    param_descriptions={}
)
builder.write("jobs.json")
