from simulation import JobSuiteBuilder
from math import log, sqrt

N_LIST = [
    20,
    22,
    24,
    26,
    28,
    30,
    34,
    36,
    40,
    42,
    46,
    50,
    56,
    60,
    66,
    72,
    78,
    84,
    92,
    100,
]

REPS = 1000
BUDGET_FN = lambda n: 30 * n * log(n)
BUDGET_DESC = "30*n*log(n)"
BENCH = "MIVS"  # <<— benchmark key

builder = JobSuiteBuilder()

# ── helper: add one algorithm on ALL n in N_LIST ───────────────────────
def add_algo(
    algorithm: str, *, lamb_fn=None, chi_fn=None, mu_fn=None, key_suffix: str = ""
) -> None:

    algo_args, param_desc = {}, {}

    if lamb_fn is not None:
        algo_args["lamb"] = lamb_fn
        param_desc["lamb"] = key_suffix or "λ(n)"

    if chi_fn is not None:  # for OPLLGA
        algo_args["chi"] = chi_fn
        param_desc["chi"] = key_suffix or "χ(n)"

    if mu_fn is not None:  # for UMDA
        algo_args["mu"] = mu_fn
        param_desc["mu"] = key_suffix or "μ(n)"

    builder.add_sweep(  # explicit-n sweep
        algorithm=algorithm,
        algo_args=algo_args,
        benchmark_key=BENCH,
        n_values=N_LIST,
        reps=REPS,
        budget=BUDGET_FN,
        param_descriptions=param_desc,
        budget_description=BUDGET_DESC,
    )


add_algo("DEGA_Limit", lamb_fn=lambda n: log(n), key_suffix="log(n)")
add_algo("DEGA_Limit", lamb_fn=lambda n: n ** (2 / 3), key_suffix="n^(2/3)")
add_algo("TPOGA")  # (2+1)-GA, no extra params

add_algo(
    "UMDA",
    lamb_fn=lambda n: sqrt(n) * log(n),
    mu_fn=lambda n: log(n),
    key_suffix="sqrt(n)*log(n)",
)

add_algo("DEGA_A")
add_algo("DEGA_B")

add_algo(
    "OPLLGA",
    lamb_fn=lambda n: sqrt(log(n)),
    chi_fn=lambda n: sqrt(log(n)),
    key_suffix="sqrt(log(n))",
)

builder.write("jobs_figs_8_9.json")
