class Algo(Enum):
    dega_1 = "DEGA_1"
    dega_2 = "DEGA_2"
    dega_3 = "DEGA_3"
    umda = "UMDA"
    tpoga = "TPOGA"
    opllga = "OPLLGA"
    opoga = "OPOGA"


def mivs_opt(n):
    d = {
        20: 8.93,
        22: 10.05,
        24: 10.75,
        26: 11.67,
        28: 12.48,
        30: 13.37,
        34: 15.06,
        36: 15.96,
        40: 17.69,
        42: 18.78,
        46: 20.14,
        50: 21.94,
        56: 24.65,
        60: 26.33,
        66: 28.93,
        72: 31.45,
        78: 34.21,
        84: 36.6,
        92: 40.09,
        100: 43.61
    }
    return int(d[n] - 0.5)


class Benchmark(Enum):
    lo = (leading_ones, "LO", lambda n: n)
    om = (one_max, "OM", lambda n: n)
    mivs = (mivs, "MIVS", mivs_opt)
    lfhw = (linear_harmonic, "LFHW", lambda n: n * (n + 1) // 2)
    labs = (labs, "LABS", lambda n: int(1e14))
    rr = (royal_road, "RR", lambda n: n // 5)