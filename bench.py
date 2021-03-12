#!/usr/bin/env python
import os
from scipy import sparse
from sparse_dot_topn import awesome_dense_cossim_topn, awesome_cossim_topn
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import seaborn as sns

class Timer(object):

    def __init__(self, label=None):
        self.label=label or 'elapsed'

    def __enter__(self):
        self.t0 = timer()
        return self

    def __exit__(self, *exc_info):
        self.t1 = timer()
        self.delta = self.t1 - self.t0
        print(f"{self.label} : {self.delta:.4f} sec.")


def create_experiment(n_row=10**4, m_row=2*10**4, feat=100, density=0.15, n_top=3, random_state=1):
    np.random.seed(random_state)
    sA = sparse.random(n_row, feat, density=density, format='csr')
    sB = sparse.random(feat, m_row, density=density, format='csr')

    A = sA.toarray()
    B = sB.toarray()

    timings = {}

    with Timer(label=f'awesome_dense_cossim_topn(density={density:.2f})') as t:
        t0 = timer()
        _ = awesome_dense_cossim_topn(A, B, n_top)
        t1 = timer()
    timings['dense'] = (t1 - t0)

    with Timer(label='awesome_cossim_topn') as t:
        t0 = timer()
        _ = awesome_cossim_topn(sA, sB, n_top)
        t1 = timer()
    timings['sparse'] = (t1 - t0)

    return timings


TIMINGS_CSV_FILENAME = "/tmp/timings.csv"
TIMINGS_PNG_FILENAME = "/tmp/timings.png"
n_row = 1*10**4
m_row = 2*10**4
feat = 500
if os.path.exists(TIMINGS_CSV_FILENAME):
    res = pd.read_csv(TIMINGS_CSV_FILENAME)
else:
    res = []
    for density in np.arange(0.01, 0.201, 0.01):
        curr = create_experiment(n_row=n_row, m_row=m_row, feat=feat, density=density)
        curr['density'] = density
        res.append(curr)
    res = pd.DataFrame(res)
    res.to_csv(TIMINGS_CSV_FILENAME, index=False)

sns.set()
fig, ax = plt.subplots(figsize=(10, 5))
res.plot(x='density', y=['dense', 'sparse'], ax=ax)
ax.legend()
ax.set_ylabel("time (sec.)")
ax.set_title(f"Sparse vs Dense Dot Topn (names_to_match={n_row} gt={m_row} feat={feat})")
sns.despine(ax=ax)
plt.savefig(TIMINGS_PNG_FILENAME, dpi=100)
plt.show()