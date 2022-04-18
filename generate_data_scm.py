"""
Code adapted from: https://github.com/ijmbarr/causalgraphicalmodels/blob/master/notebooks/cgm-examples.ipynb
"""

from causalgraphicalmodels import StructuralCausalModel
import numpy as np

# Sample from the notebook (https://github.com/ijmbarr/causalgraphicalmodels/blob/master/notebooks/cgm-examples.ipynb)
# scm = StructuralCausalModel({
#     "x1": lambda     n_samples: np.random.binomial(n=1,p=0.7,size=n_samples),
#     "x2": lambda x1, n_samples: np.random.normal(loc=x1, scale=0.1),
#     "x3": lambda x2, n_samples: x2 ** 2,
# })

# ds = scm.sample(n_samples=100)
# dot = scm.cgm.draw()
# print(dot)
# dot.render('out', format='jpg', cleanup=True)
# print(ds)

# Create an actual SCM
# scm = StructuralCausalModel({
#     "x1": lambda     n_samples: np.random.normal(loc=0, scale=0.1, size=n_samples),
#     "x2": lambda x1, n_samples: x1 * 2,
#     "x3": lambda x1, n_samples: x1 * 3,
#     "x4": lambda x2, x3, n_samples: (x2 * 0.5 + x3 * 0.5),
#     "x5": lambda x4, n_samples: x4 * 2,
#     "x6": lambda x4, n_samples: x4 * 3,
#     "x7": lambda x5, x6, n_samples: (x5 * 0.5 + x6 * 0.5),
#     "x8": lambda x7, n_samples: x7 * 2,
#     "x9": lambda x7, n_samples: x7 * 3,
#     "x10": lambda x8, x9, n_samples:  (x8 * 0.5 + x9 * 0.5),
# })

scm = StructuralCausalModel({
    "x1": lambda     n_samples: np.random.normal(loc=1, scale=0.1, size=n_samples),
    "x2": lambda     n_samples: np.random.normal(loc=0, scale=0.2, size=n_samples),
    "x3": lambda x1, x2, n_samples: (0.5 * x1) + x2,
    "x4": lambda     n_samples: np.random.normal(loc=-1, scale=0.1, size=n_samples),
    "x5": lambda x3, x4, n_samples: x3 + x4,
    "x6": lambda     n_samples: np.random.normal(loc=0, scale=0.5, size=n_samples),
    "x7": lambda x5, x6, n_samples: x5 + x6,
    "x8": lambda     n_samples: np.random.normal(loc=-1, scale=0.2, size=n_samples),
    "x9": lambda     n_samples: np.random.normal(loc=1, scale=0.2, size=n_samples),
    "x10": lambda x7, x8, x9, n_samples: x7 + x8 + x9,
})

ds = scm.sample(n_samples=100)
ds.to_csv("train_dataset.csv", index=False, header=True)
dot = scm.cgm.draw()
print(dot)
dot.render('out_real', format='jpg', cleanup=True)
print(ds)

ds = scm.sample(n_samples=1000)
ds.to_csv("test_dataset.csv", index=False, header=True)
