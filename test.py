#%%
from joblib import Parallel, delayed
import numpy as np 
from math import sqrt
# %%
def comp(i):
    a = np.zeros(3,dtype=float)
    a[0] = i
    a[1] = i*i
    a[2] = i*i*i
    return a
# %%
a = np.array(
    Parallel(n_jobs=-1,verbose=1)(
        delayed(comp)(
            i
        ) for i in range(10)  
    )
) 

print(a)
# %%
