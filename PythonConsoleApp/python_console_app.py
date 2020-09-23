import numpy as np
import pandas as pd

def double(a):
    return a * 2

a = np.array([2, 3, 4])
print(a)

df = pd.DataFrame({'Name', ['cat', 'human'],
                   'Legs', [4, 2]})
print(df)

print("Done.")
