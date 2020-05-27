>>> import pandas as pd
>>> dataset = pd.read_csv("FlavorDB_manual_none.csv")
>>> print dataset["compounds"][0]
1-Aminopropan-2-Ol
>>> print dataset["flavor"][0]
fishy
>>> df = pd.DataFrame(data = dataset)
>>> import cirpy
>>> for i in range (0,2371):
...     print(df["compounds"][i], cirpy.resolve(df["compounds"][i], 'smiles'))
...
