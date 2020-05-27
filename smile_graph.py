from pysmiles import read_smiles
import networkx as nx
import pandas as pd
smiles = pd.read_csv("FlavorDB_manual_none.csv")
#print smiles["smiles"][0]
df = pd.DataFrame(data = smiles)
#print df
for i in range(2371):
    mol = read_smiles(df['smiles'][i])
    print (mol.nodes(data='element'))
    print (nx.to_numpy_matrix(mol))
