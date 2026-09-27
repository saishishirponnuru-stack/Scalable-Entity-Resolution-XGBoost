import pandas as pd
from tabulate import tabulate
df = pd.read_csv('test\\test_source1.tsv', sep='\t')
df2 = pd.read_csv('test\\test_source2.tsv', sep='\t')
df3 = pd.read_csv('test\\test_source3.tsv', sep='\t')
df = pd.DataFrame(df)
df2 = pd.DataFrame(df2)
df3 = pd.DataFrame(df3)
# print(df.head().to_markdown())
print(tabulate(df.head(), headers='keys', tablefmt='psql'))
print(tabulate(df2.head(), headers='keys', tablefmt='psql'))
print(tabulate(df3.head(), headers='keys', tablefmt='psql'))