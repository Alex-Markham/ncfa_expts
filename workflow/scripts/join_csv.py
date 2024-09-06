from os import path

import pandas as pd

# join results
results = []
for path in snakemake.input:
    results.append(pd.read_csv(path))

eval_df = pd.concat(results)

# output
eval_df.to_csv(snakemake.output[0], index=False)
