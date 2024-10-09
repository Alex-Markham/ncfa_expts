from os import path

import pandas as pd

# join results
cv_losses = []
for path in snakemake.input:
    cv_losses.append(pd.read_csv(path))

cv_losses_df = pd.concat(cv_losses)
cv_losses_df.reset_index(drop=True, inplace=True)
selected_df = cv_losses_df.loc[[cv_losses_df["avg_cv_loss"].idxmin()]]

# output
cv_losses_df.to_csv(snakemake.output["cv_losses"], index=False)
selected_df.to_csv(snakemake.output["selected_hyperparams"], index=False)
