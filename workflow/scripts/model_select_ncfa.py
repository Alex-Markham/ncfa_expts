from os import path
import shutil

import pandas as pd

# join results
cv_losses = []
for path in snakemake.input.cv_loss:
    cv_losses.append(pd.read_csv(path))

cv_losses_df = pd.concat(cv_losses)
cv_losses_df.reset_index(drop=True, inplace=True)

selected_idx = cv_losses_df["avg_cv_loss"].idxmin()
selected_df = cv_losses_df.loc[[selected_idx]]

selected_biadj_weights = snakemake.input.pooled_biadj_weights[selected_idx]

# output
cv_losses_df.to_csv(snakemake.output["cv_losses"], index=False)
selected_df.to_csv(snakemake.output["selected_hyperparams"], index=False)
shutil.copyfile(selected_biadj_weights, snakemake.output["selected_biadj_weights"])
