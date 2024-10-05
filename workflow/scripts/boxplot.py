import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main(sfd_files, output_file):
    results = pd.concat([pd.read_csv(f) for f in sfd_files])
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='idx', y='sfd', data=results)
    plt.title('SFD between True Graph and NCFA-Learned Structure')
    plt.xlabel('Graph Index')
    plt.ylabel('Structural Frobenius Difference (SFD)')
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    sfd_files = snakemake.input.sfd_files
    output_file = snakemake.output[0]
    main(sfd_files, output_file)