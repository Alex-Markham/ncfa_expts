import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from codebase.models.mask_vae_flow import CausalVAE
from utils import _h_A, get_batch_unin_dataset_withlabel
from torchvision.utils import save_image
import os
import logging

logging.basicConfig(filename=snakemake.log[0], level=logging.INFO)
logger = logging.getLogger()

def run_causalvae_discovery(dataset_path, z_dim=16, num_epochs=100, lr=1e-3, batch_size=64):
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the dataset
    data = np.loadtxt(dataset_path, delimiter=",")
    dataset = torch.FloatTensor(data)
    
    # Create DataLoader
    dataloader = DataLoader(TensorDataset(dataset), batch_size=batch_size, shuffle=True)
    
    # Initialize the model
    model = CausalVAE(name='causalvae', z_dim=z_dim).to(device)
    
    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Training
    for epoch in range(num_epochs):
        total_loss = 0
        total_kl = 0
        total_rec = 0
        
        for batch in dataloader:
            # Get batch data
            u = batch[0].to(device)
            
            # Create dummy labels (assuming no labels are needed)
            l = torch.zeros(u.size(0)).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            L, kl, rec, reconstructed_image, _ = model.negative_elbo_bound(u, l, sample=False)
            
            # Get DAG parameters and calculate h(A)
            dag_param = model.dag.A
            h_a = _h_A(dag_param, dag_param.size()[0])
            
            # Add DAG constraint to loss
            L = L + 3*h_a + 0.5*h_a*h_a
            
            # Backward pass and optimization
            L.backward()
            optimizer.step()
            
            # Track losses
            total_loss += L.item()
            total_kl += kl.item()
            total_rec += rec.item()
        
        # Calculate average losses
        avg_loss = total_loss / len(dataloader)
        avg_kl = total_kl / len(dataloader)
        avg_rec = total_rec / len(dataloader)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, KL: {avg_kl:.4f}, Rec: {avg_rec:.4f}')
    
    # Extract biadjacency matrix from the trained model
    biadj = model.dag.A.detach().cpu().numpy()
    
    # Perform intervention
    perform_intervention(model, device, snakemake.output.intervention_dir)

    # Save model
    torch.save(model.state_dict(), snakemake.output.model)

    return model, biadj

def perform_intervention(model, device, intervention_dir):
    """
    Perform intervention using the pre-trained model and save the reconstructed images.
    """
    if not os.path.exists(intervention_dir):
        os.makedirs(intervention_dir)

    dataset_dir = './causal_data/flow_noise'
    train_dataset = get_batch_unin_dataset_withlabel(dataset_dir, 100, dataset="train")
    
    count = 0
    for u, l in train_dataset:
        for i in range(4):
            for j in range(-5, 5):
                L, kl, rec, reconstructed_image, _ = model.negative_elbo_bound(u.to(device), l.to(device), i, adj=j*0)
                save_image(reconstructed_image[0], f'{intervention_dir}/reconstructed_image_{i}_{count}.png', range=(0, 1))
        
        save_image(u[0], f'{intervention_dir}/true_{count}.png') 
        count += 1
        if count == 10:
            break

def main():
    # Run CausalVAE
    model, est_biadj = run_causalvae_discovery(
        dataset_path=snakemake.input.dataset,
        z_dim=snakemake.params.z_dim,
        num_epochs=snakemake.params.num_epochs,
        lr=snakemake.params.lr,
        batch_size=snakemake.params.batch_size
    )
    
    # Save estimated biadjacency matrix
    np.save(snakemake.output.est_biadj, est_biadj)

if __name__ == "__main__":
    main()