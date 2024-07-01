import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA

class FeaturePlane(nn.Module):
    def __init__(self, hidden_dim, resolution):
        super(FeaturePlane, self).__init__()
        self.features = nn.Parameter(torch.randn(1, hidden_dim, resolution, resolution))

    def forward(self):
        return self.features

class TriplaneRepresentation(nn.Module):
    def __init__(self, hidden_dim, resolution):
        super(TriplaneRepresentation, self).__init__()
        self.H = hidden_dim
        self.resolution = resolution
        
        self.planes = nn.ModuleList([FeaturePlane(hidden_dim, resolution) for _ in range(3)])

    def forward(self, mu, batch_size=10000):
        # Ensure mu is in [0, 1) range
        mu = torch.clamp(mu, 0, 1 - 1e-6)
        
        # Scale mu to [0, R)
        mu_scaled = mu * self.resolution
        
        all_features = []
        for i in range(0, mu.shape[0], batch_size):
            batch_mu = mu_scaled[i:i+batch_size]
            batch_features = []
            
            for j, plane in enumerate(self.planes):
                proj = self.project(batch_mu, j)
                
                # Normalize coordinates to [-1, 1] for grid_sample
                proj_normalized = (proj / (self.resolution - 1)) * 2 - 1
                
                # Interpolate features from the plane
                interp = F.grid_sample(plane().expand(batch_mu.shape[0], -1, -1, -1),
                                       proj_normalized.view(batch_mu.shape[0], 1, 1, 2),
                                       align_corners=True).squeeze(-1).squeeze(-1)
                
                batch_features.append(interp)
            
            # # Stack features from different planes
            # combined = torch.stack(batch_features, dim=1)
            # taking hadamard (element-wise) product acroos features of all planes
            combined = torch.prod(torch.stack(batch_features, dim=1), dim=1)
            all_features.append(combined)
        
        return torch.cat(all_features, dim=0)
    
    def project(self, mu, plane_idx):
        if plane_idx == 0:  # xy plane
            return mu[:, [0, 1]]
        elif plane_idx == 1:  # yz plane
            return mu[:, [1, 2]]
        else:  # zx plane
            return mu[:, [2, 0]]

def visualize_orthographic_embeddings(point_cloud, embeddings):
    pca = PCA(n_components=3)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plane_names = ['Frontal (XY)', 'Overhead (YZ)', 'Side (ZX)']
    projections = [
        lambda p: p[:, [0, 1]],  # XY
        lambda p: p[:, [1, 2]],  # YZ
        lambda p: p[:, [2, 0]]   # ZX
    ]
    
    for i in range(3):
        proj_points = projections[i](point_cloud)
        proj_embeddings = embeddings[:, i, :]
        
        # Apply PCA to reduce embedding dimension to 3
        pca_embeddings = pca.fit_transform(proj_embeddings.detach().cpu().numpy())
        
        # Normalize PCA embeddings to [0, 1] for RGB visualization
        pca_normalized = (pca_embeddings - pca_embeddings.min(axis=0)) / (pca_embeddings.max(axis=0) - pca_embeddings.min(axis=0))
        
        scatter = axes[i].scatter(proj_points[:, 0], proj_points[:, 1], c=pca_normalized, s=1)
        axes[i].set_title(f'{plane_names[i]} View')
        axes[i].set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

# Load point cloud
print("Loading point cloud...")
pcd = o3d.io.read_point_cloud("point_cloud.ply")
mu = torch.tensor(np.asarray(pcd.points)).float()
print(f"Point cloud shape: {mu.shape}")

# Normalize point cloud to [0, 1] range
print("Normalizing point cloud...")
mu_min, _ = torch.min(mu, dim=0)
mu_max, _ = torch.max(mu, dim=0)
mu = (mu - mu_min) / (mu_max - mu_min)
print(f"Normalized point cloud range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")

# Initialize TriplaneRepresentation
print("Initializing TriplaneRepresentation...")
hidden_dim = 64
resolution = 128
triplane = TriplaneRepresentation(hidden_dim, resolution)

# Get embeddings for the point cloud
print("Computing embeddings...")
embeddings = triplane(mu)
print(f"Embeddings shape: {embeddings.shape}")

# Visualize orthographic projected embeddings
print("Visualizing orthographic projected embeddings...")
visualize_orthographic_embeddings(mu, embeddings)
