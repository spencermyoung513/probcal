from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.transforms import ToTensor


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self, root_dir: str | Path, batch_size: int, num_workers: int, persistent_workers: bool
    ):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = Path(root_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def setup(self, stage: str):
        transform = Compose(
            [
                ToTensor(),
                Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.mnist_test = MNIST(self.root_dir, train=False, download=True, transform=transform)
        self.mnist_predict = MNIST(self.root_dir, train=False, download=True, transform=transform)
        mnist_full = MNIST(self.root_dir, train=True, download=True, transform=transform)
        self.mnist_train, self.mnist_val = random_split(
            mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(1998)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            shuffle=False,
        )


# import torch
# from torchvision.transforms import functional as tF
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from collections import defaultdict
# from tqdm import tqdm  # For progress bars
# import open_clip

# def get_all_clip_embeddings(datamodule, clip_model, device, max_samples=1000):
#     """Get CLIP embeddings for all MNIST images up to max_samples."""
#     datamodule.setup('fit')
#     dataset = datamodule.mnist_train

#     # Storage for embeddings and labels
#     embeddings_by_label = defaultdict(list)
#     count_by_label = defaultdict(int)

#     # Create a DataLoader to process in batches
#     loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

#     with torch.no_grad():
#         for images, batch_labels in tqdm(loader, desc="Getting embeddings"):
#             # Check if we have enough samples for all labels
#             if all(count_by_label[label] >= max_samples for label in range(10)):
#                 break

#             for idx, label in enumerate(batch_labels):
#                 label = label.item()
#                 if count_by_label[label] < max_samples:
#                     # Process image
#                     image = images[idx:idx+1].to(device)  # Keep batch dimension
#                     image_3channel = image.repeat(1, 3, 1, 1)
#                     image_resized = tF.resize(image_3channel, size=[224, 224], antialias=True)
#                     embedding = clip_model.encode_image(image_resized)

#                     embeddings_by_label[label].append(embedding.cpu())
#                     count_by_label[label] += 1

#     # Convert lists to tensors
#     for label in embeddings_by_label:
#         embeddings_by_label[label] = torch.cat(embeddings_by_label[label])

#     return embeddings_by_label

# def compute_all_pairwise_similarities(embeddings_dict):
#     """Compute average similarities between all label pairs."""
#     num_labels = 10
#     avg_similarities = torch.zeros((num_labels, num_labels))
#     std_similarities = torch.zeros((num_labels, num_labels))

#     for label1 in tqdm(range(num_labels), desc="Computing similarities"):
#         for label2 in range(label1, num_labels):  # Only compute upper triangle
#             similarities = []

#             # Get embeddings for both labels
#             embeddings1 = embeddings_dict[label1]
#             embeddings2 = embeddings_dict[label2]

#             # Compute similarities between all pairs
#             for emb1 in embeddings1:
#                 for emb2 in embeddings2:
#                     sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
#                     similarities.append(sim.item())

#             # Convert to tensor and compute statistics
#             similarities = torch.tensor(similarities)
#             avg_sim = similarities.mean().item()
#             std_sim = similarities.std().item()

#             # Store in matrices (make symmetric)
#             avg_similarities[label1, label2] = avg_sim
#             avg_similarities[label2, label1] = avg_sim
#             std_similarities[label1, label2] = std_sim
#             std_similarities[label2, label1] = std_sim

#     return avg_similarities, std_similarities

# def plot_similarity_matrices(avg_similarities, std_similarities, filename_prefix='similarities'):
#     """Plot heatmaps for average similarities and standard deviations."""
#     # Plot average similarities
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         avg_similarities,
#         annot=True,
#         fmt='.3f',
#         cmap='RdYlBu_r',
#         xticklabels=range(10),
#         yticklabels=range(10)
#     )
#     plt.title('Average CLIP Embedding Similarities Between MNIST Digits')
#     plt.xlabel('Digit')
#     plt.ylabel('Digit')
#     plt.tight_layout()
#     plt.savefig(f'{filename_prefix}_avg.png', bbox_inches='tight', dpi=300)
#     plt.close()

#     # Plot standard deviations
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(
#         std_similarities,
#         annot=True,
#         fmt='.3f',
#         cmap='viridis',
#         xticklabels=range(10),
#         yticklabels=range(10)
#     )
#     plt.title('Standard Deviation of CLIP Embedding Similarities')
#     plt.xlabel('Digit')
#     plt.ylabel('Digit')
#     plt.tight_layout()
#     plt.savefig(f'{filename_prefix}_std.png', bbox_inches='tight', dpi=300)
#     plt.close()

# def analyze_dataset_similarities(datamodule, clip_model, device, max_samples=500):
#     """Analyze similarities across all digits in the dataset."""
#     print("Getting embeddings for all digits...")
#     embeddings_dict = get_all_clip_embeddings(datamodule, clip_model, device, max_samples)
#     for k in embeddings_dict:
#         print(f"{k}: {len(embeddings_dict[k])}")

#     print("Computing pairwise similarities...")
#     avg_similarities, std_similarities = compute_all_pairwise_similarities(embeddings_dict)

#     print("Creating visualizations...")
#     plot_similarity_matrices(avg_similarities, std_similarities)

#     # Print some interesting findings
#     print("\nInteresting findings:")

#     # Find most similar pairs (excluding self-similarities)
#     avg_similarities_np = avg_similarities.numpy()
#     np.fill_diagonal(avg_similarities_np, -1)  # Exclude self-similarities
#     most_similar_idx = np.unravel_index(np.argmax(avg_similarities_np), avg_similarities_np.shape)
#     print(f"Most similar digits: {most_similar_idx[0]} and {most_similar_idx[1]} "
#           f"(avg similarity: {avg_similarities_np[most_similar_idx]:.3f} ± "
#           f"{std_similarities[most_similar_idx[0], most_similar_idx[1]]:.3f})")

#     # Find least similar pairs
#     least_similar_idx = np.unravel_index(np.argmin(avg_similarities_np), avg_similarities_np.shape)
#     print(f"Least similar digits: {least_similar_idx[0]} and {least_similar_idx[1]} "
#           f"(avg similarity: {avg_similarities_np[least_similar_idx]:.3f} ± "
#           f"{std_similarities[least_similar_idx[0], least_similar_idx[1]]:.3f})")

#     return avg_similarities, std_similarities

# def main():
#     # Setup
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Initialize your datamodule
#     datamodule = MNISTDataModule('../../data/mnist', 32, 1, False)

#     # Initialize CLIP model (you'll need to add your CLIP model initialization here)
#     clip_model, _, _ = open_clip.create_model_and_transforms(
#                 model_name="ViT-B-32",
#                 pretrained="laion2b_s34b_b79k",
#                 device=device,
#             )

#     # Analyze entire dataset
#     avg_similarities, std_similarities = analyze_dataset_similarities(
#         datamodule, clip_model, device, max_samples=250
#     )

# if __name__ == '__main__':
#     main()
