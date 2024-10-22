"""
Calculate the correlation matrix and Cramer's V matrix for the CUB dataset.
"""

import os
import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
from PIL import Image
from types import SimpleNamespace

import seaborn as sns
import matplotlib.pyplot as plt
import io

from datasets.CUB_dataset import get_CUB_dataloaders
from scipy.stats import chi2_contingency

config = SimpleNamespace(
    seed = 25,
    img_size = 224,
    num_classes = 112,
    data_path = '.',
    train_path = '.',
    test_path = '.',
    train_batch_size = 80,
    test_batch_size = 100,
    train_push_batch_size = 75,
    gpuid = 0,
    attr_index = None,
    group_by = "color",
    num_groups = 27,                              
    num_groups_tot = 27, # 28 if oracle, 27 if color
    random_groups = False,
    masked_groups = [],
)

def cramers_v(confusion_matrix):
    """Calculate Cramer's V statistic for the given confusion matrix."""

    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    r, k = confusion_matrix.shape

    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)

    return np.sqrt(phi2corr / min((k_corr-1), (r_corr-1)))

def get_empirical_covariance(dataloader):
    data = []
    for _, concepts in dataloader:
        data.append(concepts)

    data = torch.cat(data)  # Concatenate all data into a single tensor
    # data_logits = torch.logit(0.05 + 0.9 * data)
    data = data.transpose(0, 1)
    covariance = torch.cov(data)

    corr = (covariance / covariance.diag().sqrt()).transpose(dim0=0, dim1=1) / covariance.diag().sqrt()
    correlation_matrix = corr.cpu().numpy()

    # Cramer's V correlation (for categorical variables)
    num_vars = data.shape[0]
    cramers_v_matrix = np.zeros((num_vars, num_vars))
    
    for i in range(num_vars):
        for j in range(i, num_vars):
            if i == j:
                cramers_v_matrix[i, j] = 1.0
            else:
                # Create contingency table
                contingency_table = np.histogram2d(data[i], data[j], bins=2)[0]
                
                # Calculate Cramer's V
                cramers_v_value = cramers_v(contingency_table)
                cramers_v_matrix[i, j] = cramers_v_value
                cramers_v_matrix[j, i] = cramers_v_value
                
    return correlation_matrix, cramers_v_matrix

train_loader, test_loader, train_push_loader = get_CUB_dataloaders(config)

matrix, cramer = get_empirical_covariance(train_loader)

def get_concept_groups(config):

    if config.group_by == "oracle":
        # Oracle grouping based on concept type for CUB
        with open(
            os.path.join(config.data_path, "CUB_200_2011/concept_names.txt"),
            "r",
        ) as f:
            concept_names = []
            for line in f:
                concept_names.append(line.replace("\n", "").split("::"))

        group_names = []
        for c in concept_names:
            if c[0] not in group_names:
                group_names.append(c[0])
        groups = np.zeros((len(group_names), len(concept_names)), dtype=np.float32)
        for i, gn in enumerate(group_names):
            for j, cn in enumerate(concept_names):
                if cn[0] == gn:
                    groups[i, j] = 1.0

        if config.random_groups:
            random_indices = torch.randperm(len(groups))[: config.num_groups]
        else:
            random_indices = torch.tensor([num for num in list(range(config.num_groups_tot)) if num not in config.masked_groups])

        if config.num_groups == 1:
            keep_mask = groups[random_indices, :]
        else:
            keep_mask = np.sum(groups[random_indices, :], axis=0)
        
        concept_names = [
            label for label, keep in zip(concept_names, keep_mask.astype(bool)) if keep
        ]
        concept_names_graph = [": ".join(name) for name in concept_names]

    elif config.group_by == "color":
        # Color grouping based on color CUB
        with open(
            os.path.join(config.data_path, "CUB_200_2011/concept_names.txt"),
            "r",
        ) as f:
            concept_names = []
            for line in f:
                concept_names.append(line.replace("\n", "").split("::"))

        group_names = []
        for c in concept_names:
            if c[1] not in group_names:
                group_names.append(c[1])
        groups = np.zeros((len(group_names), len(concept_names)), dtype=np.float32)
        for i, gn in enumerate(group_names):
            for j, cn in enumerate(concept_names):
                if cn[1] == gn:
                    groups[i, j] = 1.0

        if config.random_groups:
            random_indices = torch.randperm(len(groups))[: config.num_groups]
        else:
            random_indices = torch.tensor([num for num in list(range(config.num_groups_tot)) if num not in config.masked_groups])

        if config.num_groups == 1:
            keep_mask = groups[random_indices, :]
        else:
            keep_mask = np.sum(groups[random_indices, :], axis=0)
            
        concept_names = [
            label for label, keep in zip(concept_names, keep_mask.astype(bool)) if keep
        ]
        concept_names_graph = [": ".join(name) for name in concept_names]
                 
    groups = groups[: config.num_groups, keep_mask.astype(bool)]
    return groups, concept_names_graph

groups, concept_names_graph = get_concept_groups(config)

num_categories, num_vars = groups.shape
categories = np.zeros(num_vars, dtype=int)
for var_index in range(num_vars):
    category_indices = np.where(groups[:, var_index] != 0)[0]
    if category_indices.size > 0:
        categories[var_index] = category_indices[0] + 1

sorted_indices = np.argsort(categories)

# Reorder the correlation matrix based on sorted indices
sorted_cramer = cramer[sorted_indices, :][:, sorted_indices]
sorted_matrix = matrix[sorted_indices, :][:, sorted_indices]
sorted_concept_names = [concept_names_graph[i] for i in sorted_indices]

def plot_heatmap(matrix, variable_names, save_name="heatmap.png"):
    """Plot the heatmap of the correlation matrix using Seaborn and save it if save_path is provided."""
    sns.set_theme(style="white")
    plt.figure(figsize=(24, 20))  # Increased figure size to fit labels
    ax = sns.heatmap(
        matrix,
        cmap="RdBu_r", 
        vmin=-1,  # Set vmin to -1 for correlation values
        vmax=1,  # Set vmax to 1 for correlation values
        square=True,
        linewidths=0.5,
        annot=False,  # Do not show annotations (numbers) in the heatmap
        cbar_kws={"shrink": .8},
        xticklabels=variable_names,
        yticklabels=variable_names
    )
    plt.xticks(rotation=90)  # Rotate x labels for better readability
    plt.yticks(rotation=0)  # Keep y labels horizontal
    # ax.set_title(title)
    
    if save_name:
        plt.savefig(save_name)
        plt.close()

plot_heatmap(matrix, concept_names_graph, save_name="heatmap_corr.png")
plot_heatmap(cramer, concept_names_graph, save_name="heatmap_cramer.png")

plot_heatmap(sorted_matrix, sorted_concept_names, save_name="heatmap_sorted_corr.png")
plot_heatmap(sorted_cramer, sorted_concept_names, save_name="heatmap_sorted_cramer.png")