"""
Calculate the heatmap of the number of shared prototypes between concepts
"""
import os
import pickle
import numpy as np
from itertools import combinations, product
import torch
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

from types import SimpleNamespace

config = SimpleNamespace(
    seed = 25,

    data_path = '.',
    group_by = "color",
    num_groups = 27,                              
    num_groups_tot = 27, # 28 if oracle, 27 if color
    random_groups = False,
    masked_groups = [],

    root_dir = '.',
    run_dir = 'awa2_resnet50_l11e-5_clst0.8_sep-0.08'

)

def get_concept_groups_CUB(config):
    """
    Get the concept groups for CUB dataset
    """
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


with open(config.root_dir + config.run_dir + '/prototype_to_concept_dict.pkl', 'rb') as f:
    x = pickle.load(f)

keys_with_empty_lists = [key for key, value in x.items() if not value]
print(len(keys_with_empty_lists)) # number of empty prototypes

category_matrix = np.zeros((224, 224), dtype=int)

for categories in x.values():
    for cat1, cat2 in combinations(categories, 2):
        category_matrix[cat1, cat2] += 1
        category_matrix[cat2, cat1] += 1  # Ensure the matrix is symmetric

groups, concept_names_graph = get_concept_groups_CUB(config)

num_categories, num_vars = groups.shape
categories = np.zeros(num_vars, dtype=int)
for var_index in range(num_vars):
    category_indices = np.where(groups[:, var_index] != 0)[0]
    if category_indices.size > 0:
        categories[var_index] = category_indices[0] + 1

sorted_indices = np.argsort(categories)
sorted_concept_names = [concept_names_graph[i] for i in sorted_indices]
sorted_indices = np.append(sorted_indices, sorted_indices + 112)
sorted_concept_names = np.append(sorted_concept_names, ["NOT" + i for i in sorted_concept_names])

# Clipping values bigger than 10
clipped_matrix = np.clip(category_matrix, 0, 10)
sorted_clipped_matrix = clipped_matrix[sorted_indices, :][:, sorted_indices]

cmap = ListedColormap(['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b', '#f03b20'])
bounds = np.arange(11)
norm = BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(60, 48))
sns.heatmap(sorted_clipped_matrix, annot=False, fmt="d", cmap=cmap, norm=norm, cbar_kws={"ticks": bounds},
            xticklabels=sorted_concept_names,
            yticklabels=sorted_concept_names)
plt.xticks(rotation=90)  # Rotate x labels for better readability
plt.yticks(rotation=0)  # Keep y labels horizontal

plt.savefig(f"heatmaps/heatmap_{config.run_dir}.png")