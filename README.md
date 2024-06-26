# HintNet
HintNet: Hierarchical Knowledge Transfer Networks for Traffic Accident Forecasting on Heterogeneous Spatio-Temporal Data
Proceedings of the 2022 SIAM International Conference on Data Mining (SDM)

# Abstract
Traffic accident forecasting is a significant problem for transportation management and public safety. However, this problem is challenging due to the spatial heterogeneity of the environment and the sparsity of accidents in space and time. The occurrence of traffic accidents is affected by complex dependencies among spatial and temporal features. Recent traffic accident prediction methods have attempted to use deep learning models to improve accuracy. However, most of these methods either focus on small-scale and homogeneous areas such as populous cities or simply use sliding-window-based ensemble methods, which are inadequate to handle heterogeneity in large regions. To address these limitations, this paper proposes a novel Hierarchical Knowledge Transfer Network (HintNet) model to better capture irregular heterogeneity patterns. HintNet performs a multi-level spatial partitioning to separate sub-regions with different risks and learns a deep network model for each level using spatio-temporal and graph convolutions. Through knowledge transfer across levels, HintNet archives both higher accuracy and higher training efficiency. Extensive experiments on a real-world accident dataset from the state of Iowa demonstrate that HintNet outperforms the state-of-the-art methods on spatially heterogeneous and large-scale areas.

# Environment
- python 3.7.0
- torch 1.12.1
- matplotlib 3.5.2
- numpy 1.21.5
- sklearn 1.1.1
- seaborn 0.11.2

# Run HintNet
Please follow the readme page \
Start with main.py

# Citation
```
@inbook{doi:10.1137/1.9781611977172.38,
author = {Bang An and Amin Vahedian and Xun Zhou and W. Nick Street and Yanhua Li},
title = {HintNet: Hierarchical Knowledge Transfer Networks for Traffic Accident Forecasting on Heterogeneous Spatio-Temporal Data},
booktitle = {Proceedings of the 2022 SIAM International Conference on Data Mining (SDM)},
chapter = {},
pages = {334-342},
doi = {10.1137/1.9781611977172.38},
URL = {https://epubs.siam.org/doi/abs/10.1137/1.9781611977172.38},
eprint = {https://epubs.siam.org/doi/pdf/10.1137/1.9781611977172.38},
}
```
