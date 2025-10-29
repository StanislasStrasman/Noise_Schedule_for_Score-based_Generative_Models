<div align="center">

# An Analysis of the Noise Schedule for Score-Based Generative Models

Official code repository for the paper:  [**An Analysis of the Noise Schedule for Score-Based Generative Models**](https://arxiv.org/abs/2402.04650)  published in *Transactions on Machine Learning Research (TMLR)*.

This repository contains the code to experiments and figures from the paper, as well as modular implementations of diffusion processes, noise schedules and theoretical bound approximations for Score-based Generative Models.

</div>

##  Repository structure

- `src/noise_diff/` – Python package implementing diffusion processes, samplers, and analytical tools.  
- `examples/` – Jupyter notebooks reproducing the main numerical experiments.  
- `requirements.txt` – Dependencies required to reproduce results.  
- `pyproject.toml` – Package metadata for editable installation.

## Installation & Reproducing results

Clone the repository and install dependencies:

```bash
git clone https://github.com/StanislasStrasman/Noise_Schedule_for_Score-based_Generative_Models.git
cd Noise_Schedule_for_Score-based_Generative_Models
pip install -r requirements.txt
pip install -e .
```
An illustrated example is given in `examples/illustrated_example.ipynb`. It provides a complete walkthrough of the setup, training and analysis of synthetic data examples. All scripts rely on the noise_diff package defined in src/. You can open it locally or directly [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StanislasStrasman/Noise_Schedule_for_Score-based_Generative_Models/blob/main/examples/illustrated_example.ipynb)

A version of this repository including the **example on a real dataset** using **pre-trained diffusion models** from  
```bibtex
@inproceedings{Karras2022edm,
  author    = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
  title     = {Elucidating the Design Space of Diffusion-Based Generative Models},
  booktitle = {Advances in Neural Information Processing Systems 35 (NeurIPS 2022)},
  year      = {2022},
  url       = {https://arxiv.org/abs/2206.00364}
}
```
will be added **soon**.  


## Overview

| Module | Description |
|---------|-------------|
| `diffusion.py` | Implements forward diffusion SDEs and associated training losses. |
| `functions.py` | Utility functions for training, evaluation, and visualization. |
| `synthetic_data.py` | Synthetic datasets (e.g., Gaussian mixtures, Funnel). |
| `sampler.py` | Numerical samplers for score-based generative models. |
| `decoder.py` | Neural decoder architectures. |
| `upperbounds.py` | Analytical upper bounds and Gaussian approximations. |


## Documentation

A short module-level overview and description is provided in `short_documentation.ipynb` for easy navigation. For further theoretical context and mathematical derivations, please refer to
  **Appendix E**](https://arxiv.org/abs/2402.04650) of the paper.


## If you use this code please cite 

```bibtex
@article{
strasman2025an,
title={An analysis of the noise schedule for score-based generative models},
author={Stanislas Strasman and Antonio Ocello and Claire Boyer and Sylvain Le Corff and Vincent Lemaire},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
}
```



