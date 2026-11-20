# Deep Learning Projects
 
A collection of hands-on deep learning projects, built mostly from scratch in PyTorch to understand how the models actually work: from a tiny autograd engine to CNNs and character-level language models.
 
## Contents
 
| Folder | Project | What it covers |
|---|---|---|
| `Micro grid/` | **micrograd** | A small autograd engine: scalar `Value` objects, backpropagation through a computation graph, and a minimal neural network built on top of it |
| `CNN projects/` | **MNIST** | Handwritten digit classification with a convolutional network |
| `CNN projects/` | **Fashion-MNIST** | Clothing image classification, a harder drop-in benchmark than MNIST |
| `CNN projects/` | **Dogs vs. Cats** | Binary image classification of dog and cat photos with a convolutional network |
| `CNN projects/` | **Chest X-ray Pneumonia Classification** | Medical image classification: detecting pneumonia from chest X-ray images |
| `RNN projects/Name-Generator-MLP/` | **Character-level name generator (MLP)** | Next-character prediction with a fixed context window, learned embeddings, a hidden `tanh` layer, BatchNorm, and L2 regularization. Generates new names by sampling one character at a time |
| `RNN projects/Name classification/` | **Name classification** | Classifying surnames by language of origin (18 languages) with a recurrent network |
| `RNN projects/Dinosaurus_Island_Character_level_language_model/` | **Dinosaurus Island** | A character-level RNN language model that invents new dinosaur names |
 
## Highlights
 
- **Built from first principles.** Training loops, losses, and gradient updates are written out explicitly instead of hidden behind high-level APIs.
- **Proper train / dev / test splits** (90 / 5 / 5) so that tuning decisions are made on the dev set and the test set stays untouched until the end.
- **Overfitting handling.** Monitoring train vs. dev loss, weight regularization on `W1` / `W2`, and Batch Normalization with running statistics for inference.
- **Autoregressive generation.** Models trained on "previous characters → next character" so they can generate new text without needing any future context.
## Getting started
 
```bash
git clone https://github.com/KamalA7med/Deep-Learning-Projects.git
cd Deep-Learning-Projects
pip install torch numpy pandas matplotlib jupyter
jupyter notebook
```
 
Then open any notebook from the folders above. A GPU is optional. The notebooks pick `cuda` automatically if it is available and fall back to CPU otherwise.
 
## Requirements
 
- Python 3.9+
- PyTorch
- NumPy, pandas, matplotlib
- Jupyter Notebook or JupyterLab (or open the notebooks in Google Colab)
## Repository structure
 
```
Deep-Learning-Projects/
├── CNN projects/
│   ├── MNIST.ipynb
│   ├── FashionMNIST.ipynb
│   ├── dogs-vs-cats.ipynb
│   └── chest-x-ray-pneumonia-classification.ipynb
├── Micro grid/
│   └── micrograd.ipynb
└── RNN projects/
    ├── Dinosaurus_Island_Character_level_language_model/
    ├── Name classification/
    └── Name-Generator-MLP/
```
 
## Notes
 
- Some notebooks read data files (for example `names.txt`) from a relative path. If a notebook was written for Google Colab, update the path to point at the file in its folder.
- Results vary slightly between runs unless you fix the random seed (`torch.manual_seed(...)`).
## Acknowledgements
 
The micrograd and character-level name generator projects follow the approach taught in Andrej Karpathy's *Neural Networks: Zero to Hero* series. The name classification and Dinosaurus Island projects follow well-known tutorial and course exercises. All code here was written and run as part of my own learning.
 
## Author
 
**Kamal Ahmed** ([@KamalA7med](https://github.com/KamalA7med))
 

 
