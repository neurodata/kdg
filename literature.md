### TOWARDS NEURAL NETWORKS THAT PROVABLY KNOW WHEN THEY DON’T KNOW
[![arXiv](https://img.shields.io/badge/arXiv-2004.12908-red.svg?style=flat)](https://arxiv.org/pdf/1909.12180.pdf)

#### Notes 
- Uses generative model.
- Assumes that samples from an out-distribution are given.
- Decomposes the the overall posterior $P(y|x)$ into $P(y|x,i)$ and $P(y|x,o)$. For in-distribution $P(y|x,i) >> P(y|x,o)$ and $P(y|x) = p(y|x,i)$. For OOD, $P(y|x,i) << P(y|x,o)$ and $P(y|x) = 1/M$.
- Uses GMM to learn $P(y|x,i)$ and $P(y|x,o)$.
- Uses 80 million tiny image dataset to learn $p(y|x,o)$.
