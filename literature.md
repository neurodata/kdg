### [TOWARDS NEURAL NETWORKS THAT PROVABLY KNOW WHEN THEY DON’T KNOW](https://arxiv.org/pdf/1909.12180.pdf)
#### Notes 
- Uses generative model.
- Assumes that samples from an out-distribution are given.
- Decomposes the the overall posterior $P(y|x)$ into $P(y|x,i)$ and $P(y|x,o)$. For in-distribution $P(y|x,i) >> P(y|x,o)$ and $P(y|x) = p(y|x,i)$. For OOD, $P(y|x,i) << P(y|x,o)$ and $P(y|x) = \frac{1}{M}$.
- Uses GMM to learn $P(y|x,i)$ and $P(y|x,o)$.
- Uses 80 million tiny image dataset to learn $p(y|x,o)$.
- Gives guarantees of performance far away like ours.
- Treats OOD detection as binary class classification using confidence as criteria and reports AUC.

### [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf)

#### Notes 
- 