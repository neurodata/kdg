### [TOWARDS NEURAL NETWORKS THAT PROVABLY KNOW WHEN THEY DON’T KNOW](https://arxiv.org/pdf/1909.12180.pdf)
#### Notes 
- Uses generative model.
- Assumes that samples from an out-distribution are given.
- Decomposes the the overall posterior $P(y|x)$ into $P(y|x,i)$ and $P(y|x,o)$. For in-distribution $P(y|x,i) >> P(y|x,o)$ and $P(y|x) = p(y|x,i)$. For OOD, $P(y|x,i) << P(y|x,o)$ and $P(y|x) = \frac{1}{M}$.
- Uses GMM to learn $P(y|x,i)$ and $P(y|x,o)$.
- Uses 80 million tiny image dataset to learn $P(y|x,o)$.
- Gives guarantees of performance far away like ours.
- Treats OOD detection as binary class classification using confidence as criteria and reports AUC.

### [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://proceedings.neurips.cc/paper/2018/file/abdeb6f575ac5c6676b747bca8d09cc2-Paper.pdf)

#### Notes 
- Fits GMM with respect to features extracted at different lvels of deep-nets.
- Considers the closest Gaussian in terms of Mahalanobis distance (We consider Euclidean distance as for small regions over the manifold in a polytope the manifold can be approaximated as Euclidean).
- Does not need OOD samples to train on like ours.
- Uses AUROC of threshold-based detector using the confidence score.
- Uses DenseNet-100, ResNet-34 rained on CIFAR-10 and tests on TinyImageNet, LSUN, SVHN and adversarial (DeepFool) samples.
Used statistics: AUROC, AUPR, TNR at 95% TPR, dtection accuracy.

### [Rethinking Feature Distribution for Loss Functions in Image Classification Weitao](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wan_Rethinking_Feature_Distribution_CVPR_2018_paper.pdf)
#### Notes
- Assumes the learned features of the training set to follow a Gaussian Mixture (GM) distribution, with each component representing a class.
- Transforms the projected feature space using a loss function with regularization from deep-net as Gausian.