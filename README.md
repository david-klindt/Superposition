# QUANTIFYING POLYSEMANTICITY IN CONVOLUTIONAL NEURAL NETWORKS

A recent hypothesis suggests that neural networks represent semantically mean-
ingful input patterns (i.e., features) not with individual neurons but in superposi-
tion. Theory suggests that this may be an advantageous encoding strategy when
features are sparse, however, empirical evidence in support of the theory has so
far been limited to language models. Here, we propose a framework for study-
ing feature superposition (i.e., polysemanticity) in convolutional neural networks
(CNNs) that process natural images. Our quantitative metric is closely inspired
by recent psychophysical experiments that operationalize semantics through hu-
man perceptual judgements. We rigorously study the statistical assumptions and
philosophical implications behind these measures and discuss how they evolve
across the hierarchy of visual processing. Equipped with these new methods, we
do indeed find polysemantic neurons in CNNs, confirming prior work. We then
directly test the superposition hypothesis by simple k-Means clustering in activa-
tion space. This approach yields clusters of activity that are more monosemantic
than individual neurons, confirming the existence of monosemantic features in su-
perposition. We show how this corresponds to sparse activation on the natural
image manifold, how this interacts with lower level features such as colour, and
higher level features such as object category. Moreover, in an effort of mechanis-
tic interpretability, we discover pairs of neurons with feature synergy, i.e., sparse
combinations in activation space that yield more monosemantic features than ei-
ther of the constituent parts. These results have implications for neural network
interpretability, system identifiabiltiy, and more broadly for the representations in
neural systems.
