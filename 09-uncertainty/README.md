## (a) Dropout as a Bayesian Approximation
  - **Title:** Dropout as a Bayesian Approximation: Representing Model Uncertainty in
  Deep Learning
  - **Authors:** Yarin Gal, Zoubin Ghahramani
  - **Year:** 2015
  - **Link:** http://arxiv.org/abs/1506.02142v6
  - **Abstract:** Deep learning tools have gained tremendous attention in applied machine learning. However such tools for regression and classification do not capture model uncertainty. In comparison, Bayesian models offer a mathematically grounded framework to reason about model uncertainty, but usually come with a prohibitive computational cost. In this paper we develop a new theoretical framework casting dropout training in deep neural networks (NNs) as approximate Bayesian inference in deep Gaussian processes. A direct result of this theory gives us tools to model uncertainty with dropout NNs -- extracting information from existing models that has been thrown away so far. This mitigates the problem of representing uncertainty in deep learning without sacrificing either computational complexity or test accuracy. We perform an extensive study of the properties of dropout's uncertainty. Various network architectures and non-linearities are assessed on tasks of regression and classification, using MNIST as an example. We show a considerable improvement in predictive log-likelihood and RMSE compared to existing state-of-the-art methods, and finish by using dropout's uncertainty in deep reinforcement learning.

## (b) Knowing What We Don't Know (Gal Thesis, Ch. 1)
  - **Title:** The Importance of Knowing What We Don't Know (Thesis Chapter 1)
  - **Authors:** Yarin Gal
  - **Year:** 2016
  - **Link:** http://mlg.eng.cam.ac.uk/yarin/thesis/1_introduction.pdf
  - **Intro Paragraph:** In the Bayesian machine learning community we work with probabilistic models and uncertainty. Models such as Gaussian processes, which define probability distributions over functions, are used to learn the more likely and less likely ways to generalise from observed data. This probabilistic view of machine learning offers confidence bounds for data analysis and decision making, information that a biologist for example would rely on to analyse her data, or an autonomous car would use to decide whether to brake or not. In analysing data or making decisions, it is often necessary to be able to tell whether a model is certain about its output, being able to ask "maybe I need to use more diverse data? or change the model? or perhaps be careful when making a decision?". Such questions are of fundamental concern in Bayesian machine learning, and have been studied extensively in the field [Ghahramani, 2015]. When using deep learning models on the other hand [Goodfellow et al., 2016], we generally only have point estimates of parameters and predictions at hand. The use of such models forces us to sacrifice our tools for answering the questions above, potentially leading to situations where we can't tell whether a model is making sensible predictions or just guessing at random.

## (c) The Language of Uncertainty (Gal Thesis, Ch. 2)
  - **Title:** The Language of Uncertainty (Chapter 2 of Thesis)
  - **Authors:** Yarin Gal
  - **Year:** 2016
  - **Link:** http://mlg.eng.cam.ac.uk/yarin/thesis/2_language_of_uncertainty.pdf
  - **Intro Paragraph:** To formalise our discussion of model uncertainty we will rely on probabilistic modelling, and more specifically on Bayesian modelling. Bayesian probability theory offers us the machinery we need to develop our tools. Together with techniques for approximate inference in Bayesian models, in the next chapter we will present the main results of this work. But prior to that, let us review the main ideas underlying Bayesian modelling, approximate inference, and a model of key importance to us: the Bayesian neural network.

## (d) Bayesian Deep Learning (Gal Thesis, Ch. 3)
  - **Title:** Bayesian Deep Learning (Chapter 3 of Thesis)
  - **Authors:** Yarin Gal
  - **Year:** 2016
  - **Link:** http://mlg.eng.cam.ac.uk/yarin/thesis/3_bayesian_deep_learning.pdf
  - **Intro Paragraph:** In previous chapters we reviewed Bayesian neural networks (BNNs) and historical techniques for approximate inference in these, as well as more recent approaches. We discussed the advantages and disadvantages of different techniques, examining their practicality. This, perhaps, is the most important aspect of modern techniques for approximate inference in BNNs. The field of deep learning is pushed forward by practitioners, working on real-world problems. Techniques which cannot scale to complex models with potentially millions of parameters, scale well with large amounts of data, need well studied models to be radically changed, or are not accessible to engineers, will simply perish. In this chapter we will develop on the strand of work of [Graves, 2011; Hinton and Van Camp, 1993], but will do so from the Bayesian perspective rather than the information theory one. Developing Bayesian approaches to deep learning, we will tie approximate BNN inference together with deep learning stochastic regularisation techniques (SRTs) such as dropout. These regularisation techniques are used in many modern deep learning tools, allowing us to offer a practical inference technique.

## (e) Applications of Bayesian Deep Learning (Gal Thesis, Ch. 5)
  - **Title:** Applications (Chapter 5 of Thesis)
  - **Authors:** Yarin Gal
  - **Year:** 2016
  - **Link:** http://mlg.eng.cam.ac.uk/yarin/thesis/5_applications.pdf
  - **Intro Paragraph:** In previous chapters we linked stochastic regularisation techniques (SRTs) to approximate inference in Bayesian neural networks (NNs), and studied the resulting model uncertainty for popular SRTs such as dropout. We have yet to give any real-world applications stemming from this link though, leaving it somewhat in the realms of "theoretical work". But a theory is worth very little if we can’t use it to obtain new tools, or shed light on existing ones. In this chapter we will survey recent literature making use of the tools developed in the previous chapters in fields ranging from language processing to computer vision to biomedical domains. This is followed by more use cases I have worked on recently with the help of others. We will see how model uncertainty can be used to choose what data to learn from (joint work with Riashat Islam as part of his Master’s project). Switching to applications in deep reinforcement learning, we will see how model uncertainty can help exploration. This is then followed by the development of a data efficient framework in deep reinforcement learning (joint work with Rowan McAllister and Carl Rasmussen). A study of the implications of this work on our understanding of existing tools will be given in the next chapter.

## (f) Predictive Uncertainty Estimation with Ensembles
  - **Title:** Simple and Scalable Predictive Uncertainty Estimation using Deep
  Ensembles
  - **Authors:** Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell
  - **Year:** 2017
  - **Link:** http://arxiv.org/abs/1612.01474v3
  - **Abstract:** Deep neural networks (NNs) are powerful black box predictors that have recently achieved impressive performance on a wide spectrum of tasks. Quantifying predictive uncertainty in NNs is a challenging and yet unsolved problem. Bayesian NNs, which learn a distribution over weights, are currently the state-of-the-art for estimating predictive uncertainty; however these require significant modifications to the training procedure and are computationally expensive compared to standard (non-Bayesian) NNs. We propose an alternative to Bayesian NNs that is simple to implement, readily parallelizable, requires very little hyperparameter tuning, and yields high quality predictive uncertainty estimates. Through a series of experiments on classification and regression benchmarks, we demonstrate that our method produces well-calibrated uncertainty estimates which are as good or better than approximate Bayesian NNs. To assess robustness to dataset shift, we evaluate the predictive uncertainty on test examples from known and unknown distributions, and show that our method is able to express higher uncertainty on out-of-distribution examples. We demonstrate the scalability of our method by evaluating predictive uncertainty estimates on ImageNet.

## (g) Bayesian Deep Learning for Computer Vision
  - **Title:** What Uncertainties Do We Need in Bayesian Deep Learning for Computer
  Vision?
  - **Authors:** Alex Kendall, Yarin Gal
  - **Year:** 2017
  - **Link:** http://arxiv.org/abs/1703.04977v2
  - **Abstract:** There are two major types of uncertainty one can model. Aleatoric uncertainty captures noise inherent in the observations. On the other hand, epistemic uncertainty accounts for uncertainty in the model -- uncertainty which can be explained away given enough data. Traditionally it has been difficult to model epistemic uncertainty in computer vision, but with new Bayesian deep learning tools this is now possible. We study the benefits of modeling epistemic vs. aleatoric uncertainty in Bayesian deep learning models for vision tasks. For this we present a Bayesian deep learning framework combining input-dependent aleatoric uncertainty together with epistemic uncertainty. We study models under the framework with per-pixel semantic segmentation and depth regression tasks. Further, our explicit uncertainty formulation leads to new loss functions for these tasks, which can be interpreted as learned attenuation. This makes the loss more robust to noisy data, also giving new state-of-the-art results on segmentation and depth regression benchmarks.

## (h) Using Uncertainty in Disease Detection
  - **Title:** Leveraging uncertainty information from deep neural networks for disease detection
  - **Authors:** Christian Leibig, Vaneeda Allken, Murat Seçkin Ayhan, Philipp Berens & Siegfried Wahl
  - **Year:** 2017
  - **Link:** https://www.nature.com/articles/s41598-017-17876-z
  - **Abstract:** Deep learning (DL) has revolutionized the field of computer vision and image processing. In medical imaging, algorithmic solutions based on DL have been shown to achieve high performance on tasks that previously required medical experts. However, DL-based solutions for disease detection have been proposed without methods to quantify and control their uncertainty in a decision. In contrast, a physician knows whether she is uncertain about a case and will consult more experienced colleagues if needed. Here we evaluate drop-out based Bayesian uncertainty measures for DL in diagnosing diabetic retinopathy (DR) from fundus images and show that it captures uncertainty better than straightforward alternatives. Furthermore, we show that uncertainty informed decision referral can improve diagnostic performance. Experiments across different networks, tasks and datasets show robust generalization. Depending on network capacity and task/dataset difficulty, we surpass 85% sensitivity and 80% specificity as recommended by the NHS when referring 0−20% of the most uncertain decisions for further inspection. We analyse causes of uncertainty by relating intuitions from 2D visualizations to the high-dimensional image space. While uncertainty is sensitive to clinically relevant cases, sensitivity to unfamiliar data samples is task dependent, but can be rendered more robust.

## (i) Out-of-Distribution Detection
  - **Title:** Learning Confidence for Out-of-Distribution Detection in Neural Networks
  - **Authors:** Terrance DeVries, Graham W. Taylor
  - **Year:** 2018
  - **Link:** http://arxiv.org/abs/1802.04865v1
  - **Abstract:** Modern neural networks are very powerful predictive models, but they are often incapable of recognizing when their predictions may be wrong. Closely related to this is the task of out-of-distribution detection, where a network must determine whether or not an input is outside of the set on which it is expected to safely perform. To jointly address these issues, we propose a method of learning confidence estimates for neural networks that is simple to implement and produces intuitively interpretable outputs. We demonstrate that on the task of out-of-distribution detection, our technique surpasses recently proposed techniques which construct confidence based on the network's output distribution, without requiring any additional labels or access to out-of-distribution examples. Additionally, we address the problem of calibrating out-of-distribution detectors, where we demonstrate that misclassified in-distribution examples can be used as a proxy for out-of-distribution examples.

## (j) Uncertainty for Segmentation
  - **Title:** Leveraging Uncertainty Estimates for Predicting Segmentation Quality
  - **Authors:** Terrance DeVries, Graham W. Taylor
  - **Year:** 2018
  - **Link:** http://arxiv.org/abs/1807.00502v1
  - **Abstract:** The use of deep learning for medical imaging has seen tremendous growth in the research community. One reason for the slow uptake of these systems in the clinical setting is that they are complex, opaque and tend to fail silently. Outside of the medical imaging domain, the machine learning community has recently proposed several techniques for quantifying model uncertainty (i.e. a model knowing when it has failed). This is important in practical settings, as we can refer such cases to manual inspection or correction by humans. In this paper, we aim to bring these recent results on estimating uncertainty to bear on two important outputs in deep learning-based segmentation. The first is producing spatial uncertainty maps, from which a clinician can observe where and why a system thinks it is failing. The second is quantifying an image-level prediction of failure, which is useful for isolating specific cases and removing them from automated pipelines. We also show that reasoning about spatial uncertainty, the first output, is a useful intermediate representation for generating segmentation quality predictions, the second output. We propose a two-stage architecture for producing these measures of uncertainty, which can accommodate any deep learning-based medical segmentation pipeline.
