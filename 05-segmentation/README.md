## 1. FCN
  - **Title:** Fully Convolutional Networks for Semantic Segmentation
  - **Authors:** Jonathan Long, Evan Shelhamer, Trevor Darrell
  - **Year:** 2015
  - **Link:** http://arxiv.org/abs/1411.4038v2
  - **Abstract:** Convolutional networks are powerful visual models that yield hierarchies of features. We show that convolutional networks by themselves, trained end-to-end, pixels-to-pixels, exceed the state-of-the-art in semantic segmentation. Our key insight is to build "fully convolutional" networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning. We define and detail the space of fully convolutional networks, explain their application to spatially dense prediction tasks, and draw connections to prior models. We adapt contemporary classification networks (AlexNet, the VGG net, and GoogLeNet) into fully convolutional networks and transfer their learned representations by fine-tuning to the segmentation task. We then define a novel architecture that combines semantic information from a deep, coarse layer with appearance information from a shallow, fine layer to produce accurate and detailed segmentations. Our fully convolutional network achieves state-of-the-art segmentation of PASCAL VOC (20% relative improvement to 62.2% mean IU on 2012), NYUDv2, and SIFT Flow, while inference takes one third of a second for a typical image.

## 2. U-Net
  - **Title:** U-Net: Convolutional Networks for Biomedical Image Segmentation
  - **Authors:** Olaf Ronneberger, Philipp Fischer, Thomas Brox
  - **Year:** 2015
  - **Link:** http://arxiv.org/abs/1505.04597v1
  - **Abstract:** There is large consent that successful training of deep networks requires many thousand annotated training samples. In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks. Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these categories by a large margin. Moreover, the network is fast. Segmentation of a 512x512 image takes less than a second on a recent GPU. The full implementation (based on Caffe) and the trained networks are available at http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net .

## 3. DeepLab
  - **Title:** DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
  Atrous Convolution, and Fully Connected CRFs
  - **Authors:** Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille
  - **Year:** 2017
  - **Link:** http://arxiv.org/abs/1606.00915v2
  - **Abstract:** In this work we address the task of semantic image segmentation with Deep Learning and make three main contributions that are experimentally shown to have substantial practical merit. First, we highlight convolution with upsampled filters, or 'atrous convolution', as a powerful tool in dense prediction tasks. Atrous convolution allows us to explicitly control the resolution at which feature responses are computed within Deep Convolutional Neural Networks. It also allows us to effectively enlarge the field of view of filters to incorporate larger context without increasing the number of parameters or the amount of computation. Second, we propose atrous spatial pyramid pooling (ASPP) to robustly segment objects at multiple scales. ASPP probes an incoming convolutional feature layer with filters at multiple sampling rates and effective fields-of-views, thus capturing objects as well as image context at multiple scales. Third, we improve the localization of object boundaries by combining methods from DCNNs and probabilistic graphical models. The commonly deployed combination of max-pooling and downsampling in DCNNs achieves invariance but has a toll on localization accuracy. We overcome this by combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF), which is shown both qualitatively and quantitatively to improve localization performance. Our proposed "DeepLab" system sets the new state-of-art at the PASCAL VOC-2012 semantic image segmentation task, reaching 79.7% mIOU in the test set, and advances the results on three other datasets: PASCAL-Context, PASCAL-Person-Part, and Cityscapes. All of our code is made publicly available online.

## 4. DeepLabv3
  - **Title:** Rethinking Atrous Convolution for Semantic Image Segmentation
  - **Authors:** Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
  - **Year:** 2017
  - **Link:** http://arxiv.org/abs/1706.05587v3
  - **Abstract:** In this work, we revisit atrous convolution, a powerful tool to explicitly adjust filter's field-of-view as well as control the resolution of feature responses computed by Deep Convolutional Neural Networks, in the application of semantic image segmentation. To handle the problem of segmenting objects at multiple scales, we design modules which employ atrous convolution in cascade or in parallel to capture multi-scale context by adopting multiple atrous rates. Furthermore, we propose to augment our previously proposed Atrous Spatial Pyramid Pooling module, which probes convolutional features at multiple scales, with image-level features encoding global context and further boost performance. We also elaborate on implementation details and share our experience on training our system. The proposed 'DeepLabv3' system significantly improves over our previous DeepLab versions without DenseCRF post-processing and attains comparable performance with other state-of-art models on the PASCAL VOC 2012 semantic image segmentation benchmark.

## 5. DeepLabv3+
  - **Title:** Encoder-Decoder with Atrous Separable Convolution for Semantic Image
  Segmentation
  - **Authors:** Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam
  - **Year:** 2018
  - **Link:** http://arxiv.org/abs/1802.02611v2
  - **Abstract:** Spatial pyramid pooling module or encode-decoder structure are used in deep neural networks for semantic segmentation task. The former networks are able to encode multi-scale contextual information by probing the incoming features with filters or pooling operations at multiple rates and multiple effective fields-of-view, while the latter networks can capture sharper object boundaries by gradually recovering the spatial information. In this work, we propose to combine the advantages from both methods. Specifically, our proposed model, DeepLabv3+, extends DeepLabv3 by adding a simple yet effective decoder module to refine the segmentation results especially along object boundaries. We further explore the Xception model and apply the depthwise separable convolution to both Atrous Spatial Pyramid Pooling and decoder modules, resulting in a faster and stronger encoder-decoder network. We demonstrate the effectiveness of the proposed model on the PASCAL VOC 2012 semantic image segmentation dataset and achieve a performance of 89% on the test set without any post-processing. Our paper is accompanied with a publicly available reference implementation of the proposed models in Tensorflow at https://github.com/tensorflow/models/tree/master/research/deeplab.

## (Extra) Depthwise Separable Convolution
  - **Title:** Xception: Deep Learning with Depthwise Separable Convolutions
  - **Authors:** François Chollet
  - **Year:** 2017
  - **Link:** http://arxiv.org/abs/1610.02357v3
  - **Abstract:** We present an interpretation of Inception modules in convolutional neural networks as being an intermediate step in-between regular convolution and the depthwise separable convolution operation (a depthwise convolution followed by a pointwise convolution). In this light, a depthwise separable convolution can be understood as an Inception module with a maximally large number of towers. This observation leads us to propose a novel deep convolutional neural network architecture inspired by Inception, where Inception modules have been replaced with depthwise separable convolutions. We show that this architecture, dubbed Xception, slightly outperforms Inception V3 on the ImageNet dataset (which Inception V3 was designed for), and significantly outperforms Inception V3 on a larger image classification dataset comprising 350 million images and 17,000 classes. Since the Xception architecture has the same number of parameters as Inception V3, the performance gains are not due to increased capacity but rather to a more efficient use of model parameters.

## (Extra) Neuromation Blogpost
  - **Title:** DeepGlobe Challenge: Three Papers from Neuromation Accepted!
  - **Authors:** Neuromation, Sergey Golovanov
  - **Year:** 2018
  - **Link:** https://medium.com/neuromation-io-blog/deepglobe-challenge-three-papers-from-neuromation-accepted-fe09a1a7fa53
  - **Abstract:** In our solution for the segmentation of buildings, we used an architecture similar to U-Net. As the encoder, we used the SE-ResNeXt-50 pretrained on ImageNet. We chose it because this classifier is of high enough quality and does not require too much memory, which is important to maintain a large batch size during training. To the encoder, we added Atrous Spatial Pyramid Pooling with Image Pooling. The decoder also contains four blocks, each of which is a sequence of convolution, deconvolution, and another convolution. Besides, following the U-Net idea we added skip-connections from the encoder at each level of the decoder.
