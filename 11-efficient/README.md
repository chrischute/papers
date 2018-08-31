## 1. Deep Compression
  - **Title:** Deep Compression: Compressing Deep Neural Networks with Pruning, Trained
  Quantization and Huffman Coding
  - **Authors:** Song Han, Huizi Mao, William J. Dally
  - **Year:** 2016
  - **Link:** http://arxiv.org/abs/1510.00149v5
  - **Abstract:** Neural networks are both computationally intensive and memory intensive, making them difficult to deploy on embedded systems with limited hardware resources. To address this limitation, we introduce "deep compression", a three stage pipeline: pruning, trained quantization and Huffman coding, that work together to reduce the storage requirement of neural networks by 35x to 49x without affecting their accuracy. Our method first prunes the network by learning only the important connections. Next, we quantize the weights to enforce weight sharing, finally, we apply Huffman coding. After the first two steps we retrain the network to fine tune the remaining connections and the quantized centroids. Pruning, reduces the number of connections by 9x to 13x; Quantization then reduces the number of bits that represent each connection from 32 to 5. On the ImageNet dataset, our method reduced the storage required by AlexNet by 35x, from 240MB to 6.9MB, without loss of accuracy. Our method reduced the size of VGG-16 by 49x from 552MB to 11.3MB, again with no loss of accuracy. This allows fitting the model into on-chip SRAM cache rather than off-chip DRAM memory. Our compression method also facilitates the use of complex neural networks in mobile applications where application size and download bandwidth are constrained. Benchmarked on CPU, GPU and mobile GPU, compressed network has 3x to 4x layerwise speedup and 3x to 7x better energy efficiency.

## 2. Quantized Neural Networks
  - **Title:** Quantized Neural Networks: Training Neural Networks with Low Precision
  Weights and Activations
  - **Authors:** Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio
  - **Year:** 2016
  - **Link:** http://arxiv.org/abs/1609.07061v1
  - **Abstract:** We introduce a method to train Quantized Neural Networks (QNNs) --- neural networks with extremely low precision (e.g., 1-bit) weights and activations, at run-time. At train-time the quantized weights and activations are used for computing the parameter gradients. During the forward pass, QNNs drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations. As a result, power consumption is expected to be drastically reduced. We trained QNNs over the MNIST, CIFAR-10, SVHN and ImageNet datasets. The resulting QNNs achieve prediction accuracy comparable to their 32-bit counterparts. For example, our quantized version of AlexNet with 1-bit weights and 2-bit activations achieves 51% top-1 accuracy. Moreover, we quantize the parameter gradients to 6-bits as well which enables gradients computation using only bit-wise operation. Quantized recurrent neural networks were tested over the Penn Treebank dataset, and achieved comparable accuracy as their 32-bit counterparts using only 4-bits. Last but not least, we programmed a binary matrix multiplication GPU kernel with which it is possible to run our MNIST QNN 7 times faster than with an unoptimized GPU kernel, without suffering any loss in classification accuracy. The QNN code is available online.

## 3. SqueezeNet
  - **Title:** SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB
  model size
  - **Authors:** Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer
  - **Year:** 2016
  - **Link:** http://arxiv.org/abs/1602.07360v4
  - **Abstract:** Recent research on deep neural networks has focused primarily on improving accuracy. For a given accuracy level, it is typically possible to identify multiple DNN architectures that achieve that accuracy level. With equivalent accuracy, smaller DNN architectures offer at least three advantages: (1) Smaller DNNs require less communication across servers during distributed training. (2) Smaller DNNs require less bandwidth to export a new model from the cloud to an autonomous car. (3) Smaller DNNs are more feasible to deploy on FPGAs and other hardware with limited memory. To provide all of these advantages, we propose a small DNN architecture called SqueezeNet. SqueezeNet achieves AlexNet-level accuracy on ImageNet with 50x fewer parameters. Additionally, with model compression techniques we are able to compress SqueezeNet to less than 0.5MB (510x smaller than AlexNet).   The SqueezeNet architecture is available for download here: https://github.com/DeepScale/SqueezeNet

## 4. MobileNet
  - **Title:** MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
  Applications
  - **Authors:** Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
  - **Year:** 2017
  - **Link:** http://arxiv.org/abs/1704.04861v1
  - **Abstract:** We present a class of efficient models called MobileNets for mobile and embedded vision applications. MobileNets are based on a streamlined architecture that uses depth-wise separable convolutions to build light weight deep neural networks. We introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. These hyper-parameters allow the model builder to choose the right sized model for their application based on the constraints of the problem. We present extensive experiments on resource and accuracy tradeoffs and show strong performance compared to other popular models on ImageNet classification. We then demonstrate the effectiveness of MobileNets across a wide range of applications and use cases including object detection, finegrain classification, face attributes and large scale geo-localization.

## 5. ShuffleNet
  - **Title:** ShuffleNet: An Extremely Efficient Convolutional Neural Network for
  Mobile Devices
  - **Authors:** Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun
  - **Year:** 2017
  - **Link:** http://arxiv.org/abs/1707.01083v2
  - **Abstract:** We introduce an extremely computation-efficient CNN architecture named ShuffleNet, which is designed specially for mobile devices with very limited computing power (e.g., 10-150 MFLOPs). The new architecture utilizes two new operations, pointwise group convolution and channel shuffle, to greatly reduce computation cost while maintaining accuracy. Experiments on ImageNet classification and MS COCO object detection demonstrate the superior performance of ShuffleNet over other structures, e.g. lower top-1 error (absolute 7.8%) than recent MobileNet on ImageNet classification task, under the computation budget of 40 MFLOPs. On an ARM-based mobile device, ShuffleNet achieves roughly 13x actual speedup over AlexNet while maintaining comparable accuracy.

## (Extra) Binary Neural Networks
  - **Title:** Binarized Neural Networks: Training Deep Neural Networks with Weights
  and Activations Constrained to +1 or -1
  - **Authors:** Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio
  - **Year:** 2016
  - **Link:** http://arxiv.org/abs/1602.02830v3
  - **Abstract:** We introduce a method to train Binarized Neural Networks (BNNs) - neural networks with binary weights and activations at run-time. At training-time the binary weights and activations are used for computing the parameters gradients. During the forward pass, BNNs drastically reduce memory size and accesses, and replace most arithmetic operations with bit-wise operations, which is expected to substantially improve power-efficiency. To validate the effectiveness of BNNs we conduct two sets of experiments on the Torch7 and Theano frameworks. On both, BNNs achieved nearly state-of-the-art results over the MNIST, CIFAR-10 and SVHN datasets. Last but not least, we wrote a binary matrix multiplication GPU kernel with which it is possible to run our MNIST BNN 7 times faster than with an unoptimized GPU kernel, without suffering any loss in classification accuracy. The code for training and running our BNNs is available on-line.

## (Extra) Ternary Neural Networks
  - **Title:** Trained Ternary Quantization
  - **Authors:** Chenzhuo Zhu, Song Han, Huizi Mao, William J. Dally
  - **Year:** 2017
  - **Link:** http://arxiv.org/abs/1612.01064v3
  - **Abstract:** Deep neural networks are widely used in machine learning applications. However, the deployment of large neural networks models can be difficult to deploy on mobile devices with limited power budgets. To solve this problem, we propose Trained Ternary Quantization (TTQ), a method that can reduce the precision of weights in neural networks to ternary values. This method has very little accuracy degradation and can even improve the accuracy of some models (32, 44, 56-layer ResNet) on CIFAR-10 and AlexNet on ImageNet. And our AlexNet model is trained from scratch, which means it's as easy as to train normal full precision model. We highlight our trained quantization method that can learn both ternary values and ternary assignment. During inference, only ternary values (2-bit weights) and scaling factors are needed, therefore our models are nearly 16x smaller than full-precision models. Our ternary models can also be viewed as sparse binary weight networks, which can potentially be accelerated with custom circuit. Experiments on CIFAR-10 show that the ternary models obtained by trained quantization method outperform full-precision models of ResNet-32,44,56 by 0.04%, 0.16%, 0.36%, respectively. On ImageNet, our model outperforms full-precision AlexNet model by 0.3% of Top-1 accuracy and outperforms previous ternary models by 3%.

## (Extra) Mobile NASNet
  - **Title:** MnasNet: Platform-Aware Neural Architecture Search for Mobile
  - **Authors:** Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Quoc V. Le
  - **Year:** 2018
  - **Link:** http://arxiv.org/abs/1807.11626v1
  - **Abstract:** Designing convolutional neural networks (CNN) models for mobile devices is challenging because mobile models need to be small and fast, yet still accurate. Although significant effort has been dedicated to design and improve mobile models on all three dimensions, it is challenging to manually balance these trade-offs when there are so many architectural possibilities to consider. In this paper, we propose an automated neural architecture search approach for designing resource-constrained mobile CNN models. We propose to explicitly incorporate latency information into the main objective so that the search can identify a model that achieves a good trade-off between accuracy and latency. Unlike in previous work, where mobile latency is considered via another, often inaccurate proxy (e.g., FLOPS), in our experiments, we directly measure real-world inference latency by executing the model on a particular platform, e.g., Pixel phones. To further strike the right balance between flexibility and search space size, we propose a novel factorized hierarchical search space that permits layer diversity throughout the network. Experimental results show that our approach consistently outperforms state-of-the-art mobile CNN models across multiple vision tasks. On the ImageNet classification task, our model achieves 74.0% top-1 accuracy with 76ms latency on a Pixel phone, which is 1.5x faster than MobileNetV2 (Sandler et al. 2018) and 2.4x faster than NASNet (Zoph et al. 2018) with the same top-1 accuracy. On the COCO object detection task, our model family achieves both higher mAP quality and lower latency than MobileNets.
