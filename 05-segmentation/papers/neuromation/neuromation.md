## Blogpost
  - **Title:** DeepGlobe Challenge: Three Papers from Neuromation Accepted!
  - **Authors:** Neuromation, Sergey Golovanov
  - **Year:** 2018
  - **Link:** https://medium.com/neuromation-io-blog/deepglobe-challenge-three-papers-from-neuromation-accepted-fe09a1a7fa53
  - **Abstract:** In our solution for the segmentation of buildings, we used an architecture similar to U-Net. As the encoder, we used the SE-ResNeXt-50 pretrained on ImageNet. We chose it because this classifier is of high enough quality and does not require too much memory, which is important to maintain a large batch size during training. To the encoder, we added Atrous Spatial Pyramid Pooling with Image Pooling. The decoder also contains four blocks, each of which is a sequence of convolution, deconvolution, and another convolution. Besides, following the U-Net idea we added skip-connections from the encoder at each level of the decoder.
