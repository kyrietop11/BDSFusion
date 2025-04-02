# *<center>BDSFusion: Bidirectional-Driven Saliency Fusion Network for Infrared Dual-Band Images</center>*
This is the official code for **“BDSFusion: Bidirectional-Driven Saliency Fusion Network for Infrared Dual-Band Images”**. **Note:** Currently, we have only made the core modules of the model's code public. Once our paper is accepted, we will make the full code publicly available.

## Dataset
The dataset used in our paper is a semi-simulated dataset with real background  MWIR and LWIR images and synthesized small target motion, appearance, and intensity through a semi-simulated approach. This dataset consisted of 15,331 pairs of meticulously aligned MWIR and LWIR images, each with a resolution of 640×512 pixels and target sizes ranging from 5 to 15 pixels. It encompassed various complex environments including high-brightness clouds and sea clutter. The data characteristics are as follows:
![SCR](https://github.com/kyrietop11/BDSFusion/blob/main/figures/SCR.png) ![Scene](https://github.com/kyrietop11/BDSFusion/blob/main/figures/Scene.png)

## Recommended Environment
 - [ ] torch  1.13.1
 - [ ] cudatoolkit 11.7
 - [ ] torchvision 0.14.1
 - [ ] mamba-ssm 1.0.1
 - [ ] causal-conv1d 1.0.0

## Qualitative Results of Infrared Small Target Detection

#### Detection results on ACM-Net
![image](https://github.com/kyrietop11/BDSFusion/blob/main/figures/Detection%20results%20on%20ACM-Net.svg)


#### Detection results on IRTransDet
![image](https://github.com/kyrietop11/BDSFusion/blob/main/figures/Detection%20results%20on%20IRTransDet.svg)

## Contact
**Welcome to raise issues or email to [nwpu_lys@mail.nwpu.edu.cn](nwpu_lys@mail.nwpu.edu.cn) for any question regarding our BDSFusion.**
