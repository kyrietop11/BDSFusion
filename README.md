# *<center>BDSFusion: Bidirectional-Driven Saliency Fusion Network for Infrared Dual-Band Images</center>*
This is the official code for **“BDSFusion: Bidirectional-Driven Saliency Fusion Network for Infrared Dual-Band Images”**. **Note:** Currently, we have only made the core modules of the model's code public. Once our paper is accepted, we will make the full code publicly available.

## Motivation
<table>
  <tr>
    <td style="text-align: center; width: 50%;">
      <img src="https://github.com/kyrietop11/BDSFusion/blob/main/figures/Transformer.gif" style="width: 90%;" />
      <strong>Transformer</strong>
    </td>
    <td style="text-align: center; width: 50%;">
      <img src="https://github.com/kyrietop11/BDSFusion/blob/main/figures/Mamba.gif" style="width: 100%;" />
      <strong>Mamba</strong>
    </td>
  </tr>
</table>

## Dataset
The dataset used in our paper is a semi-simulated dataset with real background  MWIR and LWIR images and synthesized small target motion, appearance, and intensity through a semi-simulated approach. This dataset consisted of 15,331 pairs of meticulously aligned MWIR and LWIR images, each with a resolution of 640×512 pixels and target sizes ranging from 5 to 15 pixels. It encompassed various complex environments including high-brightness clouds and sea clutter. The data statistics are as follows:
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/kyrietop11/BDSFusion/blob/main/figures/Scene.png" style="width: 45%; margin: 5px;" />
    <img src="https://github.com/kyrietop11/BDSFusion/blob/main/figures/SCR.png" style="width: 45%; margin: 5px;" />
</div>

## Qualitative Results of Infrared Small Target Detection

#### Detection results on ACM-Net
![image](https://github.com/kyrietop11/BDSFusion/blob/main/figures/Detection%20results%20on%20ACM-Net.svg)


#### Detection results on IRTransDet
![image](https://github.com/kyrietop11/BDSFusion/blob/main/figures/Detection%20results%20on%20IRTransDet.svg)


## Recommended Environment
 - [ ] torch  1.13.1
 - [ ] cudatoolkit 11.7
 - [ ] torchvision 0.14.1
 - [ ] mamba-ssm 1.0.1
 - [ ] causal-conv1d 1.0.0

## Contact
**Welcome to raise issues or email to [nwpu_lys@mail.nwpu.edu.cn](nwpu_lys@mail.nwpu.edu.cn) for any question regarding our BDSFusion.**
