# *<center>BDSFusion: Bidirectional-Driven Saliency Fusion Network for Infrared Dual-Band Images</center>*
This is the official code for **”BDSFusion: Bidirectional-Driven Saliency Fusion Network for Infrared Dual-Band Images”** (Information Fusion, 2025).

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

## Overall Architecture

![image](https://github.com/kyrietop11/BDSFusion/blob/main/figures/Overall%20Architecture.png)

## Dataset
The dataset used in our paper is a semi-simulated dataset with real background  MWIR and LWIR images and synthesized small target motion, appearance, and intensity through a semi-simulated approach. This dataset consisted of 15,331 pairs of meticulously aligned MWIR and LWIR images, each with a resolution of 640×512 pixels and target sizes ranging from 5 to 15 pixels. It encompassed various complex environments including high-brightness clouds and sea clutter. The data statistics are as follows:
<div style="display: flex; justify-content: center;">
    <img src="https://github.com/kyrietop11/BDSFusion/blob/main/figures/Scene.png" style="width: 45%; margin: 5px;" />
    <img src="https://github.com/kyrietop11/BDSFusion/blob/main/figures/SCR.png" style="width: 45%; margin: 5px;" />
</div>

#### Comparison of target and background clutter characteristics in MWIR and LWIR images. 
![image](https://github.com/kyrietop11/BDSFusion/blob/main/figures/Fig1.png)


## Quantitative Results of Infrared Small Target Detection

![image](https://github.com/kyrietop11/BDSFusion/blob/main/figures/Quantitative%20results.png)

## Qualitative Results of Infrared Small Target Detection

#### Detection results on ACM-Net
![image](https://github.com/kyrietop11/BDSFusion/blob/main/figures/Detection%20results%20on%20ACM-Net.svg)


#### Detection results on IRTransDet
![image](https://github.com/kyrietop11/BDSFusion/blob/main/figures/Detection%20results%20on%20IRTransDet.svg)


## Extension to infrared-visible image fusion for small target detection
![image](https://github.com/kyrietop11/BDSFusion/blob/main/figures/Fig_vis_inf_qualitative.png)

## Recommended Environment
 - [ ] Python 3.10+
 - [ ] torch 2.3.0+
 - [ ] torchvision 0.18.0+
 - [ ] mamba-ssm 1.1.3+
 - [ ] causal-conv1d 1.2.0+
 - [ ] timm 0.9.0+
 - [ ] einops
 - [ ] pyyaml
 - [ ] pillow

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2025bdsfusion,
  title={BDSFusion: A bidirectionally driven saliency fusion network for enhancing small target detection in infrared dual-band images},
  author={Li, Shaoyi and Li, Yusong and Niu, Saisai and Yang, Junyan and Yang, Xi and Yue, Xiaokui},
  journal={Information Fusion},
  pages={103600},
  year={2025},
  publisher={Elsevier}
}
```