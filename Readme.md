<p align="center">
  <h1 align="center">Spectroformer: Multi-Domain Query Cascaded Transformer Network For Underwater Image Enhancement</h1>
  <p align="center">
    <a href="https://mdraqibkhan.github.io">Md Raqib Khan</a>
    ·
    <a href="https://scholar.google.com/citations?user=UcUMYe8AAAAJ&hl=en&oi=sra">Priyanka Mishra</a>
    ·
    <a href="https://scholar.google.com/citations?user=WwdYdlUAAAAJ&hl=en&oi=sra">Nancy Mehta </a>
    ·
    <a href="https://scholar.google.com/citations?user=HgX8wb8AAAAJ&hl=en&oi=sra">Shruti S. Phutke</a>
    ·
    <a href="https://visionintelligence.github.io">Santosh Kumar Vipparthi</a>
    ·
    <a href="https://www.iitg.ac.in/sukumar/">Sukumar Nandi</a>
    ·
    <a href="http://www.cvlibs.net/](https://www.scss.tcd.ie/~muralas/">Subrahmanyam Murala</a>
  </p>
  <h3 align="center">WACV-2024</h3>
  <h3 align="center"><a href="https://openaccess.thecvf.com/content/WACV2024/supplemental/Khan_Spectroformer_Multi-Domain_Query_WACV_2024_supplemental.pdf">Paper</h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="Spectroformer.jpg" alt="Logo" width="100%">
  </a>
</p>



## The sample degraded images for testing are provided in "dataset/dataset_name/".
## Chekpoints are provided in "checkpoints/dataset-name/".
## After successful execution the results will be stored in "results/dataset-name" folder.

## To test on UIEB dataset. run
python test.py --dataset datasets/UIEB/ --save_path Results/UIEB 
##To test on U-45 dataset. run
python test.py --dataset dataset/U-45/ --save_path Results/U-45

## To test on SQUID dataset. run
python test.py --dataset dataset/SQUID/ --save_path Results/SQUID
## To test on UCCS dataset. run
python test.py --dataset dataset/UCCS/ --save_path Results/UCCS


## Traing
1. Structure of data for training should be like 
 uw_data/
    ├── train/
    │   ├── a/  # Input images
    │   └── b/  # Reference (ground truth) images
    └── test/
        ├── a/  # Input images
        └── b/  # Reference (ground truth) images

2. run
  pyhthon train.py



If you find this work helpful, please reference it as follows:

,,,
@inproceedings{khan2024spectroformer,
  title={Spectroformer: A Multi-Domain Query Cascaded Transformer Network for Underwater Image Enhancement},
  author={Khan, Raqib and Mishra, Priyanka and Mehta, Nancy and Phutke, Shruti S and Vipparthi, Santosh Kumar and Nandi, Sukumar and Murala, Subrahmanyam},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1454--1463},
  year={2024}}
,,,
