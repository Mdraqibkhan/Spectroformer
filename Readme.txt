# The sample degraded images for testing are provided in "dataset/dataset_name/".
# Chekpoints are provided in "checkpoints/dataset-name/".
# After successful execution the results will be stored in "results/dataset-name" folder.

# To test on UIEB dataset. run
python test.py --dataset datasets/UIEB/ --save_path Results/UIEB 
# To test on U-45 dataset. run
python test.py --dataset dataset/U-45/ --save_path Results/U-45

# To test on SQUID dataset. run
python test.py --dataset dataset/SQUID/ --save_path Results/SQUID
# To test on UCCS dataset. run
python test.py --dataset dataset/UCCS/ --save_path Results/UCCS


Traing
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

@inproceedings{khan2024spectroformer,
  title={Spectroformer: A Multi-Domain Query Cascaded Transformer Network for Underwater Image Enhancement},
  author={Khan, Raqib and Mishra, Priyanka and Mehta, Nancy and Phutke, Shruti S and Vipparthi, Santosh Kumar and Nandi, Sukumar and Murala, Subrahmanyam},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1454--1463},
  year={2024}}
