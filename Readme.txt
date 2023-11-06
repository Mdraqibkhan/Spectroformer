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

