# DeepPathMetabol
DeepPathMetabol (DPM) is a deep learning based method using H&E stain info and optimal transport theory to predict metabolite distributions and reconstruct high-resolution MSI data from small-sample inputs for cost-efficient, high-precision spatial metabolomics research.
Developer is Bo Yao from School of Life Sciences, Xiamen University, Xiamen, Fujian 361102, China.

# Overview of DeepPathMetabol

<div align=center>
<img src="https://github.com/Simon-BoY/DeepPathMetabol/blob/master/img/DPM.png" width="800" height="550" /><br/>
</div>

__Overview of DeepPathMetabol architecture__. (a) Tissue sections are subjected to MSI and H&E staining experiments. (b) The DPM alignment model registers the H&E images with the MSI images. (c) The DPM transport model uses a CNN backbone to project the information of H&E images into a latent space, along with their corresponding spatial location information. Then utilizes optimal transport module to find the best transport solution between the two and use this solution to predict the MSI of the target tissue section.

# Requirement

    Python == 3.10.15

    Anndata == 0.11.3
        
    Leidenalg == 0.10.2
    
    Matplotlib == 3.10.0
    
    POT == 0.9.5
    
    Scanpy == 1.10.4
    
    SimpleITK == 2.4.1

    Numpy == 1.26.4
    
    Torch == 2.1.1
    
    Torchvision == 0.16.1  

    Openslide-bin == 4.0.0.8

    Openslide-python == 1.4.2

# Quickly start

## Input
To utilize the DeepPathMetabol model, you need to prepare the following inputs based on the mode you choose:

(Step 1) For Dpm_Alignment (H&E-MSI registration): (a) img_path: Path to the H&E-stained image file of the tissue section. (b) msi_path: Path to the MSI data file corresponding to the tissue section. (c) cut_threshold: Threshold value for image processing (default: 220).

(Step 2) For Dpm_Transport (MSI prediction for adjacent slices): (a) h5ad_path: Path to the h5ad file containing the spatial transcriptomics data obtained from Dpm_Alignment. (b) image_path: Path to the H&E-stained image file of the adjacent tissue slice for which MSI data is to be predicted.

(Step 1&2) For running both Dpm_Alignment and Dpm_Transport sequentially (All mode): (a) img_path: Path to the source H&E-stained image file used for alignment. (b) image_path: Path to the target H&E-stained image file for which MSI data is to be predicted. (c) msi_path: Path to the MSI data file corresponding to the source tissue section.

## Run DeepPathMetabol model
cd to the DeepPathMetabol fold

If you want to perform Dpm_Alignment for H&E image and MSI image registration, taking mouse kidney section as an example, run：

    python run.py Align --img_path ./data/Kidney/img/03.tif --msi_path ./data/Kidney/count/03.txt --output_prefix ./data/processed_data/k03_80um.h5ad --cut_threshold 210 --n 0
  
If you want to perform Dpm_transport for prediction of MSI data for adjacent slices, taking mouse kidney section as an example, run:

    python run.py Predict --h5ad_path ./data/processed_data/k03_80um.h5ad --image_path ./data/Kidney/img/12.tif --output_path ./data/processed_data/k12_80um_Dpm.h5ad --spot_size 70
  
If you want to perform both Dpm_Alignment (H&E-MSI registration) and Dpm_transport (MSI prediction for adjacent slices) sequentially in one go, taking mouse kidney section as an example, run

    python run.py All --img_path ./data/Kidney/img/03.tif --msi_path ./data/Kidney/count/03.txt --output_prefix ./data/processed_data/k03_80um.h5ad --image_path ./data/Kidney/img/12.tif --pred_output_path ./data/processed_data/k12_80um_Dpm.h5ad --cut_threshold 210 --n 0 --spot_size 70

## Contact
Please contact me if you have any help: doggyfox@163.com
