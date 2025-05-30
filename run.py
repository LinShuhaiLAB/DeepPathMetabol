from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from Dpm_OT import *
import argparse
import scanpy as sc

def DPM_Alignment(args):
    n = args.n
    img, hemask = dpm_process_image(args.img_path, n, cut_threshold=args.cut_threshold)
    msimask = dpm_process_msi(args.msi_path)
    scale_factors = find_init_scale(msimask, hemask)
    msimask, moving_resampled_result, final_factor, final_transform = Dpm_align(msimask, hemask, factor=scale_factors, epochs=100)
    dpm_trans = final_transform.GetParameters()
    img = resample_high(img, msimask, dpm_trans, size=final_factor)
    anndata = create_anndata_from_file(args.msi_path, img, final_factor=final_factor)
    anndata.var_names_make_unique()
    anndata.layers["counts"] = anndata.X.copy()
    anndata.write_h5ad(args.output_prefix)

def DPM_Transport(args):
    # Read spatial transcriptomics data
    adata = sc.read_h5ad(args.h5ad_path)
    adata.X = adata.layers['counts']

    # Link image and perform DPM-OT analysis
    adata, adataP = Dpm_ot(adata, args.image_path, 0, spot_size=args.spot_size,cut_threshold=args.cut_threshold)

    # Save the processed data
    adataP.write_h5ad(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DPM Alignment and Transport Analysis.')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Parser for DPM_Alignment mode
    parser_align = subparsers.add_parser('Align', help='Run DPM_Alignment')
    parser_align.add_argument('--img_path', type=str, required=True, help='Path to the image file')
    parser_align.add_argument('--msi_path', type=str, required=True, help='Path to the MSI file')
    parser_align.add_argument('--output_prefix', type=str, required=True, help='Prefix for output files')
    parser_align.add_argument('--cut_threshold', type=int, default=220, help='Threshold for image processing')
    parser_align.add_argument('--n', type=int, default=0, help='Parameter for image processing')

    # Parser for DPM_Transport mode
    parser_transport = subparsers.add_parser('Predict', help='Run DPM_Transport')
    parser_transport.add_argument('--h5ad_path', type=str, required=True, help='Path to the h5ad file containing spatial transcriptomics data.')
    parser_transport.add_argument('--image_path', type=str, required=True, help='Path to the tissue image file.')
    parser_transport.add_argument('--spot_size', type=int, default=100, help='Size of the spots in micrometers.')
    parser_transport.add_argument('--cut_threshold', type=int, default=220, help='Threshold for image processing')
    parser_transport.add_argument('--output_path', type=str, required=True, help='Path to save the processed data.')

    # Parser for All mode
    parser_all = subparsers.add_parser('All', help='Run DPM_Alignment followed by DPM_Transport')
    parser_all.add_argument('--img_path', type=str, required=True, help='Path to the source image file')
    parser_all.add_argument('--image_path', type=str, required=True, help='Path to the target image file')
    parser_all.add_argument('--msi_path', type=str, required=True, help='Path to the MSI file')
    parser_all.add_argument('--output_prefix', type=str, required=True, help='Prefix for output files from alignment')
    parser_all.add_argument('--cut_threshold', type=int, default=220, help='Threshold for image processing')
    parser_all.add_argument('--n', type=int, default=0, help='Parameter for image processing')
    parser_all.add_argument('--spot_size', type=int, default=100, help='Size of the spots in micrometers.')
    parser_all.add_argument('--pred_output_path', type=str, required=True, help='Path to save the final processed data from transport.')

    args = parser.parse_args()

    if args.mode == 'Align':
        DPM_Alignment(args)
    elif args.mode == 'Predict':
        DPM_Transport(args)
    elif args.mode == 'All':
        # Run DPM_Alignment
        DPM_Alignment(args)
        # Run DPM_Transport using the output from alignment
        alignment_output_path = args.output_prefix
        args.h5ad_path = alignment_output_path

        args.output_path = args.pred_output_path
        DPM_Transport(args)

# python run.py Align --img_path './data/Kidney/img/03.tif' --msi_path './data/Kidney/count/03.txt' --output_prefix './data/processed_data/k03_80um.h5ad' --cut_threshold 210 --n 0
# python run.py Predict --h5ad_path './data/processed_data/k03_80um.h5ad' --image_path './data/Kidney/img/12.tif' --output_path './data/processed_data/k12_80um_Dpm.h5ad' --spot_size 100
# python run.py All --img_path './data/Kidney/img/03.tif' --msi_path './data/Kidney/count/03.txt' --output_prefix './data/processed_data/k03_80um.h5ad' --image_path './data/Kidney/img/12.tif' --pred_output_path './data/processed_data/k12_80um_Dpm.h5ad' --cut_threshold 210 --n 0 --spot_size 70
