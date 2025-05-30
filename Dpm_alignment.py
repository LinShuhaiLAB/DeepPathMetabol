import math
from utils import *
OUTPUT_DIR = "Output"





def Dpm_align(hemask, msimask, factor=8, epochs=100):

    rate = -(factor/100)
    msimask_bc = msimask
    msimask = scale_msimask(msimask_bc, factor)

    msimask = sitk.GetImageFromArray(msimask)
    msimask = sitk.Cast(msimask, sitk.sitkFloat32)
    hemask = sitk.GetImageFromArray(hemask)
    hemask = sitk.Cast(hemask, sitk.sitkFloat32)

    loss = float('inf')  # 初始化损失为无穷大
    loss_history = []  # 用于存储每次迭代的损失值

    for i in range(epochs):
        hemask, moving_resampled, loss_current, final_transform = AfRegis(hemask, msimask)

        moving_resampled = sitk.GetArrayFromImage(moving_resampled)
        # moving_resampled = crop_border_zeros(moving_resampled)

        output_array = combine_masks(hemask, sitk.GetImageFromArray(moving_resampled))
        # 显示叠加后的图像
        plt.imshow(output_array)
        plt.axis('off')  # 关闭坐标轴显示
        plt.savefig(f"./data/output/outputarray_{loss_current}.svg", format="svg")
        # 展示图像
        plt.pause(1)  # 显示2秒
        plt.close()



        if loss_current < loss:
            loss = loss_current
            # msimask = moving_resampled
            factor += rate
            msimask = scale_msimask(msimask_bc, factor)

            msimask = sitk.GetImageFromArray(msimask)
            msimask = sitk.Cast(msimask, sitk.sitkFloat32)
            moving_resampled1 = moving_resampled
            final_transform1 = final_transform
        else:
            moving_resampled = sitk.GetImageFromArray(moving_resampled1)
            final_transform = final_transform1
            factor -= rate
            break  # 如果损失没有减少，则结束循环

    return hemask, moving_resampled, factor, final_transform



def create_anndata_from_file(file_path, moving_resampled, library_id="breast", final_factor=1):
    """
    Function to read a tab-separated values file and create an AnnData object
    with spatial coordinates and data.

    Parameters:
    - file_path: str, the path to the input file.

    Returns:
    - adata: AnnData object containing the data and spatial coordinates.
    """
    # Read the file
    final_factor = 1/final_factor
    data_df = pd.read_csv(file_path, sep='\t')  # Assuming the data is tab-separated

    # Get the first and second rows of data (excluding the index)
    first_row = data_df.iloc[:, 1]  # Assuming the first row's index is 0, skipping the first column
    second_row = data_df.iloc[:, 2]  # Same as above, assuming the second row's index is 1

    # Convert the first and second rows to spatial coordinates
    X_spatial = np.stack((first_row.to_numpy(), second_row.to_numpy()), axis=-1)

    # Create an index
    index = first_row.astype(str) + 'x' + second_row.astype(str)
    data_df.index = index

    # Drop the first four columns as they have been used for indexing
    data_df = data_df.drop(data_df.columns[0:3], axis=1)
    # data_df = data_df.applymap(lambda x: float(x / 10))
    # data_df = data_df.astype(int)
    # Create an AnnData object
    adata = ad.AnnData(X=data_df)

    # Add the DataFrame's row index as the AnnData's obs (observations) index
    adata.obs = pd.DataFrame(index=data_df.index)

    # Add the DataFrame's column index as the AnnData's var (variables) index
    adata.var = pd.DataFrame(index=data_df.columns)

    # Add spatial coordinates
    adata.obsm['spatial'] = X_spatial * final_factor

    adata.uns["spatial"] = {library_id: {}}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns['spatial'][library_id]['images']['hires'] = moving_resampled
    adata.uns['spatial'][library_id]['use_quality'] = 'hires'
    adata.uns['spatial'][library_id]['scalefactors'] = {"tissue_hires_scalef": 1}
    adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = final_factor

    return adata



def dpm_process_msi(msi_path, plot=True):

    msimask,_ = create_mask_from_txt(msi_path)
    #msimask = scale_msimask(msimask, 30)

    if plot:
        plt.imshow(msimask)
        print(msimask.shape)
        plt.title('MSIMask')
        plt.savefig("./data/output/msimask.svg", format="svg")
        plt.pause(2)  # 显示2秒
        plt.close()

    return msimask


def find_init_scale(msimask, hemask):
    """
    计算两个图像之间的缩放倍数。

    :param msimask: SimpleITK图像对象，第一个图像。
    :param hemask: NumPy数组，第二个图像。
    :return: 一个整数，表示向下取整后的最小缩放倍数。
    """
    # 获取第一个图像的尺寸
    image1_shape = msimask.shape
    # 第二个图像的尺寸已经以NumPy数组的形式给出
    image2_shape = hemask.shape

    # 计算高度和宽度的缩放倍数
    height_scale = image2_shape[0] / image1_shape[0]
    width_scale = image2_shape[1] / image1_shape[1]

    # 选择较小的缩放倍数
    scale = min(height_scale, width_scale)

    # 向下取整
    scale = math.floor(scale)
    scale = 1 / scale
    # 返回缩放倍数
    return scale



def dpm_process_image(image_path, n=3, plot=True,cut=True,cut_threshold=240,flip=False):
    # 打开图像文件
    slide = opslide.open_slide(image_path)
    # 读取图像区域
    img = np.array(slide.read_region((0, 0), n, slide.level_dimensions[n]))
    # 翻转图像
    if flip:
        img = np.fliplr(img)

    # 去除白背景
    if cut is True:
        img = remove_white_background(img)
    else:
        img = sitk.GetImageFromArray(img)
    # 将图像转换为SimpleITK数组视图
    slice_all = sitk.GetArrayViewFromImage(img)
    # 提取特定层
    slice = slice_all[:, :, 1]
    # 生成掩码
    hemask = np.zeros_like(slice)
    hemask[slice < cut_threshold] = 1

    if plot:
        # 显示原始图像
        myshow(img)
        plt.title('Original Image')
        plt.pause(2)  # 显示2秒
        plt.close()  # 显示图像并阻塞，直到关闭窗口

        # 显示掩码
        plt.imshow(hemask)
        plt.title('HEMask')
        plt.savefig("./data/output/hemask.svg", format="svg")
        plt.pause(2)  # 显示2秒
        plt.close()  # 显示图像并阻塞，直到关闭窗口

    # 返回img和hemask
    return img, hemask






































