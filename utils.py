import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy.ndimage import rotate, shift
import anndata as ad
import openslide as opslide


def myshow(img):
    nda = sitk.GetArrayViewFromImage(img)
    plt.imshow(nda)


def remove_white_background(image_path, threshold=[100, 100, 100, 200]):
    # 读取图片
    img = sitk.GetImageFromArray(image_path)
    img = sitk.Cast(img, sitk.sitkVectorUInt8)

    # 将SimpleITK图像转换为NumPy数组
    nda = sitk.GetArrayViewFromImage(img)

    # 找到非白色像素的边界
    # 假设白色背景的RGB值接近(255, 255, 255)
    non_white_mask = np.any(nda < threshold, axis=2)

    # 找到非白色区域的最小和最大坐标
    rows = np.any(non_white_mask, axis=1)
    cols = np.any(non_white_mask, axis=0)
    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    # 裁剪图片
    cropped_img = nda[min_row:max_row + 1, min_col:max_col + 1]

    # 将裁剪后的NumPy数组转换回SimpleITK图像
    cropped_img_sitk = sitk.GetImageFromArray(cropped_img)
    # cropped_img_sitk.CopyInformation(img)  # 复制原始图像的信息

    return cropped_img_sitk


def TileTissueMask(heMask, tile_size, threshold=0.95):
    """
    remove H&E image tiles that belong to the background
    """

    positionList = []
    mask = np.zeros((int(heMask.shape[0] / tile_size), int(heMask.shape[1] / tile_size)))
    n_rows = int(heMask.shape[0] / tile_size)
    n_cols = int(heMask.shape[1] / tile_size)
    for i in range(n_rows):
        for j in range(n_cols):
            # print(i,j)
            if ((np.sum(heMask[i * int(tile_size):(i + 1) * int(tile_size),
                        j * int(tile_size):(j + 1) * int(tile_size)],
                        axis=(0, 1)) / (tile_size * tile_size)) > threshold):
                mask[i, j] = 1
                positionList.append((i, j))
    return mask, positionList


def crop_border_zeros(image_array):
    """
    裁剪掉NumPy数组周围值为0的边界。

    参数:
    image_array (np.ndarray): 输入的NumPy数组。

    返回:
    np.ndarray: 裁剪后的数组。
    """
    # 找到非零值的最小和最大索引
    rows = np.any(image_array, axis=1)
    cols = np.any(image_array, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    # 使用这些索引来裁剪数组
    cropped_array = image_array[top:bottom + 1, left:right + 1]

    return cropped_array


def create_mask_from_txt(file_path):
    """
    读取文本文件，创建掩码，并返回掩码和位置列表。

    参数:
    file_path -- 文本文件的路径。

    返回:
    mask -- 根据 'x' 和 'y' 列创建的掩码。
    positionList -- 掩码中值为1的位置列表。
    """
    # 读取文本文件为 DataFrame
    data = pd.read_table(file_path,sep='\t')

    # 获取 'x' 和 'y' 列的最大值，确定掩码的大小
    mask_size_x = int(data.loc[:, 'y'].max() + 1)
    mask_size_y = int(data.loc[:, 'x'].max() + 1)

    # 创建初始值为0的掩码
    mask = np.zeros((mask_size_x, mask_size_y), dtype=int)

    # 将 'x' 和 'y' 列对应的位置在掩码中设为1
    for index, row in data.iterrows():
        mask[int(row['y']), int(row['x'])] = 1

    # 创建位置列表，包含掩码中值为1的所有位置
    positionList = list(zip(np.nonzero(mask)[0], np.nonzero(mask)[1]))

    return mask, positionList


def display_2D_images(fixed_image_sitk, moving_image_sitk, titles=('Fixed Image', 'Moving Image')):
    """
    显示两个2D图像（固定图像和移动图像）。

    参数:
    fixed_image_sitk -- 固定图像的SimpleITK对象。
    moving_image_sitk -- 移动图像的SimpleITK对象。
    titles -- 图像的标题，包括固定图像和移动图像的标题，默认为 ('Fixed Image', 'Moving Image')。
    """
    # 将SimpleITK图像转换为NumPy数组
    fixed_array = sitk.GetArrayFromImage(fixed_image_sitk)
    moving_array = sitk.GetArrayFromImage(moving_image_sitk)

    # 创建图形窗口
    plt.figure(figsize=(10, 5))

    # 显示固定图像
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_array, cmap='gray')
    plt.title(titles[0])
    plt.axis('off')

    # 显示移动图像
    plt.subplot(1, 2, 2)
    plt.imshow(moving_array, cmap='gray')
    plt.title(titles[1])
    plt.axis('off')

    # 显示图形
    plt.show()


def scale_msimask(msimask, factor):
    # 检查factor是否大于0，因为0或负数的缩放因子没有意义
    if factor <= 0:
        raise ValueError("Factor must be greater than 0.")

    # 沿着每个维度缩放msimask的元素
    scaled_msimask = zoom(msimask, zoom=[factor, factor], order=3)  # order=1 表示最近邻插值

    return scaled_msimask


def AfRegis(hemask, msimask):
    initial_transform = sitk.CenteredTransformInitializer(
        hemask,
        msimask,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY, )

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=30)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(1)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1e-3,
        numberOfIterations=200,
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=5,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    # registration_method.AddCommand(
    #     sitk.sitkMultiResolutionIterationEvent, update_multires_iterations
    # )
    # registration_method.AddCommand(
    #     sitk.sitkIterationEvent, lambda: plot_values(registration_method)
    # )

    final_transform = registration_method.Execute(
        hemask, msimask
    )

    moving_resampled = sitk.Resample(
        msimask,
        hemask,
        final_transform,
        sitk.sitkLinear,
        0.0,
        hemask.GetPixelID(),
    )

    loss = registration_method.GetMetricValue()
    print('Current Loss:',loss)
    return hemask, moving_resampled, loss, final_transform


def combine_masks(hemask_result, moving_resampled_result):
    """
    将两个SimpleITK图像（mask）叠加，并根据特定规则为不同区域分配颜色。

    参数:
    - hemask_result: SimpleITK.Image对象，第一个mask。
    - moving_resampled_result: SimpleITK.Image对象，第二个mask。
    - save_path: 保存文件的路径。如果为None，则不保存文件。否则保存为指定路径。

    返回:
    - output_array: NumPy数组，表示叠加后的图像。
    """
    # 将SimpleITK图像转换为NumPy数组
    hemask_array = sitk.GetArrayFromImage(hemask_result)
    moving_array = sitk.GetArrayFromImage(moving_resampled_result)

    # 创建一个与输入图像相同大小的空白图像
    output_array = np.zeros(hemask_array.shape + (3,), dtype=np.uint8)  # +(3,)是为了创建一个三通道的RGB图像

    # 根据规则为不同区域分配颜色
    output_array[(hemask_array == 1) & (moving_array == 0)] = [46, 99, 161]
    output_array[(hemask_array == 0) & (moving_array == 1)] = [135, 56, 62]
    output_array[(hemask_array == 1) & (moving_array == 1)] = [255, 255, 255]


    return output_array



def resample_low(slice_all, final_factor, final_transform, msimask):
    # 翻转图像并提取RGB通道
    slice_all = sitk.GetArrayViewFromImage(slice_all)
    slice_allf = slice_all
    channels = [slice_allf[:, :, i] for i in range(4)]

    # 缩放和重采样每个通道
    def process_channel(channel, index):
        scaled_channel = scale_msimask(channel, final_factor)
        channel_image = sitk.GetImageFromArray(scaled_channel)
        channel_image = sitk.Cast(channel_image, sitk.sitkFloat64)
        resampled_channel = sitk.Resample(
            channel_image, msimask, final_transform, sitk.sitkLinear, 255.0, channel_image.GetPixelID()
        )
        return sitk.GetArrayFromImage(resampled_channel)

    # 处理所有通道并合并为一个RGB图像
    moving_resampled = np.stack([process_channel(channel, i) for i, channel in enumerate(channels)], axis=-1)
    moving_resampled = moving_resampled / 255.0  # 归一化到[0, 1]范围
    return moving_resampled


def resample_high(slice_all, msimask, dpm_trans, size=1.0):
    """
    Applies rotation and translation transformations to the slice_all data.

    Parameters:
    slice_all (np.ndarray): Input array with shape (3571, 4940, 4)
    msimask (SimpleITK.Image): Mask image used for cropping
    dpm_trans (tuple): Tuple containing rotation radians and translation parameters (rotation_radians, translation_x, translation_y)
    size (float): Scaling factor, default is 1.0

    Returns:
    np.ndarray: Transformed array
    """
    print("Starting the resample_high function.")  # Monitoring point 1

    size = 1 / size
    slice_all = sitk.GetArrayViewFromImage(slice_all)

    # Extract parameters
    rotation_angle = dpm_trans[0]  # Rotation in radians
    translation_x = -dpm_trans[1] * size  # Translation in x-direction
    translation_y = -dpm_trans[2] * size  # Translation in y-direction

    print(f"Parameters extracted, rotation angle: {rotation_angle}, translation parameters: ({translation_x}, {translation_y})")  # Monitoring point 2

    # Convert radians to degrees
    rotation_angle_degrees = np.degrees(rotation_angle)

    # Rotate the image first
    print("Starting image rotation...")  # Monitoring point 3
    rotated_slice_all = rotate(slice_all, angle=rotation_angle_degrees, axes=(0, 1), reshape=False)
    print("Image rotation completed.")  # Monitoring point 4

    # Then perform translation
    print("Starting image translation...")  # Monitoring point 5
    translated_slice_all = shift(rotated_slice_all, shift=(translation_y, translation_x, 0))
    print("Image translation completed.")  # Monitoring point 6

    # Convert msimask to a NumPy array
    msimask_array = sitk.GetArrayFromImage(msimask)

    # Scale the mask according to the size
    msimask_array = scale_msimask(msimask_array, size)
    print(f"Mask scaling completed, scaling factor: {size}")  # Monitoring point 7

    # Crop translated_slice_all to the same shape as msimask_array
    shape_msimask = msimask_array.shape
    print(f"Mask shape: {shape_msimask}")  # Monitoring point 8

    # Crop from the top-left corner to match the size of msimask
    cropped_slice_all = translated_slice_all[:shape_msimask[0], :shape_msimask[1], :]
    cropped_slice_all = cropped_slice_all / 255.0
    print("Image cropping completed.")  # Monitoring point 9

    return cropped_slice_all


def get_rotation(composite_tx):
    composite_tx = sitk.CompositeTransform(composite_tx)
    composite_tx.FlattenTransform()
    tx_dim = composite_tx.GetDimension()
    A = np.eye(tx_dim)
    t = np.zeros(tx_dim)
    c = np.zeros(tx_dim)
    for i in range(composite_tx.GetNumberOfTransforms() - 1, -1, -1):
        curr_tx = composite_tx.GetNthTransform(i)
        if isinstance(curr_tx, sitk.TranslationTransform):
            A_curr = np.eye(tx_dim)
            t_curr = np.asarray(curr_tx.GetOffset())
            c_curr = np.zeros(tx_dim)
        else:
            A_curr = np.asarray(curr_tx.GetMatrix()).reshape(tx_dim, tx_dim)  # type: ignore [attr-defined]
            get_transl = getattr(curr_tx, "GetTranslation", None)
            if get_transl is not None:
                t_curr = np.asarray(get_transl())
            else:
                t_curr = np.zeros(tx_dim)
            c_curr = np.asarray(curr_tx.GetCenter())  # type: ignore [attr-defined]
        A = np.dot(A_curr, A)
        t = np.dot(A_curr, t + c - c_curr) + t_curr + c_curr - c
    affine = sitk.AffineTransform(A.flatten().tolist(), t.tolist())
    affine = np.array(affine.GetMatrix()).reshape(2, 2)
    return affine



