import scanpy as sc
import umap
from Dpm_hepredictor import *
from Alignment_tools import *
from Dpm_alignment import *
from utils import *


def Dpm_ot(adataA, img_path, n, spot_size=10, poslist=False, cut=True, threshold=0.95, cut_threshold=240,coor_key="predict",use_rep='spatial'):
    print("Starting image processing...")
    img, hemask = dpm_process_image(img_path, n, cut=cut, cut_threshold=cut_threshold)
    print("Image processing completed. Extracting slice data...")

    slice_all = sitk.GetArrayViewFromImage(img).copy()
    slice_all = slice_all[:, :, :3]

    if poslist is False:
        print("Generating position list...")
        hemask, poslist = TileTissueMask(hemask, spot_size, threshold=threshold)
    else:
        print("Using provided position list...")
        poslist = poslist

    print("Initializing AnnData object...")
    X = np.ones((len(poslist), adataA.X.shape[1]))
    adataB = ad.AnnData(X=X)
    poslist = np.array(poslist).copy()
    poslist_swapped = poslist[:, [1, 0]]
    adataB.obsm['spatial'] = (poslist_swapped + 1) * spot_size
    adataB.uns["spatial"] = {'HE': {}}
    adataB.uns["spatial"]['HE']["images"] = {}
    adataB.uns['spatial']['HE']['images']['hires'] = slice_all
    adataB.uns['spatial']['HE']['use_quality'] = 'hires'
    adataB.uns['spatial']['HE']['scalefactors'] = {"tissue_hires_scalef": 1}
    adataB.uns["spatial"]['HE']["scalefactors"]["spot_diameter_fullres"] = spot_size
    adataAA = adataA.copy()
    adataBB = adataB.copy()
    print("Extracting and processing patches...")
    adataA.uns['spatial']['breast']['images']['hires'][
        adataA.uns['spatial']['breast']['images']['hires'] == 0
        ] = 1
    adataA.uns['spatial']['breast']['images']['hires'] = (adataA.uns['spatial']['breast']['images']['hires'][...,
                                                          :3]) * 255
    patches = extract_patches_from_adata(adataA)
    processed_patches = process_patches_with_model(patches)
    adataA.obsm['predict'] = np.array(processed_patches)

    adataB.uns['spatial']['HE']['images']['hires'][
        adataB.uns['spatial']['HE']['images']['hires'] == 0
        ] = 1
    adataB.uns['spatial']['HE']['images']['hires'] = (adataB.uns['spatial']['HE']['images']['hires'][...,
                                                          :3]) * 255
    patches = extract_patches_from_adata(adataB)
    processed_patches = process_patches_with_model(patches)
    adataB.obsm['predict'] = np.array(processed_patches)

    # print("Performing PCA and UMAP analysis...")
    adataP = ad.AnnData(X=adataA.obsm['predict'])
    adataP.obsm['predict'] = adataA.obsm['predict']
    adataP.uns['spatial'] = adataA.uns['spatial']
    adataP.obsm['spatial'] = adataA.obsm['spatial']

    col1 = adataP.obsm['spatial'][:, 1]
    sorted_col1 = np.sort(col1)  # 对第1列进行排序
    second_min_val = sorted_col1[1]  # 第二小的值

    # 对第1列进行归一化
    normalized_col1 = (adataP.obsm['spatial'][:, 1] - adataP.obsm['spatial'][:, 1].min()) / (second_min_val-adataP.obsm['spatial'][:, 1].min())

    col1 = adataP.obsm['spatial'][:, 0]
    sorted_col1 = np.sort(col1)  # 对第1列进行排序
    second_min_val = sorted_col1[0]  # 第二小的值
    # 对第0列进行归一化
    normalized_col0 = (adataP.obsm['spatial'][:, 0] - adataP.obsm['spatial'][:, 0].min()) / (second_min_val-adataP.obsm['spatial'][:, 0].min())

    # 将归一化后的结果组合成新的数组
    adataP.obsm['X_spatial'] = np.column_stack((normalized_col0, normalized_col1))
    # sc.pp.pca(adataP)
    # sc.pp.neighbors(adataP)
    # sc.tl.umap(adataP)

    adataQ = ad.AnnData(X=adataB.obsm['predict'])
    adataQ.obsm['predict'] = adataB.obsm['predict']
    adataQ.uns['spatial'] = adataB.uns['spatial']
    adataQ.obsm['spatial'] = adataB.obsm['spatial']

    col1 = adataQ.obsm['spatial'][:, 1]
    sorted_col1 = np.sort(col1)  # 对第1列进行排序
    second_min_val = sorted_col1[1]  # 第二小的值

    # 对第1列进行归一化
    normalized_col1 = (adataQ.obsm['spatial'][:, 1] - adataQ.obsm['spatial'][:, 1].min()) / (
                second_min_val - adataQ.obsm['spatial'][:, 1].min())

    col1 = adataQ.obsm['spatial'][:, 0]
    sorted_col1 = np.sort(col1)  # 对第1列进行排序
    second_min_val = sorted_col1[0]  # 第二小的值
    # 对第0列进行归一化
    normalized_col0 = (adataQ.obsm['spatial'][:, 0] - adataQ.obsm['spatial'][:, 0].min()) / (
                second_min_val - adataQ.obsm['spatial'][:, 0].min())

    # 将归一化后的结果组合成新的数组
    adataQ.obsm['X_spatial'] = np.column_stack((normalized_col0, normalized_col1))


    print("Performing alignment...")
    pi = pairwise_align_paste(
        adataP,
        adataQ,
        alpha=0.5,
        dissimilarity='euclidean',
        use_rep=use_rep,
        G_init=None,
        a_distribution=None,
        b_distribution=None,
        norm=False,
        numItermax=200,
        backend=ot.backend.NumpyBackend(),
        use_gpu=False,
        return_obj=False,
        verbose=False,
        gpu_verbose=False,
        coor_key=coor_key)

    print("Updating adataB data...")
    adataB.X = (adataA.X.T @ pi).T
    adataB.var.index = adataA.var.index
    adataA.uns['spatial'] = adataAA.uns['spatial']
    adataB.uns['spatial'] = adataBB.uns['spatial']
    print("Function execution completed. Returning results...")
    return adataA, adataB





def Dpm_ot_val(adataA, adataB,spot_diameter=None,alpha=0.1):


    print("Extracting and processing patches...")
    patches = extract_patches_from_adata(adataA,spot_diameter=spot_diameter)
    processed_patches = process_patches_with_model(patches)
    adataA.obsm['predict'] = np.array(processed_patches)

    patches = extract_patches_from_adata(adataB,spot_diameter=spot_diameter)
    processed_patches = process_patches_with_model(patches)
    adataB.obsm['predict'] = np.array(processed_patches)

    # print("Performing PCA and UMAP analysis...")
    adataP = ad.AnnData(X=adataA.obsm['predict'])
    adataP.obsm['predict'] = adataA.obsm['predict']
    adataP.uns['spatial'] = adataA.uns['spatial']
    adataP.obsm['spatial'] = adataA.obsm['spatial']
    # sc.pp.pca(adataP)
    # sc.pp.neighbors(adataP)
    # sc.tl.umap(adataP)

    adataQ = ad.AnnData(X=adataB.obsm['predict'])
    adataQ.obsm['predict'] = adataB.obsm['predict']
    adataQ.uns['spatial'] = adataB.uns['spatial']
    adataQ.obsm['spatial'] = adataB.obsm['spatial']
    # sc.pp.pca(adataQ)
    # sc.pp.neighbors(adataQ)
    # sc.tl.umap(adataQ)

    print("Combining feature matrices and performing UMAP...")
    combined_X = np.vstack((adataP.X, adataQ.X))
    reducer = umap.UMAP(n_neighbors=100, n_components=2, metric='euclidean', random_state=42)
    combined_umap = reducer.fit_transform(combined_X)

    N_p = adataP.X.shape[0]
    adataP_umap = combined_umap[:N_p]
    adataQ_umap = combined_umap[N_p:]

    adataP.obsm['X_he'] = adataP_umap
    adataQ.obsm['X_he'] = adataQ_umap

    print("Performing alignment...")
    pi = pairwise_align_paste(
        adataP,
        adataQ,
        alpha=alpha,
        dissimilarity='euclidean',
        use_rep='spatial',
        G_init=None,
        a_distribution=None,
        b_distribution=None,
        norm=False,
        numItermax=200,
        backend=ot.backend.NumpyBackend(),
        use_gpu=False,
        return_obj=False,
        verbose=False,
        gpu_verbose=False,
        coor_key="X_he")

    print("Updating adataB data...")
    adataB.X = (adataA.X.T @ pi).T
    adataB.var.index = adataA.var.index

    print("Function execution completed. Returning results...")
    return adataA, adataB
