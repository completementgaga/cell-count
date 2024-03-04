# Toolkit for cell counting
def display(img,v_max=2**12):
    """Display the image with pyplot

    Args:
        img (np.ndarray): 2d image to display
        v_max (float| int, optional): Value by which the array is divided
            before conversion to float. Defaults to 2**12.
    """
    fig, ax = plt.subplots(figsize=(20,20))
    ax.imshow(img.astype('float64')/v_max,cmap='gray')


def background_image(img:np.ndarray,cell_radius)->np.ndarray:
    """Return background image.
        Perform the rolling ball algorithm to determine the background levels.
        substracting these levels to img will homogenize the exposure.

    Args:
        img (np.ndarray): image of cells 
        cell_radius (_type_): cell/nuclei radius

    Returns:
        np.ndarray: background image with the same number of channels as img.
    """
    # the background is computed from a smoothened version of img
    # to avoid influence of pepper noise 
    footprint=np.ones((3,3))
    ball_kernel=skimage.restoration.ball_kernel(2*cell_radius, 2)
    if len(img.shape)==3:
        footprint=np.ones((3,3,1))
        ball_kernel=ball_kernel.reshape(ball_kernel.shape+(1,))
    smooth_img=skimage.filters.rank.median(img,footprint=footprint)

    background=skimage.restoration.rolling_ball(smooth_img,kernel=ball_kernel)
    background=np.minimum(background,img)
    return background

def tissue_mask(img,threshold=40,cell_radius=7):
    mask=skimage.morphology.binary_opening(img>40,footprint=np.ones((cell_radius,cell_radius)))
    mask=skimage.morphology.binary_dilation(mask,footprint=np.ones((cell_radius*3,cell_radius*3)))
    return mask


def display_histograms(img, start_val=0, end_val=None, mask=None):
    """Display histograms of intensity for all channels in img.
        Also display the corresponding distribution functions.

    Args:
        img (np.ndarray): greyscale or multichannel 2d image given as integer array
        start_val (int, optional): lower bound of interval over which the
            histograms are displayed.Defaults to 0.
        end_val (_type_, optional): upper bound of interval over which the
            histograms are displayed.Defaults to 0.. Defaults to max value of img.
        mask (np.ndarray, optional): boolean 2d mask, if provided, only True pixels
        are considered in histogram computation. Otherwise, all pixels are used.
    """
    if mask is None:
        mask = np.ones(img.shape[:2], dtype="bool")
    if len(img.shape) == 2:
        view = img.reshape((img.shape + (1,)))
    elif len(img.shape) == 3:
        view = img
    else:
        raise ValueError("The passed img is not a 2d image")

    altered_image = view.copy()
    # we first set all pixels to zero outside the mask
    for i in range(view.shape[2]):
        altered_image[:, :, i] = mask * homogenized_view[:, :, i]

    if end_val is None:
        end_val = np.amax(altered_image) + 1
    histograms = [
        skimage.exposure.histogram(
            altered_image[:, :, i], source_range="dtype"
        )[0].reshape(-1, 1)
        for i in range(view.shape[2])
    ]
    for i in range(view.shape[2]):
        histograms[i][0] -= np.count_nonzero(1 - mask)

    cum_sums = [np.cumsum(histogram) for histogram in histograms]
    distributions = [
        (cum_sum / cum_sum[-1]).reshape(-1, 1) for cum_sum in cum_sums
    ]
    df = pandas.DataFrame(
        np.hstack(
            [np.arange(start_val, end_val).reshape(-1, 1)]
            + [histogram[start_val:end_val] for histogram in histograms]
            + [distribution[start_val:end_val] for distribution in distributions]
        ),
        columns=["intensity"]
        + ["counts for channel %i" % i for i in range(view.shape[2])]
        + ["distributions for channel %i" % i for i in range(view.shape[2])],
    )
    df.set_index(["intensity"], inplace=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    df.plot(
        y=["counts for channel %i" % i for i in range(view.shape[2])],
        ax=ax[0],
    )
    df.plot(
        y=["distributions for channel %i" % i for i in range(view.shape[2])],
        ax=ax[1],
    )

def check_threshold(img,channel,threshold,val_bound=2**12,color='magenta'):
    """Display thresholded img[:,:channel] overlayed over img[:,:,channel]
         and img[:,:,channel].

    Args:
        img (np.ndarray): 2d image
        channel (int): considered channel
        threshold (int|float): used threshold
        val_bound (_type_, optional): Value by which the array is divided
            before conversion to float. Defaults to 2**12.
        color (str, optional):color of binary mask. Defaults to 'magenta'.
    """
    view=img[:,:,channel]
    mask=view>threshold
    fig, ax = plt.subplots(1,2,figsize=(40, 80))
    ax[0].imshow(skimage.color.label2rgb(mask,image=1-view/val_bound,bg_label=False,bg_color=(.9,.9,.9),alpha=0.2,colors=[color]), cmap="gray")
    ax[1].imshow(view/val_bound, cmap="gray")

def watershed(
    bin_image: np.ndarray, cell_radius=7, ignored_radius=1
) -> np.ndarray:
    """Watershed to separate cells in bin_imaga

    Args:
        bin_image (np.ndarray): binary image to treat
        cell_radius (optional): Estimated cell radius in pixels. Defaults to 7.
        ignored_radius (optional): max radius of clusters to be ignored.

    Returns:
        np.ndarray: transformed image
    """
    # the first labels allow to avoid influence of min distance between different
    # connected components
    first_labels = skimage.measure.label(bin_image)
    distance = scipy.ndimage.distance_transform_edt(bin_image)

    local_max_coords = skimage.feature.peak_local_max(
        distance,
        labels=first_labels,
        min_distance=(cell_radius)//2,
        threshold_abs=ignored_radius + 1,
    )
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.transpose())] = True
    markers = skimage.measure.label(local_max_mask)

    return skimage.segmentation.watershed(-distance, markers, mask=bin_image)

def check_segmentation(img,channel,labels,val_bound=2**12):
    """Display labels overlayed over img[:,:,channel] and img[:,:,channel].

    Args:
        img (np.ndarray): 2d image
        channel (int): considered channel
        threshold (int|float): used threshold
        val_bound (_type_, optional): Value by which the array is divided
            before conversion to float. Defaults to 2**12.
    """
    view=img[:,:,channel]
    fig, ax = plt.subplots(1,2,figsize=(40, 80))
    ax[0].imshow(skimage.color.label2rgb(labels,image=1-view/val_bound,bg_label=False,bg_color=(.9,.9,.9),alpha=0.2), cmap="gray")
    ax[1].imshow(view/val_bound, cmap="gray")




@njit(parallel=True)
def _destroy_labels_flat(labeled_img: np.ndarray, unique_labels: np.ndarray):
    """Replace a copy of labeled_img where elements of unique_labels are set to 0

    Args:
        labeled_img (np.ndarray): falt array of labels
        unique_labels (np.ndarray): 1d array of labels to be replaced by zero
    """
    A = labeled_img.copy()
    for i in unique_labels:
        A[labeled_img == i] = 0
    return A


def destroy_labels(labeled_img: np.ndarray, unique_labels: np.ndarray):
    """Replace a copy of labeled_img where elements of unique_labels are set to 0

    Args:
        labeled_img (np.ndarray): array of labels
        unique_labels (np.ndarray): 1d array of labels to be replaced by zero
    """
    return _destroy_labels_flat(labeled_img.flatten(), unique_labels).reshape(
        labeled_img.shape
    )
