""" Toolkit for cell counting """

import scipy
import pandas
import numpy as np
from numba import njit
import skimage
import matplotlib.pyplot as plt
import tifffile


def display(img, v_max=2**12):
    """Display the image with pyplot

    Args:
        img (np.ndarray): 2d image to display
        v_max (float| int, optional): Value by which the array is divided
            before conversion to float. Defaults to 2**12.
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(img.astype("float64") / v_max, cmap="gray")


def background_image(img: np.ndarray, cell_radius) -> np.ndarray:
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
    footprint = np.ones((3, 3))
    ball_kernel = skimage.restoration.ball_kernel(2 * cell_radius, 2)
    if len(img.shape) == 3:
        footprint = np.ones((3, 3, 1))
        ball_kernel = ball_kernel.reshape(ball_kernel.shape + (1,))
    smooth_img = skimage.filters.rank.median(img, footprint=footprint)

    background = skimage.restoration.rolling_ball(
        smooth_img, kernel=ball_kernel
    )
    background = np.minimum(background, img)
    return background


def tissue_mask(img, threshold=40, cell_radius=7):
    mask = skimage.morphology.binary_opening(
        img > threshold, footprint=np.ones((cell_radius, cell_radius))
    )
    mask = skimage.morphology.binary_dilation(
        mask, footprint=np.ones((cell_radius * 3, cell_radius * 3))
    )
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
        altered_image[:, :, i] = mask * img[:, :, i]

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
            + [
                distribution[start_val:end_val]
                for distribution in distributions
            ]
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


def check_threshold(
    img, channel, threshold, val_bound=2**12, color="magenta", figsize=(40, 80)
):
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
    view = img[:, :, channel]
    mask = view > threshold
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(
        skimage.color.label2rgb(
            mask,
            image=1 - view / val_bound,
            bg_label=False,
            bg_color=(0.9, 0.9, 0.9),
            alpha=0.2,
            colors=[color],
        ),
        cmap="gray",
    )
    ax[1].imshow(view / val_bound, cmap="gray")


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
        min_distance=(cell_radius) // 2,
        threshold_abs=ignored_radius + 1,
    )
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.transpose())] = True
    markers = skimage.measure.label(local_max_mask)

    return skimage.segmentation.watershed(-distance, markers, mask=bin_image)


def check_segmentation(
    img, channel, labels, val_bound=2**12, figsize=(40, 80)
):
    """Display labels overlayed over img[:,:,channel] and img[:,:,channel].

    Args:
        img (np.ndarray): 2d image
        channel (int): considered channel
        threshold (int|float): used threshold
        val_bound (_type_, optional): Value by which the array is divided
            before conversion to float. Defaults to 2**12.
    """
    view = img[:, :, channel]
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(
        skimage.color.label2rgb(
            labels,
            image=1 - view / val_bound,
            bg_label=False,
            bg_color=(0.9, 0.9, 0.9),
            alpha=0.2,
        ),
        cmap="gray",
    )
    ax[1].imshow(view / val_bound, cmap="gray")


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


def sobel_based_mask(
    img, contrast_threshold, footprint_size: int | None = None
):
    """Define a contrast based 'no cell' mask.

    The locus where the gradient intensity is higher that contrast_threshold
    is used to separate various components in the image. A dilation of
    the biggest component is returned. If footprint_size is provided, a closing filter with a
    square footprint of this size is used to enhance the contouring property
    of the above locus, prior to computing the components.

    Args:
        img (np.ndarray): 1 channel image
        contrast_threshold (float): used contrast threshold
        footprint_size (int | None, optional): footprint size. Defaults to None.
            If None, no closing is performed.

    Returns:
        np.ndarray: binary image.
    """
    sobel = skimage.filters.sobel(img)
    contours = sobel > contrast_threshold
    if footprint_size is not None:
        contours = skimage.morphology.binary_closing(
            contours, footprint=np.ones((footprint_size,) * 2)
        )
    labels = skimage.measure.label(1 - contours)
    counts = np.bincount(labels.flatten())
    mask_label = np.argmax(counts)
    mask = labels == mask_label

    return skimage.morphology.binary_dilation(mask, footprint=np.ones((3, 3)))


@njit(parallel=True)
def _njit_blur_level(img: np.ndarray, mask, size=5):
    radius = size // 2
    output = img.copy() * mask
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not mask[i, j]:
                continue
            y_start = max(0, i - radius)
            y_end = i + radius + 1
            x_start = max(0, j - radius)
            x_end = j + radius + 1
            img_view = img[y_start:y_end, x_start:x_end]
            mask_view = mask[y_start:y_end, x_start:x_end]
            mean = np.sum(img_view * mask_view) / np.count_nonzero(mask_view)
            var = np.sum(
                (img_view - mean) ** 2 * mask_view
            ) / np.count_nonzero(mask_view)
            output[i, j] = var**0.5
    return output


def blur_level(
    img: np.ndarray, mask: np.ndarray | None = None, size: int = 5
) -> np.ndarray:
    """Compute the blur level at every point of img.

        We define the blur level as the standard deviation of the
        intensity of img in the square of size 'size around the given point.

    Args:
        img (np.ndarray): The grayscale image whose blur levels we compute.
        mask (np.ndarray | None): The mask that defines which values are used for the
            blur level computation. Defaults to None. If None, every value is used.
        size (int, optional): Odd integer. Size of the filter square footprint.
            Defaults to 5.

    Returns:
        np.ndarray: blur level array, same shape as img.
    """
    if mask is None:
        mask = np.ones(img.shape, dtype="bool")
    output = np.zeros(img.shape)
    locus = np.where(mask)
    y_start, y_end = np.amin(locus[0]), np.amax(locus[0]) + 1
    x_start, x_end = np.amin(locus[1]), np.amax(locus[1]) + 1
    output_view = output[y_start:y_end, x_start:x_end]
    mask_view = mask[y_start:y_end, x_start:x_end]
    img_view = img[y_start:y_end, x_start:x_end]
    output_view[:, :] = _njit_blur_level(
        img_view.astype("float64"), mask=mask_view, size=size
    )
    return output


def no_blur_sobel_mask(
    img,
    contrast_threshold,
    blur_threshold,
    dilation_footprint_size: int | None = None,
    blur_footprint_size: int = 5,
):
    """Compute sobel_based_mask and extend it to blurry regions of its complement.

        The arguments that do not start by 'blur' are passed to sobel_based_mask.
        The resulting mask is then treated as follows.
        blur levels of img are computed outside mask in square footprints of
        size blur_footprint_size using the standard deviation of intensity.
        The points with blur>blur_threshold are added to mask.

    Args:
        img (_type_): image to be treated
        contrast_threshold (_type_): contrast threshold
        dilation_footprint_size (int | None, optional): See sobel_based_doc.
            Defaults to None.
        blur_footprint_size (int, optional):  blur footprint size. Defaults to 5.
    """
    mask = sobel_based_mask(img, contrast_threshold, dilation_footprint_size)
    labels = skimage.measure.label(1 - mask)
    for label in np.unique(labels):
        blur_levels = blur_level(img, labels=label, size=blur_footprint_size)
        mask[blur_levels > blur_threshold] = True
    return mask


def sobel_segmentation(
    img: np.ndarray,
    contrast_threshold: int | float,
    footprint_size: int | None = None,
    cell_radius=7,
    ignored_radius=3,
):
    """Watershed foreground defined by sobel_based_mask result as background.

        The three first arguments are passed to sobel_based_mask to get mask.
        1-mask is segmented by watershed using the last optional arguments.



    Args:
        img (np.darray): grayscale Image to be treated.
        contrast_threshold (_type_): see sobel_based_mask doc
        footprint_size (int | None, optional): see sobel_based_mask doc.
            Defaults to None.
        cell_radius (int, optional): see watershed doc. Defaults to 7.
        ignored_radius (int, optional): see watershed doc. Defaults to 3.

    Returns:
        labels for segmentation as np.ndarray
    """

    mask = sobel_based_mask(img, contrast_threshold)
    labels = watershed(
        1 - mask, ignored_radius=ignored_radius, cell_radius=cell_radius
    )
    return labels


# shape considerations

def line_in_shape(shape,line_eq):
    """Give pixels of shape that belong to line_eq

    Args:
        shape (np.ndarray): binary  image
        line_eq (tuple[float,float,float]): triple (a,b,c) defining the line
            as a x i + b x j = c (i for row, j for column)
    """
    a,b,c=line_eq

    if b==0:
        if a<0:
            a,c=-a,-c
        locus=np.zeros(shape.shape,dtype='bool')
        locus[c//a]=1

    if b<0:
        a,b,c=-a,-b,-c
    
    if b>0:
        indices=np.indices(shape.shape)
        i=indices[0,:,:]
        j=indices[1,:,:]

        if a>0:
            diff=c-(a*i+b*j)
            locus=(diff<(a+b))*(diff>=0)

        if a<0:
            diff=c-(a*(i+1)+b*j)
            locus=(diff<(b-a))*(diff>=0)
        
    if a==0:
        locus=np.zeros(shape.shape,dtype='bool')
        locus[:,c//b]=1

    return locus*shape

@njit(parallel=True)
def max_dist(shape,center):
    indices=np.indices(shape.shape)
    i=indices[0,:,:]
    j=indices[1,:,:]
    distances=(i-center[0])**2+(j-center[1])**2
    distances=distances*shape
    return (np.amax(distances))**.5



def shape_dims(shape):
    m=skimage.measure.moments(shape,order=1)
    center=m[1,0]/m[0,0],m[0,1]/m[0,0]
    M=skimage.measure.moments_central(shape,order=2)
    if M[0,2]==M[2,0]:
        slope_0=0
    else:
        axis_theta_0=1/2*np.arctan(2*M[1,1]/(M[2,0]-M[0,2]))
        slope=np.tan(axis_theta_0)

    line_eq_0=(slope,1,slope*center[0]+center[1])
    line_eq_1=(1,-slope,center[0]-slope*center[1])

    locus_0=line_in_shape(shape,line_eq_0)
    locus_1=line_in_shape(shape,line_eq_1)

    dims=[max_dist(locus_0,center),max_dist(locus_1,center)]
    dims.sort()

    return dims

def eccentricity(shape):
    """ Return eccentricity of shape as defined in Burge-Burger 

    Args:
        shape (np.ndarray): binary image

    Raises:
        ValueError: 'empty shape' 

    Returns:
        float : the eccentricty of the shape, a^2/b^2 if the shape is an
            ellipse (x/a)^2+(y/b)^2 = 1 with a>=b.
    """
    height,width=shape_dims(shape)
    if width==0:
        raise ValueError('shape is empty')
    return (height/width)**2


