import pickle
import time
import skimage
import numpy as np
from np_gui import np_clickable_image

# from toolkit import toolkit


channel = 2
threshold = 1000
timestamp = str(time.time()).split(".")[0]
y_start, y_end = 4000, 6000
x_start, x_end = 4000, 6000



with open("homogenized_s4.pickle", "rb") as f:
    sample = pickle.load(f)[y_start:y_end,x_start:x_end]

labels = skimage.measure.label(sample[:, :, channel] > threshold)

with open(
    "cell_stat_labeled_image_%i_%i_%s.pickle"
    % (channel, threshold, timestamp),
    "wb",
) as f:
    pickle.dump(labels, f)


def overlay_red(img, mask, alpha, vmax=2**12):
    output = img.astype("float64")
    correction = np.zeros(img.shape[:2], dtype="float64")
    correction[mask] = alpha * vmax
    output[:, :, 0] = (1 - alpha) * output[:, :, 0] * mask + output[
        :, :, 0
    ] * (1 - mask)
    output[:, :, 0] = output[:, :, 0] + correction
    return output


def pop_yn(img: np.ndarray, instruction: str):
    radius = 5
    constant = np_clickable_image.ClickableImage(img, img.shape, [], [], {})
    radio = np_clickable_image.RadioButton(radius)
    print(instruction)
    centered_radio = radio.center_in_shape((constant.shape[0], 3 * radius + 4))
    final_clickable = np_clickable_image.ClickableImage.hstack(
        [constant, centered_radio]
    )
    return final_clickable.use()["radio_toggle"][0]


def select_random_cells(
    n_cells,
    img,
    channel,
    labeled,
    min_size=120,
    max_size=400,
    margin_width=25,
    vmax=2**12,
):
    fine_labels = np.where(
        (np.bincount(labels.flatten()) <= max_size)
        * (np.bincount(labels.flatten()) >= min_size)
    )[0]
    selected = []
    tested = []
    while len(selected) < n_cells:
        if len(tested) == fine_labels.size:
            print("not enough fine labels to provide %i sample." % n_cells)
            return selected
        index = np.random.randint(low=0, high=fine_labels.size - 1)
        label=fine_labels[index]
        if label not in tested:
            cell_mask = labeled == label
            locus = np.where(cell_mask)
            y_start, y_end = np.amin(locus[0]), np.amax(locus[0]) + 1
            x_start, x_end = np.amin(locus[1]), np.amax(locus[1]) + 1

            img_view = img[
                y_start - margin_width : y_end + margin_width,
                x_start - margin_width : x_end + margin_width,
                channel,
            ]
            mask_view = cell_mask[
                y_start - margin_width : y_end + margin_width,
                x_start - margin_width : x_end + margin_width,
            ]
            img_3d = np.stack([img_view] * 3, axis=2)
            overlayed = overlay_red(img_3d, mask_view, 0.5)
            displayed_img = np.hstack([overlayed / vmax, img_3d / vmax])
            text = "Check the radio button if you accept the cell. After this choice, close the window."
            if 0 in displayed_img.shape:
                continue
            accept = pop_yn(displayed_img, text)

            if accept:
                selected.append(label)
            tested.append(label)

    return selected


cell_labels = select_random_cells(30, sample, channel, labels)


with open(
    "chosen_cells_labels_%i_%i_%s.pickle" % (channel, threshold, timestamp),
    "wb",
) as f:
    pickle.dump(cell_labels, f)
