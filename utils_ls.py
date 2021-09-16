#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hiremas1
"""

import sys
import h5py as hf
import numpy as np
import pandas as pd
import config as cfg
import numpy.ma as ma
np.set_printoptions(edgeitems=30, linewidth=100000,
                    formatter=dict(float=lambda x: "%.4f" % x))

np.random.seed(5)


def load_hdf_and_csv(image_path: str, attributes_path: str):
    """
    load landsat satellite images from hdf file
    """
    # load attributes
    col_dtypes = {
        "new_ID": int,
        "year": int,
        "orig_ID": str,
        "loss": int,
        "area": int,
        "plantcode": str,
        "speciescod": str,
        "farmid": str,
    }
    df = pd.read_csv(attributes_path, dtype=col_dtypes)
    ids = df["new_ID"].values
    # load images
    images = []
    with hf.File(image_path, "r") as f:
        for key in ids:
            im = f[str(key)][:]
            images.append(im)
    images = np.array(images)
    return images, df


def compute_median_field(images: list) -> (list):
    nimages = len(images)
    medians = []
    for i in np.arange(nimages):
        if i % 5000 == 0:
            print(i, end=",")
        im = images[i]
        im = ma.masked_values(im, cfg.fill_value)
        temp1 = append_ndvi_band(im)
        mdn = ma.median(temp1, axis=(2, 3))
        medians.append(ma.filled(mdn, cfg.fill_value))
    return medians


# def compute_mean_and_median_field(images:list) -> (list, list):
##    print("Computing mean and median")
#    nimages = len(images)
#    MEANS, medians = [], []
#    for i in np.arange(nimages):
#        if i % 5000 == 0:
#            print(i, end=',')
#        im = images[i]
#        im = ma.masked_values(im, cfg.fill_value)
#        temp1 = append_ndvi_band(im)
#        avg = ma.mean(temp1, axis=(2, 3))
#        mdn = ma.median(temp1, axis=(2, 3))
#        MEANS.append(ma.filled(avg, cfg.fill_value))
#        medians.append(ma.filled(mdn, cfg.fill_value))
# print("Finished")
#    return MEANS, medians


def append_ndvi_band(im_seq):
    """
    Append NDVI band to images
    INPUTS:
        1) im_seq: ndarray image of size (seq_len, nbands, h, w)
    OUTPUTS:
        1) im_seq_out: appended sequence
    """
    seq_len, nbands, h, w = im_seq.shape
    im_seq_out = np.zeros((seq_len, nbands + 1, h, w))
    for t, im_t in enumerate(im_seq):
        ndvi_t = compute_ndvi(im_t)
        ndvi_t = np.expand_dims(ndvi_t, axis=0)
        im_t = np.vstack((im_t, ndvi_t))
        im_seq_out[t] = im_t
    return im_seq_out


def append_vi_bands(im_seq):
    """
    Append NDVI,SAVI and EVI to images
    INPUT:
    - im_seq: ndarray image of size (seq_len, nbands, h, w)
    OUTPUT:
    - im_seq_out: appended sequence (seq_len, nbands+(ndvi,savi,evi, ndwi,
                                                      vsdi,tcv,tcb,tcw),h,w)
    """
    seq_len, nbands, h, w = im_seq.shape
    im_seq_out = np.zeros((seq_len, nbands + 8, h, w))
    for t, im_t in enumerate(im_seq):
        ndvi_t = compute_ndvi(im_t)
        savi_t = compute_savi(im_t)
        evi_t = compute_evi(im_t)
        ndwi_t = compute_ndwi(im_t)
        vsdi_t = compute_vsdi(im_t)
        tcv_t = compute_tcv(im_t)
        tcb_t = compute_tcb(im_t)
        tcw_t = compute_tcw(im_t)
        ndvi_t = np.expand_dims(ndvi_t, axis=0)
        savi_t = np.expand_dims(savi_t, axis=0)
        evi_t = np.expand_dims(evi_t, axis=0)
        ndwi_t = np.expand_dims(ndwi_t, axis=0)
        vsdi_t = np.expand_dims(vsdi_t, axis=0)
        tcv_t = np.expand_dims(tcv_t, axis=0)
        tcb_t = np.expand_dims(tcb_t, axis=0)
        tcw_t = np.expand_dims(tcw_t, axis=0)
        im_t = np.vstack(
            (im_t, ndvi_t, savi_t, evi_t, ndwi_t, vsdi_t, tcv_t, tcb_t, tcw_t)
        )
        # im_t = np.vstack((im_t,tcv_t,tcb_t,tcw_t))
        im_seq_out[t] = im_t
    return im_seq_out


def binarize_qa_mask(qa_band):
    """
    convert QA band to binary mask
    """
    clear_pixels = qa_band
    clear_pixels[(clear_pixels == 66) | (clear_pixels == 130)] = 999
    clear_pixels[clear_pixels != 999] = 0
    clear_pixels[clear_pixels == 999] = 1
    clear_pixels = clear_pixels.astype(np.bool)
    return clear_pixels


def apply_qa_mask(in_list):
    """
    Given a list of image sequences, apply 'quality cloud_mask' to each sequence.
    2) cloud maks is applied to retain only valid pixels within the boundary
    INPUTS:
        1) in_list: list containing image sequences
    OUTPUTS:
        1) out_list: list of containing masked image sequences
    """

    out_list = []
    for im_seq in in_list:
        im_seq_masked = np.zeros_like(im_seq)
        for t, im_t in enumerate(im_seq):
            cloud_mask = im_t[-1]
            cloud_mask = binarize_qa_mask(cloud_mask)
            # apply cloud mask to all bands
            cloud_mask = np.tile(cloud_mask, (im_t.shape[0], 1))
            cloud_mask = cloud_mask.reshape((im_t.shape))
            im_t = im_t * cloud_mask
            im_seq_masked[t] = im_t
        out_list.append(im_seq_masked)
    return out_list


def window_cumsum2d(x, right_edges):
    """
    (non-overlapping) windowed cummulative an array.
    Vectorized implementation to a 2d array where we compute the windowed
    cummulative sum along the column for each row.
    """
    _, ncols = x.shape
    x_cumsum = np.cumsum(x, axis=0, dtype=float)
    edge_vals = x_cumsum[right_edges]
    edge_vals = np.vstack((np.zeros(ncols), edge_vals))
    edge_vals = edge_vals[1:, :] - edge_vals[:-1, :]
    return edge_vals


def window_mean2d(x, right_edges):
    """
    caclulate window mean of masked/valid NDVI values.
    Valid NDVI range is [-1, 1]. Note that 0 is a valid NDVI value, so
    1) first create a mask for valied entries within the averaging window.
    2) set invalid entries to 0; we use 0 for computational simpliciyt of
    cummulative sum
    3) use the mask to get the counts of valid values within the window which
    inturn is used for computing the average.
    """
    # NDVI values ranges from [-1, 1]
    mask = (x >= -1) & (x <= 1)  # valid ndvi ranges
    x[~mask] = 0
    window_cumsum = window_cumsum2d(x, right_edges)
    window_counts = window_cumsum2d(mask, right_edges)
    res = np.divide(
        window_cumsum,
        window_counts,
        out=np.zeros_like(window_cumsum),
        where=window_counts != 0,
    )
    res = np.round(res, decimals=cfg.decimal_precision)
    res[res == 0] = cfg.fill_value
    return res


def compute_masked_mean(X):
    X = ma.masked_values(X.astype(float), cfg.fill_value)
    avg = ma.mean(X, axis=0)
    return avg.data


def apply_masked_mean(X, avg):
    X = ma.masked_values(X.astype(float), cfg.fill_value)
    ret = ma.filled(X, avg)
    return ret


def masked_mean_transform(X):
    avg = compute_masked_mean(X)
    X = apply_masked_mean(X, avg)
    return X


def compute_ndvi(im):
    """
    compute NDVI for a given image
    INPUTS:
        1) im: ndarray of size (nbands, h, w)
    OUTPUTS:
        1) ndvi: ndarray of size (h, w) containing the ndvi values.
                The HDVI values are clipped in range [0, 1]
    """
    red = im[2]
    nir = im[3]
    a = nir - red
    b = nir + red + sys.float_info.epsilon
    ndvi = np.round(np.divide(a, b), 3)
    ndvi = np.clip(ndvi, -1, 1)
    return ndvi


def compute_savi(im):
    """
    SAVI = ((NIR - R) / (NIR + R + L)) * (1 + L)
    compute SAVI for given image
    INPUT:
        - im: ndarray (nbands,h,w)
    OUTPUT:
        - savi: ndarray (h,w)
    """
    red = im[2]
    nir = im[3]
    L = 0.5
    precision = 3
    a = nir - red
    b = nir + red + L + sys.float_info.epsilon
    c = 1 + L
    savi = np.multiply(np.divide(a, b), c)
    savi = np.round(savi, precision)
    # savi = 10000 + savi
    return savi


def compute_evi(im):
    """
    EVI = G * ((NIR - R) / (NIR + C1 * R – C2 * B + L))
    compute EVI for given image
    INPUT:
        - im: ndarray (nbands,h,w)
    OUTPUT:
        - evi: ndarray (h,w)
    """
    precision = 3

    G = 2.5
    C1 = 6
    C2 = 7.5
    L = 1

    red = im[2]
    nir = im[3]
    blue = im[0]

    a = nir - red
    b = nir + C1 * red - C2 * blue + L + sys.float_info.epsilon
    evi = np.multiply(G, np.divide(a, b))
    evi = np.round(evi, precision)
    # evi = 10000+ evi

    return evi


def compute_ndwi(im):
    """
    Normalized Difference Water Index (Fletcher)

    NDWI = (green-nir)/(green+nir)
    INPUT:
        - im: ndarray (nbands,h,w)
    OUTPUT:
        - ndwi: ndarray (h,w)

    """

    precision = 3

    green = im[1]
    nir = im[3]

    a = green - nir
    b = green + nir + sys.float_info.epsilon
    ndwi = np.round(np.divide(a, b), precision)
    # ndwi = 10000 + ndwi
    return ndwi


def compute_vsdi(im):
    """
    Visible and Shortwave infrared Drought Index
    (https://link.springer.com/article/10.1007%2Fs13753-013-0008-8)

    VSDI = 1 − [( SWIR − BLUE ) + ( RED − BLUE )]

    INPUT:
        - im: ndarray (nbands,h,w)
    OUTPUT:
        - vsdi: ndarray (h,w)

    """
    precision = 3

    swir = im[5]
    blue = im[0]
    red = im[2]

    a = (swir - blue) + (red - blue)
    vsdi = 1 - a
    vsdi = np.round(vsdi, precision)
    # vsdi = 10000 + vsdi

    return vsdi


def compute_tcv(im):
    """
    Tasseled Cap Vegetation/Greenness
    (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0147121)



    INPUT:
        - im: ndarray (nbands,h,w)
    OUTPUT:
        - tcv: ndarray (h,w)

    """

    precision = 3

    blue = im[0]
    green = im[1]
    red = im[2]
    nir = im[3]
    swir1 = im[4]
    swir2 = im[5]

    tcv = (
        -0.1603 * blue
        + 0.2819 * green
        - 0.4934 * red
        + 0.7940 * nir
        - 0.0002 * swir1
        - 0.1446 * swir2
    )

    tcv = np.round(tcv, precision)

    # tcv = 10000 + tcv

    return tcv


def compute_tcb(im):
    """
    Tasseled Cap Brightness
    (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0147121)


    INPUT:
        - im: ndarray (nbands,h,w)
    OUTPUT:
        - tcb: ndarray (h,w)

    """

    precision = 3

    blue = im[0]
    green = im[1]
    red = im[2]
    nir = im[3]
    swir1 = im[4]
    swir2 = im[5]

    tcb = (
        0.2043 * blue
        + 0.04158 * green
        + 0.5524 * red
        + 0.5741 * nir
        + 0.3124 * swir1
        + 0.2303 * swir2
    )

    tcb = np.round(tcb, precision)

    # tcb = 10000 + tcb

    return tcb


def compute_tcw(im):
    """
    Tasseled Cap Wetness
    (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0147121)



    INPUT:
        - im: ndarray (nbands,h,w)
    OUTPUT:
        - tcw: ndarray (h,w)

    """

    precision = 3

    blue = im[0]
    green = im[1]
    red = im[2]
    nir = im[3]
    swir1 = im[4]
    swir2 = im[5]

    tcw = (
        0.0315 * blue
        + 0.2021 * green
        + 0.3102 * red
        + 0.1594 * nir
        - 0.6806 * swir1
        - 0.6109 * swir2
    )

    tcw = np.round(tcw, precision)

    # tcw = 10000 + tcw

    return tcw


def normalize(im):
    """
    normalize image values between 0-1
    """
    precision = 3
    im = im.astype(np.float16)
    a = im - np.min(im)
    b = np.max(im) - np.min(im) + sys.float_info.epsilon
    d = np.round(np.divide(a, b, where=b != 0), precision)
    return d


def compute_rgb(im):
    """
    Take rgb spectral bands and stack them in appropriate order for display
    INPUTS:
        1) im: ndarray of size (nbands, h, w)
    OUTPUTS:
        1) rgb: ndarray with the bands display order
    """
    assert im.ndim == 3, print(
        "Error: input image should have"
        "3 dimensions(nbands, h, w) but"
        "it has {} dimensions".format(im.ndim)
    )
    b = normalize(im[0])  # band 1 is red so index 0
    g = normalize(im[1])  # band 2 is red so index 1
    r = normalize(im[2])  # band 3 is red so index 2
    rgb = np.dstack((r, g, b))
    rgb = (rgb * 255).astype(np.uint8)
    return rgb


def missing_data_rate(im_seq_list):
    total_seq_len = 0
    total_zeros = 0
    for im_seq in im_seq_list:
        total_seq_len += len(im_seq)
        binary_qa_mask = im_seq[-1]
        total_zeros += np.sum(np.sum(binary_qa_mask, axis=(1, 2)) == 0)
    return total_zeros / total_seq_len
