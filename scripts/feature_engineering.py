import numpy as np
from numpy.polynomial import Polynomial
import logger
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

def fit_polynomial(pulse, degree):
    x = np.arange(len(pulse))
    y = pulse
    p = Polynomial.fit(x, y, degree)
    coeffs = p.convert().coef
    return coeffs

def feature_engineer_simple(data, params):
    np.random.seed(42)
    X, y, groups, all_scans = data

    logger.log("Shape of data after feature engineering: ", X.shape)
    logger.log("")

    logger.log("Shape of data after downsampling: ", X.shape)
    logger.log("")


    return X, y, groups, all_scans


def feature_engineer(data, params):
    np.random.seed(42)
    X, y, groups = data

    degree = params["degree_of_polynomial"]

    # Parallel fitting polynomial to extract features
    polynomial_coefficients = Parallel(n_jobs=-1)(
        delayed(fit_polynomial)(pulse, degree) for pulse in tqdm(X, desc="Fitting polynomial to extract features")
    )
    
    polynomial_coefficients = np.array(polynomial_coefficients)

    X = polynomial_coefficients

    logger.log("Shape of data after polynomial fit: ", polynomial_coefficients.shape)
    logger.log("")
    
    return X, y, groups