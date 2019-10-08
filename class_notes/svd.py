# Using SVD for Linear Regression
"""
M = m x n
U = m x m
S = m x n
V = n x n
Mx = y: use y (labels) to find x (weights) and then use weights to classify new data points
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def main():
    x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
    y = np.array([15, 11, 2, 8, 25, 32])

    transformer = PolynomialFeatures(degree=2, include_bias=False)

    transformer.fit(x)

    x_ = transformer.transform(x)

    # x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

    model = LinearRegression().fit(x_, y)

    print('intercept:', model.intercept_)
    print('coefficients:', model.coef_)

    y_pred = model.predict(x_)

    # Do using psuedo inverse
    from numpy.linalg import pinv as pinv

    # Appending bias column
    ones = np.array([1, 1, 1, 1, 1, 1]).reshape((6, 1))
    x_ = np.concatenate((ones, x_), 1)
    # Calculate pseudo inverse and get weights for each feature
    pinv_x_ = pinv(x_)
    np.matmul(pinv_x_, y)

    # Another way using built in functions
    x_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(x)

    pinv_x_ = pinv(x_)
    np.matmul(pinv_x_, y)

    # Calculate psuedoinverse using SVD
    from numpy.linalg import svd as svd

    U, s, VT = svd(x_)

    Sigma = np.zeros(x_.shape)
    Sigma[:x_.shape[1], :x_.shape[1]] = np.diag(s)

    reconstruct_x_ = U.dot(Sigma).dot(VT)

    Sigma_inv = np.zeros((x_.shape[1], x_.shape[0]))
    Sigma_inv[:x_.shape[1], :x_.shape[1]] = np.diag(1 / s)

    pinv2 = np.transpose(VT).dot(Sigma_inv).dot(np.transpose(U))

# def image():
#     ###Fun with image
#     # from PIL import Image
#     from matplotlib import pyplot as plt
#
#     plt.style.use('classic')
#     img = Image.open('svd_image.JPG')
#     # convert image to grayscale
#     imggray = img.convert('LA')
#     # convert to numpy array
#     imgmat = np.array(list(imggray.getdata(band=0)), float)
#     # Reshape according to orginal image dimensions
#     imgmat.shape = (imggray.size[1], imggray.size[0])
#
#     plt.figure(figsize=(9, 6))
#     plt.imshow(imgmat, cmap='gray')
#     plt.show()
#
#     imgmat.size
#
#     rank = 100
#     U, s, VT = svd(imgmat)
#
#     blurry = U[:, :rank].dot(np.diag(s[:rank])).dot((VT[:rank, :]))
#     plt.imshow(blurry, cmap='gray')
#     plt.show()

if __name__ == '__main__':
    main()