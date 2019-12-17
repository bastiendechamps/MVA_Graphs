import matplotlib.pyplot as plt
from imageio import imread
import numpy as np
import cv2
import os

from helper import *

from harmonic_function_solution import *


def brightness(face, value=100):
    shadow = 0
    highlight = 255 + value
    alpha_b = (highlight - shadow) / 255
    gamma_b = shadow
    return cv2.addWeighted(face, alpha_b, face, 0, gamma_b)


def offline_face_recognition():
    """
    TO BE COMPLETED

    Function to test offline face recognition.
    """

    # Parameters
    cc = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))
    frame_size = 96
    # Loading images
    images = np.zeros((100, frame_size ** 2))
    labels = np.zeros(100)

    for i in np.arange(10):
        for j in np.arange(10):
            im = imread("data/10faces/%d/%02d.jpg" % (i, j + 1))
            box = cc.detectMultiScale(im)
            top_face = {"area": 0}

            for cfx, cfy, clx, cly in box:
                face_area = clx * cly
                if face_area > top_face["area"]:
                    top_face["area"] = face_area
                    top_face["box"] = [cfx, cfy, clx, cly]

            fx, fy, lx, ly = top_face["box"]
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray_face = gray_im[fy:fy + ly, fx:fx + lx]

            """
            Apply preprocessing to balance the image (color/lightning), such as filtering 
            (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and
            equalization (cv.equalizeHist).   
            """
            gray_face = brightness(gray_face, 170)
            gray_face = cv2.equalizeHist(gray_face)

            # gray_face = cv2.boxFilter(gray_face, -1, (5, 5))
            # gray_face = cv2.GaussianBlur(gray_face, (5, 5), 0)
            # gray_face = cv2.bilateralFilter(gray_face, 9, 75, 75)

            gf = cv2.resize(gray_face, (frame_size, frame_size)).astype(np.float)
            gf -= gf.mean()
            gf /= gf.max()
            gray_face = gf
            # resize the face and reshape it to a row vector, record labels
            images[j * 10 + i] = gray_face.reshape((-1))
            labels[j * 10 + i] = i + 1

    """
     If you want to plot the dataset, set the following variable to True
    """
    plot_the_dataset = False

    if plot_the_dataset:
        plt.figure(1)
        for i in range(100):
            plt.subplot(10, 10, i+1)
            plt.axis('off')
            plt.imshow(images[i].reshape(frame_size, frame_size))
            r='{:d}'.format(i+1)
            if i < 10:
                plt.title('Person '+r)
        plt.show()

    """
    select 4 random labels per person and reveal them  
    Y_masked: (n x 1) masked label vector, where entries Y_i take a values in [1, ..., num_classes] if the node is  
              labeled, or 0 if the node is unlabeled (masked)   
    """
    mlabels = labels.copy()
    for i in range(10):
        mask = np.arange(10)
        np.random.shuffle(mask)
        mask = mask[:6]
        for m in mask:
            mlabels[m * 10 + i] = 0

    """
     Choose the experiment parameter and compute hfs solution using either soft_hfs or hard_hfs  
    """
    gamma = .95
    var = 10000.0
    eps = 0.3
    k = 0
    laplacian_regularization = gamma
    laplacian_normalization = 'sym'
    c_l = 1e-2
    c_u = 1e-3

    # hard or soft HFS
    hard_rlabels = hard_hfs(images, mlabels, laplacian_regularization, var, eps, k, laplacian_normalization)
    soft_rlabels = soft_hfs(images, mlabels, c_l, c_u, laplacian_regularization, var, eps, k, laplacian_normalization)

    # Plots #
    plt.subplot(121)
    plt.imshow(hard_rlabels.reshape((10, 10)))
    plt.title("Hard HFS : Acc: {}".format(np.equal(hard_rlabels, labels).mean()))

    plt.subplot(122)
    plt.imshow(soft_rlabels.reshape((10, 10)))
    plt.title("Soft HFS : Acc: {}".format(np.equal(soft_rlabels, labels).mean()))

    plt.show()

    
def offline_face_recognition_augmented():
    """
    TO BE COMPLETED.
    """

    # Parameters
    cc = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))
    frame_size = 96
    gamma = .95
    nbimgs = 50
    # Loading images
    images = np.zeros((10 * nbimgs, frame_size ** 2))
    labels = np.zeros(10 * nbimgs)
    var = 10000

    for i in np.arange(10):
        imgdir = "data/extended_dataset/%d" % i
        imgfns = os.listdir(imgdir)
        for j, imgfn in enumerate(np.random.choice(imgfns, size=nbimgs)):
            im = imread("{}/{}".format(imgdir, imgfn))
            box = cc.detectMultiScale(im)
            top_face = {"area": 0, "box": (0, 0, *im.shape[:2])}

            for cfx, cfy, clx, cly in box:
                face_area = clx * cly
                if face_area > top_face["area"]:
                    top_face["area"] = face_area
                    top_face["box"] = [cfx, cfy, clx, cly]

            fx, fy, lx, ly = top_face["box"]
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray_face = gray_im[fy:fy + ly, fx:fx + lx]

            """
            Apply preprocessing to balance the image (color/lightning), such as filtering 
            (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and
            equalization (cv.equalizeHist).   
            """
            gray_face = brightness(gray_face, 170)
            gray_face = cv2.equalizeHist(gray_face)

            # gray_face = cv2.boxFilter(gray_face, -1, (5, 5))
            # gray_face = cv2.GaussianBlur(gray_face, (5, 5), 0)
            # gray_face = cv2.bilateralFilter(gray_face, 9, 75, 75)

            gf = cv2.resize(gray_face, (frame_size, frame_size)).astype(np.float)
            gf -= gf.mean()
            gf /= gf.max()
            gray_face = gf


            # resize the face and reshape it to a row vector, record labels
            images[j * 10 + i] = gray_face.reshape((-1))
            labels[j * 10 + i] = i + 1

    """
     If you want to plot the dataset, set the following variable to True
    """
    plot_the_dataset = False
    if plot_the_dataset:

        plt.figure(1)
        for i in range(10 * nbimgs):
            plt.subplot(nbimgs,10,i+1)
            plt.axis('off')
            plt.imshow(images[i].reshape(frame_size, frame_size))
            r='{:d}'.format(i+1)
            if i < 10:
                plt.title('Person ' + r)
        plt.show()

    """
    select 4 random labels per person and reveal them  
    Y_masked: (n x 1) masked label vector, where entries Y_i take a values in [1, ..., num_classes] if the node is  
              labeled, or 0 if the node is unlabeled (masked)   
    """
    mlabels = labels.copy()
    for i in range(10):
        mask = np.arange(nbimgs)
        np.random.shuffle(mask)
        mask = mask[:nbimgs - 4]
        for m in mask:
            mlabels[m * 10 + i] = 0

    """
     Choose the experiment parameter and compute hfs solution using either soft_hfs or hard_hfs  
    """
    gamma = .95
    var = 200.0
    eps = 0
    k = 7
    laplacian_regularization = gamma
    laplacian_normalization = 'sym'
    c_l = 1e-2
    c_u = 1e-3

    # hard or soft HFS
    # rlabels = hard_hfs(images, mlabels, laplacian_regularization, var, eps, k, laplacian_normalization)
    rlabels = soft_hfs(images, mlabels, c_l, c_u, laplacian_regularization, var, eps, k, laplacian_normalization)

    """
    Plots
    """

    plt.subplot(311)
    plt.imshow(labels.reshape((-1, 10)).T)
    plt.axis('off')
    plt.title('True labels')

    plt.subplot(312)
    plt.imshow(mlabels.reshape((-1, 10)).T)
    plt.axis('off')
    plt.title("Masked labels")

    plt.subplot(313)
    plt.imshow(rlabels.reshape((-1, 10)).T)
    plt.axis('off')
    plt.title("HFS: Acc: {}".format(np.equal(rlabels, labels).mean()))

    plt.show()


if __name__ == '__main__':
    offline_face_recognition_augmented()
