import numpy as np
import astropy.modeling.functional_models
import MakeFourierTransform
from matplotlib import pyplot
import PIL
from PIL import ImageDraw

import importlib
importlib.reload(MakeFourierTransform)


def make_FFT_ready(A):
    return A[:A.shape[0]-1,:A.shape[0]-1] # make it FFT ready


def rotate_vector_cartesian_z(v, theta):
    """
    Rotates a cartesian vector along the z-axis.
    U ... np.array a 2 or 3D vector that should be rotated
    theta ... float rotation in degree, theta > 0 counter-clock-wise
    retruns ... np.array of the rotated vector
    """
    theta = np.deg2rad(theta)
    if v.shape == (2,):
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    elif v.shape == (3,):
        rotation = np.array([[np.cos(theta), -np.sin(theta),0], [np.sin(theta), np.cos(theta),0],[0,0,1]])
    else:
        raise IndexError("Shape ", v.shape, " is incomaptible with rotation.")
    return np.dot(rotation,v)


def make_gauss(amplitude, sigmaX, sigmaY):
    """
    Creates an array that contains a 2D guass function at the origin.
    amplitude ... float height of the gauss function
    sigmaX ... float standard deviation in x direction
    sigmaY ... float standard deviation in y direction
    """
    gauss = astropy.modeling.functional_models.Gaussian2D(amplitude, 0, 0, sigmaX, sigmaY)

    # plotting the gauss function into an array
    x = np.arange(-4,4.1,0.1)
    y = np.arange(-4,4.1,0.1)
    z = np.zeros([len(x),len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            z[i,j] = gauss(x[i],y[j])
    # z=np.roll(z,10) # I have no idea what this did here, maybe I was testing soemthing?
    return make_FFT_ready(z)


def gauss_array_sigmas(amplitude, sigmaMin, sigmaMax, sigmaStep, folder):
    """
    Creates multiple 2D Gauss functions and varrying their sigmas in x and y direction.
    amplitude ... float height of the gauss function
    sigmaMin ... float smallest standard deviation
    sigmaMax ... float largest standard deviation
    sigmaStep ... int number of steps for varrying between sigmaMin and sigmaMax
    folder ... str location where to save images to
    """
    sX = np.arange(sigmaMin[0], sigmaMax[0], sigmaMax[0] / sigmaStep[0])
    sY = np.arange(sigmaMin[1], sigmaMax[1], sigmaMax[1] / sigmaStep[1])

    for i in range(len(sX)):
        for j in range(len(sY)):
            z = make_gauss(amplitude, sX[i], sY[j])
            fileName = "gauss_A" + str(amplitude) + "_sx" + str(sX[i]) + "_sy" + str(sY[j])
            MakeFourierTransform.transform_data(z, 50, False, folder, fileName, dataColorMap="Reds")


def gauss_array_translation(amplitude, translationMin, translationMax, translationStep, folder):
    """
    Creates multiple 2D Gauss functions and varrying the offset.
    Note: using the roll function instead of the gauss function's offset is not the smartest way to do it.
    amplitude ... float height of the gauss function
    translationMin ... int lower end of translation
    translationMax ... int upper end of translation
    translationStep ... int number of steps for varrying between translationMin and translationMax
    folder ... str location where to save images to
    """
    # calculating x and y translation steps
    tX = np.arange(translationMin[0], translationMax[0], translationMax[0] / translationStep[0])
    tY = np.arange(translationMin[1], translationMax[1], translationMax[1] / translationStep[1])
    # apply translation
    for i in range(len(tX)):
        for j in range(len(tY)):
            z = make_gauss(amplitude, 0.3, 0.3)
            z = np.roll(z,int(tX[i]),axis=1)
            z = np.roll(z,int(tY[j]),axis=0)
            fileName = "gauss_A" + str(amplitude) + "_tx" + str(tX[i]) + "_ty" + str(tY[j])
            MakeFourierTransform.transform_data(z, 5, False, folder, fileName, dataColorMap="Reds")

def make_centerd_ngon(vertices, size, radius, startAngle):
    """
    Creates an FFT-ready array of zeros with a polygon shape of ones located at the center.
    vertices ... int number of the polygon's vertices
    size ... int edge length in pixel of the image
    radius ... diameter of the polygon in pixel
    startAngle ... int degrees between the y-axis and center-to-1st-vertex-vector
    """
    offset = size // 2
    slice = 360 / vertices
    points = []
    # calulate vertex positions
    for i in range(vertices):
        newVector = rotate_vector_cartesian_z(np.array([0,-radius]), slice * i + startAngle)
        points.append((newVector[0] + offset, newVector[1] + offset))

    # drawing the ngon
    image = PIL.Image.new('L',(size,size),0)
    ImageDraw.Draw(image).polygon(points,outline=1,fill=1)
    mask = np.array(image)
    return make_FFT_ready(mask)


def ngon_array_rotation(vertices, size, radius, folder, thumbnailSize=None):
    """
    Makes an array of rotatet ngons
    vertices ... list where each entry describes the vertices of an ngon
    size ... int edge length in pixel of the image
    radius ... diameter of the polygon in pixel
    folder ... str location where to save images to
    thumbnailSize ... int when given, cuts out a center part of the given size in pixel of the original image
    """
    for n in vertices:
        for angleStep in range(4):
            angle = 360 / n / 4 * angleStep
            ngon = make_centerd_ngon(n,size,radius,angle)
            fileName = str(n) + "-gon_rot" + str(angle)
            if thumbnailSize != None:
                thumbnailSize = np.array((thumbnailSize,thumbnailSize))
            MakeFourierTransform.transform_data(ngon, 100, show=False, outputFolder=folder, baseName=fileName, dataColorMap="Reds", thumbnail=thumbnailSize)