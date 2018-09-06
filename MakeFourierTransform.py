__author__ = "Arkadiy Simonov, Gregor Hofer"


"""
This is a cleaned up version of Arkadiy's python notebook version.
Added comments, applied python's style guide and expanded variable names to improve readability.
Made to streamline the making of Fourier transforms.
"""

import numpy as np
import husl
import scipy.interpolate
import scipy.misc
import os.path
from matplotlib import pyplot


def vectorize_husl(f):
    """
    Does something.
    f ... ?
    returns ... ?
    """
    def result(H,S,V): # hue, saturation, value
        size = H.shape
        N = np.prod(size)
        [H,S,V] = map(np.ravel, (H, S, V))
        result = np.zeros(np.hstack((N, 3)))
        for i in range(N):
            result[i] = f(H[i], S[i], V[i])
        result = np.reshape(result, np.hstack((size, 3)))
        return result
    return result


def complex_to_color(lookup, a, A):
    """
    Transfomrs the complex values into colors.
    lookup ... ?
    a ... ?
    A ... ?
    """
    size = a.shape
    N = np.prod(size)
    [a,A] = map(np.ravel, (a,A))
    result = np.zeros(np.hstack((N,3)))
    for i in range(N):
        result[i] = lookup[int(a[i] / 4), int(A[i] * 100)]
    result = np.reshape(result, np.hstack((size, 3)))
    return result


def rainbow_image(data,colorLimit,colorMapType='light_husl'):
    """
    False-colors a two-dimensional complex array into an image
    where color intensity corresponds to the absolute value of a complex number
    while the color encodes the phase of the complex number
    data ... two dimensional complex array
    colorLimit ... the value at which full color saturation is achieved. Any values above colorLimit will be indestinguishable.
    colorMapType ... select version of the colormap potions light_husl, dark_husl, light_huslp, dark_huslp.
    """

    data = data / colorLimit

    angles = np.angle(data, deg=True)
    angles[angles<0] = angles[angles<0] + 360
    A = np.absolute(data)
    A[A>1] = 1

    ar = np.arange(0,361,4)
    Ar = np.linspace(0,1,101)
    al, Al = np.meshgrid(ar,Ar)
    al = al.T
    Al = Al.T

    # matching the color scheme
    if colorMapType == 'light_husl':
        lookup = vectorize_husl(husl.husl_to_rgb)(al, Al * 100, 100 - 30 * Al)
    elif colorMapType == 'dark_husl':
        lookup = vectorize_husl(husl.husl_to_rgb)(al, Al * 100, 70 * Al)
    elif colorMapType == 'light_huslp':
        lookup = vectorize_husl(husl.huslp_to_rgb)(al, Al * 100, 100 - 30 * Al)
    elif colorMapType == 'dark_huslp':
        lookup = vectorize_husl(husl.huslp_to_rgb)(al, Al * 100, 70 * Al)
    else:
        raise("dunno this colormap")

    return complex_to_color(lookup, angles, A)

    #var col = HUSL.toRGB(phi/Math.PI*180,val*100,100-30*val);
    # //var col = HUSL.p.toRGB(phi/Math.PI*180,val*100,100-100*val);
    # //var col = HUSL.p.toRGB(phi/Math.PI*180,val*100,70*val);


def rainbow_image_gamma(data, colorLimit, light, gamma=1):
    """
    False-colors a two-dimensional complex array into an image
    with a color adjustment of the gamma value
    where color intensity corresponds to the absolute value of a complex number
    while the color encodes the phase of the complex number
    data ... two dimensional complex array
    colorLimit ... the value at which full color saturation is achieved. Any values above colorLimit will be indestinguishable.
    returns ... ?
    """
    data = data / colorLimit

    angles = np.angle(data, deg=True)
    angles[angles<0] = angles[angles<0] + 360

    amplitude = np.absolute(data) ** gamma

    if light:
        return husl.hsv2rgb(angles, amplitude, 1 - amplitude * 0.0)
    else:
        return husl.hsv2rgb(angles, 1, amplitude)


def to_black_and_white(image):
    """
    Converts color image to black-and-white.
    image ... array like image data
    returns ... ??? what are these calculations?
    """
    if len(image.shape) == 2:
        return image
    else:
        return 0.21 * image[:,:,0] + 0.72 * image[:,:,1] + 0.07 * image[:,:,2]


def get_rectangular_section(image):
    # checking wether image is landscape or portrait
    if image.shape[0] != image.shape[1]:
        if image.shape[0] > image.shape[1]:
            size = image.shape[1]
        else:
            size = image.shape[0]
        # finding the centered position
        xOffset = image.shape[0] // 2 - size // 2 # floor division
        yOffset = image.shape[1] // 2 - size // 2 # floor division
        image = image[xOffset:xOffset+size, yOffset:yOffset+size]
    return image

def transform_data(inputData, colorLimit, show=True, outputFolder=None, baseName=None, dataColorMap='Greys_r'):
    """
    Makes the Fourier transform of an image from a given file.
    show ... bool if true, then the resulting Fourier transform will be displayed using pyplot.
    """
    if type(inputData) == str:
        image = scipy.misc.imread(inputData)
        data = to_black_and_white(image) # convert to black and white
        # check image format
        if len(data.shape) != 2:
            raise TypeError("The provided image has more or less than two dimensions!")
    elif type(inputData) == np.ndarray:
        data = inputData
    data = get_rectangular_section(data) # make it square image

    # make the Fourier transform
    FT = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(data))) # calculate Fourier transform
    rainbow = rainbow_image(FT, colorLimit=colorLimit, colorMapType='light_husl')

    # draw fourier transform
    if show:
        pyplot.imshow(rainbow)
        pyplot.show()

    # write data out
    if outputFolder != None and baseName != None:
        if type(inputData) == str:
            scipy.misc.imsave(os.path.join(outputFolder,baseName + "_col.png"), image)
        pyplot.imsave(os.path.join(outputFolder,baseName + "_BaW.png"), data, cmap=dataColorMap)
        scipy.misc.imsave(os.path.join(outputFolder,baseName + "_FTC.png"), rainbow)
        scipy.misc.imsave(os.path.join(outputFolder,baseName + "_FTA.png"), to_black_and_white(rainbow))


#save the original image with its pixels, and not the representation
#imsave('fft.png', rainbow_image(ft, cl=cl, kind='light_husl'))




def rainbow_draw(data, colorLimit, colorMapType='light_husl'):
    """
    A quick function in order to draw complex arrays
    """
    RGB = rainbow_image(data, colorLimit, colorMapType)
    pyplot.imshow(RGB,interpolation='nearest')
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.axis('off')
    pyplot.show()