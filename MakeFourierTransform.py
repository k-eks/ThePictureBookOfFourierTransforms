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
import scipy.fftpack
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


def rainbow_image(data,colorLimit,color_map_type='light_husl'):
    """
    False-colors a two-dimensional complex array into an image
    where color intensity corresponds to the absolute value of a complex number
    while the color encodes the phase of the complex number
    data ... two dimensional complex array
    colorLimit ... the value at which full color saturation is achieved. Any values above colorLimit will be indestinguishable.
    color_map_type ... select version of the colormap potions light_husl, dark_husl, light_huslp, dark_huslp.
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
    if color_map_type == 'light_husl':
        lookup = vectorize_husl(husl.husl_to_rgb)(al, Al * 100, 100 - 30 * Al)
    elif color_map_type == 'dark_husl':
        lookup = vectorize_husl(husl.husl_to_rgb)(al, Al * 100, 70 * Al)
    elif color_map_type == 'light_huslp':
        lookup = vectorize_husl(husl.huslp_to_rgb)(al, Al * 100, 100 - 30 * Al)
    elif color_map_type == 'dark_huslp':
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


def transform_image_from_path(imageFilePath, colorLimit, show=True):
    """
    Makes the Fourier transform of an image from a given file.
    imageFilePath ... string path to the image location.
    show ... bool if true, then the resulting Fourier transform will be displayed using pyplot.
    """
    image = scipy.misc.imread(imageFilePath)
    image = image[:,:387] # make it square image
    image_data = to_black_and_white(image) # convert to black and white

    # make the Fourier transform
    FT = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(image_data))) # calculate Fourier transform

    # draw fourier transform
    if show:
        rainbow_draw(FT,
                     colorLimit=colorLimit, # colorLimit gives saturation
                     color_map_type='light_husl') # There are four possible colormaps: light_husl, dark_husl, light_huslp, dark_huslp


#save the original image with its pixels, and not the representation
#imsave('fft.png', rainbow_image(ft, cl=cl, kind='light_husl'))




def rainbow_draw(data, colorLimit, color_map_type='light_husl'):
    """
    A quick function in order to draw complex arrays
    """
    RGB = rainbow_image(data, colorLimit, color_map_type)
    pyplot.imshow(RGB,interpolation='nearest')
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.axis('off')
    pyplot.show()