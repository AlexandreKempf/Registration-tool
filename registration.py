###
###     Modules import
###

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
import pyfftw
import h5py
from scipy.ndimage.filters import gaussian_filter
from vispy import scene, app
import imageio
from pyfftw.interfaces.numpy_fft import fftn
from joblib import Parallel, delayed

###
###     todo list
###

# TODO: test the different way to import fftn if pyfftw is not installed
# TODO: faire l'implementation de reikna sur windows
# TODO: save template as png

###
###     Functions
###

def shift(img,x,y):
    """
    Apply a shift to an image
    """
    offset_image = fourier_shift(pyfftw.interfaces.numpy_fft.fftn(img), (x,y))
    offset_image = pyfftw.interfaces.numpy_fft.ifftn(offset_image).real
    return offset_image

def shifts(imgs,x,y):
    """
    Apply a serie of shifts to a serie of images (time should be the first dim)
    """
    return np.array([shift(imgs[i,:,:],x[i],y[i]) for i in np.arange(len(imgs))])


def template_generation(imgs):
    avgimg = np.median(imgs,0)
    avgimg_fft = fftn(avgimg)
    # Estimation of the difference for few trials in order to create a new template
    datafft = fftn(imgs, axes=(1,2))
    mov = Parallel(n_jobs=4)(delayed(register_translation)(avgimg_fft,datafft[i,:,:], 100, 'fourier') for i in np.arange(len(imgs)))
    x = [element[0][0] for element in mov]
    y = [element[0][1] for element in mov]
    # Creation of the new template
    newimgs = shifts(imgs,x,y)
    # Run the algorythm to estimate registration between frames
    template = np.median(newimgs, 0)
    return template

def registerPP(data,template=[],size=30):
    """
    Register multiple images according to a template, if the template does not exist create one with the first SIZE images.
    """
    #size of the first bunch of time to create the template
    nt,nx,ny = data.shape
    datasmooth = gaussian_filter(data,(0,2,2))
    # First approximation of the template
    if len(template) == 0:
        template = template_generation(data[:size,:,:])
    template_fft = fftn(template)
    mov = Parallel(n_jobs=4)(delayed(register_translation)(template_fft,fftn(data[i,:,:]), 100, 'fourier') for i in np.arange(nt))
    x = [element[0][0] for element in mov]
    y = [element[0][1] for element in mov]
    return x,y



def crop(folder):
    """
    Crop the image accordings to the maximum movement in all the direction
    """
    if folder[-1] == "/":
        listofmov=glob.glob(folder + "mov_*.npy")
        listoffile=glob.glob(folder + "data_*.npz")
    else:
        listofmov=glob.glob(folder + "/mov_*.npy")
        listoffile=glob.glob(folder + "/data_*.npz")

    minx, maxx, miny, maxy = -1, -1, -1, -1
    for i in np.arange(len(listofmov)):
        x,y=np.load(listofmov[i])
        minx = np.minimum(minx,x.min())
        maxx = np.maximum(maxx,x.max())
        miny = np.minimum(miny,y.min())
        maxy = np.maximum(maxy,y.max())

    for i in np.arange(len(listoffile)):
        with np.load(listoffile[0]) as data:
            data = data["arr_0"]
            data = data[:,int(np.ceil(maxx)):int(np.floor(minx)),int(np.ceil(maxy)):int(np.floor(miny))]
            template = np.load(folder + "template.npy")
            template_crop = template[int(np.ceil(maxx)):int(np.floor(minx)),int(np.ceil(maxy)):int(np.floor(miny))]
            np.save(folder + "template_crop.npy", template_crop)
            np.savez_compressed(listoffile[i],data)



def fullregistration(folder):
    listfiles = glob.glob(folder + "*.mesc")
    name_idx=0
    for mescfile in np.sort(listfiles):
        a=h5py.File(mescfile,'r')
        print("Using the file " + mescfile.split("/")[-1] + ":")
        try :
            template = np.load(folder + "template.npy")
        except:
            template=[]
        for i in np.arange(len(a["MSession_0"])):
            print("Processing capture "+ str(i) + "/" + str(len(a["MSession_0"])) + " ...", end="\r")
            data=np.array(a["MSession_0"]["MUnit_"+str(i)]["Channel_0"], dtype="uint16")
            # data = data[:int((1./3)*len(data))] # Restrict the data in size / remove for real use
            x,y = registerPP(data,template=template, size=50)
            result = shifts(data,x,y)
            template = np.median(result,0)
            np.save(folder + "template.npy",template)
            np.savez_compressed(folder + "data_" + str(name_idx) + ".npz",result)
            np.save(folder + "mov_" + str(name_idx) + ".npy",np.concatenate([[x],[y]]))
            name_idx += 1
    crop(folder)





def plot(data):
    canvas = scene.SceneCanvas(keys='interactive')
    canvas.size = 800, 600
    canvas.show()
    view = canvas.central_widget.add_view()
    global i
    i = 10
    image = scene.visuals.Image(-np.mean(data[(i-10):i,:,:],0), interpolation='nearest', parent=view.scene, cmap= "viridis")
    view.camera = scene.PanZoomCamera(aspect=1)
    view.camera.flip = (0, 1, 0)
    view.camera.set_range()

    def update(ev):
        global i
        i+=1
        image.set_data(-data[i,:,:])
        image.update()
        if i>(data.shape[0]-2):
            i=0

    timer = app.Timer(.02, connect=update, start=True)
    app.run()


if __name__=="__main__":
    # folder= "/media/alexandre/alex_DD/data/2p/test_tigre/x10/"
    folder= "/run/user/1001/gvfs/smb-share:server=157.136.60.205,share=rawdata/RAW DATA/Alex/Auditory/151029_Al0013/"

    if not os.path.isfile(folder+"data_0.npz"):
        fullregistration(folder=folder)

    listoffile=glob.glob(folder+"data_0.npz")
    with np.load(listoffile[0]) as data:
        # template_crop = np.load(folder + "template_crop.npy")
        data2 = data["arr_0"] #- np.array([template_crop])
        plot(data2)
