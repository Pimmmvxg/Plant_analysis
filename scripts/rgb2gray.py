import os
from plantcv.plantcv import params
from plantcv.plantcv._debug import _debug
from plantcv.plantcv._helpers import _rgb2gray

def rgb2gray(rgb_img):
    
    gray_img = _rgb2gray(rgb_img)
    _debug(visual=gray_img, filename=os.path.join(params.debug_outdir, "rgb2gray.png"), cmap="gray")
    
    return gray_img