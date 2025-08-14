import cv2
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
from plantcv.plantcv.analyze import color as analyze_color

color_img, path, filename = pcv.readimage(filename=r"C:\Plant_analysis\notebooks\datasets\top_view\multi2.jpg")
a = pcv.rgb2gray_lab(rgb_img=color_img, channel='a')
gaus = pcv.threshold.gaussian(gray_img=a, ksize=5000, offset=10,
                                object_type='dark')
a_fill = pcv.fill(bin_img=gaus, size=200)
rois1 = pcv.roi.multi(img=color_img, coord=(100,200), radius=200,
                      spacing=(500,400), nrows=3, ncols=2)
lbl_mask, n_lbls = pcv.create_labels(mask=a_fill, rois=rois1)

analyze_color(rgb_img=color_img, labeled_mask=lbl_mask, colorspaces='all')

pcv.outputs.save_results(filename="test.json")
results_dicts = pcv.outputs.observations
print(results_dicts)