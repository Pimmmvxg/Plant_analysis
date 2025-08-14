
import matplotlib
from plantcv import plantcv as pcv
from plantcv.utils import tabulate_bayes_classes
from plantcv.parallel import WorkflowInputs
import numpy as np

args = WorkflowInputs(
    images=[r"C:\Plant_analysis\notebooks\datasets\side_view\07.jpg"],
    names="image1",
    result="side07.json",
    debug="plot"
)
pcv.params.debug = args.debug
pcv.params.dpi = 100
pcv.params.text_size = 2
pcv.params.text_thickness = 10
img, path, filename = pcv.readimage(filename=args.image1)

colorspaces = pcv.visualize.colorspaces(rgb_img=img, original_img=False)
a = pcv.rgb2gray_lab(rgb_img=img, channel='a')

hist = pcv.visualize.histogram(img=a)

thresh = pcv.threshold.gaussian(gray_img=a, ksize=9000, offset=10,
                                object_type='dark')

roi = pcv.roi.rectangle(img=thresh, x=620, y=600, w=480, h=460)
kept_mask = pcv.roi.filter(mask=thresh, roi=roi, roi_type='cutto')
cropped_mask = kept_mask[500:1200, 600:1400] #[y:y2, x1:x2]
cropped_img = img[500:1200, 600:1400]
pcv.plot_image(cropped_mask)

mask_dilated = pcv.dilate(gray_img=cropped_mask, ksize=2, i=1)

mask_fill = pcv.fill(bin_img=mask_dilated, size=30)
#mask_fill = pcv.fill_holes(bin_img=mask_fill)

skeleton = pcv.morphology.skeletonize(mask=mask_fill)

pruned_skel, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=100, mask=mask_fill)

pruned_skel, seg_img, edge_objects = pcv.morphology.prune(skel_img=pruned_skel, size=50, mask=mask_fill)

leaf_obj, stem_obj= pcv.morphology.segment_sort(skel_img=pruned_skel, objects=edge_objects, mask=mask_fill)


filled_img = pcv.morphology.fill_segments(mask=mask_fill, objects=leaf_obj, label="default")
branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=pruned_skel, mask=cropped_mask, label="default")

segmented_img, labeled_img = pcv.morphology.segment_id(skel_img=pruned_skel,
                                                       objects=leaf_obj,
                                                       mask=cropped_mask)
labeled_img = pcv.morphology.segment_euclidean_length(segmented_img=segmented_img, 
                                                      objects=leaf_obj, label="default")
shape_img = pcv.analyze.size(img=cropped_img, labeled_mask=mask_fill, label="default")
