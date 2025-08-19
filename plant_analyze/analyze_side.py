import numpy as np
from plantcv import plantcv as pcv

def analyze_one_side(slot_mask, sample_name, rgb_img):
    # 1) skeletonize
    base_skel = pcv.morphology.skeletonize(mask=slot_mask)

    # 2) prune sizes (สเกลง่าย ๆ แบบ safe-list)
    sizes = [50, 75, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400]
    last_err = None
    pruned_skel, edge_objects = None, None
    
    for sz in sizes:
        try:
            ret = pcv.morphology.prune(skel_img=base_skel if pruned_skel is None else pruned_skel,
                                       size=sz, mask=slot_mask)
            if isinstance(ret, tuple) and len(ret) == 3:
                pruned_skel, seg_img, edge_objects = ret
            elif isinstance(ret, tuple) and len(ret) == 2:
                pruned_skel, edge_objects = ret
            else:
                pruned_skel = ret
                edge_objects = None
                
            lo = pcv.morphology.segment_sort(skel_img=pruned_skel, objects=edge_objects, mask=slot_mask)
            leaf_obj = lo[0] if isinstance(lo, tuple) else lo
            stem_obj = lo[1] if isinstance(lo, tuple) and len(lo) > 1 else None


            sid = pcv.morphology.segment_id(skel_img=pruned_skel, objects=leaf_obj, mask=slot_mask)
            segmented_img = sid[0] if isinstance(sid, tuple) else sid
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise last_err if last_err else RuntimeError("Failed to prune skeleton.")
    
    _ = pcv.morphology.fill_segments(mask=slot_mask, objects=leaf_obj, label=sample_name)
    branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=pruned_skel, mask=slot_mask, label=sample_name)
    _ = pcv.morphology.segment_euclidean_length(segmented_img=segmented_img, objects=leaf_obj, label=sample_name)
    _ = pcv.analyze.size(img=rgb_img, labeled_mask=slot_mask, label=sample_name)


    num_leaf_segments = int(len(leaf_obj)) if leaf_obj is not None else 0
    num_stem_segments = int(len(stem_obj)) if stem_obj is not None else 0
    num_branch_points = int(np.count_nonzero(branch_pts_mask)) if branch_pts_mask is not None else 0


    pcv.outputs.add_observation(sample=sample_name, variable='num_leaf_segments',
                                trait='count', method='pcv.morphology.segment_sort',
                                scale='count', datatype=int, value=num_leaf_segments, label='leaf segments')
    pcv.outputs.add_observation(sample=sample_name, variable='num_stem_segments',
                                trait='count', method='pcv.morphology.segment_sort',
                                scale='count', datatype=int, value=num_stem_segments, label='stem segments')
    pcv.outputs.add_observation(sample=sample_name, variable='num_branch_points',
                                trait='count', method='pcv.morphology.find_branch_pts',
                                scale='count', datatype=int, value=num_branch_points, label='branch points')