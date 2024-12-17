import json
import os
import os.path as osp
from tqdm import tqdm


def convert_facemask_detectons_to_coco(facemask_dir_annotations: str):
    coco_annotations = {"images": [], "annotations": [], "categories": []}
    categories = {}
    
    for ann_id, ann_name in enumerate(tqdm(os.listdir(facemask_dir_annotations))):
        ann_path = osp.join(facemask_dir_annotations, ann_name)
        with open(ann_path) as io:
            ann_data = json.load(io)
    
        # create image info
        if len(ann_data["objects"]) == 0:
            continue
        # image_id = ann_data["tags"][0]["id"]
        image_id = ann_id + 1
        image_name = ann_name.split(".json")[0]
        image_info = {
                      "file_name": image_name,
                      "height":    ann_data["size"]["height"],
                      "width":     ann_data["size"]["width"],
                      "id":        image_id      
                     }    
        coco_annotations["images"].append(image_info.copy())
    
        
        for ann_obj in ann_data["objects"]:       
            if ann_obj['classTitle'] not in categories: 
                categories.update({ann_obj['classTitle']: len(categories)+1})
    
            category_id = categories[ann_obj['classTitle']]
            top_left_coord = ann_obj['points']['exterior'][0]
            bottom_right_coord = ann_obj['points']['exterior'][1]
            x_tl, y_tl = top_left_coord
            x_br, y_br = bottom_right_coord
            w_obj = x_br - x_tl
            h_obj = y_br - y_tl
            area_obj = w_obj*h_obj
    
            # create coco annotations
            ann_coco = {
                        "segmentation": [[x_tl, y_tl, x_br, y_tl, x_br, y_br, x_tl, y_br]],
                        "area":         area_obj,
                        "iscrowd":      0,
                        "image_id":     image_id,
                        "bbox":         [x_tl, y_tl, w_obj, h_obj],
                        "category_id":  category_id,
                        "id":           ann_obj["id"]
                        }
            coco_annotations["annotations"].append(ann_coco.copy())

    for category_name, category_id in categories.items():
        coco_annotations["categories"].append({"id": category_id,
                                               "name": category_name})

    return coco_annotations