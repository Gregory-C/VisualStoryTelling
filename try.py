import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import io
import os
import json
import base64
import numpy as np
import cv2
import torch
from vist import *


# Show the image in ipynb
from IPython.display import clear_output, Image, display
import PIL.Image
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


def load_vist_dii(img_dir, annotation_dir):
    dii = Description_in_Isolation(img_dir, annotation_dir)
    albums = list(dii.albums)
    imgids = {'train': [], 'test': [], 'val': []}
    for a in albums:
        split = a['split']
        imgids[split].extend(a['img_ids'])
    return dii, imgids


# Load VG Classes
data_path = 'data/genome/1600-400-20'

vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

vg_attrs = []
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for object in f.readlines():
        vg_attrs.append(object.split(',')[0].lower().strip())

MetadataCatalog.get("vg").thing_classes = vg_classes
MetadataCatalog.get("vg").attr_classes = vg_attrs


cfg = get_cfg()
cfg.merge_from_file("../configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
# VG Weight
cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr.pkl"
predictor = DefaultPredictor(cfg)

NUM_OBJECTS = 36

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, \
    fast_rcnn_inference_single_image


def doit(raw_image):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        #print("Original image size: ", (raw_height, raw_width))

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        #print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        #proposal = proposals[0]
        #print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        #print('Pooled features size:', feature_pooled.shape)

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(
            feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)

        # Note: BUTD uses raw RoI predictions,
        #       we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:],
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break

        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label

        #print(instances)

        return instances, roi_features


def generate(split, dii, imgids, img_dir):
    captions = []
    feature_lines = []
    label_lines = []
    total = len(imgids[split])

    for count, img_id in enumerate(imgids[split]):
        if count % 10 == 0:
            print(count, '\\', total)
        torch.cuda.empty_cache()
        img_file = os.path.join(img_dir, split, img_id + '.jpg')
        img = cv2.imread(img_file)
        if img is None:
            continue
        try:
            instances, features = doit(img)
        except:
            continue

        for id in dii.Images[img_id]['sent_ids']:
            entry = {'image_id': img_id, 'id': id, 'caption': dii.Sents[id]['original_text']}
            '''xxx'''
            captions.append(entry)

        instances = instances.to('cpu')
        features = features.to('cpu')

        boxes = instances.pred_boxes.tensor.numpy()
        labels = instances.pred_classes.numpy()
        confidence = instances.scores.numpy()

        label_line_content = []
        wh = np.random.randn(len(labels), 2)
        for it in range(len(labels)):
            label_line_content.append(
                {"class": vg_classes[int(labels[it])], "rect": boxes[it].tolist(),
                 "conf": float(confidence[it])})
            wh[it][0] = boxes[it][2] - boxes[it][0]
            wh[it][1] = boxes[it][3] - boxes[it][1]
        label_line = str(img_id) + '\t' + str(json.dumps(label_line_content)) + '\n'

        boxes_feat = np.concatenate((boxes, wh), 1)
        features = np.concatenate((features, boxes_feat), 1).astype('float32')
        feat_b = base64.b64encode(features.tobytes())
        feature_line = str(img_id) + '\t' + str(
            json.dumps({"num_boxes": len(labels), "features": str(feat_b)[2:-1]})) + '\n'

        feature_lines.append(feature_line)
        label_lines.append(label_line)

    f_feature = open('/freespace/local/xc295/data/' + split + '.feature.tsv', 'w')
    f_label = open('/freespace/local/xc295/data/' + split + '.label.tsv', 'w')
    f_caption = open('/freespace/local/xc295/data/' + split + '.caption.json', 'w')

    f_caption.write(json.dumps(captions))
    f_feature.writelines(feature_lines)
    f_label.writelines(label_lines)



if __name__ == '__main__':
    img_dir = '/freespace/local/xc295/images'
    annotations_dir = '/freespace/local/xc295/annotations'
    dii, imgids = load_vist_dii(img_dir, annotations_dir)
    generate('val', dii, imgids, img_dir)
    generate('test', dii, imgids, img_dir)
    generate('train', dii, imgids, img_dir)

