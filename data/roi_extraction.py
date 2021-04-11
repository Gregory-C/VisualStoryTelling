from preprocessing import *
import cv2
import torch
import torchvision
import base64
import os


names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}


def load_vist_dii(img_dir, annotation_dir):
    dii = Description_in_Isolation(img_dir, annotation_dir)
    albums = list(dii.albums)
    imgids = {'train': [], 'test': [], 'val': []}
    for a in albums:
        split = a['split']
        imgids[split].extend(a['img_ids'])
    return dii, imgids


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def generate(split, dii, imgids, img_dir):
    captions = []
    feature_lines = []
    label_lines = []

    with torch.no_grad():
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cuda()
        model.eval()
        print(len(imgids[split]))
        count = 0
        for img_ids_batch in batch(imgids[split], 8):
            torch.cuda.empty_cache()
            count += 1
            print('--------------------------------------------------', count)
            faster_rcnn_input = []
            err = []
            for img_id in img_ids_batch:
                img_file = osp.join(img_dir, split, img_id + '.jpg')
                img = cv2.imread(img_file)
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except:
                    print(img_id, img_file)
                    img_file = osp.join(img_dir, split, img_id + '.png')
                    img = cv2.imread(img_file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                height, width, channels = img.shape
                new_height = 500
                new_width = 500 * width // height
                if width > height:
                    new_width = 500
                    new_height = 500 * height // width
                img = cv2.resize(img, (new_width, new_height))
                img = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
                faster_rcnn_input.append(img)

                for id in dii.Images[img_id]['sent_ids']:
                    entry = {'image_id': img_id, 'id': id, 'caption': dii.Sents[id]['original_text']}
                    '''xxx'''
                    captions.append(entry)
            batchid = []
            for i in img_ids_batch:
                if i in err:
                    continue
                batchid.append(i)
            outputs = []
            hook = model.backbone.register_forward_hook(
                lambda self, input, output: outputs.append(output))
            res = model(faster_rcnn_input)
            hook.remove()
            box_features = model.roi_heads.box_roi_pool(outputs[0], [r['boxes'] for r in res],
                                                        [i.shape[-2:] for i in faster_rcnn_input])
            box_features = model.roi_heads.box_head(box_features).detach().cpu().numpy()

            index = 0
            for i, img_id in enumerate(batchid):
                # for i in range(count):
                rois = res[i]
                boxes = rois['boxes'].detach().cpu().numpy()
                labels = rois['labels'].detach().cpu().numpy()
                confidence = rois['scores'].detach().cpu().numpy()

                wh = np.random.randn(len(labels), 2)
                label_line_content = []
                for it in range(len(labels)):
                    label_line_content.append(
                        {"class": names[str(int(labels[it]))], "rect": boxes[it].tolist(),
                         "conf": float(confidence[it])})
                    wh[it][0] = boxes[it][2] - boxes[it][0]
                    wh[it][1] = boxes[it][3] - boxes[it][1]

                boxes_feat = np.concatenate((boxes, wh), 1)
                features = box_features[index: index + len(labels)]
                features = np.concatenate((features, boxes_feat), 1)
                feat_b = base64.b64encode(features.tobytes())

                '''xxx'''
                feature_line = str(img_id) + '  ' + str(
                    json.dumps({"num_boxes": len(labels), "features": str(feat_b)})) + '\n'
                label_line = str(img_id) + '  ' + str(json.dumps(label_line_content)) + '\n'

                feature_lines.append(feature_line)
                label_lines.append(label_line)

                index += len(labels)

        f_feature = open(split + '.feature.tsv', 'w')
        f_label = open(split + '.label.tsv', 'w')
        f_caption = open(split + '.caption.json', 'w')

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
