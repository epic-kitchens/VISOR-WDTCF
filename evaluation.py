import os
import numpy as np
import cv2
import argparse
import json
from tqdm import tqdm

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--VISOR_anno_path', default='VISOR/', type=str)
    parser.add_argument('--WDTCF_anno_path', default='VISOR/WDTCF_GT.json', type=str)
    parser.add_argument('--prediction_path', default='Prediction/', type=str)
    parser.add_argument('--EPIC_100_noun_classes', default='EPIC_100_noun_classes.csv', type=str)

    return parser

def transfer_noun(noun):
    if ':' not in noun:
        return noun
    List = noun.split(':')
    return ' '.join(List[1:]) + ' ' + List[0]

def get_category(entity_name, key_dict):
    entity_name = entity_name.strip()
    entity_name = entity_name.lower()

    for kind, kval in key_dict.items():
        if entity_name in kval['instances']:
            return kind

    print(f"Error: entity name not in csv: {entity_name}")
    return None

def get_cats(csv_path='EPIC_100_noun_classes.csv'):
    import csv
    key_dict = {}
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            key_idx = int(row['id']) + 1
            key = transfer_noun(row['key'])
            instances = [x.strip()[1:-1] for x in row['instances'][1:-1].split(',')]
            inst_ls = [transfer_noun(inst) for inst in instances]
            category = row['category']

            key_dict[key_idx] = {}
            key_dict[key_idx]['key'] = key
            key_dict[key_idx]['instances'] = inst_ls
            key_dict[key_idx]['category'] = category

    categories = [{'id': kind, 'name': kval['key']} for kind, kval in key_dict.items()]
    return key_dict, categories

def compute_IoU(mask1, mask2):
    import numpy as np
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    if (np.sum(union) == 0):
        return -1
    else:
        return (np.round(iou_score, 3))

def evaluate(WDTCF_anno_path, VISOR_anno_path, prediction_path, key_dict):
    num_correct_query_pred = 0
    num_correct_source_pred = 0
    query_mask_ious = []
    source_mask_ious = []

    if(not os.path.exists(os.path.join(prediction_path, 'WDTCF_preds.json'))):
        print('WDTCF_preds.json not found!')
        return
    WDTCF_preds = json.load(open(os.path.join(prediction_path, 'WDTCF_preds.json'), 'r'))

    with open (WDTCF_anno_path, 'r') as f:
        WDTCF_GT = json.load(f)
        for query in tqdm(WDTCF_GT):
            query_object = query.split('_')[-1]
            answer = WDTCF_GT[query]['answer'][-1]
            query_class_id = get_category(query_object, key_dict)
            answer_class_id = get_category(answer, key_dict)

            query_pred_class_id = WDTCF_preds[query]['query_pred']
            answer_pred_class_id = WDTCF_preds[query]['answer_pred']

            evidence_frame = WDTCF_GT[query]['evidence']
            evidence_frame_pred = WDTCF_preds[query]['evidence_frame_pred']

            ## check prediction of query
            if(not query_pred_class_id==query_class_id):
                query_mask_ious.append(0.0)
                source_mask_ious.append(0.0)
                continue
            else:
                num_correct_query_pred += 1


            video_id = '_'.join(evidence_frame.split('/')[-1].split('_')[0:-2])

            json_load_path = os.path.join(VISOR_anno_path, 'train', video_id+'.json')
            if(not os.path.exists(json_load_path)):
                json_load_path = os.path.join(VISOR_anno_path, 'val', video_id + '.json')
                if (not os.path.exists(json_load_path)):
                    print('{} not found!'.format(json_load_path))
                    return

            with open(json_load_path, 'r') as f:
                annos = json.load(f)['video_annotations']
                query_mask = np.zeros([1080, 1920], dtype=np.uint8)
                source_mask = np.zeros([1080, 1920], dtype=np.uint8)

                for anno in annos:
                    if (not anno['image']['name'] == evidence_frame):
                        continue
                    entities = anno['annotations']
                    for entity in entities:
                        entity_class = get_category(entity['name'], key_dict)
                        if (entity_class == answer_class_id):
                            object_annotations = entity["segments"]
                            polygons = []
                            polygons.append(object_annotations)
                            ps = []
                            for polygon in polygons:
                                for poly in polygon:
                                    if poly == []:
                                        poly = [[0.0, 0.0]]
                                    ps.append(np.array(poly, dtype=np.int32))

                            cv2.fillPoly(source_mask, ps, (1, 1, 1))

                        if (entity_class == query_class_id):
                            object_annotations = entity["segments"]
                            polygons = []
                            polygons.append(object_annotations)

                            ps = []
                            for polygon in polygons:
                                for poly in polygon:
                                    if poly == []:
                                        poly = [[0.0, 0.0]]
                                    ps.append(np.array(poly, dtype=np.int32))

                            cv2.fillPoly(query_mask, ps, (1, 1, 1))

                    if(not os.path.exists(os.path.join(prediction_path, query+'_query_pred.png'))):
                        print('{} not found!'.format(os.path.join(prediction_path, query+'_query_pred.png')))
                        return

                    query_pred_mask = cv2.imread(os.path.join(prediction_path, query+'_query_pred.png'), cv2.IMREAD_UNCHANGED).astype(np.uint8)

                    # check prediction of evidence
                    if (evidence_frame_pred == evidence_frame):
                        query_iou = compute_IoU(query_pred_mask, query_mask)
                        query_mask_ious.append(query_iou)

                        if (answer_pred_class_id == answer_class_id):
                            num_correct_source_pred += 1

                            if (not os.path.exists(os.path.join(prediction_path, query + '_source_pred.png'))):
                                print('{} not found!'.format(os.path.join(prediction_path, query + '_source_pred.png')))
                                return

                            source_pred_mask = cv2.imread(os.path.join(prediction_path, query+'_source_pred.png'), cv2.IMREAD_UNCHANGED).astype(np.uint8)
                            source_iou = compute_IoU(source_pred_mask, source_mask)
                            source_mask_ious.append(source_iou)
                        else:
                            source_mask_ious.append(0.0)
                    else:
                        query_mask_ious.append(0.0)
                        source_mask_ious.append(0.0)

                    break

        print('Query prediction Acc: %.3f, Source Acc:%.3f, Query mask IOU: %.3f, source IOU: %.3f' % (\
            num_correct_query_pred / len(WDTCF_GT), num_correct_source_pred / len(WDTCF_GT),\
            np.average(query_mask_ious), np.average(source_mask_ious)))

if __name__ == '__main__':
    parser = get_parse()
    opts = parser.parse_args()
    
    key_dict, categories = get_cats(opts.EPIC_100_noun_classes)
    evaluate(opts.WDTCF_anno_path, opts.VISOR_anno_path, opts.prediction_path, key_dict)

