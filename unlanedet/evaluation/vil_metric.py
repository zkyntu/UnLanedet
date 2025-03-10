from functools import partial
import numpy as np
from sklearn.linear_model import LinearRegression
import json, os
import cv2
from tqdm import tqdm
from p_tqdm import t_map, p_map
from .vil_cal_metric import culane_metric
from .vil_utils import RES_MAPPING


def load_vil_data(data_dir, file_list_name:list):
    data = []
    res_each_prediction = []
    for path in tqdm(file_list_name):
        img_data = load_vil_img_data(os.path.join(data_dir,path))
        data.append(img_data)
        res = get_img_shape_info(path.split('/')[0])
        res_each_prediction.append(res)
    return data,res_each_prediction

def get_img_shape_info(sub_name):
    h,w = RES_MAPPING[sub_name][0],RES_MAPPING[sub_name][1]
    return np.array([h,w,3])

def load_vil_img_data(path):
    with open(path, 'r') as data_file:
        img_data = data_file.readlines()
    img_data = [line.split() for line in img_data]
    img_data = [list(map(float, lane)) for lane in img_data]
    img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)] for lane in img_data]
    img_data = [lane for lane in img_data if len(lane) >= 2]

    return img_data
def cal_by_res(predictions,annotations,res,official=True, sequential=False,width=30):
    if sequential:
        results = t_map(partial(culane_metric, width=width, official=official, img_shape=res), predictions,
                        annotations)
    else:
        results = p_map(partial(culane_metric, width=width, official=official, img_shape=res), predictions,
                        annotations)
    # results = culane_metric( predictions,annotations, width=width, official=official, img_shape=res)
    total_tp = sum(tp for tp, _, _, _, _ in results)
    total_fp = sum(fp for _, fp, _, _, _ in results)
    total_fn = sum(fn for _, _, fn, _, _ in results)
    return total_tp,total_fp,total_fn

def eval_predictions(pred_dir, anno_dir, list_names:list, width=30, official=True, sequential=False,iou_thresholds=np.linspace(0.5, 0.95, 10)):
    # print('List: {}'.format(list_path))
    print('Loading prediction data...')
    predictions,res = load_vil_data(pred_dir, list_names)
    print('Loading annotation data...')
    annotations,res = load_vil_data(anno_dir, list_names)
    print('Calculating metric {}...'.format('sequentially' if sequential else 'in parallel'))
    if sequential:
        results = t_map(partial(culane_metric, width=width, official=official,iou_thresholds=iou_thresholds), predictions,
                        annotations,res)
    else:
        results = p_map(partial(culane_metric, width=width, official=official,iou_thresholds=iou_thresholds), predictions,
                        annotations,res)
    # total_tp = sum(tp for tp, _, _, _, _ in results)
    # total_fp = sum(fp for _, fp, _, _, _ in results)
    # total_fn = sum(fn for _, _, fn, _, _ in results)
    # if total_tp == 0:
    #     precision = 0
    #     recall = 0
    #     f1 = 0
    # else:
    #     precision = float(total_tp) / (total_tp + total_fp)
    #     recall = float(total_tp) / (total_tp + total_fn)
    #     f1 = 2 * precision * recall / (precision + recall)
    mean_f1, mean_prec, mean_recall, total_tp, total_fp, total_fn = 0, 0, 0, 0, 0, 0
    ret = {}
    for thr in iou_thresholds:
        tp = sum(m[thr][0] for m in results)
        fp = sum(m[thr][1] for m in results)
        fn = sum(m[thr][2] for m in results)
        precision = float(tp) / (tp + fp) if tp != 0 else 0
        recall = float(tp) / (tp + fn) if tp != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if tp !=0 else 0
        mean_f1 += f1 / len(iou_thresholds)
        mean_prec += precision / len(iou_thresholds)
        mean_recall += recall / len(iou_thresholds)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        ret[thr] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    if len(iou_thresholds) > 2:
        ret['mean'] = {
            'TP': total_tp,
            'FP': total_fp,
            'FN': total_fn,
            'Precision': mean_prec,
            'Recall': mean_recall,
            'F1': mean_f1
        }
    return ret

def get_x(points):
    return [x for x,_ in points]
def get_y(points):
    return [y for _,y in points]

class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta


    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)


    def get_pred_lanes(filename):
        data = load_vil_img_data(filename)
        res = get_img_shape_info(filename.split('/')[-2])
        img_height = res[0]

        LaneEval.img = np.zeros(res)
        # for k in range(len(data)):
        #     imshow_by_xy(LaneEval.img,xs=get_x(data[k]),ys=get_y(data[k]),color=(255,0,0))
        # cv2.imwrite('bf.jpg',LaneEval.img)

        param = [np.polyfit(get_y(data[k]), get_x(data[k]), 2).tolist() for k in range(len(data))]
        
        return param, img_height



    def get_gt_lanes(gt_dir, filename, height):
        gt_json = json.load(open(os.path.join(gt_dir, filename))).get('annotations')['lane']
        img_height = height
        lanex_points = []
        laney_points = []
        for i in gt_json:
            for key, value in i.items():
                if key == 'points' and value != []:
                    lanex = []
                    laney = []
                    for item in value:
                        lanex.append(item[0])
                        laney.append(item[1])
                    lanex_points.append(lanex)
                    laney_points.append(laney)
        return lanex_points,laney_points


    def calculate_results(param, gtx, gty):
        angles = [LaneEval.get_angle(np.array(gtx[i]), np.array(gty[i])) for i in range(len(gty))]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.

        for index, (x_gts,thresh) in enumerate(zip(gtx, threshs)):
            accs = []
            for x_preds in param:
                x_pred =  (x_preds[0] * np.array(gty[index]) * np.array(gty[index]) + x_preds[1] * np.array(gty[index]) + x_preds[2]).tolist()
                
                # imshow_by_xy(LaneEval.img,xs=x_gts,ys=gty[index],color=(0,255,0))
                # imshow_by_xy(LaneEval.img,xs=x_pred,ys=gty[index],color=(0,0,255))
                accs.append(LaneEval.line_accuracy(np.array(x_pred), np.array(x_gts), thresh))
            # print(accs)
            # cv2.imwrite('af.jpg',LaneEval.img)
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(param) - matched
        if len(gtx) > 8 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gtx) > 8:
            s -= min(line_accs)
        return s / max(min(8.0, len(gtx)), 1.), fp / len(param) if len(param) > 0 else 0., fn / max(min(len(gtx), 8.), 1.)


    def calculate_return(pre_dir_name, json_dir_name):
        Preditction = pre_dir_name
        Json = json_dir_name
        num, accuracy, fp, fn = 0., 0., 0., 0.
        list_preditction = os.listdir(Preditction)
        list_preditction.sort()
        for filename in list_preditction:
            pred_files = os.listdir(os.path.join(Preditction, filename))
            json_files = os.listdir(os.path.join(Json, filename))
            pred_files.sort()
            json_files.sort()

            for pfile, jfile in zip(pred_files, json_files):
                pfile_name = os.path.join(Preditction, filename, pfile)
                param, height = LaneEval.get_pred_lanes(pfile_name)
                # print('pred_image_name:', pfile_name)
                # print('json_file_name:', os.path.join(Json, filename, jfile))
                lanex_points, laney_points = LaneEval.get_gt_lanes(os.path.join(Json, filename), jfile, height)

                try:
                    a, p, n = LaneEval.calculate_results(param, lanex_points, laney_points)
                except BaseException as e:
                    raise Exception('Format of lanes error.')
                accuracy += a
                fp += p
                fn += n
                num += 1


        accuracy = accuracy / num
        fp = fp / num
        fn = fn / num
        return accuracy, fp, fn
