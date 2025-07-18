import os
from tqdm import tqdm
import datetime
import logging
import time
import json
import numpy as np
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler

from . import culane_metric
from .tusimple_metric import LaneEval
from .vil_metric import eval_predictions,LaneEval as VILLaneEval
from .vil_utils import RES_MAPPING
from ..utils.comm import get_world_size, is_main_process
from ..utils.logger import log_every_n_seconds
from ..model.module.core.lane import Lane

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class TusimpleEvaluator(DatasetEvaluator):
    """
    Evaluator of the Tusimple dataset
    """
    def __init__(self,
                 ori_img_h,
                 ori_img_w,
                 test_json_file,
                 output_basedir="",
                 metric = None):
        self.ori_img_h = ori_img_h
        self.h_samples = list(range(160, 720, 10))
        self.ori_img_w = ori_img_w
        self.data_infos = None
        self.test_json_file = test_json_file
        self.output_basedir = output_basedir

    def pred2lanes(self, pred):
        # old version. If you meet errors, you can switch to the old version
        # ys = np.array(self.h_samples) / self.ori_img_h
        # lanes = []
        # for lane in pred:
        #     xs = lane(ys)
        #     invalid_mask = xs < 0
        #     lane = (xs * self.ori_img_w).astype(int)
        #     lane[invalid_mask] = -2
        #     lanes.append(lane.tolist())
        # return lanes    

        if len(pred) and isinstance(pred[0], Lane):    #  List[Lane0, Lane1, ...]
            ys = np.array(self.h_samples) / self.ori_img_h
            lanes = []
            for lane in pred:
                xs = lane(ys)
                invalid_mask = xs < 0
                lane = (xs * self.ori_img_w).astype(int)
                lane[invalid_mask] = -2
                lanes.append(lane.tolist())
        else:       # List[(N0, 2),  (N1, 2), ...]
            # for BezierNet
            lanes = []
            for lane in pred:
                lane = lane.astype(np.int)
                lanes.append(lane[:, 0].tolist())
        
        return lanes 
    
    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)
    
    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions, runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def process(self, inputs, outputs):
        return super().process(inputs, outputs)

    def evaluate(self, predictions, runtimes=None):
        pred_filename = os.path.join(self.output_basedir, 'tusimple_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        result, acc = LaneEval.bench_one_submit(pred_filename, self.test_json_file)
        print(result)
        return dict(acc=acc)
    
class CULaneEvaluator(DatasetEvaluator):
    """
    Evaluator of the CULane dataset
    """
    def __init__(self,
                 data_root,
                 ori_img_h,
                 ori_img_w,
                 output_basedir="",
                 cfg=None,
                 metric = None):
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.ori_img_w = ori_img_w
        self.ori_img_h = ori_img_h
        self.output_basedir = output_basedir
        LIST_FILE = {
            'train': 'list/train_gt.txt',
            'val': 'list/test.txt',
            'test': 'list/test.txt',
        } 
        self.list_path = os.path.join(data_root, LIST_FILE['test'])
        self.cfg = cfg
        self.load_annotations()

    def load_annotations(self):
        self.logger.info('Loading CULane annotations for evaluation...')
        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
                infos = self.load_annotation(line.split())
                self.data_infos.append(infos)

    def load_annotation(self, line):
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == '/' else 0::]
        img_path = os.path.join(self.data_root, img_line)
        infos['img_name'] = img_line 
        infos['img_path'] = img_path
        if len(line) > 1:
            mask_line = line[1]
            mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
            mask_path = os.path.join(self.data_root, mask_line)
            infos['mask_path'] = mask_path

        if len(line) > 2:
            exist_list = [int(l) for l in line[2:]]
            infos['lane_exist'] = np.array(exist_list)

        anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt
        with open(anno_path, 'r') as anno_file:
            data = [list(map(float, line.split())) for line in anno_file.readlines()]
        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
                 for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes if len(lane) > 3]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        infos['lanes'] = lanes

        return infos

    def get_prediction_string(self, pred):
        ys = np.array(list(self.cfg.sample_y))[::-1] / self.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join(['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions):
        print('Generating prediction output...')
        for idx, pred in enumerate(tqdm(predictions)):
            output_dir = os.path.join(self.output_basedir, os.path.dirname(self.data_infos[idx]['img_name']))
            output_filename = os.path.basename(self.data_infos[idx]['img_name'])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)
            with open(os.path.join(output_dir, output_filename), 'w') as out_file:
                out_file.write(output)
        result = culane_metric.eval_predictions(self.output_basedir, self.data_root, self.list_path, official=True)
        return result

class VILEvaluator(DatasetEvaluator):
    def __init__(self,output_basedir,data_root,split,metric=None):
        super().__init__()
        self.output_basedir = output_basedir
        dbfile = os.path.join(data_root, 'data', 'db_info.yaml')
        self.imgdir = os.path.join(data_root, 'JPEGImages')
        self.annodir = os.path.join(data_root, 'Annotations')
        self.jsondir = os.path.join(data_root,'Json')
        self.root = data_root
        self.data_infos = []
        self.folder_all_list = []
        self.sub_folder_name = []
        self.max_lane = 0
        self.data_root = data_root

        with open(dbfile, 'r') as f:
            db = yaml.load(f, Loader=yaml.Loader)['sequences']
            self.info = db
            self.videos = [info['name'] for info in db if info['set'] == split]
        self.load_annotations()
    
    def get_json_path(self,vid_path):
        json_paths = []
        for root, _, files in os.walk(vid_path):
            for file in files:
                if file.endswith(".json"):
                    json_paths.append(os.path.join(root, file))
        return json_paths

    def load_annotations(self):
        json_paths = []
        self.all_file_name = []
        print("Searching annotation files...")
        for vid in self.videos:
            json_paths.extend(self.get_json_path(os.path.join(self.jsondir,vid)))
        print("Found {} annotations".format(len(json_paths)))
        for json_path in tqdm(json_paths):
            with open(json_path,'r') as jfile:
                data = json.load(jfile)
            self.load_annotation(data)
            self.all_file_name.append(json_path.replace(self.jsondir+'/','')[:-9]+'.lines.txt')
        print('Max lane: {}'.format(self.max_lane))
        # print(self.mapping)
        
    def load_annotation(self,data):
        points = []
        lane_id_pool =[]
        image_path = data['info']["image_path"]
        # width,height = cv2.imread(os.path.join(self.imgdir,image_path)).shape[:2]
        mask_path = image_path.split('.')[0] + '.png'
        for lane in data['annotations']['lane']:
            # if lane['lane_id'] not in lane_id_pool:
            points.append(lane['points'])
                # lane_id_pool.append(lane['lane_id'])
        self.data_infos.append(
            dict(
                img_name = os.path.join('JPEGImages',image_path),
                # img_size = [width,height],
                img_path = os.path.join(self.imgdir,image_path),
                mask_path = os.path.join(self.annodir,mask_path),
                lanes = points
            )
        )
        sub_folder = image_path.split('/')[0]
        if sub_folder not in self.sub_folder_name:
            self.sub_folder_name.append(sub_folder)
            # self.mapping.update({sub_folder:[width,height]})
        # using index
        idx = self.sub_folder_name.index(sub_folder)
        self.folder_all_list.append(idx)
        
        
        if len(points) > self.max_lane:
            self.max_lane = len(points)


    def get_prediction_string(self, pred,sub_name):
        ori_img_h,ori_img_w = RES_MAPPING[sub_name]
        ys = np.arange(ori_img_h) / ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join(['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != '':
                out.append(lane_str)
        return '\n'.join(out)

    def save_tusimple_predictions(self, idx,prediction,sub_name,runtime):
        line = self.pred2tusimpleformat(idx, prediction, runtime,sub_name)
        self.tu_lines.append(line)

    
    def pred2tusimpleformat(self, idx, pred, runtime,sub_name):
        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred,sub_name)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)
    
    def pred2lanes(self, pred,sub_name):
        ori_img_h,ori_img_w = RES_MAPPING[sub_name]
        ys = np.arange(ori_img_h) / ori_img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes
    def evaluate(self, predictions):
        print('Generating prediction output...')
        output_basedir = os.path.join(self.output_basedir,'preds')
        os.makedirs(output_basedir, exist_ok=True)
        for idx, pred in enumerate(tqdm(predictions)):
            sub_name = self.data_infos[idx]['img_name'].split('/')[1]
            output_dir = os.path.join(output_basedir, sub_name)
            output_filename = os.path.basename(self.data_infos[idx]['img_name'])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred,sub_name)
            with open(os.path.join(output_dir, output_filename), 'w') as out_file:
                out_file.write(output)
        txt_path = os.path.join(self.data_root,'anno_txt')
        # output_basedir = '/home/xly/SPGnet/pred_txt'
        accuracy, fp, fn = VILLaneEval.calculate_return(output_basedir, self.jsondir)
        result = eval_predictions(output_basedir, txt_path, self.all_file_name, official=False,iou_thresholds=[0.5])
        
        return result[0.5]

class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(
    model,
    data_loader,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    callbacks=None,
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.
        callbacks (dict of callables): a dictionary of callback functions which can be
            called at each stage of inference.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    # Import comm functions for distributed support
    from ..utils.comm import get_world_size, get_rank, is_main_process, all_gather, gather, synchronize
    
    num_devices = get_world_size()
    current_rank = get_rank()
    logger = logging.getLogger(__name__)
    
    # Only log from main process to avoid duplicate logs
    if is_main_process():
        logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    dataset = data_loader.dataset
    sampler = data_loader.sampler
    batch_sampler = data_loader.batch_sampler
    evaluator.data_infos = dataset.data_infos
    evaluator.reset()
    is_view = evaluator.view if hasattr(evaluator,"view") else False

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    prediction = []
    prediction_indices = []  # Store indices for reordering
    
    # Synchronize all processes before starting inference
    synchronize()
    
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        dict.get(callbacks or {}, "on_start", lambda: None)()
        
        # Iterate through data_loader with batch indices
        for idx, (batch_indices, inputs) in enumerate(zip(batch_sampler, data_loader)):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            dict.get(callbacks or {}, "before_inference", lambda: None)()
            outputs = model(inputs)
            if hasattr(model,"get_lanes"):
                outputs = model.get_lanes(outputs)
            if num_devices > 1 and hasattr(model.module,"get_lanes"):
                outputs = model.module.get_lanes(outputs)
            dict.get(callbacks or {}, "after_inference", lambda: None)()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            prediction.extend(outputs)
            prediction_indices.extend(batch_indices)  # Store batch indices
            
            if is_view:
                dataset.view(outputs,inputs['meta'],"viz")
            # evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            
            # Only log from main process to reduce log spam
            if is_main_process() and (idx >= num_warmup * 2 or compute_seconds_per_iter > 5):
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()
        dict.get(callbacks or {}, "on_end", lambda: None)()

    # Synchronize all processes before gathering results
    synchronize()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    
    # Only log from main process
    if is_main_process():
        logger.info(
            "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )
    
    # Gather predictions from all processes for distributed evaluation
    if num_devices > 1:
        # Check if distributed sampler is being used
        is_using_distributed_sampler = isinstance(sampler, DistributedSampler)
        
        if is_using_distributed_sampler:
            # With DistributedSampler: each process has different data, need to gather
            all_predictions = gather(prediction)
            all_indices = gather(prediction_indices)
            
            if is_main_process():
                # Flatten and create index-prediction pairs
                flat_predictions = [pred for pred_list in all_predictions for pred in pred_list]
                flat_indices = [idx for idx_list in all_indices for idx in idx_list]
            
                # Reorder predictions according to original dataset order
                if flat_indices and flat_predictions:
                    indexed_predictions = list(zip(flat_indices, flat_predictions))
                    indexed_predictions.sort(key=lambda x: x[0])
                    prediction = [pred for idx, pred in indexed_predictions]
                else:
                    prediction = flat_predictions
            else:
                prediction = []
            
            if is_main_process():
                logger.info(f"Distributed evaluation: gathered and reordered {len(prediction)} predictions from {num_devices} processes")
        else:
            # Without DistributedSampler: all processes have same data, causing duplication
            logger.warning(
                "Distributed training detected but DistributedSampler not used! "
                "This will cause data duplication and incorrect evaluation results. "
                "Consider using DistributedSampler for proper distributed evaluation."
            )
            # Only use predictions from main process to avoid duplication
            if not is_main_process():
                prediction = []
            else:
                # Still reorder predictions on main process if indices are available
                if prediction_indices and prediction:
                    indexed_predictions = list(zip(prediction_indices, prediction))
                    indexed_predictions.sort(key=lambda x: x[0])
                    prediction = [pred for idx, pred in indexed_predictions]
                logger.info(f"Using predictions from main process only to avoid duplication: {len(prediction)} predictions")
    else:
        # Single process: reorder predictions according to batch indices
        if prediction_indices and prediction:
            indexed_predictions = list(zip(prediction_indices, prediction))
            indexed_predictions.sort(key=lambda x: x[0])
            prediction = [pred for idx, pred in indexed_predictions]
    
    # Synchronize before evaluation
    synchronize()
    
    # for evaluation - only evaluate on main process or if evaluator supports distributed evaluation
    if is_main_process() or getattr(evaluator, '_distributed', False):
        results = evaluator.evaluate(prediction)
    else:
        results = None
    
    # Synchronize after evaluation
    synchronize()
    
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
