import copy
import functools
import logging
import os
import pickle
import sys
from pathlib import Path

import mmcv.parallel
import mmcv.runner
import mmdet.core
import mmdet.datasets
import mmdet.models
import torch
from torchprofile import profile_macs

from .bbox_tools import get_bbox_indices, get_bboxes_single
from .nms_tools import multiclass_nms
from tqdm import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class Node:
    random_input = torch.randn(1, 3, 300, 300)

    @staticmethod
    def factory(tree, removed_anchors):
        if issubclass(type(tree), AnchorPruningTreeSlow):
            return NodeSlow(tree, removed_anchors)
        else:
            return NodeFast(tree, removed_anchors)

    def __init__(self, tree,  removed_anchors):
        self.removed_anchors = removed_anchors
        self.base_cfg = copy.deepcopy(tree.base_cfg)
        self.cfg = copy.deepcopy(tree.base_cfg)
        self.experiment_dir = tree.experiment_dir
        self.test_dataset = tree.test_dataset

        anchors_to_remove_list = sorted(list(removed_anchors), reverse=True)
        for anchor in anchors_to_remove_list:
            new_scale_ratios = []
            for layer_i, scale_ratio_layer in enumerate(
                    self.cfg['model']['bbox_head']['anchor_generator']['scale_ratios']):
                scale_ratio_layer_list = list(scale_ratio_layer)
                if 0 <= anchor < len(scale_ratio_layer):
                    scale_ratio_layer_list.pop(anchor)
                anchor -= len(scale_ratio_layer)
                new_scale_ratios.append(scale_ratio_layer_list)
            self.cfg['model']['bbox_head']['anchor_generator']['scale_ratios'] = new_scale_ratios

    def bbox_results(self, overwrite=False):
        raise NotImplemented

    @functools.cached_property
    def mAP(self):
        bbox_results = self.bbox_results()
        key = self.base_cfg.evaluation.get("checkpoint_metric", None)
        if key is None:
            key = self.base_cfg.evaluation.get("metric", None)
        if key == "bbox":
            key = "bbox_mAP"
        return bbox_results[key]

    @functools.cached_property
    def macs(self):
        new_backbone = mmdet.models.build_backbone(self.cfg["model"]["backbone"])
        new_backbone.eval()
        backbone_output = new_backbone(Node.random_input)
        if "neck" in self.cfg["model"] and self.cfg["model"]["neck"] is not None:
            new_neck = mmdet.models.build_neck(self.cfg["model"]["neck"])
            new_neck.eval()
            head_input = new_neck(backbone_output)
        else:
            head_input = backbone_output
        bbox_head = mmdet.models.build_head(self.cfg["model"]["bbox_head"])
        bbox_head.eval()
        macs = profile_macs(bbox_head, (head_input,))
        return macs

    @functools.cached_property
    def number_bboxes(self):
        new_backbone = mmdet.models.build_backbone(self.cfg["model"]["backbone"])
        new_backbone.eval()
        backbone_output = new_backbone(Node.random_input)
        if "neck" in self.cfg["model"] and self.cfg["model"]["neck"] is not None:
            new_neck = mmdet.models.build_neck(self.cfg["model"]["neck"])
            head_input = new_neck(backbone_output)
        else:
            head_input = backbone_output
        n_bboxes = 0
        for i, out in enumerate(head_input):
            n_bboxes += out.shape[-2] * out.shape[-1] * len(
                self.cfg['model']['bbox_head']['anchor_generator']['scale_ratios'][i])
        return n_bboxes

    def _anchors(self, round_to_int=True):
        raw_model = mmdet.models.build_detector(self.cfg.model, train_cfg=None,
                                                test_cfg=self.cfg.test_cfg)
        anchor_list = []
        for base_anchors in raw_model.bbox_head.anchor_generator.base_anchors:
            anchor_layer = []
            for anchor_box in base_anchors:
                anchor_w = (anchor_box[2] - anchor_box[0]).item()
                anchor_h = (anchor_box[3] - anchor_box[1]).item()
                if round_to_int:
                    anchor_w = round(anchor_w)
                    anchor_h = round(anchor_h)
                anchor_layer.append((anchor_w, anchor_h))
            anchor_list.append(anchor_layer)
        return anchor_list

    @functools.cached_property
    def anchors(self):
        return self._anchors()

    def __str__(self):
        setstr = repr((sorted(list(self.removed_anchors)))).replace('[', '{').replace(']', '}')
        return f"{setstr}"

    def __repr__(self):
        setstr = repr((sorted(list(self.removed_anchors)))).replace('[', '{').replace(']', '}')
        return f"Node with {setstr} pruned. map: {self.mAP}, macs: {self.macs}"


class NodeFast(Node):
    def __init__(self, tree, removed_anchors):
        super().__init__(tree, removed_anchors)
        self.anchor_generator = tree.anchor_generator
        self.device = tree.device
        self.featmap_sizes_per_img = tree.featmap_sizes_per_img
        self.mlvl_bboxes_per_img = tree.mlvl_bboxes_per_img
        self.mlvl_scores_per_img = tree.mlvl_scores_per_img

    def bbox_results(self, overwrite=False):
        setstr = repr((sorted(list(self.removed_anchors)))).replace('[', '{').replace(']', '}')

        map_results_path = self.experiment_dir.joinpath(f"results/results_{setstr}.pickle")
        if map_results_path.exists() and not overwrite:
            logging.debug(f'Loading results for {self}')
            with open(map_results_path, 'rb') as map_results_file:
                bbox_map = pickle.load(map_results_file)
            logging.debug(f'{bbox_map}')
        else:
            logging.debug(f'Calculating results for {self}')
            reproduced_results = []
            for mlvl_bboxes, mlvl_scores, featmap_sizes in zip(
                    self.mlvl_bboxes_per_img, self.mlvl_scores_per_img, self.featmap_sizes_per_img):
                if torch.cuda.is_available():
                    mlvl_bboxes = copy.deepcopy(mlvl_bboxes).cuda()
                    mlvl_scores = copy.deepcopy(mlvl_scores).cuda()
                else:
                    mlvl_bboxes = copy.deepcopy(mlvl_bboxes)
                    mlvl_scores = copy.deepcopy(mlvl_scores)

                mlvl_anchors = self.anchor_generator.grid_anchors(
                    featmap_sizes, device=self.device)
                filtered_bboxes = []
                for anchor in self.removed_anchors:
                    filtered_bboxes.extend(get_bbox_indices(anchor, mlvl_anchors, featmap_sizes))
                filtered_bboxes.sort()
                mlvl_scores[filtered_bboxes, :] = 0
                test_cfg = self.base_cfg.model.get('test_cfg')
                det_bboxes, det_labels = mmdet.core.multiclass_nms(mlvl_bboxes, mlvl_scores, test_cfg.score_thr,
                                                                   test_cfg.nms, test_cfg.max_per_img)
                bbox_results = mmdet.core.bbox2result(det_bboxes, det_labels, self.base_cfg.model.bbox_head.num_classes)
                reproduced_results.append(bbox_results)
            bbox_map = self.test_dataset.evaluate(reproduced_results)
            map_results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(map_results_path, 'wb+') as map_results_file:
                pickle.dump(bbox_map, map_results_file)
        return bbox_map


class NodeSlow(Node):
    def __init__(self, tree, removed_anchors):
        super().__init__(tree, removed_anchors)
        self.checkpoint_path = tree.checkpoint_path

    def bbox_results(self, overwrite=False):
        setstr = repr((sorted(list(self.removed_anchors)))).replace('[', '{').replace(']', '}')

        map_results_path = self.experiment_dir.joinpath(f"results/results_{setstr}.pickle")
        if map_results_path.exists() and not overwrite:
            logging.debug(f'Loading results for {self}')
            with open(map_results_path, 'rb') as map_results_file:
                bbox_map = pickle.load(map_results_file)
        else:
            logging.debug(f'Calculating results for {self}')
            reproduced_results = []
            test_dataloader = mmdet.datasets.build_dataloader(
                self.test_dataset,
                samples_per_gpu=1,
                workers_per_gpu=self.base_cfg.data.workers_per_gpu,
                dist=False,
                shuffle=False
            )
            raw_model = mmdet.models.build_detector(self.base_cfg.model)
            checkpoint = mmcv.runner.load_checkpoint(raw_model, str(self.checkpoint_path), map_location='cpu')
            raw_model.CLASSES = test_dataloader.dataset.CLASSES
            model = mmcv.parallel.MMDataParallel(raw_model, device_ids=[0])
            raw_model.eval()
            bbox_coder = mmdet.core.build_bbox_coder(self.base_cfg.model.bbox_head.bbox_coder)
            rescale = True
            for i, data in tqdm(enumerate(test_dataloader)):
                with torch.no_grad():
                    x = raw_model.extract_feat(data['img'][0].cuda())
                    outs = raw_model.bbox_head(x)
                    bbox_inputs = outs + (data['img_metas'][0].data[0], raw_model.test_cfg, True)
                    cls_scores, bbox_preds, img_metas, _, _ = bbox_inputs

                    num_levels = len(cls_scores)
                    device = cls_scores[0].device
                    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
                    mlvl_anchors = raw_model.bbox_head.anchor_generator.grid_anchors(
                        featmap_sizes, device=device)
                    for img_id in range(len(img_metas)):
                        cls_score_list = [
                            cls_scores[i][img_id].detach() for i in range(num_levels)
                        ]
                        bbox_pred_list = [
                            bbox_preds[i][img_id].detach() for i in range(num_levels)
                        ]
                        img_shape = img_metas[img_id]['img_shape']
                        scale_factor = img_metas[img_id]['scale_factor']

                        mlvl_bboxes, mlvl_scores = get_bboxes_single(cls_score_list, bbox_pred_list,
                                                                     mlvl_anchors, img_shape,
                                                                     self.base_cfg, bbox_coder)

                        if rescale:
                            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
                        use_sigmoid = False
                        if "loss_cls" in self.base_cfg.model.bbox_head:
                            use_sigmoid = self.base_cfg.model.bbox_head.loss_cls.get('use_sigmoid', False)
                        if use_sigmoid:
                            # Add a dummy background class to the backend when using sigmoid
                            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
                            # BG cat_id: num_class
                            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
                            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
                        filtered_bboxes = []
                        for anchor in self.removed_anchors:
                            filtered_bboxes.extend(get_bbox_indices(anchor, mlvl_anchors, featmap_sizes))
                        filtered_bboxes.sort()
                        mlvl_scores[filtered_bboxes, :] = 0
                        test_cfg = self.base_cfg.model.test_cfg
                        det_bboxes, det_labels = mmdet.core.multiclass_nms(mlvl_bboxes, mlvl_scores, test_cfg.score_thr,
                                                                           test_cfg.nms, test_cfg.max_per_img)
                        bbox_results = mmdet.core.bbox2result(det_bboxes, det_labels, self.base_cfg.model.bbox_head.num_classes)
                reproduced_results.append(bbox_results)
            bbox_map = self.test_dataset.evaluate(reproduced_results)
            map_results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(map_results_path, 'wb+') as map_results_file:
                pickle.dump(bbox_map, map_results_file)
        return bbox_map


def update_pareto_front(pareto_front, new_node):
    logging.debug(pareto_front)
    logging.debug(repr(new_node))
    added_new_node = True
    updated_pareto_front = []
    # First check if any of the current pareto nodes are no longer pareto optimal
    for front_node in pareto_front:
        if (front_node.mAP < new_node.mAP) and (front_node.macs >= new_node.macs):
            logging.debug(f"Removing {repr(front_node)}")
            continue
        elif (front_node.bbox_results() == new_node.bbox_results()) and (front_node.macs > new_node.macs):
            logging.debug(f"Removing {repr(front_node)}")
            continue
        else:
            updated_pareto_front.append(front_node)
    # Check if the new node is pareto optimal (could not be determined by previous loop alone)
    for front_node in updated_pareto_front:
        if (new_node.mAP <= front_node.mAP) and (new_node.macs >= front_node.macs):
            added_new_node = False
    if added_new_node:
        updated_pareto_front.append(new_node)
        logging.debug(updated_pareto_front)
    return updated_pareto_front, added_new_node


class AnchorPruningTree:
    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)
        config_files = list(self.experiment_dir.glob('*.py'))
        self.checkpoint_path = self.experiment_dir.joinpath('best_checkpoint.pth')
        config_path = self.experiment_dir.joinpath(config_files[0].name)

        cfg = mmcv.Config.fromfile(config_path)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        self.base_cfg = cfg

        self.test_dataset = mmdet.datasets.build_dataset(cfg.data.test)
        test_dataloader = mmdet.datasets.build_dataloader(
            self.test_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False
        )
        raw_model = mmdet.models.build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        checkpoint = mmcv.runner.load_checkpoint(raw_model, str(self.checkpoint_path), map_location='cpu')
        raw_model.CLASSES = test_dataloader.dataset.CLASSES
        model = mmcv.parallel.MMDataParallel(raw_model, device_ids=[0])
        raw_model.eval()
        bbox_coder = mmdet.core.build_bbox_coder(cfg.model.bbox_head.bbox_coder)
        rescale = True
        self.mlvl_bboxes_per_img = []
        self.mlvl_scores_per_img = []
        self.featmap_sizes_per_img = []
        for i, data in enumerate(test_dataloader):
            with torch.no_grad():
                x = raw_model.extract_feat(data['img'][0].cuda())
                outs = raw_model.bbox_head(x)
                bbox_inputs = outs + (data['img_metas'][0].data[0], raw_model.test_cfg, True)
                cls_scores, bbox_preds, img_metas, _, _ = bbox_inputs

                num_levels = len(cls_scores)
                device = cls_scores[0].device
                featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
                mlvl_anchors = raw_model.bbox_head.anchor_generator.grid_anchors(
                    featmap_sizes, device=device)
                for img_id in range(len(img_metas)):
                    cls_score_list = [
                        cls_scores[i][img_id].detach() for i in range(num_levels)
                    ]
                    bbox_pred_list = [
                        bbox_preds[i][img_id].detach() for i in range(num_levels)
                    ]
                    img_shape = img_metas[img_id]['img_shape']
                    scale_factor = img_metas[img_id]['scale_factor']

                    mlvl_bboxes, mlvl_scores = get_bboxes_single(cls_score_list, bbox_pred_list,
                                                                 mlvl_anchors, img_shape,
                                                                 cfg, bbox_coder)

                    if rescale:
                        mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
                    use_sigmoid = False
                    if "loss_cls" in self.base_cfg.model.bbox_head:
                        use_sigmoid = self.base_cfg.model.bbox_head.loss_cls.get('use_sigmoid', False)
                    if use_sigmoid:
                        # Add a dummy background class to the backend when using sigmoid
                        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
                        # BG cat_id: num_class
                        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
                        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
                    self.mlvl_bboxes_per_img.append(mlvl_bboxes.cpu().detach())
                    self.mlvl_scores_per_img.append(mlvl_scores.cpu().detach())
                    self.featmap_sizes_per_img.append(featmap_sizes)

        self.anchor_generator = raw_model.bbox_head.anchor_generator
        self.device = device

        self.stack = {Node.factory(self, set())}
        self.pareto_front = [Node.factory(self, set())]
        self.visited_nodes = set()
        self.hashable_visited_nodes = set()

    def get_children(self, node):
        n_anchors = 0
        for layer in self.base_cfg['model']['bbox_head']['anchor_generator']['scale_ratios']:
            n_anchors += len(layer)
        anchors = set(range(n_anchors)) - node.removed_anchors
        children = []
        for anchor in anchors:
            anchors_to_remove = node.removed_anchors | {anchor}
            try:
                child = Node.factory(self, anchors_to_remove)
            except ValueError:
                continue
            if set(node.removed_anchors) != set(child.removed_anchors):
                children.append(child)
        return children

    def process(self, min_map=0.2):
        try:
            while len(self.stack):
                node_to_process = self.stack.pop()
                logging.info(f"Processing {node_to_process}")
                if str(node_to_process) in self.hashable_visited_nodes:
                    logging.info("Node already processed.")
                    continue
                children = self.get_children(node_to_process)
                pareto_children = []
                for child in children:
                    if child.mAP > node_to_process.mAP:
                        logging.info(f'{child} performs best of all children and removing these anchors IMPROVES the score,'
                              f'so we remove them immediately.')
                        pareto_children = [child]
                        break
                    elif child.bbox_results() == node_to_process.bbox_results():
                        logging.info(f'{child} performs equal to parent, so updating pareto front immediately.')
                        pareto_children = [child]
                        break
                    elif child.mAP >= min_map:
                        logging.info(f'{child} does not perform better but reaches threshold so will be compared to pareto front.')
                        pareto_children, flag = update_pareto_front(pareto_children, child)
                    else:
                        logging.info(f'{child} does not reach the {min_map} threshold, so not further processed.')
                for child in pareto_children:
                    logging.debug(f"Potentially updating pareto front with {repr(child)}")
                    self.pareto_front, flag = update_pareto_front(self.pareto_front, child)
                    if flag:
                        self.stack.add(child)
                        logging.debug(f"Added {repr(child)} to stack")
                self.hashable_visited_nodes.add(str(node_to_process))
                self.visited_nodes.add(node_to_process)
        except KeyboardInterrupt:
            print("Interrupted but not cancelled.")

    def __repr__(self):
        return f"Tree for {self.experiment_dir}"


class AnchorPruningTreeSlow(AnchorPruningTree):
    def __init__(self, experiment_dir):
        self.experiment_dir = Path(experiment_dir)
        config_files = list(self.experiment_dir.glob('*.py'))
        self.checkpoint_path = self.experiment_dir.joinpath('best_checkpoint.pth')
        config_path = self.experiment_dir.joinpath(config_files[0].name)

        cfg = mmcv.Config.fromfile(config_path)
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        self.base_cfg = cfg

        self.test_dataset = mmdet.datasets.build_dataset(cfg.data.test)

        self.stack = {Node.factory(self, set())}
        self.pareto_front = [Node.factory(self, set())]
        self.hashable_visited_nodes = set()
        self.visited_nodes = set()


class AnchorPruningTreeSharedLayers(AnchorPruningTreeSlow):
    def get_children(self, node):
        n_layers = len(self.base_cfg['model']['bbox_head']['anchor_generator']['scale_ratios'])
        n_anchors = len(self.base_cfg['model']['bbox_head']['anchor_generator']['scale_ratios'][0])
        for layer in range(n_layers):
            if len(node.cfg['model']['bbox_head']['anchor_generator']['scale_ratios'][layer]) != 0:
                removed_anchors = {removed_anchor % n_anchors for removed_anchor in node.removed_anchors if
                                   n_anchors*layer <= removed_anchor < n_anchors * (layer+1)}
                break

        anchors = set(range(n_anchors)) - removed_anchors
        children = []
        # Remove anchor on each layer
        for anchor in anchors:
            new_anchors_to_remove = {anchor + i * n_anchors for i in range(n_layers)}
            anchors_to_remove = node.removed_anchors | new_anchors_to_remove
            if len(anchors_to_remove) >= n_anchors * n_layers:
                continue
            try:
                child = Node.factory(self, anchors_to_remove)
            except ValueError:
                continue
            children.append(child)
        # or remove an entire layer
        for layer in range(n_layers):
            new_anchors_to_remove = {n_anchors * layer + i for i in range(n_anchors)}
            anchors_to_remove = node.removed_anchors | new_anchors_to_remove
            if len(anchors_to_remove) >= n_anchors * n_layers:
                continue
            try:
                child = Node.factory(self, anchors_to_remove)
            except ValueError:
                continue
            if set(node.removed_anchors) != set(child.removed_anchors):
                children.append(child)
        return children


class AnchorPruningTreeLayerwise(AnchorPruningTree):
    def get_children(self, node, layer=None, min_children=0):
        n_anchors = 0
        begin_anchor = 0
        for i, scale_ratio_layer in enumerate(
                self.base_cfg['model']['bbox_head']['anchor_generator']['scale_ratios']):
            if layer == i:
                begin_anchor = n_anchors
            n_anchors += len(scale_ratio_layer)
            if layer == i:
                break
        anchors = set(range(begin_anchor, n_anchors)) - node.removed_anchors
        children = []
        if len(anchors) <= min_children:
            return children
        for anchor in anchors:
            anchors_to_remove = node.removed_anchors | {anchor}
            try:
                child = Node.factory(self, anchors_to_remove)
            except ValueError:
                continue
            if set(node.removed_anchors) != set(child.removed_anchors):
                children.append(child)
        return children

    def process(self, max_anchors_layer=4, min_map=0.2):
        n_layers = len(self.base_cfg['model']['bbox_head']['anchor_generator']['scale_ratios'])
        for i in range(n_layers):
            next_stack = set()
            while (len(self.stack)):
                node_to_process = self.stack.pop()
                print(f"Processing {node_to_process}")
                children = self.get_children(node_to_process, i, max_anchors_layer)
                if not children:
                    next_stack.add(node_to_process)
                pareto_children = []
                for child in children:
                    self.hashable_visited_nodes.add(child)
                    if child.mAP >= min_map:
                        pareto_children, flag = update_pareto_front(pareto_children, child)
                        if child.bbox_results() == node_to_process.bbox_results():
                            print(f'{child} performs equal to parent, so removing immediately.')
                            pareto_children = [child]
                            break

                for child in pareto_children:
                    self.pareto_front, flag = update_pareto_front(self.pareto_front, child)
                    if flag:
                        self.stack.add(child)
            self.stack = next_stack
        nodes_to_remove = set()
        self.pareto_front = set(self.pareto_front)
        for node in self.pareto_front:
            for i in range(n_layers):
                if len(node.cfg['model']['bbox_head']['anchor_generator']['scale_ratios'][i]) > max_anchors_layer:
                    nodes_to_remove.add(node)
        self.pareto_front -= nodes_to_remove
        self.pareto_front = list(self.pareto_front)


def remove_anchor_from_checkpoint(checkpoint, scale_ratio_index, num_classes):
    new_checkpoint = copy.deepcopy(checkpoint)
    new_indices = list(range(0, scale_ratio_index[1] * 4)) + list(range(scale_ratio_index[1] * 4 + 4, len(
        new_checkpoint['state_dict'][f'bbox_head.reg_convs.{scale_ratio_index[0]}.weight'])))
    new_checkpoint['state_dict'][f'bbox_head.reg_convs.{scale_ratio_index[0]}.weight'] = torch.index_select(
        new_checkpoint['state_dict'][f'bbox_head.reg_convs.{scale_ratio_index[0]}.weight'], 0,
        torch.LongTensor(new_indices))
    new_indices = list(range(0, scale_ratio_index[1] * 4)) + list(range(scale_ratio_index[1] * 4 + 4, len(
        new_checkpoint['state_dict'][f'bbox_head.reg_convs.{scale_ratio_index[0]}.bias'])))
    new_checkpoint['state_dict'][f'bbox_head.reg_convs.{scale_ratio_index[0]}.bias'] = torch.index_select(
        new_checkpoint['state_dict'][f'bbox_head.reg_convs.{scale_ratio_index[0]}.bias'], 0,
        torch.LongTensor(new_indices))
    new_indices = list(range(0, scale_ratio_index[1] * num_classes)) + list(
        range(scale_ratio_index[1] * num_classes + num_classes,
              len(new_checkpoint['state_dict'][f'bbox_head.cls_convs.{scale_ratio_index[0]}.weight'])))
    new_checkpoint['state_dict'][f'bbox_head.cls_convs.{scale_ratio_index[0]}.weight'] = torch.index_select(
        new_checkpoint['state_dict'][f'bbox_head.cls_convs.{scale_ratio_index[0]}.weight'], 0,
        torch.LongTensor(new_indices))
    new_indices = list(range(0, scale_ratio_index[1] * num_classes)) + list(
        range(scale_ratio_index[1] * num_classes + num_classes,
              len(new_checkpoint['state_dict'][f'bbox_head.cls_convs.{scale_ratio_index[0]}.bias'])))
    new_checkpoint['state_dict'][f'bbox_head.cls_convs.{scale_ratio_index[0]}.bias'] = torch.index_select(
        new_checkpoint['state_dict'][f'bbox_head.cls_convs.{scale_ratio_index[0]}.bias'], 0,
        torch.LongTensor(new_indices))
    return new_checkpoint


def results_with_anchor_dropped(experiment_dir, overwrite=False):
    # create model
    checkpoint_path = experiment_dir.joinpath('best_checkpoint.pth')
    config_files = list(Path(experiment_dir).glob('*.py'))
    config_path = os.path.join(experiment_dir, config_files[0].name)

    cfg = mmcv.Config.fromfile(config_path)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    test_dataset = mmdet.datasets.build_dataset(cfg.data.test)
    test_dataloader = mmdet.datasets.build_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    raw_model = mmdet.models.build_detector(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    checkpoint = mmcv.runner.load_checkpoint(raw_model, str(checkpoint_path), map_location='cpu')
    raw_model.CLASSES = test_dataset.CLASSES
    og_model = mmcv.parallel.MMDataParallel(raw_model, device_ids=[0])
    raw_model.eval()

    anchors = []
    for base_anchors in raw_model.bbox_head.anchor_generator.base_anchors:
        for anchor_box in base_anchors:
            anchor_w = (anchor_box[2] - anchor_box[0])
            anchor_h = (anchor_box[3] - anchor_box[1])
            anchors.append((anchor_w, anchor_h))

    map_results_path = experiment_dir.joinpath("before_nms_filter.pickle")
    before_nms_filter_results = {}
    if map_results_path.exists() and not overwrite:
        with open(map_results_path, 'rb') as map_results_file:
            before_nms_filter_results = pickle.load(map_results_file)
    else:
        for j, anchor in enumerate(anchors):
            reproduced_results = []
            bbox_coder = mmdet.core.build_bbox_coder(cfg.model.bbox_head.bbox_coder)
            rescale=True
            for i, data in enumerate(test_dataloader):
                with torch.no_grad():
                    x = raw_model.extract_feat(data['img'][0].cuda())
                    outs = raw_model.bbox_head(x)
                    bbox_inputs = outs + (data['img_metas'][0].data[0], raw_model.get('test_cfg'), True)
                    cls_scores, bbox_preds, img_metas, _, _ = bbox_inputs

                    num_levels = len(cls_scores)
                    device = cls_scores[0].device
                    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
                    mlvl_anchors = raw_model.bbox_head.anchor_generator.grid_anchors(
                        featmap_sizes, device=device)
                    assert len(img_metas) == 1
                    img_id = 0
                    cls_score_list = [
                        cls_scores[i][img_id].detach() for i in range(num_levels)
                    ]
                    bbox_pred_list = [
                        bbox_preds[i][img_id].detach() for i in range(num_levels)
                    ]
                    img_shape = img_metas[img_id]['img_shape']
                    scale_factor = img_metas[img_id]['scale_factor']

                    mlvl_bboxes, mlvl_scores = get_bboxes_single(cls_score_list, bbox_pred_list,
                                                                                     mlvl_anchors, img_shape,
                                                                                     cfg, bbox_coder)

                    if rescale:
                        mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
                    filtered_bboxes = get_bbox_indices(j, mlvl_anchors, featmap_sizes)
                    mlvl_scores[filtered_bboxes, :] = 0
                    det_bboxes, det_labels, bbox_indices = multiclass_nms(mlvl_bboxes, mlvl_scores, cfg.test_cfg.score_thr, cfg.test_cfg.nms, cfg.test_cfg.max_per_img)
                    bbox_results = mmdet.core.bbox2result(det_bboxes, det_labels, cfg.model.bbox_head.num_classes)
                    reproduced_results.append(bbox_results)

            print(anchor)
            before_nms_filter_results[anchor] = test_dataset.evaluate(reproduced_results)
            print('***********************')
        with open(map_results_path, 'wb') as map_results_file:
            pickle.dump(before_nms_filter_results, map_results_file)
    return before_nms_filter_results
