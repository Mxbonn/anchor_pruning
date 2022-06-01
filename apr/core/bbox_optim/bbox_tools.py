import torch
import mmdet.core


def get_bboxes_single(cls_score_list, bbox_pred_list, mlvl_anchors, img_shape, cfg, bbox_coder):
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)

        use_sigmoid_cls = False
        if "loss_cls" in cfg.model.bbox_head:
            use_sigmoid_cls = cfg.model.bbox_head.loss_cls.get('use_sigmoid', False)
        if use_sigmoid_cls:
            num_classes = cfg.model.bbox_head.num_classes
        else:
            num_classes = cfg.model.bbox_head.num_classes + 1
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, num_classes)
            if use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bboxes = bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        return mlvl_bboxes, mlvl_scores


def get_anchor_idx(bbox_idx, mlvl_anchors, featmap_sizes):
    n_levels = len(mlvl_anchors)
    base_anchor_idx = 0
    for level in range(n_levels):
        n_anchors_level = mlvl_anchors[level].shape[0]
        n_default_anchors = n_anchors_level / (featmap_sizes[level][0] * featmap_sizes[level][1])
        if bbox_idx < n_anchors_level:
            return int(base_anchor_idx + (bbox_idx % n_default_anchors))
        else:
            base_anchor_idx += n_default_anchors
            bbox_idx -= n_anchors_level
    raise ValueError(f"bbox idx {bbox_idx} out of scope.")


def get_bbox_indices(anchor_idx, mlvl_anchors, featmap_sizes):
    n_levels = len(mlvl_anchors)
    base_bbox_idx = 0
    for level in range(n_levels):
        n_bboxes_level = mlvl_anchors[level].shape[0]
        n_default_anchors = n_bboxes_level / (featmap_sizes[level][0] * featmap_sizes[level][1])
        if anchor_idx < n_default_anchors:
            return [i + base_bbox_idx for i in range(0, n_bboxes_level) if i % n_default_anchors == anchor_idx]
        else:
            anchor_idx -= n_default_anchors
            base_bbox_idx += n_bboxes_level
    raise ValueError(f"Anchor idx {anchor_idx} out of scope.")


def get_bboxes_scores_anchor_ids(bboxes, scores, mlvl_anchors, featmap_sizes, score_thr):
    filtered_bboxes = []
    filtered_scores = []
    anchor_ids = []
    assert len(bboxes) == len(scores)
    for i, (bbox, bbox_scores) in enumerate(zip(bboxes, scores)):
        max_score = bbox_scores[:-1].max()
        if max_score > score_thr:
            filtered_bboxes.append(bbox)
            filtered_scores.append(max_score)
            anchor_ids.append(get_anchor_idx(i, mlvl_anchors, featmap_sizes))
    return filtered_bboxes, filtered_scores, anchor_ids


def get_mlvl_bboxes_and_scores(detector, dataloader, cfg, rescale=False):
    mlvl_bboxes_list = []
    mlvl_scores_list = []
    for i, data in enumerate(dataloader):
            mlvl_bboxes, mlvl_scores = get_single_data_mlvl_bboxes_and_scores(detector, data, cfg, rescale)
            mlvl_bboxes_list.append(mlvl_bboxes)
            mlvl_scores_list.append(mlvl_scores)
    return mlvl_bboxes_list, mlvl_scores_list


def get_single_data_mlvl_bboxes_and_scores(detector, data, cfg, rescale=False):
    bbox_coder = mmdet.core.build_bbox_coder(cfg.model.bbox_head.bbox_coder)
    with torch.no_grad():
        x = detector.extract_feat(data['img'][0].cuda())
        outs = detector.bbox_head(x)
        bbox_inputs = outs + (data['img_metas'][0].data[0], detector.test_cfg, True)
        cls_scores, bbox_preds, img_metas, _, _ = bbox_inputs

        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = detector.bbox_head.anchor_generator.grid_anchors(
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
        mlvl_bboxes, mlvl_scores = mlvl_bboxes.detach().cpu().numpy(), mlvl_scores.detach().cpu().numpy()
    return mlvl_bboxes, mlvl_scores
