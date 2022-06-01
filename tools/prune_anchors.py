import pickle
from argparse import ArgumentParser
import sys
import torch

import apr


def main(args):
    if len(args.input_shape) == 1:
        input_shape = (1, 3, args.input_shape[0], args.input_shape[0])
    elif len(args.input_shape) == 2:
        input_shape = (1, 3) + tuple(args.input_shape)
    else:
        raise ValueError('invalid input shape')
    apr.core.bbox_optim.Node.random_input = torch.randn(input_shape)
    if args.layerwise:
        tree = apr.core.bbox_optim.AnchorPruningTreeLayerwise(args.input)
        tree.process(min_map=args.min_map, max_anchors_layer=args.anchors_layer)
    else:
        if args.sharedlayers:
            tree = apr.core.bbox_optim.AnchorPruningTreeSharedLayers(args.input)
        elif args.slow:
            tree = apr.core.bbox_optim.AnchorPruningTreeSlow(args.input)
        else:
            tree = apr.core.bbox_optim.AnchorPruningTree(args.input)
        tree.process(min_map=args.min_map)

    with open(args.output, 'wb+') as f:
        print("writing tree to file...")
        results = {'pareto_front': [], 'visited_nodes': []}
        for node in tree.pareto_front:
            dict_node = {'removed_anchors': node.removed_anchors,
                         'mAP': node.mAP,
                         'macs': node.macs}
            results['pareto_front'].append(dict_node)

        for node in tree.visited_nodes:
            dict_node = {'removed_anchors': node.removed_anchors,
                         'mAP': node.mAP,
                         'macs': node.macs}
            results['visited_nodes'].append(dict_node)
        pickle.dump(results, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to directory with experiment to prune.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to file where to store the serialized '
                                                                        'pruning tree.')
    parser.add_argument('--layerwise', action='store_true', help='Use layerwise pruning tree')
    parser.add_argument('--sharedlayers', action='store_true', help='Use layerwise pruning tree')
    parser.add_argument('--slow', action='store_true', help='Use layerwise pruning tree')
    parser.add_argument('--min_map', type=float, default=0.2, help='Minimum mAP to consider in pareto front.')
    parser.add_argument('--anchors_layer', type=float, help='Max number of anchors in a layer when using layerwise')
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs='+',
        default=[300, 300],
        help='input image size')

    args = parser.parse_args()
    main(args)
