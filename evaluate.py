import os
import sys
import logging

import tensorflow as tf
import tflearn
import numpy as np
import cv2

from IP import reconstructFloorplan
from floornet_custom.eval_utils import filter_points
from floornet_custom.eval_utils import reconstruct_room
from floornet_custom.metric import evaluate_corners
from floornet_custom.metric import evaluate_edges
from floornet_custom.metric import evaluate_rooms
from floornet_custom.metric import evaluate_room_plus_plus
from RecordReader import getDatasetVal
from train import build_graph, build_loss
from utils import sigmoid, softmax
from utils import segmentation2Heatmaps
from utils import ColorPalette

from floorplan_utils import getOrientationCorners
from floorplan_utils import NUM_WALL_CORNERS, NUM_ICONS, NUM_ROOMS


CORNER_DIST_THRESHOLD = 10
ROOM_IOU_THRESHOLD = 0.7
FORMAT = '%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)

def evaluate(options):
    logging.info('Test output directory: %s' % options.test_dir)
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s" % options.test_dir)
    print('I dont use any cache')
    tf.reset_default_graph()

    filenames = []
    filenames.append('data/Tango_val.tfrecords')
    logging.info('Evaluate data files: %s' % filenames)

    assert options.batchSize == 1
    assert options.numTestingImages == 20, 'The length of Tango_val dataset'

    dataset = getDatasetVal(filenames, '', '4' in options.branches, options.batchSize)
    iterator = dataset.make_one_shot_iterator()
    input_dict, gt_dict = iterator.get_next()

    pred_dict, debug_dict = build_graph(options, input_dict)
    dataset_flag = input_dict['flags'][0, 0]
    flags = input_dict['flags'][:, 1]
    loss, loss_list = build_loss(options, pred_dict, gt_dict, dataset_flag, debug_dict, input_dict['flags'])
    var_to_restore = [v for v in tf.global_variables()]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tflearn.is_training(False)
        var_to_restore = [v for v in var_to_restore if 'is_training' not in v.name]
        # Restore network weight
        CHECKPOINT_PATH = './checkpoint/floornet_hybrid1_branch0123_wsf/checkpoint.ckpt'
        loader = tf.train.Saver(var_to_restore)
        loader.restore(sess, CHECKPOINT_PATH)
        logging.info('Restore model weight from ' + CHECKPOINT_PATH)

        for i in xrange(options.numTestingImages):
            total_loss, losses, dataset, image_flags, gt, pred, debug, inp = \
                sess.run([loss, loss_list, dataset_flag, flags, gt_dict, pred_dict, debug_dict, input_dict])
            gt = {
                'corner': gt['corner'],
                'corner_values': gt['corner_values'],
                'icon': gt['icon'],
                'room': gt['room'],
                'density': debug['x0_topdown'][:, :, :, -1],
                'image_path': inp['image_path'],
                'num_corners': gt['num_corners'],
                'image_flags': image_flags
            }
            # Concatenate all prediction outputs and ground truth
            if i == 0:
                gt_all = gt
                pred_all = pred
            else:
                for k, v in gt.iteritems():
                    gt_all[k] = np.concatenate([gt_all[k], v], axis=0)
                for k, v in pred.iteritems():
                    pred_all[k] = np.concatenate([pred_all[k], v], axis=0)

    evaluate_batch(options, gt_all, pred_all)

def evaluate_batch(options, gt_dict, pred_dict):
    if options.separateIconLoss:
        # pred_icon_separate = softmax(np.load(options.test_dir.replace('wsf', 'wsf_loss3') + '/dummy/pred_dict.npy')[()]['icon'])
        raise NotImplementedError

    stat = {key: [0, 0, 0] for key in ('corner', 'edge', 'room', 'room++')}
    
    pred_wc = pred_dict['corner'][:, :, :, :NUM_WALL_CORNERS]
    pred_oc = pred_dict['corner'][:, :, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4]
    pred_ic = pred_dict['corner'][:, :, :, NUM_WALL_CORNERS + 4:NUM_WALL_CORNERS + 8]

    if options.branches != '5':
        pred_wc = sigmoid(pred_wc)
        pred_oc = sigmoid(pred_oc)
        pred_ic = sigmoid(pred_ic)
    else:
        raise NotImplementedError

    for batch_idx in xrange(gt_dict['corner'].shape[0]):
        if options.evaluateImage and gt_dict['image_flags'][batch_idx] == 0:
            raise NotImplementedError

        # Build density map for visualization
        density = np.minimum(gt_dict['density'][batch_idx] * 255, 255).astype(np.uint8)
        density = np.stack([density, density, density], axis=-1)
        
        pred_icon = softmax(pred_dict['icon'][batch_idx])
        pred_room = softmax(pred_dict['room'][batch_idx])

        if options.separateIconLoss:
            raise NotImplementedError

        print(batch_idx, 'start reconstruction', gt_dict['image_path'][batch_idx])
        debug_dir = os.path.join(options.test_dir, 'gt_debug_')
        output_dir = os.path.join(options.test_dir, 'gt_output_')
        num_corners = gt_dict['num_corners'][batch_idx]
        orientationCorners = getOrientationCorners(gt_dict['corner_values'][batch_idx][:num_corners])
        result_gt = reconstructFloorplan(
            orientationCorners[:NUM_WALL_CORNERS],
            orientationCorners[NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4],
            orientationCorners[NUM_WALL_CORNERS + 4:NUM_WALL_CORNERS + 8],
            segmentation2Heatmaps(gt_dict['icon'][batch_idx], NUM_ICONS),
            segmentation2Heatmaps(gt_dict['room'][batch_idx], NUM_ROOMS),
            output_prefix=output_dir,
            densityImage=density[:, :, 0],
            gt=True,
            debug_prefix=debug_dir)

        debug_dir = os.path.join(options.test_dir, 'pred_debug_')
        output_dir = os.path.join(options.test_dir, 'pred_output_')
        result_pred = reconstructFloorplan(
            pred_wc[batch_idx],
            pred_oc[batch_idx],
            pred_ic[batch_idx],
            pred_icon,
            pred_room,
            output_prefix=output_dir,
            densityImage=density[:, :, 0],
            gt_dict=result_gt,
            gt=False, debug_prefix=debug_dir)

        # TODO: Visualize result and fix visualize in reconstructFloorplan

        # Evaluate corner metric
        result = eval_corner_wrapper(
            result_pred, result_gt,
            dist_threshold=CORNER_DIST_THRESHOLD
        )
        stat['corner'][0] += result[0]
        stat['corner'][1] += result[1]
        stat['corner'][2] += result[2]
        
        result = eval_edges_wrapper(
            result_pred, result_gt,
            dist_threshold=CORNER_DIST_THRESHOLD
        )
        stat['edge'][0] += result[0]
        stat['edge'][1] += result[1]
        stat['edge'][2] += result[2]

        result = eval_room_wrapper(result_pred, result_gt, ROOM_IOU_THRESHOLD)
        stat['room'][0] += result[0]
        stat['room'][1] += result[1]
        stat['room'][2] += result[2]

        result = eval_room_pp_wrapper(result_pred, result_gt, ROOM_IOU_THRESHOLD)
        stat['room++'][0] += result[0]
        stat['room++'][1] += result[1]
        stat['room++'][2] += result[2]

        # visualize pred
        background = density[:, :, :3].copy()
        result_path = os.path.join(options.test_dir, '%s_pred_result.png' % batch_idx)
        result_img = draw_result(result_pred, background)
        cv2.imwrite(result_path, result_img)
        logging.info('Write visualization result to %s' % result_path)
        # visualize GT
        background = density[:, :, :3].copy()
        result_path = os.path.join(options.test_dir, '%s_gt_result.png' % batch_idx)
        result_img = draw_result(result_gt, background)
        cv2.imwrite(result_path, result_img)
        logging.info('Write visualization result to %s' % result_path)
    logging.info(stat)


def draw_result(result_dict, background):
    corner_color = np.array([255, 0, 0])
    wall_color = np.array([0, 255, 0])
    thinkness = 3
    point_list, wall_list, label_lsit = result_dict['wall']
    room_masks, _ = reconstruct_room(point_list, wall_list, label_lsit)

    color_map = ColorPalette(len(room_masks)).getColorMap()
    for i, room_mask in enumerate(room_masks):
        xs, ys = room_mask.nonzero()
        background[xs, ys, :] = color_map[i, :]

    for i, (x1, x2) in enumerate(wall_list):
        x1 = tuple(int(z) for z in point_list[x1][:2])
        x2 = tuple(int(z) for z in point_list[x2][:2])
        cv2.line(
            background, x1, x2,
            color=tuple(wall_color),
            thickness=thinkness)
    valid_point_list = filter_points(point_list, wall_list)

    for point in valid_point_list:
        x, y = int(point[1]), int(point[0])
        background[x-2:x+3, y-2:y+3, :] = corner_color

    return background

def eval_corner_wrapper(pred_dict, gt_dict, dist_threshold):
    # dict['wall']: (wall_points, wall_line, wall_labels)
    points_pred, walls_pred, _ = pred_dict['wall']
    points_gt, walls_gt, _ = gt_dict['wall']

    # Filter corners points that at least has one wall
    valid_points_gt = filter_points(points_gt, walls_gt)
    valid_points_pred = filter_points(points_pred, walls_pred)

    return evaluate_corners(valid_points_pred, valid_points_gt, dist_threshold)


def eval_edges_wrapper(pred_dict, gt_dict, dist_threshold):
    # dict['wall']: (wall_points, wall_line, wall_labels)
    points_pred, walls_pred, _ = pred_dict['wall']
    points_gt, walls_gt, _ = gt_dict['wall']

    return evaluate_edges(
        points_pred, points_gt, walls_pred, walls_gt, dist_threshold)


def eval_room_wrapper(pred_dict, gt_dict, iou_threshold):
    points_gt, walls_gt, labels_gt = gt_dict['wall']
    room_masks_gt, _ = reconstruct_room(points_gt, walls_gt, labels_gt)
    points_pred, walls_pred, labels_pred = pred_dict['wall']
    room_masks_pred, _ = reconstruct_room(points_pred, walls_pred, labels_pred)

    # Extract list of room_masks from room_info (dictionary)
    return evaluate_rooms(room_masks_pred, room_masks_gt, iou_threshold)


def eval_room_pp_wrapper(pred_dict, gt_dict, iou_threshold):
    points_gt, walls_gt, labels_gt = gt_dict['wall']
    room_masks_gt, room_adj_gt = reconstruct_room(points_gt, walls_gt, labels_gt)
    points_pred, walls_pred, labels_pred = pred_dict['wall']
    room_masks_pred, room_adj_pred = reconstruct_room(points_pred, walls_pred, labels_pred)

    return evaluate_room_plus_plus(
        room_masks_pred, room_adj_pred,
        room_masks_gt, room_adj_gt, iou_threshold
    )
