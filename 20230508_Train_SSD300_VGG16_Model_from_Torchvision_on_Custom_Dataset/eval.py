import torch
from torchvision.ops import box_iou

from tqdm import tqdm
from config import DEVICE, NUM_CLASSES, NUM_WORKERS
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import create_model
from datasets import create_valid_dataset, create_valid_loader
import numpy as np

# Evaluation function
def validate(valid_data_loader, model):
    print('Validating')
    model.eval()
    
    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        filtered_output_list = []

        for output in outputs:
            # Filter the scores and corresponding boxes for each output object
            valid_mask = output['scores'] >= 0.3
            filtered_output = {
                'boxes': output['boxes'][valid_mask],
                'scores': output['scores'][valid_mask],
                'labels': output['labels'][valid_mask]
            }
            filtered_output_list.append(filtered_output)

        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = filtered_output_list[i]['boxes'].detach().cpu()
            preds_dict['scores'] = filtered_output_list[i]['scores'].detach().cpu()
            preds_dict['labels'] = filtered_output_list[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
        #####################################

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary


def validate_accuracy(valid_data_loader, model, iou_threshold=0.5):
    print('Validating')
    model.eval()

    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    total_targets = 0
    correct_predictions = 0

    for data in prog_bar:
        images, targets = data

        images = [image.to(DEVICE) for image in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images, targets)

        for i in range(len(images)):
            # Get predicted and target boxes.
            pred_boxes = outputs[i]['boxes'].detach().cpu()
            true_boxes = targets[i]['boxes'].detach().cpu()

            if len(pred_boxes) == 0 or len(true_boxes) == 0:
                continue

            # Compute IoU between predicted and target boxes.
            iou_matrix = box_iou(true_boxes, pred_boxes)

            # Check for each target if it has at least one predicted box with IoU >= threshold.
            for j in range(len(true_boxes)):
                iou_values = iou_matrix[j]

                # Check if any predicted box overlaps with this target box based on IoU threshold.
                if (iou_values >= iou_threshold).any():
                    correct_predictions += 1

            total_targets += len(true_boxes)

    # Calculate accuracy as the percentage of targets that were correctly predicted.
    accuracy = correct_predictions / total_targets if total_targets > 0 else 0
    return accuracy

if __name__ == '__main__':
    # Load the best model and trained weights.
    model = create_model(num_classes=NUM_CLASSES, size=640)
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    test_dataset = create_valid_dataset(
        'data/test'
    )
    test_loader = create_valid_loader(test_dataset, num_workers=NUM_WORKERS)

    # metric_summary = validate(test_loader, model)
    custom_summary = validate_accuracy(test_loader, model, 0.25)
    # print(f"mAP_50: {metric_summary['map_50']*100:.3f}")
    # print(f"mAP_50_95: {metric_summary['map']*100:.3f}")
    print(custom_summary)