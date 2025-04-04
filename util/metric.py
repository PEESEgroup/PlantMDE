import torch
import numpy as np

def align_to_range_torch(arr, target_range, mask):
    X_min, X_max = torch.min(arr[mask]), torch.max(arr[mask])  # Ignore NaNs in min/max calculation
    Y_min, Y_max = torch.min(target_range[mask]), torch.max(target_range[mask])  # Range from target array
    
    # Create a copy of the array to preserve NaN values
    normalized_arr = arr.clone()

    # Avoid division by zero if X_max == X_min
    if X_max == X_min:
        normalized_arr[mask] = (Y_max + Y_min) / 2
    else:
        normalized_arr[mask] = (arr[mask] - X_min) / (X_max - X_min) * (Y_max - Y_min) + Y_min
    
    return normalized_arr, target_range

def align_to_range(arr, target_range, mask):
    X_min, X_max = np.nanmin(arr), np.nanmax(arr)  # Ignore NaNs in min/max calculation
    Y_min, Y_max = np.nanmin(target_range), np.nanmax(target_range)  # Range from target array
    
    # Create a copy of the array to preserve NaN values
    normalized_arr = np.copy(arr)

    # Avoid division by zero if X_max == X_min
    if X_max == X_min:
        normalized_arr[mask] = (Y_max + Y_min) / 2
    else:
        normalized_arr[mask] = (arr[mask] - X_min) / (X_max - X_min) * (Y_max - Y_min) + Y_min
    
    return normalized_arr

def align_by_lstsq(pred, target, valid_mask):
    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred)
    if not torch.is_tensor(target):
        target = torch.from_numpy(target)
        
    A_torch = torch.stack([
        pred[valid_mask],
        torch.ones_like(pred[valid_mask])
    ], dim=1)  # Shape: [N, 2]

    y_torch = target[valid_mask] 

    result = torch.linalg.lstsq(A_torch, y_torch)
    m, c = result.solution 
    scaled_pred = m * pred + c

    return scaled_pred.to(pred.device), target

def align_by_normalization(pred, target, valid_mask):
    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred)
    if not torch.is_tensor(target):
        target = torch.from_numpy(target)

    pred_min = torch.min(pred[valid_mask])
    pred_max = torch.max(pred[valid_mask])
    target_min = torch.min(target[valid_mask])
    target_max = torch.max(target[valid_mask])

    nor_pred = pred.clone()
    nor_target = target.clone()
    nor_pred[valid_mask] = (pred[valid_mask] - pred_min) / (pred_max - pred_min + 1e-6)
    nor_target[valid_mask] = (target[valid_mask] - target_min) / (target_max - target_min + 1e-6)

    return nor_pred, nor_target

def pearson_correlation(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    
    x_std = torch.std(x, unbiased=False)
    y_std = torch.std(y, unbiased=False)
    
    covariance = torch.mean((x - x_mean) * (y - y_mean))
    correlation = covariance / torch.clamp_min(x_std * y_std, 1e-10)
    
    return correlation

def eval_rel_depth(pred, target, valid_mask, noise_percentile=[0,1], align_func='lstsq'):
    assert pred.shape == target.shape

    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred)
    if not torch.is_tensor(target):
        target = torch.from_numpy(target)
    if not torch.is_tensor(valid_mask):
        valid_mask = torch.from_numpy(valid_mask)

    outlier_filtered = (pred > torch.quantile(pred[valid_mask], noise_percentile[0])) & \
                                (pred < torch.quantile(pred[valid_mask], noise_percentile[1]))
    valid_mask = valid_mask & outlier_filtered
    if align_func == 'lstsq':
        scaled_pred, _ = align_by_lstsq(pred, target, valid_mask)
    elif align_func == 'normal':
        scaled_pred, target = align_by_normalization(pred, target, valid_mask)
    elif align_func == 'mix':
        scaled_pred, target = align_by_normalization(pred, target, valid_mask)
        scaled_pred, target = align_by_lstsq(scaled_pred, target, valid_mask)
    elif align_func == 'to_range':
        scaled_pred, _ = align_to_range_torch(pred, target, valid_mask)

    scaled_pred = scaled_pred[valid_mask]
    target = target[valid_mask] 

    epsilon = 1e-10
    scaled_pred = torch.clamp(scaled_pred, min=epsilon)
    target = torch.clamp(target, min=epsilon)

    thresh = torch.max((target / scaled_pred), (scaled_pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = scaled_pred - target
    diff_log = torch.log(scaled_pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(scaled_pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}

def eval_rel_depth_segmentwise(pred, target, valid_mask, segmentation, noise_percentile=[0.05,1], retrun_segmentwise_metrics=False,):
    assert pred.shape == target.shape

    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred)
    if not torch.is_tensor(target):
        target = torch.from_numpy(target)
    if not torch.is_tensor(valid_mask):
        valid_mask = torch.from_numpy(valid_mask)
    if not torch.is_tensor(segmentation):
        segmentation = torch.from_numpy(segmentation)


    unique_segments = torch.unique(segmentation[valid_mask])
    metrics_list = []

    for segment_id in unique_segments:
        segment_mask = valid_mask & (segmentation == segment_id)
        segment_pred = pred[segment_mask]
        segment_target = target[segment_mask]
        segment_mask =  torch.ones_like(segment_mask)[segment_mask]

        outlier_filtered = (segment_pred > torch.quantile(segment_pred[segment_mask], noise_percentile[0])) & \
                                (segment_pred < torch.quantile(segment_pred[segment_mask], noise_percentile[1]))
        segment_mask = segment_mask & outlier_filtered
        if segment_mask.sum() < 100:
            continue
        # scaled_pred, scaled_target = align_to_range_torch(segment_pred, segment_target, segment_mask)
        scaled_pred, scaled_target = align_by_normalization(segment_pred, segment_target, segment_mask)
        
        scaled_pred = scaled_pred[segment_mask]
        scaled_target = scaled_target[segment_mask] 
        pearson = pearson_correlation(scaled_pred, scaled_target)

        diff = scaled_pred - scaled_target
        l1 = torch.mean(torch.abs(diff))
        rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
        metrics_list.append({'seg_l1':l1.item(), 'seg_rmse':rmse.item(), 'seg_pearson':pearson.item(), 'segment_id':segment_id.item()}) 

    if retrun_segmentwise_metrics:
        return metrics_list

    if len(metrics_list) == 0:
        return {'seg_l1':0, 'seg_rmse':0, 'seg_pearson':0}
    else:
        # Average metrics across segments
        metrics_avg = {}
        for key in metrics_list[0].keys():
            if key == 'segment_id':
                continue
            metrics_avg[key] = sum(m[key] for m in metrics_list) / len(metrics_list)

        return metrics_avg
    
def eval_rel_depth_seg(pred, target, valid_mask, segmentation, depth_range=[0.1,0.99], seg_range=[0.05,0.98]):
    assert pred.shape == target.shape

    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred)
    if not torch.is_tensor(target):
        target = torch.from_numpy(target)
    if not torch.is_tensor(valid_mask):
        valid_mask = torch.from_numpy(valid_mask)
    if not torch.is_tensor(segmentation):
        segmentation = torch.from_numpy(segmentation)

    res = eval_rel_depth(pred, target, valid_mask, depth_range, align_func='lstsq')
    res_seg = eval_rel_depth_segmentwise(pred, target, valid_mask, segmentation, seg_range)
    res.update(res_seg)
    return res
