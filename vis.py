import matplotlib
from PIL import Image
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import os

def render_depth(arr, mask=None, colormap_name="Spectral_r", grayscale=False):
    if mask is not None:
        arr[mask] = np.nan
    cmap = matplotlib.colormaps.get_cmap(colormap_name)
    depth = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
    if grayscale:
        depth = depth
    else:
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        if mask is not None:
            depth[mask] = [255,255,255]
    return depth

def depth_to_cloud_point(rgb_img, depth,focal_length_x, focal_length_y):
    width, height = rgb_img.shape[1], rgb_img.shape[0]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - width / 2) / focal_length_x
    y = - (y - height / 2) / focal_length_y
    z = - np.array(depth)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), -z), axis=-1).reshape(-1, 3)
    colors = np.array(rgb_img).reshape(-1, 3) / 255.0
    # Remove points where depth is NaN
    valid_mask = ~np.isnan(points).any(axis=1)
    points = points[valid_mask]
    colors = colors[valid_mask]

    return points.astype(np.float16), colors.astype(np.float16)

def review_pcd(points, colors, density=20):

    x = points[range(0, len(points), density), 0]
    y = - points[range(0, len(points), density), 1]
    z = points[range(0, len(points), density), 2]
    c = colors[range(0, len(points), density), :] 

    fig, axs = plt.subplots(ncols=3, figsize=(12, 3))

    axs[0].scatter(x, z, c=c, s=1)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('z')
    axs[0].set_title('Side-view (x,z)')
    axs[0].set_aspect('equal', adjustable='box')

    axs[1].scatter(y, z, c=c, s=1)
    axs[1].set_xlabel('y')
    axs[1].set_ylabel('z')
    axs[1].set_title('Side-view (y,z)')
    axs[1].set_aspect('equal', adjustable='box')

    axs[2].scatter(x, y, c=c, s=1)
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')
    axs[2].set_title('Top-view (x, y)')
    axs[2].set_aspect('equal', adjustable='box')

    # return fig


def show_pairs(rgb_path, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    img = plt.imread(rgb_path)
    # Add an alpha channel to the RGB image
    img = np.concatenate([img, np.ones((*img.shape[:2], 1))], axis=-1)
    
    # Display depth image
    depth_file = rgb_path.replace('rgb', 'depth')
    depth_img = cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR |cv2.IMREAD_UNCHANGED).astype(np.float32)        
    mask = (depth_img == 0)
    depth_img[mask] = np.nan
    norm = plt.Normalize(vmin=np.nanmin(depth_img), vmax=np.nanmax(depth_img))
    cm = plt.cm.ScalarMappable(norm=norm, cmap='Spectral_r')
    depth_rgb = cm.to_rgba(depth_img)
    depth_rgb[mask,:] = [0, 0, 0, 0.7]

    seg_file = rgb_path.replace('rgb', 'segmentation')
    seg_img = cv2.imread(seg_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_UNCHANGED)
    norm = plt.Normalize(vmin=np.nanmin(seg_img), vmax=np.nanmax(seg_img))
    cm = plt.cm.ScalarMappable(norm=norm, cmap='nipy_spectral')
    seg_rgb = cm.to_rgba(seg_img)

    combined = np.concatenate((img, depth_rgb, seg_rgb), axis=1)
    ax.imshow(combined)
    ax.axis('off')