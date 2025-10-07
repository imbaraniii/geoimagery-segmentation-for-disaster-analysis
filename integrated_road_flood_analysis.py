"""
Integrated Road-Flood Analysis Pipeline
========================================
This script combines:
1. Road segmentation from post-disaster images
2. Flood detection from pre/post image comparison
3. Safe road identification (roads not in flood zones)
4. Connected road network analysis
"""

import os
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import label

# Import existing modules
from models.siamese_change import SiameseUNetResNet50
from utils import load_tensor_rgb
from flood_filter import build_fused_score
from heatmap_vis import save_heatmap_standalone, save_heatmap_overlay

# Import road segmentation model creation
import sys
sys.path.append('predic')
from road_segmentation import create_road_segmentation_model


def predict_roads(image_path, model_path, model_type='pspnet', encoder_name='resnet50', 
                  image_size=384, device='cuda'):
    """
    Predict road segmentation from a single image.
    Returns binary road mask and probability map.
    """
    print(f"[Road Seg] Loading model from {model_path}")
    model = create_road_segmentation_model(model_type, encoder_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    # Resize to model input size
    image_resized = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    image_np = np.array(image_resized)
    
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Run prediction
    print("[Road Seg] Running prediction...")
    with torch.no_grad():
        output = model(image_tensor)
        prob_map = output.squeeze().cpu().numpy()
        pred_mask = (prob_map > 0.5).astype(np.uint8)
    
    # Resize back to original image size
    prob_map_full = cv2.resize(prob_map, original_size, interpolation=cv2.INTER_LINEAR)
    pred_mask_full = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    print(f"[Road Seg] Found {pred_mask_full.sum() / pred_mask_full.size * 100:.2f}% road pixels")
    
    return pred_mask_full, prob_map_full


def detect_flood(pre_path, post_path, args, device='cuda'):
    """
    Detect flood-affected areas using Siamese change detection.
    Returns flood probability heatmap and binary mask.
    """
    print("[Flood Det] Loading images...")
    pre_pil, pre_t = load_tensor_rgb(pre_path, args.size)
    post_pil, post_t = load_tensor_rgb(post_path, args.size)
    
    # Get original size from post image
    original_size = post_pil.size  # (W, H)
    
    # Load Siamese model
    print("[Flood Det] Loading Siamese model...")
    model = SiameseUNetResNet50(out_ch=1).to(device).eval()
    
    with torch.no_grad():
        prob_ = model(pre_t.to(device), post_t.to(device)).squeeze().cpu().numpy()
    
    # Prepare fused heat score
    pre_rgb_s = np.array(pre_pil.resize((args.size, args.size), Image.BILINEAR))
    post_rgb_s = np.array(post_pil.resize((args.size, args.size), Image.BILINEAR))
    
    score = build_fused_score(
        pre_rgb_s, post_rgb_s, prob_,
        w_L=args.w_L, w_edge=args.w_edge, w_prob=args.w_prob,
        use_water_gate=args.use_water_gate,
        blue_minus_green=args.blue_minus_green, max_r=args.max_r,
        p_low=args.p_low, p_high=args.p_high, gamma=args.gamma
    )
    
    # Resize to original size
    H, W = original_size[1], original_size[0]  # PIL size is (W, H)
    score_full = cv2.resize(score, (W, H), interpolation=cv2.INTER_LINEAR)
    
    # Binary mask
    thr_heat = float(np.percentile(score, args.bin_percentile))
    mask_heat = (score > thr_heat).astype(np.uint8)
    mask_heat_full = cv2.resize(mask_heat, (W, H), interpolation=cv2.INTER_NEAREST)
    
    print(f"[Flood Det] Found {mask_heat_full.sum() / mask_heat_full.size * 100:.2f}% flooded pixels")
    
    return score_full, mask_heat_full


def find_connected_components(binary_mask):
    """
    Find connected components in a binary mask.
    Returns labeled image and number of components.
    """
    # Use morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components
    labeled, num_features = label(cleaned_mask)
    
    return labeled, num_features


def filter_small_components(labeled_image, num_components, min_size=100):
    """
    Remove small connected components below a minimum size.
    """
    output = np.zeros_like(labeled_image)
    
    for i in range(1, num_components + 1):
        component = (labeled_image == i)
        if component.sum() >= min_size:
            output[component] = 1
    
    return output.astype(np.uint8)


def analyze_safe_roads(road_mask, flood_mask, min_component_size=100):
    """
    Identify roads that are NOT in flood zones and analyze connectivity.
    
    Args:
        road_mask: Binary road segmentation (1 = road, 0 = no road)
        flood_mask: Binary flood mask (1 = flooded, 0 = safe)
        min_component_size: Minimum pixels for a road segment to be kept
    
    Returns:
        safe_roads: Binary mask of safe roads
        connected_roads: Labeled connected road segments
        stats: Dictionary with statistics
    """
    # Find safe roads (roads that are not flooded)
    safe_roads = road_mask * (1 - flood_mask)
    
    # Find connected components in safe roads
    labeled_roads, num_components = find_connected_components(safe_roads)
    
    # Filter small disconnected segments
    filtered_roads = filter_small_components(labeled_roads, num_components, min_component_size)
    
    # Re-label after filtering
    labeled_roads_final, num_final = label(filtered_roads)
    
    # Calculate statistics
    total_road_pixels = road_mask.sum()
    flooded_road_pixels = (road_mask * flood_mask).sum()
    safe_road_pixels = safe_roads.sum()
    filtered_safe_pixels = filtered_roads.sum()
    
    stats = {
        'total_road_pixels': int(total_road_pixels),
        'flooded_road_pixels': int(flooded_road_pixels),
        'safe_road_pixels': int(safe_road_pixels),
        'filtered_safe_road_pixels': int(filtered_safe_pixels),
        'flood_impact_percent': float(flooded_road_pixels / total_road_pixels * 100) if total_road_pixels > 0 else 0,
        'num_connected_segments': int(num_final)
    }
    
    return filtered_roads, labeled_roads_final, stats


def create_visualization(post_image_path, road_mask, flood_mask, safe_roads, 
                        flood_score, output_path):
    """
    Create comprehensive visualization of the analysis.
    """
    # Load original post-disaster image
    post_img = cv2.imread(post_image_path)
    post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
    
    # Ensure all masks have the same size as the image
    h, w = post_img.shape[:2]
    road_mask = cv2.resize(road_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    flood_mask = cv2.resize(flood_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    safe_roads = cv2.resize(safe_roads.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    flood_score = cv2.resize(flood_score, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Create overlays
    # 1. All roads overlay (blue)
    road_overlay = post_img.copy()
    road_overlay[road_mask > 0] = road_overlay[road_mask > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
    
    # 2. Flood heatmap overlay
    flood_heatmap = cv2.applyColorMap((flood_score * 255).astype(np.uint8), cv2.COLORMAP_JET)
    flood_heatmap = cv2.cvtColor(flood_heatmap, cv2.COLOR_BGR2RGB)
    flood_overlay = cv2.addWeighted(post_img, 0.6, flood_heatmap, 0.4, 0)
    
    # 3. Combined overlay: Safe roads (green), Flooded roads (red)
    combined_overlay = post_img.copy()
    flooded_roads = road_mask * flood_mask
    combined_overlay[safe_roads > 0] = combined_overlay[safe_roads > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    combined_overlay[flooded_roads > 0] = combined_overlay[flooded_roads > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    
    # 4. Safe roads only
    safe_roads_overlay = post_img.copy()
    safe_roads_overlay[safe_roads > 0] = safe_roads_overlay[safe_roads > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    # Create combined figure
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(post_img)
    axes[0, 0].set_title('Post-Disaster Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(road_overlay)
    axes[0, 1].set_title('All Detected Roads (Blue)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(flood_overlay)
    axes[0, 2].set_title('Flood Affected Areas (Heatmap)', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(combined_overlay)
    axes[1, 0].set_title('Safe (Green) vs Flooded (Red) Roads', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(safe_roads_overlay)
    axes[1, 1].set_title('Safe Connected Roads Only', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Statistics panel
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[Visualization] Saved to {output_path}")
    plt.close()
    
    # Save individual outputs
    base_name = os.path.splitext(output_path)[0]
    cv2.imwrite(f"{base_name}_safe_roads.png", cv2.cvtColor(safe_roads_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{base_name}_combined.png", cv2.cvtColor(combined_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{base_name}_safe_roads_mask.png", safe_roads * 255)
    
    return road_overlay, flood_overlay, combined_overlay, safe_roads_overlay


def main(args):
    """
    Main pipeline execution.
    """
    print("=" * 80)
    print("INTEGRATED ROAD-FLOOD ANALYSIS PIPELINE")
    print("=" * 80)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output prefix
    if args.output_prefix:
        prefix = args.output_prefix
    else:
        prefix = os.path.splitext(os.path.basename(args.pre_image))[0]
    
    print(f"\n{'='*80}")
    print("STEP 1: Road Segmentation (Post-Disaster Image)")
    print(f"{'='*80}")
    road_mask, road_prob = predict_roads(
        args.post_image,
        args.road_model,
        args.model_type,
        args.encoder,
        args.image_size,
        device
    )
    
    print(f"\n{'='*80}")
    print("STEP 2: Flood Detection")
    print(f"{'='*80}")
    flood_score, flood_mask = detect_flood(args.pre_image, args.post_image, args, device)
    
    print(f"\n{'='*80}")
    print("STEP 3: Safe Road Analysis")
    print(f"{'='*80}")
    safe_roads, labeled_roads, stats = analyze_safe_roads(
        road_mask, flood_mask, args.min_road_size
    )
    
    # Print statistics
    print("\n" + "=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Total Road Pixels:          {stats['total_road_pixels']:,}")
    print(f"Flooded Road Pixels:        {stats['flooded_road_pixels']:,}")
    print(f"Safe Road Pixels:           {stats['safe_road_pixels']:,}")
    print(f"Filtered Safe Road Pixels:  {stats['filtered_safe_road_pixels']:,}")
    print(f"Flood Impact on Roads:      {stats['flood_impact_percent']:.2f}%")
    print(f"Connected Road Segments:    {stats['num_connected_segments']}")
    print("=" * 80)
    
    # Save statistics to file
    stats_path = os.path.join(args.output_dir, f"{prefix}_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write("ROAD-FLOOD ANALYSIS STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Pre-disaster Image:  {args.pre_image}\n")
        f.write(f"Post-disaster Image: {args.post_image}\n")
        f.write(f"Road Model:          {args.road_model}\n")
        f.write("=" * 80 + "\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n[Stats] Saved to {stats_path}")
    
    print(f"\n{'='*80}")
    print("STEP 4: Visualization")
    print(f"{'='*80}")
    output_viz_path = os.path.join(args.output_dir, f"{prefix}_analysis.png")
    create_visualization(
        args.post_image,
        road_mask,
        flood_mask,
        safe_roads,
        flood_score,
        output_viz_path
    )
    
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output prefix:    {prefix}")
    print(f"\nKey outputs:")
    print(f"  - {prefix}_analysis.png         (Complete visualization)")
    print(f"  - {prefix}_safe_roads.png       (Safe roads overlay)")
    print(f"  - {prefix}_combined.png         (Safe vs flooded roads)")
    print(f"  - {prefix}_safe_roads_mask.png  (Binary safe roads mask)")
    print(f"  - {prefix}_statistics.txt       (Detailed statistics)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Integrated Road-Flood Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python integrated_road_flood_analysis.py \\
        --pre_image pre/hurricane-florence_00000018_pre_disaster.png \\
        --post_image post/hurricane-florence_00000018_post_disaster.png \\
        --road_model predic/deepglobe_models/road_seg_pspnet_resnet50_deepglobe.pth \\
        --model_type pspnet \\
        --encoder resnet50 \\
        --output_dir output/integrated_analysis
        """
    )
    
    # Input images
    parser.add_argument('--pre_image', type=str, required=True,
                       help='Path to pre-disaster image')
    parser.add_argument('--post_image', type=str, required=True,
                       help='Path to post-disaster image')
    
    # Road segmentation model
    parser.add_argument('--road_model', type=str, required=True,
                       help='Path to trained road segmentation model')
    parser.add_argument('--model_type', type=str, default='pspnet',
                       choices=['unet', 'deeplabv3', 'pspnet'],
                       help='Road segmentation model type')
    parser.add_argument('--encoder', type=str, default='resnet50',
                       help='Encoder backbone (e.g., resnet50, efficientnet-b0)')
    parser.add_argument('--image_size', type=int, default=384,
                       help='Input size for road segmentation model')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='output/integrated_analysis',
                       help='Directory to save outputs')
    parser.add_argument('--output_prefix', type=str, default='',
                       help='Prefix for output files (default: pre-image basename)')
    
    # Processing parameters
    parser.add_argument('--size', type=int, default=256,
                       help='Processing size for flood detection')
    parser.add_argument('--min_road_size', type=int, default=100,
                       help='Minimum pixels for a road segment to be kept')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage (default: use GPU if available)')
    
    # Flood detection parameters (from infer_siamese.py)
    parser.add_argument('--w_L', type=float, default=0.45,
                       help='Weight for lightness in flood fusion')
    parser.add_argument('--w_edge', type=float, default=0.25,
                       help='Weight for edges in flood fusion')
    parser.add_argument('--w_prob', type=float, default=0.30,
                       help='Weight for model probability in flood fusion')
    parser.add_argument('--use_water_gate', action='store_true',
                       help='Use water prior gating')
    parser.add_argument('--blue_minus_green', type=int, default=6,
                       help='Blue-green threshold for water detection')
    parser.add_argument('--max_r', type=int, default=205,
                       help='Maximum red value for water detection')
    parser.add_argument('--p_low', type=float, default=2.0,
                       help='Lower percentile for contrast adjustment')
    parser.add_argument('--p_high', type=float, default=98.0,
                       help='Higher percentile for contrast adjustment')
    parser.add_argument('--gamma', type=float, default=0.6,
                       help='Gamma correction value')
    parser.add_argument('--bin_percentile', type=float, default=80.0,
                       help='Percentile threshold for binary flood mask')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.pre_image):
        print(f"Error: Pre-disaster image not found: {args.pre_image}")
        exit(1)
    if not os.path.exists(args.post_image):
        print(f"Error: Post-disaster image not found: {args.post_image}")
        exit(1)
    if not os.path.exists(args.road_model):
        print(f"Error: Road model not found: {args.road_model}")
        exit(1)
    
    main(args)
