Dataset Link: [GDRIVE](https://drive.google.com/drive/folders/1BImYeNtnf0HHtI5FbmBShatvGbMalCEK?usp=sharing)

## Installation

Install the dependencies: 
```bash
pip install -r requirements.txt
```

## Usage

### 1. Flood Detection Only
Run change detection to identify flooded areas:
```bash
python infer_siamese.py --pre_dir data/pre --post_dir data/post
```

### 2. Integrated Road-Flood Analysis (Recommended)
**NEW:** Run the complete pipeline that combines road segmentation and flood detection to identify safe, navigable roads:

```bash
python integrated_road_flood_analysis.py \
    --pre_image pre/hurricane-florence_00000018_pre_disaster.png \
    --post_image post/hurricane-florence_00000018_post_disaster.png \
    --road_model predic/deepglobe_models/road_seg_pspnet_resnet50_deepglobe.pth \
    --model_type pspnet \
    --encoder resnet50 \
    --output_dir output/integrated_analysis
```





This pipeline will:
- Detect roads in the post-disaster image
- Identify flooded areas by comparing pre/post images
- Exclude roads that are in flood zones
- Show connected, safe road networks
- Generate comprehensive visualizations and statistics

For detailed documentation and more options, see [RUN_INTEGRATED_PIPELINE.md](RUN_INTEGRATED_PIPELINE.md)

## Output

The integrated pipeline generates:
- `*_analysis.png` - Complete 6-panel visualization
- `*_safe_roads.png` - Safe roads overlay (green)
- `*_combined.png` - Safe (green) vs flooded (red) roads
- `*_safe_roads_mask.png` - Binary mask of safe roads
- `*_statistics.txt` - Detailed analysis statistics
