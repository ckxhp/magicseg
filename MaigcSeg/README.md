# MagicSeg

## Overview



1. **Training Code**: Based on  [ZegCLIP]

   

2. **Generation Code**: Pipeline for text, image, and mask generation(base on [grounded sam]: https://github.com/IDEA-Research/Grounded-Segment-Anything)

## Project Structure

```
MaigcSeg/
├── example/                 # Example data
│   ├── images/             # Original images
│   ├── images_neg/         # Negative images (filename + '-neg')
│   └── masks/              # Generated masks
├── generate/               # Generation pipeline
│   ├── generate_texts.py   # Text generation using GPT
│   ├── generate_images.py  # Image generation using SD1.5
│   └── mask_generate.py    # Mask generation using GroundingDINO+SAM
└── train_code/            # Training framework
    └── ZegCLIP-main/      # Modified ZegCLIP codebase
```

## Training Configuration

### Key Modifications

The training framework has been enhanced with the following features:

#### 1. Dynamic Text Feature Construction
- **File**: `models/segmentor/zegclip.py`
- **Function**: `forward_train()` method
- **Features**:
  - Extracts class names from image filenames (max 2 classes separated by '_')
  - Samples additional classes to reach 100 total classes per image
  - Constructs new text_feat with shape [bs, 100, dim]


#### 2. Contrastive Loss
- **File**: `models/decode_heads/decode_seg.py`
- **Function**: Added cosine similarity loss
- **Formula**: `max(0, cos(cls_token, cls_token_neg))`
- **Integration**: Added to losses dictionary

### Training Commands

```bash
bash dist_train.sh configs/magicseg/vpt_seg_fully_vit-b_512x512_20k_12_10.py Path/to/magicseg/fully

```

## Generation Pipeline

### 1. Text Generation

**File**: `generate/generate_texts.py`

```bash
python generate_texts.py
```

### 2. Image Generation

**File**: `generate/generate_images.py`

```bash
python generate_images.py
```

### 3. Mask Generation

**File**: `generate/mask_generate.py`

ref to  

[grounded sam]: https://github.com/IDEA-Research/Grounded-Segment-Anything



## Requirements

ref to 

[ZegCLIP]: https://github.com/ZiqinZhou66/ZegCLIP





## Citation

If you use MagicSeg in your research, please cite:



## License

This project is built upon ZegCLIP, Grounged-Segment-Anything and follows the same license terms. Please refer to the original ZegCLIP repository for licensing information.
