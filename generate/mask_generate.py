import argparse
import os
import glob
import json
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def extract_classes_from_filename(filename):
    """Extract class names from filename"""
    # Remove file extension
    basename = os.path.splitext(filename)[0]
    
    # Remove -neg suffix if exists
    if basename.endswith('-neg'):
        basename = basename[:-4]
    
    # Split class names by underscore
    classes = basename.split('_')
    
    return classes

def create_mask_image(masks, image_size):
    """Create mask image"""
    if masks is None or len(masks) == 0 or not isinstance(masks, (list, tuple)):
        return None
    
    # Validate image size
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        print("Error: Invalid image size")
        return None
    
    # Create blank mask
    try:
        mask_img = np.zeros(image_size, dtype=np.uint8)
        
        # Merge all masks
        valid_masks = 0
        for idx, mask in enumerate(masks):
            if mask is not None and hasattr(mask, 'cpu') and hasattr(mask, 'numpy'):
                mask_np = mask[0].cpu().numpy()
                if mask_np.shape == image_size:
                    mask_img[mask_np] = 255  # Set to white
                    valid_masks += 1
        
        if valid_masks == 0:
            print("Warning: No valid masks to merge")
            return None
            
        return mask_img
        
    except Exception as e:
        print(f"Failed to create mask image: {e}")
        return None

def save_mask_data(mask_img, output_path):
    """Save mask as PNG"""
    if mask_img is None:
        print("Error: Mask image is empty")
        return False
    
    # Validate mask image
    if not isinstance(mask_img, np.ndarray) or mask_img.dtype != np.uint8:
        print("Error: Invalid mask image format")
        return False
    
    if len(mask_img.shape) != 2 or mask_img.shape[0] == 0 or mask_img.shape[1] == 0:
        print("Error: Invalid mask image dimensions")
        return False
    
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Check file extension
        if not output_path.lower().endswith('.png'):
            output_path = output_path + '.png'
        
        # Save mask
        success = cv2.imwrite(output_path, mask_img)
        if success:
            print(f"Mask saved: {output_path}")
            return True
        else:
            print(f"Failed to save mask: Cannot write to file {output_path}")
            return False
        
    except Exception as e:
        print(f"Failed to save mask: {e}")
        return False
    

def process_single_image(image_path, dino_model, sam_predictor, output_dir="masks", 
                         box_threshold=0.3, text_threshold=0.25, device="cpu"):
    """Process single image"""
    try:
        # Extract classes from filename
        filename = os.path.basename(image_path)
        classes = extract_classes_from_filename(filename)
        
        print(f"Processing image: {filename}")
        print(f"Detected classes: {classes}")
        
        # Build text prompt
        text_prompt = ". ".join(classes)
        
        # Load image
        image_pil, image_tensor = load_image(image_path)
        
        # Use Grounding DINO to detect objects
        boxes_filt, pred_phrases = get_grounding_output(
            dino_model, image_tensor, text_prompt, box_threshold, text_threshold, device=device
        )
        
        if boxes_filt is None or len(boxes_filt) == 0:
            print(f"No objects detected: {filename}")
            return False
        
        print(f"Detected {len(boxes_filt)} objects")
        
        # Use SAM to segment objects
        image_cv2 = cv2.imread(image_path)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        sam_predictor.set_image(image_cv2)
        
        # Convert boxes format
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]
        
        boxes_filt = boxes_filt.cpu()
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image_cv2.shape[:2])
        
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        if masks is None or len(masks) == 0:
            print(f"Segmentation failed: {filename}")
            return False
        
        # Create mask image
        mask_img = create_mask_image(masks, image_cv2.shape[:2])
        
        if mask_img is None:
            print(f"Failed to create mask: {filename}")
            return False
        
        # Save mask
        mask_filename = f"{os.path.splitext(filename)[0]}_mask.png"
        mask_path = os.path.join(output_dir, mask_filename)
        
        return save_mask_data(mask_img, mask_path)
        
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="path to image directory")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="masks", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    input_dir = args.input_dir
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    print("Loading Grounding DINO model...")
    dino_model = load_model(config_file, grounded_checkpoint, device=device)
    
    print("Loading SAM model...")
    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} images")
    
    # Process images in batch
    success_count = 0
    for image_path in image_files:
        if process_single_image(image_path, dino_model, sam_predictor, output_dir, 
                              box_threshold, text_threshold, device):
            success_count += 1
    
    print(f"Processing completed! Successfully generated {success_count}/{len(image_files)} masks")

