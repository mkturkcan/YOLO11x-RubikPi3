import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate
from PIL import Image, ImageDraw
import os, time, sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def curr_ms():
    return round(time.time() * 1000)

# Paths
IMAGE_IN = 'three-people-640-480.jpg'
IMAGE_OUT = 'three-people-640-480-overlay.jpg'
MODEL_PATH = 'yolo11x.tflite'
REPEATS = 10

# If we pass in --use-qnn we offload to NPU
use_qnn = True if len(sys.argv) >= 2 and sys.argv[1] == '--use-qnn' else False

experimental_delegates = []
if use_qnn:
    experimental_delegates = [load_delegate("libQnnTFLiteDelegate.so", options={"backend_type":"htp"})]

# Load TFLite model and allocate tensors
interpreter = Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=experimental_delegates
)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === BEGIN PREPROCESSING ===

# Load an image (using Pillow) and make it in the right format that the interpreter expects (e.g. quantize)
# All AI Hub image models use 0..1 inputs to start.
def load_image_litert(interpreter, path, single_channel_behavior: str = 'grayscale'):
    d = interpreter.get_input_details()[0]
    shape = [int(x) for x in d["shape"]]  # e.g. [1, H, W, C] or [1, C, H, W]
    dtype = d["dtype"]
    scale, zp = d.get("quantization", (0.0, 0))

    if len(shape) != 4 or shape[0] != 1:
        raise ValueError(f"Unexpected input shape: {shape}")

    # Detect layout
    if shape[1] in (1, 3):   # [1, C, H, W]
        layout, C, H, W = "NCHW", shape[1], shape[2], shape[3]
    elif shape[3] in (1, 3): # [1, H, W, C]
        layout, C, H, W = "NHWC", shape[3], shape[1], shape[2]
    else:
        raise ValueError(f"Cannot infer layout from shape {shape}")

    # Load & resize
    img = Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR)
    arr = np.array(img)
    if C == 1:
        if single_channel_behavior == 'grayscale':
            # Convert to luminance (H, W)
            gray = np.asarray(Image.fromarray(arr).convert('L'))
        elif single_channel_behavior in ('red', 'green', 'blue'):
            ch_idx = {'red': 0, 'green': 1, 'blue': 2}[single_channel_behavior]
            gray = arr[:, :, ch_idx]
        else:
            raise ValueError(f"Invalid single_channel_behavior: {single_channel_behavior}")
        # Keep shape as HWC with C=1
        arr = gray[..., np.newaxis]

    # HWC -> correct layout
    if layout == "NCHW":
        arr = np.transpose(arr, (2, 0, 1))  # (C,H,W)

    # Scale 0..1 (all AI Hub image models use this)
    arr = (arr / 255.0).astype(np.float32)

    # Quantize if needed
    if scale and float(scale) != 0.0:
        q = np.rint(arr / float(scale) + int(zp))
        if dtype == np.uint8:
            arr = np.clip(q, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(q, -128, 127).astype(np.int8)

    return np.expand_dims(arr, 0)  # add batch

# This model looks like grayscale, but AI Hub inference actually takes the BLUE channel
# see https://github.com/quic/ai-hub-models/blob/8cdeb11df6cc835b9b0b0cf9b602c7aa83ebfaf8/qai_hub_models/models/face_det_lite/app.py#L70
input_data = load_image_litert(interpreter, IMAGE_IN, single_channel_behavior='blue')

# === END PREPROCESSING (input_data contains right data) ===

# Set tensor and run inference
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run once to warmup
interpreter.invoke()

# Then run 10x
start = curr_ms()
for i in range(0, REPEATS):
    interpreter.invoke()
end = curr_ms()

# === BEGIN POSTPROCESSING ===

# Grab 3 output tensors and dequantize
q_output_0 = interpreter.get_tensor(output_details[0]['index'])
scale_0, zero_point_0 = output_details[0]['quantization']
pred_boxes = ((q_output_0.astype(np.float32) - zero_point_0) * scale_0)

q_output_1 = interpreter.get_tensor(output_details[1]['index'])
scale_1, zero_point_1 = output_details[1]['quantization']
pred_scores = ((q_output_1.astype(np.float32) - zero_point_1) * scale_1)

q_output_2 = interpreter.get_tensor(output_details[2]['index'])
scale_2, zero_point_2 = output_details[2]['quantization']
pred_class_idx = ((q_output_2.astype(np.float32) - zero_point_2) * scale_2)

print(pred_boxes.shape, pred_scores.shape, pred_class_idx.shape)
print('')
print(f'Inference took (on average): {(end - start) / REPEATS}ms. per image')

# CPU: Inference took (on average): 94.6ms. per image
# Inference took (on average): 6.4ms. per image



def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    Boxes are in format [x1, y1, x2, y2]
    """
    # Intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def nms_numpy(boxes, scores, pred_class_idx, iou_threshold=0.5):
    """
    Non-Maximum Suppression using numpy.
    
    Args:
        boxes: numpy array of shape (batch, N, 4) where boxes are [x1, y1, x2, y2]
        scores: numpy array of shape (batch, N) - confidence scores
        pred_class_idx: numpy array of shape (batch, N) - predicted class indices
        iou_threshold: float - IoU threshold for suppression
    
    Returns:
        filtered_boxes: list of numpy arrays, one per batch
        filtered_scores: list of numpy arrays, one per batch
        filtered_class_idx: list of numpy arrays, one per batch
    """
    batch_size = boxes.shape[0]
    
    filtered_boxes_batch = []
    filtered_scores_batch = []
    filtered_class_idx_batch = []
    
    for b in range(batch_size):
        batch_boxes = boxes[b]
        batch_scores = scores[b]
        batch_classes = pred_class_idx[b]
        
        # Filter out boxes with zero or negative scores
        valid_mask = batch_scores > 0
        batch_boxes = batch_boxes[valid_mask]
        batch_scores = batch_scores[valid_mask]
        batch_classes = batch_classes[valid_mask]
        
        if len(batch_boxes) == 0:
            filtered_boxes_batch.append(np.array([]))
            filtered_scores_batch.append(np.array([]))
            filtered_class_idx_batch.append(np.array([]))
            continue
        
        # Perform NMS per class
        unique_classes = np.unique(batch_classes)
        keep_indices = []
        
        for cls in unique_classes:
            # Get boxes for this class
            cls_mask = batch_classes == cls
            cls_boxes = batch_boxes[cls_mask]
            cls_scores = batch_scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]
            
            # Sort by scores in descending order
            sorted_indices = np.argsort(cls_scores)[::-1]
            
            # NMS for this class
            keep_cls = []
            while len(sorted_indices) > 0:
                # Pick the box with highest score
                current_idx = sorted_indices[0]
                keep_cls.append(cls_indices[current_idx])
                
                if len(sorted_indices) == 1:
                    break
                
                # Compute IoU with remaining boxes
                current_box = cls_boxes[current_idx]
                remaining_indices = sorted_indices[1:]
                remaining_boxes = cls_boxes[remaining_indices]
                
                # Vectorized IoU computation
                ious = np.array([compute_iou(current_box, box) for box in remaining_boxes])
                
                # Keep boxes with IoU less than threshold
                keep_mask = ious < iou_threshold
                sorted_indices = remaining_indices[keep_mask]
            
            keep_indices.extend(keep_cls)
        
        # Sort keep_indices to maintain order
        keep_indices = sorted(keep_indices)
        
        # Gather results
        filtered_boxes_batch.append(batch_boxes[keep_indices])
        filtered_scores_batch.append(batch_scores[keep_indices])
        filtered_class_idx_batch.append(batch_classes[keep_indices])
    
    return filtered_boxes_batch, filtered_scores_batch, filtered_class_idx_batch


def plot_boxes(image_path, boxes, scores, class_idx, output_path='output_with_boxes.png', 
               class_names=None, score_threshold=0.0):
    """
    Plot bounding boxes on an image and save the result.
    
    Args:
        image_path: str - path to input image
        boxes: numpy array of shape (N, 4) - boxes in [x1, y1, x2, y2] format
        scores: numpy array of shape (N,) - confidence scores
        class_idx: numpy array of shape (N,) - class indices
        output_path: str - path to save output image
        class_names: list - optional list of class names
        score_threshold: float - minimum score to display
    """
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_array)
    
    # Color map for different classes
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    # Plot each box
    for i, (box, score, cls) in enumerate(zip(boxes, scores, class_idx)):
        if score < score_threshold:
            continue
            
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Select color based on class
        color = colors[int(cls) % len(colors)]
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), width, height, 
                                 linewidth=2, edgecolor=color, 
                                 facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        if class_names is not None and int(cls) < len(class_names):
            label = f'{class_names[int(cls)]}: {score:.2f}'
        else:
            label = f'Class {int(cls)}: {score:.2f}'
        
        ax.text(x1, y1-5, label, color='white', fontsize=10,
               bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=2))
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Output saved to: {output_path}")

num_boxes = pred_boxes.shape[1]
# Apply NMS
print("Applying NMS...")
filtered_boxes, filtered_scores, filtered_classes = nms_numpy(
    pred_boxes, pred_scores, pred_class_idx, iou_threshold=0.5
)

print(f"Original boxes per batch: {num_boxes}")

# Example 2: Create a simple test image with boxes
print("\nCreating example visualization...")

# est_img = Image.open(IMAGE_IN)
# Create some example boxes for visualization (from first batch)
if len(filtered_boxes[0]) > 0:
    # Take up to 10 boxes for visualization
    num_vis = min(10, len(filtered_boxes[0]))
    vis_boxes = filtered_boxes[0][:num_vis]
    vis_scores = filtered_scores[0][:num_vis]
    vis_classes = filtered_classes[0][:num_vis]
    
    # Optional: define class names
    class_names = [f'Object_{i}' for i in range(10)]
    
    # Plot and save
    plot_boxes(IMAGE_IN, 
                vis_boxes, vis_scores, vis_classes,
                output_path=IMAGE_OUT,
                class_names=class_names,
                score_threshold=0.0)
else:
    print("No boxes to visualize after NMS")

print("\nExample complete!")
