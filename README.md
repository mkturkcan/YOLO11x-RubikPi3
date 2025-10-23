# YOLO11x-RubikPi3
This repository is a walkthrough for running extra-large-scale YOLO models in real-time on Rubik Pi boards.

# Installation

In the Rubik Pi (after flashing Ubuntu),

```bash
# Create a new venv
python3 -m venv .venv
source .venv/bin/activate

# Install the LiteRT runtime (to run models) and Pillow (to parse images)
pip3 install ai-edge-litert==1.3.0 Pillow

# Download an example image
wget https://cdn.edgeimpulse.com/qc-ai-docs/example-images/three-people-640-480.jpg
wget https://cdn.edgeimpulse.com/qc-ai-docs/wheels/onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl
pip3 install onnxruntime_qnn-1.23.0-cp312-cp312-linux_aarch64.whl
```
## Create Your Own Quantized Weights (Optional)
In your host computer, after creating a Qualcomm ID and logging into Qualcomm AI Hub,
```bash
pip install qai_hub_models
qai-hub configure --api_token API_TOKEN
python -m qai_hub_models.models.yolov11_det.export --quantize w8a8 --target-runtime tflite
```

## Download Pre-Quantized Weights

We provide pre-quantized weights at https://huggingface.co/mehmetkeremturkcan/RubikPi.YOLO11x.Detection/tree/main .

## Run Script in the Rubik Pi 3

First, run on CPU to test:
```python
python run.py # runs on cpu
```
And then:
```python
python run.py --use-qnn
```
