Step 1: Instal tensort, onnx
following:https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/01-how-to-build/linux-x86_64.md

Step2: install mmdetection
- cd mmdetection
- pip install -e .

Step 3: install MMDeploy:
- cd MMDeploy
- pip install -e .

Step 4: Download model
- scp seta2023@192.168.80.167:~/MMDeploy/model_convert/end2end.engine MMDeploy/model_convert/
- pass: ask Huyen

Step 5: run inference.ipybn code
