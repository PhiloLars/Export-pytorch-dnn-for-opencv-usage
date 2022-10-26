import numpy as np
import onnxruntime
import torchvision
import torch
import cv2

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main():
    batchSize = 1 
    channels = 3
    height = 224
    width = 224
    modelPath = "model.onnx"
    device = torch.device("cpu")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval()
    dummyInput = torch.randn(batchSize, channels, height, width, requires_grad=True).to(device)
    processedDummy = model(dummyInput)

    torch.onnx.export(model, 
            dummyInput,
            modelPath,
            verbose=True,
            export_params = True,
            do_constant_folding=True,
            opset_version=11,
            dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                        'output' : {0 : 'batch_size'}}
            )

    ort_session = onnxruntime.InferenceSession(modelPath)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummyInput)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(processedDummy[0]["boxes"]), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(processedDummy[0]["scores"]), ort_outs[1], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(processedDummy[0]["labels"]), ort_outs[2], rtol=1e-03, atol=1e-05)

    cv2.dnn.readNet(modelPath)

if(__name__ == "__main__"):
    main()