import aiohttp
import cv2
import torchvision.transforms as tf
import torch
import numpy as np
from PIL import Image
from aiohttp import web

from api import utils

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
image_size = (640, 480)
CLASSES = np.asarray([
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
])
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


async def websocket_handler(request):
    logger = utils.get_app_logger(request)
    device = utils.get_app_data(request, 'device')

    transform = tf.Compose([
        tf.ToTensor(),
        tf.Normalize(*stats)])

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == 'close':
                await ws.close()
            else:
                try:
                    logger.debug('Received data')
                    data = await ws.receive_json()

                    image = utils.from_b64(data['image'])

                    orig = image.copy()

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    image = transform(image)
                    image = torch.unsqueeze(image, 0)

                    image = image.to(device)

                    model = utils.get_app_data(request, 'fasterrcnn_resnet50_fpn')

                    model.eval()
                    with torch.set_grad_enabled(False):
                        detections = model(image)[0]

                    for i in range(0, len(detections["boxes"])):
                        # extract the confidence (i.e., probability) associated with the
                        # prediction
                        confidence = detections["scores"][i]
                        # filter out weak detections by ensuring the confidence is
                        # greater than the minimum confidence
                        if confidence > 0.8:
                            # extract the index of the class label from the detections,
                            # then compute the (x, y)-coordinates of the bounding box
                            # for the object
                            idx = int(detections["labels"][i])
                            box = detections["boxes"][i].detach().cpu().numpy()
                            (startX, startY, endX, endY) = box.astype("int")
                            # display the prediction to our terminal
                            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                            # print("[INFO] {}".format(label))
                            # draw the bounding box and label on the image
                            cv2.rectangle(orig, (startX, startY), (endX, endY),
                                          COLORS[idx], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(orig, label, (startX, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                    data = utils.to_b64(orig)

                    await ws.send_json({
                        'image': data
                    })
                except Exception as ex:
                    logger.exception(ws.exception())
                    await ws.send_json({
                        'error': str(ex)
                    })
        elif msg.type == aiohttp.WSMsgType.ERROR:
            logger.exception(ws.exception())

    logger.debug('websocket connection closed')

    return ws
