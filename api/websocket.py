import time
from typing import Tuple, Optional

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
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
])
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


async def websocket_handler(request):
    logger = utils.get_app_logger(request)
    device = utils.get_app_data(request, 'device')

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

                    time_1 = time.monotonic()

                    image = utils.from_b64(data['image'])
                    origin_image = image.copy()

                    model_name = data['model']
                    # size = None
                    # if model_name == 'ssdlite320_mobilenet_v3_large':
                    #     size = (320, 320)
                    # elif model_name == 'ssd300_vgg16':
                    #     size = (300, 300)

                    image = _image_to_tensor(image).to(device)

                    model = utils.get_app_data(request, model_name)

                    boxes, scores, labels = _predict(model, image)

                    for i, box in enumerate(boxes):
                        score = scores[i]
                        if score > 0.8:
                            origin_image = _update_origin_image(origin_image, box, score, labels[i])

                    data = utils.to_b64(origin_image)

                    time_2 = time.monotonic()

                    await ws.send_json({
                        'time': time_2 - time_1,
                        'model': model_name,
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


def _update_origin_image(origin_image, box, score, label):
    """
    Добавление к изображению рамки и текста с вероятностью и классом
    :param origin_image: изображение
    :param box: предсказанная рамка
    :param score: вероятность того, что предсказание верное
    :param label: метка класса
    :return:
    """
    class_idx = int(label)

    box = box.detach().cpu().numpy()
    start_x, start_y, end_x, end_y = box.astype("int")
    class_name = CLASSES[class_idx]
    class_color = COLORS[class_idx]
    label = "{}: {:.2f}%".format(class_name, score * 100)

    cv2.rectangle(origin_image, (start_x, start_y), (end_x, end_y), class_color, 2)
    y = start_y - 15 if start_y - 15 > 15 else start_y + 15
    cv2.putText(origin_image, label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2)

    return origin_image


def _predict(model, image: torch.Tensor):
    """
    Предсказание
    :param model: модель
    :param image: изображение
    :return:
    """
    model.eval()
    with torch.set_grad_enabled(False):
        detections = model(image)[0]

    return detections["boxes"], detections["scores"], detections["labels"]


def _image_to_tensor(image, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Конвертация изображение в тензор
    :param image: изображение
    :return:
    """
    transformers = [tf.ToTensor()]

    if size:
        transformers.append(tf.Resize(size))

    transformers.append(tf.Normalize(*stats))

    transform = tf.Compose(transformers)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image)
    image = torch.unsqueeze(image, 0)

    return image
