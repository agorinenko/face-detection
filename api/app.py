import logging
import ssl

import torch
from aiohttp import web
from envparse import env
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, fasterrcnn_mobilenet_v3_large_fpn, \
    fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn, keypointrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, \
    ssdlite320_mobilenet_v3_large, ssd300_vgg16

from api.middlewares import error_middleware
from api.websocket import websocket_handler

MODELS = {
    'fasterrcnn_resnet50_fpn': fasterrcnn_resnet50_fpn,
    'fasterrcnn_mobilenet_v3_large_fpn': fasterrcnn_mobilenet_v3_large_fpn,
    'fasterrcnn_mobilenet_v3_large_320_fpn': fasterrcnn_mobilenet_v3_large_320_fpn,
    'retinanet_resnet50_fpn': retinanet_resnet50_fpn,
    'keypointrcnn_resnet50_fpn': keypointrcnn_resnet50_fpn,
    'maskrcnn_resnet50_fpn': maskrcnn_resnet50_fpn,
    'ssdlite320_mobilenet_v3_large': ssdlite320_mobilenet_v3_large,
    'ssd300_vgg16': ssd300_vgg16
}


def create_app() -> web.Application:
    ssl._create_default_https_context = ssl._create_unverified_context

    env.read_envfile(path='./.env')

    log_level: str = env.str('LOG_LEVEL', default='error')
    logging.basicConfig(level=log_level.upper())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    application = web.Application(middlewares=[error_middleware])
    application['logger.server'] = logging.getLogger('aiohttp.server')

    for name, model in MODELS.items():
        application[name] = model(pretrained=True, progress=False).to(device)

    application['device'] = device
    application.add_routes([web.get('/detector', websocket_handler)])

    return application


app = create_app()
