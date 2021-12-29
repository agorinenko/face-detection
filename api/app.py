import logging
import ssl

import torch
from aiohttp import web
from envparse import env

from api.middlewares import error_middleware
from api.websocket import websocket_handler
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def create_app() -> web.Application:
    ssl._create_default_https_context = ssl._create_unverified_context

    env.read_envfile(path='./.env')

    log_level: str = env.str('LOG_LEVEL', default='error')
    logging.basicConfig(level=log_level.upper())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    application = web.Application(middlewares=[error_middleware])
    application['logger.server'] = logging.getLogger('aiohttp.server')
    application['fasterrcnn_resnet50_fpn'] = fasterrcnn_resnet50_fpn(pretrained=True, progress=False).to(device)
    application['device'] = device
    application.add_routes([web.get('/detector', websocket_handler)])

    return application


app = create_app()
