import base64
import logging
from typing import Any

import cv2
import numpy as np
from aiohttp.web_exceptions import HTTPInternalServerError
from aiohttp.web_request import Request


def get_app_data(request: Request, key: str) -> Any:
    if key not in request.app:
        raise HTTPInternalServerError(text=f'App data "{key}" not found')
    return request.app[key]


def get_app_logger(request: Request) -> logging.Logger:
    return get_app_data(request, 'logger.server')


def from_b64(uri):
    """
    Convert from b64 uri to OpenCV image
    :param uri: base64 str, 'data:image/jpg;base64,/9j/4AAQSkZJR......'
    :return:
    """
    encoded_data = uri.split(',')[1]
    data = base64.b64decode(encoded_data)
    np_arr = np.fromstring(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def to_b64(img):
    """
    Convert from OpenCV image to b64 uri
    :param img: OpenCV image
    :return:
    """
    _, buffer = cv2.imencode('.jpg', img)
    uri = base64.b64encode(buffer).decode('utf-8')
    return f'data:image/jpg;base64,{uri}'


# def urljoin(separator: str, *args) -> str:
#     urls = list(filter(lambda url: url is not None, args))
#     urls = list(map(prepare_url, urls))
#     return separator.join(urls)
#
#
# def prepare_url(url: str) -> str:
#     if url.endswith('/'):
#         url = url[:-1]
#
#     if url.startswith('/'):
#         url = url[1:]
#
#     return url
