// Этим создадим ширину фотографии
let width = 640;
// Это будет вычисляться на основе входящего потока
let height = 0;

let video = undefined;
let loader = undefined;
let photo = undefined;
let canvas = undefined;
let scene = undefined;
let timeInSec = undefined;
let modelName = undefined;


let sceneIsHide = true;

function onLoad() {
    video = document.getElementById('video');
    loader = document.getElementById('loader');
    photo = document.getElementById('photo');
    canvas = document.getElementById('canvas');
    scene = document.getElementById('scene');
    timeInSec = document.getElementById('time-in-sec');
    modelName = document.getElementById('model-name');

    // Начало проигрывания видео потока с камеры
    navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
    }).then(function (stream) {
        video.srcObject = stream;
        video.play();
    }).catch(function (err) {
        console.log('An error occurred: ' + err);
    });
    //Можно начать воспроизведение видео
    video.addEventListener('canplay', canplay);
}

/*
* Периодическая отправка изображений в сокет
* */
function clientTask(ws) {
    const frame = takePicture()
    if (frame.length) {
        const model = document.querySelector('input[name="model"]:checked').value
        const json = JSON.stringify({
            model: model,
            image: frame
        })
        ws.send(json);
    }
}

/*
* Обработка ответа от сервера
* */
function serverHandler(jsonData) {
    photo.src = jsonData.image;
    timeInSec.innerHTML = jsonData.time
    modelName.innerHTML = jsonData.model
    if (sceneIsHide) {
        loader.classList.add('hide');
        scene.classList.remove('hide');
        sceneIsHide = false;
    }
}

/*
* Получение фрейма
* */
function takePicture() {
    if (width && height) {
        if (canvas && video) {
            const context = canvas.getContext('2d');

            canvas.width = width;
            canvas.height = height;
            context.drawImage(video, 0, 0, width, height);

            return canvas.toDataURL('image/png').toString();
        }
    }
    return ''
}


function connect(url, clientTask, serverHandler) {
    console.log('Connect to ' + url + '.');
    const ws = new WebSocket(url);

    ws.onopen = () => {
        console.log('Socket is opened.');
        setInterval(() => {
            if (ws.bufferedAmount === 0) {
                clientTask(ws);
            }
        }, 800);
    }

    ws.onmessage = message => {
        const jsonData = JSON.parse(message.data)
        if (Object.prototype.hasOwnProperty.call(jsonData, 'image')) {
            serverHandler(jsonData);
        } else if (Object.prototype.hasOwnProperty.call(jsonData, 'error')) {
            console.error(jsonData.error);
        }
    }

    ws.onerror = event => {
        console.error(event);
        ws.close();
    }

    ws.onclose = event => {
        console.log('Socket is closed. Reconnect will be attempted in 1 second.', event.reason);
        setTimeout(() => {
            connect(url, clientTask, serverHandler);
        }, 1000);
    }
}

/*
* Можно начать воспроизведение видео
* */
function canplay() {
    // Установка высоты и ширины для video и canvas
    height = video.videoHeight / (video.videoWidth / width);

    video.setAttribute('width', width.toString());
    video.setAttribute('height', height.toString());
    canvas.setAttribute('width', width.toString());
    canvas.setAttribute('height', height.toString());

    // Подключаемся к WebSocket
    // for develop
    // const ws_url = 'ws://127.0.0.1:8000/detector'
    // for nginx
    const ws_url = 'ws://' + window.location.host + '/detector'
    connect(ws_url, clientTask, serverHandler);
}