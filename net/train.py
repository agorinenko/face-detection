import datetime
import os
import pathlib

import torch
from torch.optim import lr_scheduler
from tqdm import tqdm

from net.data_set import load_coco_data_set


# stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
# image_size = 300


def load_state(checkpoint_path, optimizer, model):
    """ Функция загрузки сохраненного состояния модели и прогресса обучения """
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['losses'], checkpoint['scores']


def save_state(checkpoint_path, optimizer, model, losses, scores):
    """ Функция сохранения состояния модели и прогресса обучения """
    if checkpoint_path:
        os.remove(checkpoint_path)

    torch.save({
        'losses': losses,
        'scores': scores,
        'm_discriminator_state_dict': model.state_dict(),
        'o_generator_state_dict': optimizer.state_dict(),
    }, checkpoint_path)


def fit_epoch(model, train_loader, criterion, optimizer, device):
    """
    Обучение на одной эпохе
    :param model: модель
    :param train_loader: тренировочная выборка
    :param criterion: функция потерь
    :param optimizer: оптимизатор
    :return: суммарные потери модели и метрика качества
    """
    model.train()  # train mode
    running_loss = 0.0
    processed_data = 0

    for images, boxes, labels in train_loader:
        inputs = images.to(device)
        boxes = [torch.FloatTensor(b).to(device) for b in boxes]
        labels = [torch.FloatTensor(l).to(device) for l in labels]

        # boxes = torch.FloatTensor(boxes).to(device)
        # labels = torch.FloatTensor(labels).to(device)
        # Зануляем градиенты оптимизатора
        optimizer.zero_grad()
        # Делаем предсказания
        predicted_locs, predicted_scores = model(inputs)

        # Считаем функцию потерь
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        # Вычисляем градиенты
        loss.backward()
        # Производим оптимизацию
        optimizer.step()
        # обновляем суммарные потери
        running_loss += loss.item() * inputs.size(0)
        # обновляем информацию о том, сколько данных обработано
        processed_data += inputs.size(0)

    # в конце обучения подсчитываем среднюю функцию потерь
    train_loss = running_loss / processed_data

    # TODO: не реализовано
    train_score = 0

    return train_loss, train_score


def eval_epoch(model, val_loader, criterion, device):
    """
    Валидация модели
    :param model: модель
    :param val_loader: валидационная выборка
    :param criterion: функция потерь
    :return: суммарные потери модели и метрика качества
    """
    # Указываем модели переключиться в режим предсказаний. Это необходимо для специфичных
    # слоем сети, например, для Dropouts или BatchNorm слоев
    model.eval()
    # суммарные потери
    running_loss = 0.0
    # размер обработанных элементов
    processed_data = 0
    # выключаем вычисление градиентов
    with torch.set_grad_enabled(False):
        for images, boxes, labels in val_loader:
            inputs = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            # Делаем предсказания
            predicted_locs, predicted_scores = model(inputs)
            # reconstruction = reconstruction.view(-1, 64, 64, 3)
            # Считаем функцию потерь
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            # Подсчитываем потери и метрику качества
            running_loss += loss.item() * inputs.size(0)
            processed_data += inputs.size(0)
    # в конце подсчитываем среднюю функцию потерь
    val_loss = running_loss / processed_data

    return val_loss


def train(model, epochs, criterion, device, losses=None, scores=None):
    """
    Тренировка модели
    :param train_dataset: тренировочная выборка
    :param val_dataset: валидационная выборка
    :param model: модель
    :param epochs: количество эпох
    :param criterion: функция потерь
    :return: потери и метрики качества для тренировочной и валидационной выборок для каждой эпохи
    """
    root = pathlib.Path(__file__).parent.parent
    train_loader = load_coco_data_set(root / 'data/train2017', root / 'data/annotations/instances_train2017.json')

    # val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    train_loss_history = [] or losses
    # val_loss_history = []
    scores_history = [] or scores

    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        # инициализируем оптимизатор, передаем ему параметры модели
        model_params = model.parameters()
        optimizer = torch.optim.Adam(model_params, lr=1e-4)
        # Умножает learning_rate на 0.1 каждые 7 эпохи
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # best_model_wts = model.state_dict()
        # best_val = 0.0

        # Запускаем обучение по эпохам
        for epoch in range(epochs):
            # Учимся на тренировочной выборке
            train_loss, train_score = fit_epoch(model, train_loader, criterion, optimizer, device)

            # Вылидируемся
            # val_loss = 0
            # val_loss = eval_epoch(model, val_loader, criterion)

            # if val_loss > best_val:
            #     best_val = val_loss
            #     best_model_wts = model.state_dict()

            # Добавляем в историю обучения потери для тренировочной и валидационной выборок
            scores_history.append(train_score)
            train_loss_history.append(train_loss)
            # val_loss_history.append(val_loss)

            scheduler.step()

            checkpoint_path = f'/data/model{datetime.datetime.now()}.pt'.replace(' ', '_')

            save_state(checkpoint_path, optimizer, model, train_loss_history, scores_history)

            pbar_outer.update(1)
            tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss))

        # model.load_state_dict(best_model_wts)

    return train_loss_history, scores_history
