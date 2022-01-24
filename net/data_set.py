import torch
import torchvision
import torchvision.transforms.functional as FT


def _collate_fn(batch):
    # return batch
    images = list()
    boxes = list()
    labels = list()

    for b in batch:
        bbox_list = [_coco_to_pascal(row['bbox']) for row in b[1]]
        new_image, new_boxes = resize(b[0], bbox_list)
        new_image = FT.to_tensor(new_image)
        images.append(new_image)

        category_list = [row['category_id'] for row in b[1]]
        boxes.append(new_boxes)
        labels.append(category_list)

    images = torch.stack(images, dim=0)

    return images, boxes, labels  # tensor (N, 3, 300, 300), 2 lists of N tensors each


def _coco_to_pascal(box):
    """
    pascal_voc is a format used by the Pascal VOC dataset.
    Coordinates of a bounding box are encoded with four values in pixels: [x_min, y_min, x_max, y_max].
    x_min and y_min are coordinates of the top-left corner of the bounding box.
    x_max and y_max are coordinates of bottom-right corner of the bounding box.

    coco is a format used by the Common Objects in Context COCO dataset.
    In coco, a bounding box is defined by four values in pixels [x_min, y_min, width, height].
    They are coordinates of the top-left corner along with the width and height of the bounding box.
    :param box:
    :return:
    """
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    if boxes:
        old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
        new_boxes = torch.FloatTensor(boxes) / old_dims  # percent coordinates

        if not return_percent_coords:
            new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
            new_boxes = new_boxes * new_dims
    else:
        new_boxes = torch.FloatTensor()

    return new_image, new_boxes


def load_coco_data_set(root_folder, annotations_file) -> torch.utils.data.DataLoader:
    data_set = torchvision.datasets.CocoDetection(root=root_folder,
                                                  annFile=annotations_file,
                                                  # transform=transform
                                                  )
    data_loader = torch.utils.data.DataLoader(data_set,
                                              batch_size=5,
                                              shuffle=True,
                                              pin_memory=True,
                                              collate_fn=_collate_fn)

    return data_loader
