import os
from PIL import Image
import pickle as pkl
import re
import numpy as np
import json


def create_dataset(mode, dirPath):

    if mode == 'tr':
        folder_paths = ['best2019-r31-with-label', 'best2019-r32-with-label', 'best2019-r33-with-label',
                        'best2019-r34-with-label', 'best2019-r35-with-label', 'best2019-r36-with-label', 'best2020-r31-with-label']
    elif mode == 'te':
        folder_paths = ['best2020-r33-1001to2640-with-label']
    images = {}
    i = 0
    for file_name in folder_paths:
        for file in os.listdir(os.path.join(dirPath, file_name)):
            if file.endswith('.label'):
                try:
                    with open(os.path.join(dirPath, file_name, file), 'r', encoding='cp874') as f:
                        for line in f:
                            temp = line.split()
                            img_path = temp[0]
                            label = "".join(temp[1:])
                            img = Image.open(os.path.join(
                                dirPath, file_name, img_path))
                            images[f'writer_{i}'] = [
                                {'img': img, 'label': label}]
                            i += 1
                except UnicodeDecodeError:
                    with open(os.path.join(dirPath, file_name, file), 'r', encoding='utf_16') as f:
                        for line in f:
                            temp = line.split()
                            if len(temp) < 2:
                                continue
                            img_path = temp[0]
                            label = "".join(temp[1:])
                            if mode == 'te':
                                img_id = int(re.findall(
                                    r"-(.*).png", img_path)[0])
                                if img_id < 1001:
                                    img = Image.open(os.path.join(
                                        dirPath, "best2020-r33-1to1000", img_path))
                            else:
                                img = Image.open(os.path.join(
                                    dirPath, file_name, img_path))
                            images[f'writer_{i}'] = [
                                {'img': img, 'label': label}]
                            i += 1
                break

    return images, i


def getAlphabet(dirPath):
    folder_paths = ['best2019-r31-with-label', 'best2019-r32-with-label', 'best2019-r33-with-label',
                    'best2019-r34-with-label', 'best2019-r35-with-label', 'best2019-r36-with-label', 'best2020-r31-with-label']
    labels = []
    for file_name in folder_paths:
        for file in os.listdir(os.path.join(dirPath, file_name)):
            if file.endswith('.label'):
                try:
                    with open(os.path.join(dirPath, file_name, file), 'r', encoding='cp874') as f:
                        for line in f:
                            temp = line.split()
                            label = "".join(temp[1:])
                            labels.append(label)
                except UnicodeDecodeError:
                    with open(os.path.join(dirPath, file_name, file), 'r', encoding='utf_16') as f:
                        for line in f:
                            temp = line.split()
                            if len(temp) < 2:
                                continue
                            label = "".join(temp[1:])
                            labels.append(label)
                break
    return np.unique(np.concatenate(
        [[char for char in w_i.split()[-1]] for w_i in labels]))


path = '../../Best-Handwritten-Corpus/'
train_images, authors1 = create_dataset('tr', path)
test_images, authors2 = create_dataset('te', path)
print(authors1 + authors2)
with open('./files/best.pickle', 'wb') as f:
    pkl.dump({'train': train_images, 'test': test_images},
             f, protocol=pkl.HIGHEST_PROTOCOL)
print("".join(getAlphabet(path)))
alphabets = getAlphabet(path)
char_to_idx = {}
idx_to_char = {}
i = 1
for a in alphabets:
    char_to_idx[a] = i
    idx_to_char[i] = a
    i += 1
with open('./files/charmap.json', 'w', encoding='utf-8') as f:
    json.dump({'char_to_idx': char_to_idx,
              'idx_to_char': idx_to_char}, f, ensure_ascii=False)
