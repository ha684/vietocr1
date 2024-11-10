import sys
import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def checkImageIsValid(imageBin):
    isvalid = True
    imgH = None
    imgW = None

    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    try:
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)

        if img is None:
            isvalid = False
        else:
            imgH, imgW = img.shape[0], img.shape[1]
            if imgH * imgW == 0:
                isvalid = False
    except Exception as e:
        isvalid = False

    return isvalid, imgH, imgW

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def process_image(args):
    idx, (imageFile, label), root_dir = args
    imagePath = os.path.join(root_dir, imageFile)
    if not os.path.exists(imagePath):
        return None
    try:
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        isvalid, imgH, imgW = checkImageIsValid(imageBin)
        if not isvalid:
            return None
        data = {
            'idx': idx,
            'imageBin': imageBin,
            'label': label,
            'imageFile': imageFile,
            'imgH': imgH,
            'imgW': imgW,
        }
        return data
    except Exception:
        return None

def createDataset(outputPath, root_dir, annotation_path):
    """
    Create LMDB dataset for CRNN training.
    """
    annotation_path = os.path.join(root_dir, annotation_path)
    with open(annotation_path, 'r', encoding='utf-8') as ann_file:
        lines = ann_file.readlines()
        annotations = [l.strip().split('\t') for l in lines]

    nSamples = len(annotations)
    env = lmdb.open(outputPath, map_size=4* 1024**3)
    cache = {}
    cnt = 0
    error = 0

    pbar = tqdm(total=nSamples, ncols=100, desc='Create {}'.format(outputPath))

    max_workers = 8  # Adjust based on your system's capabilities
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, item in enumerate(annotations):
            futures.append(executor.submit(process_image, (idx, item, root_dir)))

        for future in as_completed(futures):
            pbar.update(1)
            result = future.result()
            if result is None:
                error += 1
                continue
            imageBin = result['imageBin']
            label = result['label']
            imageFile = result['imageFile']
            imgH = result['imgH']
            imgW = result['imgW']
            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            pathKey = 'path-%09d' % cnt
            dimKey = 'dim-%09d' % cnt

            cache[imageKey] = imageBin
            cache[labelKey] = label.encode()
            cache[pathKey] = imageFile.encode()
            cache[dimKey] = np.array([imgH, imgW], dtype=np.int32).tobytes()

            cnt += 1

            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}

    if cache:
        writeCache(env, cache)

    nSamples = cnt
    cache = {'num-samples': str(nSamples).encode()}
    writeCache(env, cache)

    pbar.close()
    if error > 0:
        print('Removed {} invalid images'.format(error))
    print('Created dataset with {} samples'.format(nSamples))
    sys.stdout.flush()

# Example usage:
# createDataset('output_lmdb_path', 'root_directory', 'annotation_file.txt')
