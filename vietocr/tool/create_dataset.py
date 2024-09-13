import os
import lmdb
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import msgpack
import mmap
from PIL import Image
import io

def checkImageIsValid(image_data):
    try:
        img = Image.open(io.BytesIO(image_data))
        imgW, imgH = img.size
        if imgH * imgW == 0:
            return None
        return (imgH, imgW)
    except Exception:
        return None

def processImage(args):
    i, (imageFile, label), root_dir = args
    imagePath = os.path.join(root_dir, imageFile)
    
    if not os.path.exists(imagePath):
        return None
    
    with open(imagePath, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        image_data = mm.read()
        mm.close()
    
    dim = checkImageIsValid(image_data)
    if dim is None:
        return None
    
    return {
        'image': ('image-%09d' % i, image_data),
        'label': ('label-%09d' % i, label),
        'path': ('path-%09d' % i, imageFile),
        'dim': ('dim-%09d' % i, dim)
    }

def createDataset(outputPath, root_dir, annotation_path):
    annotation_path = os.path.join(root_dir, annotation_path)
    with open(annotation_path, 'r', encoding='utf-8') as ann_file:
        annotations = [l.strip().split('\t') for l in ann_file]

    env = lmdb.open(outputPath, map_size=10 * 1024 * 1024 * 1024)  
    
    chunksize = max(1, len(annotations) // (cpu_count() * 4)) 
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(processImage, [(i, ann, root_dir) for i, ann in enumerate(annotations)], chunksize=chunksize),
            total=len(annotations),
            desc='Processing images'
        ))
    
    results = [r for r in results if r is not None]
    
    with env.begin(write=True) as txn:
        for chunk in tqdm(list(chunks(results, 1000)), desc='Writing to LMDB'):
            with txn.cursor() as curs:
                for result in chunk:
                    for k, v in result.items():
                        curs.put(k[0].encode(), msgpack.packb(v[1], use_bin_type=True))
        
        txn.put(b'num-samples', str(len(results)).encode())
    
    print(f'Created dataset with {len(results)} samples')
    print(f'Removed {len(annotations) - len(results)} invalid images')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output_path', help='Path to output LMDB dataset')
    parser.add_argument('root_dir', help='Root directory of dataset')
    parser.add_argument('annotation_path', help='Path to annotation file')
    args = parser.parse_args()

    createDataset(args.output_path, args.root_dir, args.annotation_path)