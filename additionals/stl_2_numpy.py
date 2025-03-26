import os
import numpy as np
import h5py
import stltovoxel
from PIL import Image
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
from pathlib import Path
from typing import Iterator, Optional
import logging
from logging.handlers import RotatingFileHandler
import itertools
from datetime import datetime

def setup_logging():
    logger = logging.getLogger()
    
    # 如果logger已经有处理器，说明已经被设置过，直接返回
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()

def convert_stl_to_png(stl_path: str, resolution: int = 512, pad: int = 0) -> str:
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        png_path = tmp.name
    stltovoxel.convert_file(stl_path, png_path, resolution=resolution, pad=pad, parallel=True)
    return png_path

def load_png_to_numpy(png_path: str, start_num: int, end_num: int) -> np.ndarray:
    target_size = (512, 512)
    images = []
    
    for i in range(start_num, end_num + 1):
        filename = f"{png_path[:-4]}_{i:03d}.png"
        if not os.path.exists(filename):
            logger.warning(f"File {filename} does not exist. Skipping.")
            continue
        with Image.open(filename) as img:
            resized_img = img.resize(target_size, Image.LANCZOS)
            images.append(np.array(resized_img))
    
    images_array = np.array(images)
    
    if len(images) != 512:
        padded_array = np.zeros((512, 512, 512), dtype=np.uint8)
        padded_array[:len(images)] = images_array[:512]
        return padded_array
    
    return images_array.astype(np.uint8)

def process_single_stl(stl_file: str) -> np.ndarray:
    png_path = convert_stl_to_png(stl_file)
    array = load_png_to_numpy(png_path, 0, 511)
    
    # Delete PNG files
    for png_file in Path(png_path[:-4]).glob('*.png'):
        png_file.unlink()
    
    return array

def stl_file_generator(input_dir: str, max_files: Optional[int] = None) -> Iterator[Path]:
    count = 0
    for stl_file in Path(input_dir).rglob('*.stl'):
        yield stl_file
        count += 1
        if max_files is not None and count >= max_files:
            break

def process_stl_files(input_dir: str, output_dir: str, batch_size: int = 1000, max_files: Optional[int] = None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    stl_files = stl_file_generator(input_dir, max_files)

    batch_num = 0
    file_count = 0

    while True:
        h5_filename = Path(output_dir) / f'batch_{batch_num:03d}.h5'
        
        with h5py.File(h5_filename, 'w') as h5f:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                batch_files = list(itertools.islice(stl_files, batch_size))
                if not batch_files:
                    break  # No more files to process

                futures = {executor.submit(process_single_stl, str(stl_file)): i 
                           for i, stl_file in enumerate(batch_files)}
                
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        array = future.result()
                        h5f.create_dataset(f'array_{i:03d}', data=array, compression='gzip')
                        file_count += 1
                        logger.info(f"Processed file {file_count}")
                    except Exception as e:
                        logger.error(f"Error processing file {file_count + 1}: {e}")

        logger.info(f"Batch {batch_num} saved to {h5_filename}")
        batch_num += 1

    logger.info(f"Total processed files: {file_count}")

if __name__ == "__main__":
    input_dir = r"D:\datasets\stl2\abc_0000_stl2_v00"
    output_dir = r"D:\datasets\stl2\output_h5"
    batch_size = 10
    max_files = 100  # Set this to the number of files you want to process, or None to process all files
    process_stl_files(input_dir, output_dir, batch_size=batch_size, max_files=max_files)