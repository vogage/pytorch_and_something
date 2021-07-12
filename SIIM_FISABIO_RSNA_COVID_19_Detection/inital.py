# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 12:25:57 2021

@author: Qiandehou
"""

import os
import gc
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import argparse


import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from glob import glob


print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Necessary/extra dependencies. 
#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

print('aaaaaaaaaaaaaaaaaaaaaaaa')

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
        
ROOT_PATH=r'E:\train_and_test_data\SIIM_FISABIO_RSNA_COVID_19_Detection\\'       

TRAIN_PATH =ROOT_PATH+'siim-covid19-detection\\train\\'
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 10

# --img {IMG_SIZE} \ # Input image size.
# --batch {BATCH_SIZE} \ # Batch size
# --epochs {EPOCHS} \ # Number of epochs
# --data data.yaml \ # Configuration file
# --weights yolov5s.pt \ # Model name
# --save_period 1\ # Save model after interval
# --project kaggle-siim-covid # W&B project name



# def train(hyp,  # path/to/hyp.yaml or hyp dictionary
#           opt,
#           device,
#           ):
#     save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, notest, nosave, workers, = \
#         opt.save_dir, opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
#         opt.resume, opt.notest, opt.nosave, opt.workers
                 
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[IMG_SIZE, IMG_SIZE], help='image (height, width)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--include', nargs='+', default=['torchscript', 'onnx', 'coreml'], help='include formats')
    # parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    # parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    # parser.add_argument('--train', action='store_true', help='model.train() mode')
    # parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
    # parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    # parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    # parser.add_argument('--opset-version', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--save_period', type=int, default=1, help='Save model after interval')  
    parser.add_argument('--project', type=str, default='kaggle-siim-covid', help='W&B project name') 
    opt = parser.parse_args()
    return opt

opt = parse_opt()
# Load image level csv file
df = pd.read_csv(ROOT_PATH+'/siim-covid19-detection/train_image_level.csv')

# Modify values in the id column
df['id'] = df.apply(lambda row: row.id.split('_')[0], axis=1)
# Add absolute path
df['path'] = df.apply(lambda row: ROOT_PATH+'SIIM_COVID_19_Resized_to_256px_JPG\\train\\'+row.id+'.jpg', axis=1)
# Get image level labels




dataset_dir = ROOT_PATH+'siim-covid19-detection'

dicom_paths = glob(f'{dataset_dir}/train/*/*/*.dcm')
"""
Return a list of paths matching a pathname pattern.

The pattern may contain simple shell-style wildcards a la
fnmatch. However, unlike fnmatch, filenames starting with a
dot are special cases that are not matched by '*' and '?'
patterns.

If recursive is true, the pattern '**' will match any files and
zero or more directories and subdirectories.
"""
#df['path'] = df.apply(lambda row: TRAIN_PATH+f'\{row.StudyInstanceUID}\\', axis=1)    
df['image_level'] = df.apply(lambda row: row.label.split(' ')[0], axis=1)

print(df.head(5))

  
for path in dicom_paths[:5]:
    print(path)
    
    
    
    

def dicom_dataset_to_dict(dicom_header):
    dicom_dict = {}
    repr(dicom_header)
    for dicom_value in dicom_header.values():
        if dicom_value.tag == (0x7fe0, 0x0010):
            # discard pixel data
            continue
        if type(dicom_value.value) == pydicom.dataset.Dataset:
            dicom_dict[dicom_value.name] = dicom_dataset_to_dict(dicom_value.value)
        else:
            v = _convert_value(dicom_value.value)
            dicom_dict[dicom_value.name] = v
    
    for d in dicom_dict:
        print('{} : {}'.format(d, dicom_dict[d]))


def _sanitise_unicode(s):
    return s.replace(u"\u0000", "").strip()


def _convert_value(v):
    t = type(v)
    if t in (list, int, float):
        cv = v
    elif t == str:
        cv = _sanitise_unicode(v)
    elif t == bytes:
        s = v.decode('ascii', 'replace')
        cv = _sanitise_unicode(s)
    elif t == pydicom.valuerep.DSfloat:
        cv = float(v)
    elif t == pydicom.valuerep.IS:
        cv = int(v)
    else:
        cv = repr(v)
    return cv


fig, ax = plt.subplots(1, 3, figsize=(18,6))

ds0 = pydicom.dcmread(dicom_paths[0]).pixel_array
ds1 = pydicom.dcmread(dicom_paths[1]).pixel_array
ds2 = pydicom.dcmread(dicom_paths[2]).pixel_array

ax[0].imshow(ds0)
ax[1].imshow(ds1, cmap=plt.cm.bone)
ax[2].imshow(ds2, cmap='gray')
plt.show()



# Load meta.csv file
# Original dimensions are required to scale the bounding box coordinates appropriately.
meta_df = pd.read_csv(ROOT_PATH+'/SIIM_COVID_19_Resized_to_256px_JPG/meta.csv')
train_meta_df = meta_df.loc[meta_df.split == 'train']
train_meta_df = train_meta_df.drop('split', axis=1)
train_meta_df.columns = ['id', 'dim0', 'dim1']

print(train_meta_df.head(2))



# Merge both the dataframes
df = df.merge(train_meta_df, on='id',how="left")
print(df.head(5))

# Create train and validation split.
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df.image_level.values)
"""
    stratify : array-like, default=None
        If not None, data is split in a stratified fashion, using this as
        the class labels.
        Read more in the :ref:`User Guide <stratification>`.
"""
train_df.loc[:, 'split'] = 'train'
valid_df.loc[:, 'split'] = 'valid'

df = pd.concat([train_df, valid_df]).reset_index(drop=True)
print(f'Size of dataset: {len(df)}, training images: {len(train_df)}. validation images: {len(valid_df)}')


os.makedirs(ROOT_PATH+'tmp/covid/images/train', exist_ok=True)
os.makedirs(ROOT_PATH+'tmp/covid/images/valid', exist_ok=True)

os.makedirs(ROOT_PATH+'tmp/covid/labels/train', exist_ok=True)
os.makedirs(ROOT_PATH+'tmp/covid/labels/valid', exist_ok=True)


# Move the images to relevant split folder.
for i in tqdm(range(len(df))):
    row = df.loc[i]
    if row.split == 'train':               
        copyfile(row.path, ROOT_PATH+f'tmp/covid/images/train/{row.id}.jpg')
    else:
        copyfile(row.path, f'tmp/covid/images/valid/{row.id}.jpg')
        
    """Copy data from src to dst in the most efficient way possible.

    If follow_symlinks is not set and src is a symbolic link, a new
    symlink will be created instead of copying the file it points to.

    """        
        
        
# Create .yaml file 
import yaml

data_yaml = dict(
    train = '../covid/images/train',
    val = '../covid/images/valid',
    nc = 2,
    names = ['none', 'opacity']
)

# Note that I am creating the file in the yolov5/data/ directory.
with open('tmp/yolov5/data/data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=True)
    
    
# Get the raw bounding box by parsing the row value of the label column.
# Ref: https://www.kaggle.com/yujiariyasu/plot-3positive-classes
def get_bbox(row):
    bboxes = []
    bbox = []
    for i, l in enumerate(row.label.split(' ')):
        if (i % 6 == 0) | (i % 6 == 1):
            continue
        bbox.append(float(l))
        if i % 6 == 5:
            bboxes.append(bbox)
            bbox = []  
            
    return bboxes

# Scale the bounding boxes according to the size of the resized image. 
def scale_bbox(row, bboxes):
    # Get scaling factor
    scale_x = IMG_SIZE/row.dim1
    scale_y = IMG_SIZE/row.dim0
    
    scaled_bboxes = []
    for bbox in bboxes:
        x = int(np.round(bbox[0]*scale_x, 4))
        y = int(np.round(bbox[1]*scale_y, 4))
        x1 = int(np.round(bbox[2]*(scale_x), 4))
        y1= int(np.round(bbox[3]*scale_y, 4))

        scaled_bboxes.append([x, y, x1, y1]) # xmin, ymin, xmax, ymax
        
    return scaled_bboxes

# Convert the bounding boxes in YOLO format.
def get_yolo_format_bbox(img_w, img_h, bboxes):
    yolo_boxes = []
    for bbox in bboxes:
        w = bbox[2] - bbox[0] # xmax - xmin
        h = bbox[3] - bbox[1] # ymax - ymin
        xc = bbox[0] + int(np.round(w/2)) # xmin + width/2
        yc = bbox[1] + int(np.round(h/2)) # ymin + height/2
        
        yolo_boxes.append([xc/img_w, yc/img_h, w/img_w, h/img_h]) # x_center y_center width height
    
    return yolo_boxes



# Prepare the txt files for bounding box
for i in tqdm(range(len(df))):
    row = df.loc[i]
    # Get image id
    img_id = row.id
    # Get split
    split = row.split
    # Get image-level label
    label = row.image_level
    
    if row.split=='train':
        file_name = f'tmp/covid/labels/train/{row.id}.txt'
    else:
        file_name = f'tmp/covid/labels/valid/{row.id}.txt'
        
    
    if label=='opacity':
        # Get bboxes
        bboxes = get_bbox(row)
        # Scale bounding boxes
        scale_bboxes = scale_bbox(row, bboxes)
        # Format for YOLOv5
        yolo_bboxes = get_yolo_format_bbox(IMG_SIZE, IMG_SIZE, scale_bboxes)
        
        with open(file_name, 'w') as f:
            for bbox in yolo_bboxes:
                bbox = [1]+bbox
                bbox = [str(i) for i in bbox]
                bbox = ' '.join(bbox)
                f.write(bbox)
                f.write('\n')


                 
                 

TEST_PATH = '/kaggle/input/siim-covid19-resized-to-256px-jpg/test/' # absolute path


MODEL_PATH = 'kaggle-siim-covid/exp/weights/best.pt'\

PRED_PATH = 'runs/detect/exp3/labels'


prediction_files = os.listdir(PRED_PATH)
print('Number of test images predicted as opaque: ', len(prediction_files))


# The submisison requires xmin, ymin, xmax, ymax format. 
# YOLOv5 returns x_center, y_center, width, height
def correct_bbox_format(bboxes):
    correct_bboxes = []
    for b in bboxes:
        xc, yc = int(np.round(b[0]*IMG_SIZE)), int(np.round(b[1]*IMG_SIZE))
        w, h = int(np.round(b[2]*IMG_SIZE)), int(np.round(b[3]*IMG_SIZE))

        xmin = xc - int(np.round(w/2))
        xmax = xc + int(np.round(w/2))
        ymin = yc - int(np.round(h/2))
        ymax = yc + int(np.round(h/2))
        
        correct_bboxes.append([xmin, xmax, ymin, ymax])
        
    return correct_bboxes

# Read the txt file generated by YOLOv5 during inference and extract 
# confidence and bounding box coordinates.
def get_conf_bboxes(file_path):
    confidence = []
    bboxes = []
    with open(file_path, 'r') as file:
        for line in file:
            preds = line.strip('\n').split(' ')
            preds = list(map(float, preds))
            confidence.append(preds[-1])
            bboxes.append(preds[1:-1])
    return confidence, bboxes



# Read the submisison file
sub_df = pd.read_csv('/kaggle/input/siim-covid19-detection/sample_submission.csv')
sub_df.tail()

# Id,PredictionString
# 2b95d54e4be65_study,negative 1 0 0 1 1
# 2b95d54e4be66_study,typical 1 0 0 1 1
# 2b95d54e4be67_study,indeterminate 1 0 0 1 1 atypical 1 0 0 1 1
# 2b95d54e4be68_image,none 1 0 0 1 1
# 2b95d54e4be69_image,opacity 0.5 100 100 200 200 opacity 0.7 10 10 20 20
# etc.


# Prediction loop for submission
predictions = []

for i in tqdm(range(len(sub_df))):
    row = sub_df.loc[i]
    id_name = row.id.split('_')[0]
    id_level = row.id.split('_')[-1]
    
    if id_level == 'study':
        # do study-level classification
        predictions.append("Negative 1 0 0 1 1") # dummy prediction
        
    elif id_level == 'image':
        # we can do image-level classification here.
        # also we can rely on the object detector's classification head.
        # for this example submisison we will use YOLO's classification head. 
        # since we already ran the inference we know which test images belong to opacity.
        if f'{id_name}.txt' in prediction_files:
            # opacity label
            confidence, bboxes = get_conf_bboxes(f'{PRED_PATH}/{id_name}.txt')
            bboxes = correct_bbox_format(bboxes)
            pred_string = ''
            for j, conf in enumerate(confidence):
                pred_string += f'opacity {conf} ' + ' '.join(map(str, bboxes[j])) + ' '
            predictions.append(pred_string[:-1]) 
        else:
            predictions.append("None 1 0 0 1 1")



sub_df['PredictionString'] = predictions
sub_df.to_csv('submission.csv', index=False)
sub_df.tail()                        
                        
                        
                        