# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 12:25:57 2021

@author: Qiandehou
"""
#https://www.kaggle.com/ayuraj/train-covid-19-detection-using-yolov5 

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

import yaml

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from glob import glob

import wandb



import sys
from IPython import get_ipython 
# adding Folder_2 to the system path


    
    
    
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Necessary/extra dependencies. 
#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

print('aaaaaaaaaaaaaaaaaaaaaaaa')

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
        
#Environment Setting
ROOT_PATH=r'E:\train_and_test_data\SIIM_FISABIO_RSNA_COVID_19_Detection\\'       

TRAIN_PATH =ROOT_PATH+'siim-covid19-detection\\train\\'

#Hyperparameters Setting
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 10



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
    
    
def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data    
    

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


# os.makedirs(ROOT_PATH+'tmp/covid/images/train', exist_ok=True)
# os.makedirs(ROOT_PATH+'tmp/covid/images/valid', exist_ok=True)

# os.makedirs(ROOT_PATH+'tmp/covid/labels/train', exist_ok=True)
# os.makedirs(ROOT_PATH+'tmp/covid/labels/valid', exist_ok=True)


# # Move the images to relevant split folder.
# for i in tqdm(range(len(df))):
#     row = df.loc[i]
#     if row.split == 'train':               
#         copyfile(row.path, ROOT_PATH+f'tmp/covid/images/train/{row.id}.jpg')
#     else:
#         copyfile(row.path,  ROOT_PATH+f'tmp/covid/images/valid/{row.id}.jpg')
        
#     """Copy data from src to dst in the most efficient way possible.

#     If follow_symlinks is not set and src is a symbolic link, a new
#     symlink will be created instead of copying the file it points to.

#     """        
        
        
# Create .yaml file 
import yaml

data_yaml = dict(
    train = ROOT_PATH+'tmp/covid/images/train',
    val = ROOT_PATH+'tmp/covid/images/valid',
    nc = 2,
    names = ['none', 'opacity'],
   
)

# Note that I am creating the file in the yolov5/data/ directory.
with open('yolov5/data/data.yaml', 'w') as outfile:
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
# Create the train and test labels
for i in tqdm(range(len(df))):
    row = df.loc[i]
    # Get image id
    img_id = row.id
    # Get split
    split = row.split
    # Get image-level label
    label = row.image_level
    
    if row.split=='train':
        file_name = ROOT_PATH+f'tmp/covid/labels/train/{row.id}.txt'
    else:
        file_name = ROOT_PATH+f'tmp/covid/labels/valid/{row.id}.txt'
        
    
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


sys.path.insert(0, '..\SIIM_FISABIO_RSNA_COVID_19_Detection\yolov5')


# artifact = wandb.Artifact()
run=wandb.init()
# api = wandb.Api()
# artifact = api.artifact('kaggle-siim-covid/my-dataset:v1', type='dataset')
artifact = wandb.Artifact('animals', type='dataset')
artifact.add_dir('E:\\train_and_test_data\SIIM_FISABIO_RSNA_COVID_19_Detection\SIIM_COVID_19_Resized_to_256px_JPG')
run.log_artifact(artifact)


wandb.login()
# parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
#     parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
#     parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
#     parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
#     parser.add_argument('--epochs', type=int, default=300)
#     parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
#     parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
#     parser.add_argument('--rect', action='store_true', help='rectangular training')
#     parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
#     parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
#     parser.add_argument('--notest', action='store_true', help='only test final epoch')
#     parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
#     parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
#     parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
#     parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
#     parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
#     parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
#     parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
#     parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
#     parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
#     parser.add_argument('--project', default='runs/train', help='save to project/name')
#     parser.add_argument('--entity', default=None, help='W&B entity')
#     parser.add_argument('--name', default='exp', help='save to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--quad', action='store_true', help='quad dataloader')
#     parser.add_argument('--linear-lr', action='store_true', help='linear LR')
#     parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
#     parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
#     parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
#     parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
#     parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
#     parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
#     opt = parser.parse_known_args()[0] if known else parser.parse_args()



# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', type=str, default= ROOT_PATH+'/siim-covid19-detection/yolov5s.pt', help='weights path')
#     parser.add_argument('--img-size', nargs='+', type=int, default=[IMG_SIZE, IMG_SIZE], help='image (height, width)')
#     parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='batch size')
#     parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     # parser.add_argument('--include', nargs='+', default=['torchscript', 'onnx', 'coreml'], help='include formats')
#     # parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
#     # parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
#     # parser.add_argument('--train', action='store_true', help='model.train() mode')
#     # parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
#     # parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
#     # parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
#     # parser.add_argument('--opset-version', type=int, default=12, help='ONNX: opset version')
#     parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
#     parser.add_argument('--save_period', type=int, default=1, help='Save model after interval')  
#     parser.add_argument('--project', type=str, default='kaggle-siim-covid', help='W&B project name') 
#     parser.add_argument('--save_dir', type=str, default=ROOT_PATH+'save', help='the directory for model and immediate files')
#     parser.add_argument('--single-cls',action='store_true', help='train as single-class dataset')
#     parser.add_argument('--evolve',action='store_true',help='evolve hyperparameters')
#     parser.add_argument('--data',type=str,default='..\SIIM_FISABIO_RSNA_COVID_19_Detection\yolov5\data\data.yaml',help='data.yaml path')
#     parser.add_argument('--cfg',type=str,default='..\SIIM_FISABIO_RSNA_COVID_19_Detection\yolov5\models\yolov5s.yaml',help='model.yaml path')
#     parser.add_argument('--resume',nargs='?',const=True,default=False,help='resume most recent training')
#     parser.add_argument('--notest',action='store_true',help='only test final epoch')
#     parser.add_argument('--nosave',action='store_true',help='only save final checkpoint')
#     parser.add_argument('--workers',type=int,default=8,help='maximum number of dataloader workers')
#     parser.add_argument('--entity',type=str,default='poi',help='the project by poi (mine)')
#     parser.add_argument('--upload_dataset',type=bool,default=False,help='upload the dataset for data versioning')
#     parser.add_argument('--bbox_interval',type=int,default=2,help='At the end of every bbox_interval epoches,\
#                         the output of the model on the validation set will be uploaded to W&B')
#     parser.add_argument('--artifact_alias',type=str, default="latest", help='version of dataset artifact to be used')
#     opt = parser.parse_args()
#     return opt

# opt = parse_opt()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyp='..\SIIM_FISABIO_RSNA_COVID_19_Detection\yolov5\data\data.yaml'
# Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
# hyp = { 'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
#         'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
#         'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
#         #'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
#         'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
#         'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
#         'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
#         'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
#         'box': (1, 0.02, 0.2),  # box loss gain
#         'cls': (1, 0.2, 4.0),  # cls loss gain
#         'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
#         'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
#         'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
#         'iou_t': (0, 0.1, 0.7),  # IoU training threshold
#         'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
#         #'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
#         'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
#         'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
#         'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
#         'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
#         'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
#         'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
#         'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
#         'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
#         'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
#         'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
#         'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
#         'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
#         'mixup': (1, 0.0, 1.0),  # image mixup (probability)
#         'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)




# --img {IMG_SIZE} \ # Input image size.
# --batch {BATCH_SIZE} \ # Batch size
# --epochs {EPOCHS} \ # Number of epochs
# --data data.yaml \ # Configuration file
# --weights yolov5s.pt \ # Model name
# --save_period 1\ # Save model after interval
# --project kaggle-siim-covid # W&B project name
import train
train.run(data='..\SIIM_FISABIO_RSNA_COVID_19_Detection\yolov5\data\data.yaml',img=IMG_SIZE,
          batch=BATCH_SIZE,epochs=EPOCHS)


"""  
def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          ):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, notest, nosave, workers, = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.notest, opt.nosave, opt.workers
"""



          
"""                 
# ROOT_PATH=r'E:\train_and_test_data\SIIM_FISABIO_RSNA_COVID_19_Detection\\'  
TEST_PATH = ROOT_PATH+'SIIM_COVID_19_Resized_to_256px_JPG\test' # absolute path


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
"""                            
                        
                        