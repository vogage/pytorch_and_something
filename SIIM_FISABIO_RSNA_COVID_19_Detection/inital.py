# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 12:25:57 2021

@author: Qiandehou
"""
#https://www.kaggle.com/ayuraj/train-covid-19-detection-using-yolov5 


import os
import torch 
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

    
    
    
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Necessary/extra dependencies. 
#customize iPython writefile so we can write variables
# from IPython.core.magic import register_line_cell_magic

# print('aaaaaaaaaaaaaaaaaaaaaaaa')

# @register_line_cell_magic
# def writetemplate(line, cell):
#     with open(line, 'w') as f:
#         f.write(cell.format(**globals()))
        


# #Hyperparameters Setting
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 10



import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
        """Apply a modality lookup table or rescale operation to `arr`.

        .. versionadded:: 1.4
    
        Parameters
        ----------
        arr : numpy.ndarray
            The :class:`~numpy.ndarray` to apply the modality LUT or rescale
            operation to.
        ds : dataset.Dataset
            A dataset containing a :dcm:`Modality LUT Module
            <part03/sect_C.11.html#sect_C.11.1>`.
    
        Returns
        -------
        numpy.ndarray
            An array with applied modality LUT or rescale operation. If
            (0028,3000) *Modality LUT Sequence* is present then returns an array
            of ``np.uint8`` or ``np.uint16``, depending on the 3rd value of
            (0028,3002) *LUT Descriptor*. If (0028,1052) *Rescale Intercept* and
            (0028,1053) *Rescale Slope* are present then returns an array of
            ``np.float64``. If neither are present then `arr` will be returned
            unchanged.
    
        Notes
        -----
        When *Rescale Slope* and *Rescale Intercept* are used, the output range
        is from (min. pixel value * Rescale Slope + Rescale Intercept) to
        (max. pixel value * Rescale Slope + Rescale Intercept), where min. and
        max. pixel value are determined from (0028,0101) *Bits Stored* and
        (0028,0103) *Pixel Representation*.
    
        References
        ----------
        * DICOM Standard, Part 3, :dcm:`Annex C.11.1
          <part03/sect_C.11.html#sect_C.11.1>`
        * DICOM Standard, Part 4, :dcm:`Annex N.2.1.1
          <part04/sect_N.2.html#sect_N.2.1.1>`
        """
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im

# #Environment Setting
ROOT_PATH=r'E:\\train_and_test_data\SIIM_FISABIO_RSNA_COVID_19_Detection/'       

TRAIN_PATH =ROOT_PATH+'siim-covid19-detection\\train\\'

df_train=pd.read_csv(ROOT_PATH+'siim-covid19-detection/train_image_level.csv')

# train = pd.read_csv('../input/siim-covid19-detection/train_image_level.csv')
path =ROOT_PATH+'siim-covid19-detection/train/ae3e63d94c13/288554eb6182/e00f9fe0cce5.dcm'


dicom = pydicom.read_file(path)
image_id = []
dim0 = []
dim1 = []
splits = []

# for split in ['test', 'train']:
#     save_dir =ROOT_PATH+ f'mytmp/{split}/'

#     os.makedirs(save_dir, exist_ok=True)
    
#     for dirname, dirpath, filenames in tqdm(os.walk(ROOT_PATH+f'siim-covid19-detection\{split}')):
#         for file in filenames:
#             # set keep_ratio=True to have original aspect ratio
#             xray = read_xray(os.path.join(dirname, file))
#             im = resize(xray, size=256)  
#             im.save(os.path.join(save_dir, file.replace('dcm', 'jpg')))

#             image_id.append(file.replace('.dcm', ''))
#             dim0.append(xray.shape[0])
#             dim1.append(xray.shape[1])
#             splits.append(split)


TRAIN_PATH='/kaggle/working/train/'

# Modify values in the id column
df_train['id'] = df_train.apply(lambda row: row.id.split('_')[0], axis=1)
# Add absolute path
df_train['path'] = df_train.apply(lambda row: TRAIN_PATH+row.id+'.jpg', axis=1)
# Get image level labels
df_train['image_level'] = df_train.apply(lambda row: row.label.split(' ')[0], axis=1)

df_train.head(5)



               
# # ROOT_PATH=r'E:\train_and_test_data\SIIM_FISABIO_RSNA_COVID_19_Detection\\'  
# TEST_PATH = ROOT_PATH+'SIIM_COVID_19_Resized_to_256px_JPG\test' # absolute path


# MODEL_PATH = 'C:\\Users\Qiandehou\Desktop\github\pytorch_and_something\SIIM_FISABIO_RSNA_COVID_19_Detection\yolov5\runs\train\exp11\best.pt'

# PRED_PATH = 'C:\\Users\Qiandehou\Desktop\github\pytorch_and_something\\SIIM_FISABIO_RSNA_COVID_19_Detection\\yolov5\\runs\detect\exp10\labels'


# prediction_files = os.listdir(PRED_PATH)
# print('Number of test images predicted as opaque: ', len(prediction_files))


# # The submisison requires xmin, ymin, xmax, ymax format. 
# # YOLOv5 returns x_center, y_center, width, height
# def correct_bbox_format(bboxes):
#     correct_bboxes = []
#     for b in bboxes:
#         xc, yc = int(np.round(b[0]*IMG_SIZE)), int(np.round(b[1]*IMG_SIZE))
#         w, h = int(np.round(b[2]*IMG_SIZE)), int(np.round(b[3]*IMG_SIZE))

#         xmin = xc - int(np.round(w/2))
#         xmax = xc + int(np.round(w/2))
#         ymin = yc - int(np.round(h/2))
#         ymax = yc + int(np.round(h/2))
        
#         correct_bboxes.append([xmin, xmax, ymin, ymax])
        
#     return correct_bboxes

# # Read the txt file generated by YOLOv5 during inference and extract 
# # confidence and bounding box coordinates.
# def get_conf_bboxes(file_path):
#     confidence = []
#     bboxes = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             preds = line.strip('\n').split(' ')
#             preds = list(map(float, preds))
#             confidence.append(preds[-1])
#             bboxes.append(preds[1:-1])
#     return confidence, bboxes



# # Read the submisison file
# sub_df = pd.read_csv('E:\\train_and_test_data\SIIM_FISABIO_RSNA_COVID_19_Detection\siim-covid19-detection/sample_submission.csv')
# sub_df.tail()

# # Id,PredictionString
# # 2b95d54e4be65_study,negative 1 0 0 1 1
# # 2b95d54e4be66_study,typical 1 0 0 1 1
# # 2b95d54e4be67_study,indeterminate 1 0 0 1 1 atypical 1 0 0 1 1
# # 2b95d54e4be68_image,none 1 0 0 1 1
# # 2b95d54e4be69_image,opacity 0.5 100 100 200 200 opacity 0.7 10 10 20 20
# # etc.


# # Prediction loop for submission
# predictions = []

# for i in tqdm(range(len(sub_df))):
    
#     row = sub_df.loc[i]
#     id_name = row.id.split('_')[0]
#     id_level = row.id.split('_')[-1]
    
#     if id_level == 'study':
#         # do study-level classification
#         predictions.append("Negative 1 0 0 1 1") # dummy prediction
        
#     elif id_level == 'image':
#         # we can do image-level classification here.
#         # also we can rely on the object detector's classification head.
#         # for this example submisison we will use YOLO's classification head. 
#         # since we already ran the inference we know which test images belong to opacity.
#         if f'{id_name}.txt' in prediction_files:
#             # opacity label
#             confidence, bboxes = get_conf_bboxes(f'{PRED_PATH}/{id_name}.txt')
#             bboxes = correct_bbox_format(bboxes)
#             pred_string = ''
#             for j, conf in enumerate(confidence):
#                 pred_string += f'opacity {conf} ' + ' '.join(map(str, bboxes[j])) + ' '
#             predictions.append(pred_string[:-1]) 
#         else:
#             predictions.append("None 1 0 0 1 1")



# sub_df['PredictionString'] = predictions
# sub_df.to_csv('submission.csv', index=False)
# sub_df.tail()                        
                           
                        
                        