#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import plotly.express as px
import os 
from PIL import Image
import time
import re
import csv
import torch
import segmentation_models_pytorch as smp
from torch.nn.modules.loss import _Loss
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data import Dataset as BaseDataset
import imgaug.augmenters as iaa # 圖片增強套件
import albumentations as albu  # 圖片增強套件
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from model.u2net import U2NET
from statistics import mean
from scipy.spatial import distance
import json
import math
import dash_bootstrap_components as dbc
import dash_table
import dash
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import datetime


os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
#初始圖片與選單變數設置
test_list = pd.read_csv('./Demo_list.csv',low_memory=False)
img_list = list(test_list['File_name'])
img2= Image.open('./背景公版.png')
figure = px.imshow(img, binary_compression_level=0)
figure2 = px.imshow(img2, binary_compression_level=0)

#資料集位置
x_test_dir = './Demo_data/ETT/img/'
y_test_dir = './Demo_data/ETT/mask/'


#模型使用function
image_store=[]
model_gap=[]
zero_image=[]
class Dataset(BaseDataset):
            
    CLASSES = ['null','tube']
            
    def __init__(
                    self, 
                    images_dir, 
                    masks_dir, 
                    classes=None, 
                    augmentation=None, 
                    preprocessing=None,
            ):
                # mask 分類(只有一類)
                self.ids = os.listdir(images_dir)
                self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
                self.masks_fps = [os.path.join(masks_dir, image_id.replace("img", "mask").replace(".dcm.jpg", "_Y.dcm.png")) for image_id in self.ids]         
                self.augmentation = augmentation
                self.preprocessing = preprocessing
            
    def __getitem__(self, i):
        
        
        #name = self.images_fps[i]
        image = cv2.imread(self.images_fps[i])
        mask = cv2.imread(self.masks_fps[i], 0)
        masks = [(mask > 0)]
        mask = np.stack(masks, axis=-1).astype('float')

        seq = iaa.Sequential([
            iaa.MaxPooling(3,keep_size=False),
                    #iaa.CropToFixedSize(width=384, height=512, position="center-bottom"),
            iaa.CropToFixedSize(width=384, height=512, position=(0.5,0.85)),
                    #iaa.CropToFixedSize(width=384, height=384, position="center"),
                ])
        image = seq(images=[image])[0]
        mask = seq(images=[mask])[0]
                
                # apply augmentations
        if self.augmentation:
            
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
                
                # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
                    
        return image, mask
                
    def __len__(self):
        return len(self.ids)

def cnt_area(cnt):
    area=cv2.contourArea(cnt)
    return area
        
def find_point(image):
    image = np.where(image > 0.5, 255, 0)
    image = np.array(image, np.uint8)
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = [cv2.contourArea(a) for a in contours]
    if area == []:
        return torch.tensor(0).float()
    else:
        contour = contours[area.index(max(area))].squeeze()
        if(len(contour)<3):
            return torch.tensor(0).float()
        else:
            contour = contour[contour[:, 1].argsort()]
            point = contour[-1]
            return torch.tensor(point[1]).float()
                
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 設置生成隨機數的種子
    torch.cuda.manual_seed(seed) # 設定GPU隨機種子，如果 CUDA 不可用，它會被默默地忽略。
    torch.backends.cudnn.deterministic = True # 每次返回的捲積算法將是確定的
    
seed_torch(seed=42)

def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.15, rotate_limit=5, shift_limit=0.1, p=1, border_mode=0),
        albu.CLAHE(clip_limit=[1,4],p=0.7),            ]
    return albu.Compose(train_transform)
        
        
def normalization1(x, **kwargs):
    return (x - np.mean(x)) / np.std(x)
        
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
        
        
def get_preprocessing(preprocessing_fn):            
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
            ]
    return albu.Compose(_transform)
        
ENCODER = 'timm-regnetx_160'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['tube']
ACTIVATION = 'sigmoid' 
DEVICE = 'cuda'

model = smp.UnetPlusPlus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
            encoder_depth=4,
            decoder_channels=[448,224,32,32])
        
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

c=''
H = 384
Psize=3 # pooling size

test_dataset = Dataset(
    x_test_dir, 
    y_test_dir, 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
        )
test_dataset_vis = Dataset(
    x_test_dir, y_test_dir, 
    classes=CLASSES,
        )

###氣管內管模型###
def ETT_img_Predit(i):    

    image_vis = test_dataset_vis[i][0].astype('uint8')
    i_name = test_dataset_vis.masks_fps[i].split('\\')[-1] #mask name
    image, gt_mask = test_dataset[i]          
    orig_name = test_dataset_vis.images_fps[i].split('\\')[-1] #image name
    im_gr = cv2.imread(orig_name)
    gt_mask = gt_mask.squeeze()
    x_tensor = torch.Tensor(image).to(DEVICE).unsqueeze(0)
    #模型載入
    best_model = torch.load('./model/best_model'+'5'+'.pth')
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_mask = np.where(pr_mask > 0.5, 255, 0)
    pr_mask = np.array(pr_mask, np.uint8)
    ret, binary = cv2.threshold(pr_mask, 127, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cnt_area, reverse=True)
    clonemask = np.zeros(pr_mask.shape).astype('uint8')
    su = 0          
    if(len(cnts)!=0):
        image_clone = cv2.drawContours(clonemask, [cnts[0]], -1, 255, -1)
    else:
        image_clone = np.zeros(384, dtype=int)


    a = ((gt_mask>0.5).sum(1)>0).nonzero()[0].max()
    if(np.all(image_clone == 0)):
        b = 0
        image_clone=np.zeros((512,384), dtype=int)
        zero_image.append(i_name)
    else:
        b = ((image_clone>0.5).sum(1)>0).nonzero()[0].max()


    c = abs(a-b)*3/72
    image_store.append(i_name)
    model_gap.append(c)
    su+=abs(a-b)

    if(len((image_clone[b]>0.5).nonzero()[0])==0):
        b_x=0
        px = 0
        py = 0
    else:
        b_x=round(((image_clone[b]>0.5)).nonzero()[0].mean())


    up_px = round(im_gr.shape[1]/3-192)
    all_px = (up_px-192)/10
    px = round(all_px*5) # 寬度等於:im_gr.shape[1]/3-(all_px*5+384)
    up_py = round(im_gr.shape[0]/3-256)
    all_py = (up_py-256)/10
    py = round(all_py*1.5) # 高度等於:im_gr.shape[0]/3-(all_py*8.5+512)
    a_x=round(((gt_mask[a]>0.5)).nonzero()[0].mean())
    im_Ngr = im_gr.copy()

    test_im = im_gr.copy()
    im_Ngr =  cv2.circle(im_Ngr, ((b_x+px)*3,(b+py)*3), 5, (255, 0, 0), 5)
    point_pr = ((b_x+px)*3,(b+py)*3)
    point_gr = ((a_x+px)*3,(a+py)*3)

    image_point =  cv2.circle(im_gr, ((a_x+px)*3,(a+py)*3), 5, (0, 0, 255), 5)
    image_point =  cv2.circle(im_gr, ((b_x+px)*3,(b+py)*3), 5, (255, 0, 0), 5) # Predict
    return point_pr,point_gr


###Carina模型###
def Carina(i):
    ENCODER = 'resnet101'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['tube']
    ACTIVATION = 'sigmoid'
    DEVICE = 'cuda'

    model = U2NET(3, 1)
    model.cuda()

    reprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    #載入預測模型
    best_model = torch.load('./model/Carina_best_model_622.pth')
  
    image_vis = test_dataset_vis[i][0].astype('uint8')
    i_name = test_dataset_vis.masks_fps[i].split('\\')[-1] #mask name
    image, gt_mask = test_dataset[i]          
    orig_name = test_dataset_vis.images_fps[i].split('\\')[-1] #image name
    im_gr = cv2.imread(orig_name)
    gt_mask = gt_mask.squeeze()
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().detach().numpy().round())
    pr_mask = np.where(pr_mask > 0.5, 255, 0)
    pr_mask = np.array(pr_mask, np.uint8)
    ret, binary = cv2.threshold(pr_mask, 127, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cnt_area, reverse=True)
    clonemask = np.zeros(pr_mask.shape).astype('uint8')
    su = 0          
    if(len(cnts)!=0):
        image_clone = cv2.drawContours(clonemask, [cnts[0]], -1, 255, -1)
    else:
        image_clone = np.zeros(384, dtype=int)


    a = ((gt_mask>0.5).sum(1)>0).nonzero()[0].max()
    if(np.all(image_clone == 0)):
        b = 0
        image_clone=np.zeros((512,384), dtype=int)
        zero_image.append(i_name)
    else:
        b = ((image_clone>0.5).sum(1)>0).nonzero()[0].max()


    c = abs(a-b)*3/72
    image_store.append(i_name)
    model_gap.append(c)
    su+=abs(a-b)

    if(len((image_clone[b]>0.5).nonzero()[0])==0):
        b_x=0
        px = 0
        py = 0
    else:
        b_x=round(((image_clone[b]>0.5)).nonzero()[0].mean())
        px = round((im_gr.shape[1]/3/2)-(384/2))
        py = round((im_gr.shape[0]/3/100*15)-(512/100*15))
        
    a_x=round(((gt_mask[a]>0.5)).nonzero()[0].mean())
    gx = round((im_gr.shape[1]/3/2)-(384/2))
    gy = round((im_gr.shape[0]/3/100*15)-(512/100*15))

    image = image_clone.copy()
    height, width = image.shape
    for i in range(0, height):
        list_y = image[i]
        if mean(list_y) != 0:
            y_axis_up = i+40
            break;

    for i in range(y_axis_up, height):
        list_y = image[i]
        if mean(list_y) == 0:
            y_axis_down = i-20
            break;

    y_dict = {}
    add = int((y_axis_down-y_axis_up)/2)
    for i in range(y_axis_up+add, y_axis_down):
        count=0
        list_y = image[i]
        for j in range(0,len(list_y)):
            if list_y[j] != 0:
                x_start_one = j+1
                break;

        for k in range(x_start_one, len(list_y)):
            if list_y[k] == 0:
                x_start_two = k
                break;

        for l in range(x_start_two, len(list_y)):
            if list_y[l] == 0:
                count += 1
            else:
                break;

        y_dict[i] = count

    y_axis = min(y_dict, key=lambda n: y_dict[n])
    for j in range(0, len(image[y_axis])):
        if image[y_axis][j] != 0:
                x_start_one = j+1
                break;

    for k in range(x_start_one, len(image[y_axis])):
            if image[y_axis][k] == 0:
                x_start_two = k
                break;

    x_axis = x_start_two
    x_axis = (x_axis+px)*3
    y_axis = (y_axis+py)*3
    Carina_point = (x_axis, y_axis)
    return Carina_point



#預測圖片繪圖控制
config = {
    'autosizable  ':False,
    'frameMargins':'100%',
    "modeBarButtonsToAdd": [
        "drawline",
        "eraseshape"]}

figure2 = px.imshow(img2, binary_compression_level=0)


#bootstrap載入
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


#網頁配置修改
app.layout = html.Div([

        dbc.Container(
            [
                    dbc.Row([
                    html.Div([dcc.Dropdown(img_list,id='demo-dropdown',style={
                     'background-color': '#d4d4d420',
                 })],className='col-3'),
                    
                    html.Div(id='Dist',className='col-3',style={
                                        'textAlign': 'right',
                                        'font-size': '1vw'}),
                                        
                    html.Div(id='content1',className='col-3',style={
                                        'textAlign': 'right',
                                        'font-size': '1vw'}),
                    html.Div(id='content',className='col-3',style={
                                        'textAlign': 'right',
                                        'font-size': '1vw'})],
                        className="justify-content-center",
                        justify='center',
                        style={
                            'position': 'relative'}),       
                html.Center([dcc.Graph(id='graph-with-slider',figure=figure2,config=config,style={
                            'width':'98vw',
                            'height':'95vh'})])])],
                            style={
                            'position': 'absolute'})

#預測距離圖片
@app.callback(
    Output('graph-with-slider', 'figure'),
    Output(component_id='Dist', component_property='children'),
    Input('demo-dropdown', 'value'))

def update_figure(value):
    i = test_list.loc[test_list['File_name'] == value].index[0]
    
    #ETT座標
    ETT_point = ETT_img_Predit(i)
    point_pr = ETT_point[0]
    point_gt = ETT_point[1]
    
    #Carina座標
    Carina_point = Carina(i)
    pointC_x = Carina_point[0]
    pointC_y = Carina_point[1]
    
    #距離換算
    dist_ETT =distance.euclidean(point_pr,Carina_point)
    dist_ETT = dist_ETT/72
    dist_ETT = round(dist_ETT,1)
    
    
    #預測圖片標示距離線
    print(value)
    print("ETT X",point_pr[0])
    print("ETT Y",point_pr[1])
    print("Car X",pointC_x)
    print("Car Y",pointC_y)
    img2 = Image.open('./test_data/Carina/test/{}'.format(value))
    figure2 = px.imshow(img2, binary_compression_level=0,binary_string=True)
    
    figure2.add_shape(editable=True,type='line',
              x0=point_pr[0],x1=pointC_x, y0=point_pr[1],y1=pointC_y,
              xref='x', yref='y')
   
    #預測距離判斷
    if dist_ETT >=3 and dist_ETT <=7:

        return figure2, f'插管正常，距離:\n {dist_ETT} CM'
    
    elif dist_ETT<3:
        return figure2,f'插管異常(離支氣管太近!)，\n 距離: {dist_ETT} CM'
    else:
        return figure2,f'插管異常(離支氣管太遠!)，\n 距離: {dist_ETT} CM'

#DrawLine線長
@app.callback(
    Output('content', 'children'),
    [Input('graph-with-slider', 'relayoutData')],
    [State('content', 'children')])
def shape_added(fig_data, content):
    if fig_data is None:
        return dash.no_update
    if 'shapes' in fig_data:
        line = fig_data['shapes'][-1]
        length = math.sqrt((line['x1'] - line['x0']) ** 2 +
                           (line['y1'] - line['y0']) ** 2)
        length = length/72

    return f'繪圖線長：%.1f'%length + '\n CM'
    

    
#修改預測距離
@app.callback(
    Output("content1", "children"),
    Input("graph-with-slider", "relayoutData"),
    
)
def modify_table_entries(
    graph_relayoutData,
):
    
    x0 = graph_relayoutData['shapes[0].x0']
    x1 = graph_relayoutData['shapes[0].x1']
    y0 = graph_relayoutData['shapes[0].y0']
    y1 = graph_relayoutData['shapes[0].y1']
    
    new_dist =distance.euclidean((x0,y0),(x1,y1))
    new_dist = new_dist/72
    new_dist = round(new_dist,1)
    
    
    return f'目前距離: \n{new_dist} CM'





#發布IP設定
if __name__ == '__main__':
    app.run_server(debug=False)




