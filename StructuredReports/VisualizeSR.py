import pydicom
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
from wsidicom import WsiDicom
import numpy as np
import cv2
import math
def mirror_diagonal(array):

    array=np.flip(np.flip(array, axis=0), axis=1)
    return [list(row) for row in zip(*array)]

def getDicomInfos(file):
    rows=file.Rows
    cols=file.Columns
    PixelmatrixY=file.TotalPixelMatrixRows
    PixelmatrixX=file.TotalPixelMatrixColumns
    RealRows=math.ceil(PixelmatrixY/cols)*rows
    RealCols=math.ceil(PixelmatrixX/rows)*cols

    spacing=file.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
    Width=spacing[0]*RealCols
    Height=spacing[1]*RealRows
    return float(file.TotalPixelMatrixOriginSequence[0].XOffsetInSlideCoordinateSystem),float(file.TotalPixelMatrixOriginSequence[0].YOffsetInSlideCoordinateSystem),Width,Height,spacing


FileFull=pydicom.dcmread('./4E17833F.dcm')
ImageFile_overlay=pydicom.dcmread('./low.dcm')
ImageFile=pydicom.dcmread('./highest.dcm')

X_Offset,Y_Offset,Width,Height,(spacing_x,spacing_y)=getDicomInfos(ImageFile)

downscaling_factor=4

Origin_X=X_Offset-Height
Origin_Y=Y_Offset-Width

AnnotatetObjects3=len(FileFull.ContentSequence[13].ContentSequence)
plt.figure()
tumor_list=[]
tissue_list=[]


image=np.zeros((int(Width/spacing_x),int(Height/spacing_x),3),dtype=np.uint8)######additional

for i in range(AnnotatetObjects3):
    Type=FileFull.ContentSequence[13].ContentSequence[i].ContentSequence[2].ConceptCodeSequence[0].CodeMeaning
    Coords3=FileFull.ContentSequence[13].ContentSequence[i].ContentSequence[3].GraphicData
    x_coords3 = [int((Coords3[i]-Origin_X)/spacing_x) for i in range(0, len(Coords3), 3)]
    y_coords3 = [int((Coords3[i]-Origin_Y)/spacing_y) for i in range(1, len(Coords3), 3)]
    if Type=='Tissue':
        tissue_list.append([x_coords3,y_coords3])
        color=(255,0,0)
    else:
        tumor_list.append([x_coords3,y_coords3])
        color=(0,0,255)
    contours = np.array([[[abs(x), abs(y)] for x, y in zip(x_coords3, y_coords3)]], dtype=np.int32)
    cv2.drawContours(image, contours, -1, color, cv2.FILLED)


plt.imshow(image[:,:,0])
plt.figure()
image=cv2.resize(image,(int(image.shape[1]/downscaling_factor),int(image.shape[0]/downscaling_factor)))# (16507,23460)
plt.imshow(image[:,:,0])
mirrored=mirror_diagonal(image[:,:,0])
mirrored=np.asarray(mirrored,dtype=np.uint8)
plt.figure()
plt.imshow(mirrored)

image_data=Image.fromarray(ImageFile_overlay.pixel_array)
image_data=image_data.resize((image.shape[0],image.shape[1]))#(23460,16507)
annotation_data=Image.fromarray(mirrored)
annotation_data=annotation_data.convert('RGB')

blended=Image.blend(image_data,annotation_data,0.5)
plt.figure()
plt.imshow(blended)




