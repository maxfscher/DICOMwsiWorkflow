import pydicom
import io
from io import BytesIO

import skimage.measure
from tqdm import tqdm
from time import  sleep
import h5py
import dask.array as da
from PIL import Image
from PIL import ImageCms
import numpy as np
import matplotlib.pyplot as plt
from pydicom.encaps import encapsulate
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset
from pydicom.encaps import generate_pixel_data_frame
from pydicom.dataset import Dataset, FileDataset, DataElement
from pydicom.sequence import Sequence
from random import randint
from torchvision import datasets,models,transforms
import torch.nn as nn
import torch as torch
import h5py
from scipy import ndimage
from datetime import datetime

class_names=['metastasis','non_metastasis']
MyModel=models.vgg19_bn(pretrained=True)
num_ftrs=MyModel.classifier[6].in_features
MyModel.classifier[6]=nn.Linear(num_ftrs,len(class_names))
MyModel.load_state_dict(torch.load('/home/m813r/Documents/Data_cohorts/Camelyon16/Path_ML_Analysis/classification_results_mean_norm/classification_mean_norm_best_model_ft.pt'))
MyModel=MyModel.cuda()
MyModel.eval()

def reconstruct_image(dcm_file):
    SizeInX=dcm_file.TotalPixelMatrixColumns
    SizeInY=dcm_file.TotalPixelMatrixRows
    FrameSize=dcm_file.Rows
    FramesInX=int(SizeInX/FrameSize)
    FramesInY=int(SizeInY/FrameSize)
    frame_generator=generate_pixel_data_frame(dcm_file.PixelData)
    HelperForStacking=da.from_array(np.zeros((FrameSize,FramesInX*FrameSize),dtype=np.float64))###### size of numpy array changed from 3d to 2d
    frames=[]
    SampleSize=250
    if FrameSize%SampleSize==0:
        SampleSize=SampleSize
    else:
        while FrameSize%SampleSize!=0:
            SampleSize=SampleSize+1
    predictions_1=[]
    predictions_2=[]
    for row in range(FramesInY):
        stack=da.from_array(np.zeros((FrameSize,FrameSize),dtype=np.float64),chunks=(FrameSize,FrameSize))###changed size in array from 3d to 2d and chunks
        for column in tqdm(range (FramesInX)):
            frame=next(frame_generator)
            image=Image.open(io.BytesIO(frame))
            image_array=np.asarray(image,dtype=np.float64)
            if image_array.shape[0]!=SampleSize:
                batch=[]
                for rows in range(0,image_array.shape[0],SampleSize):
                    for columns in range (0,image_array.shape[1],SampleSize):
                        sub_image=image_array[rows:(rows+SampleSize),columns:(columns+SampleSize),:]
                        ImageForClassification=transforms.ToTensor()(sub_image).type(torch.float32)
                        ImageForClassification=ImageForClassification/255
                        ImageForClassification=ImageForClassification.unsqueeze(0)
                        ImageForClassification=ImageForClassification.type(torch.float32)
                        batch.append(ImageForClassification)
                batch=torch.cat(batch,0)
                batch=batch.cuda()
                with torch.no_grad():
                    ModelOut=MyModel(batch)
                _,Prediction=torch.max(ModelOut,1)
                batch_counter=0
                for rows in range(0, image_array.shape[0], SampleSize):
                    for columns in range(0, image_array.shape[1], SampleSize):
                        sub_image = image_array[rows:(rows + SampleSize), columns:(columns + SampleSize), :]

                        sub_image[:, :, 0] = int(Prediction[batch_counter].item()) * 255
                        if int(Prediction[batch_counter].item())==0:
                            predictions_2.append(Prediction[batch_counter])
                        elif int(Prediction[batch_counter].item())==1:
                            predictions_1.append(Prediction[batch_counter])


                        image_array[rows:(rows + SampleSize), columns:(columns + SampleSize), :] = sub_image
                        batch_counter=batch_counter+1
                LabelImage=image_array[:,:,0].astype(np.uint8)

            else:
                """
                ##Apply some modification on whole image data
                """
                ##########DICOM generation here we label existing Frames
                factor=randint(0,1)
                #image_array=image_array*255#-------------------delete maybe this again
                image_array[:,:,0]=factor*255
                LabelImage=image_array[:,:,0].astype(np.uint8)




            instance_byte_string_buffer=io.BytesIO()
            SaveIMG=Image.fromarray(image_array.astype(np.uint8))
            SaveIMG.save(instance_byte_string_buffer,"JPEG",quality=75,icc_profile=SaveIMG.info.get('icc_profile'),progressive=False)
            t=instance_byte_string_buffer.getvalue()
            frames.append(t)
            ##########



            dask_array=da.from_array(LabelImage,chunks=(FrameSize,FrameSize))##was changed from 3d to 2d
            stack=[stack,dask_array]
            stack=da.concatenate(stack,axis=1)
        stack=stack[:,FrameSize:]  ####was changed from 3d to 2d
        HelperForStacking=[HelperForStacking,stack]
        HelperForStacking=da.concatenate(HelperForStacking,axis=0)
    DataFrame=HelperForStacking[FrameSize:,:] ##was changed from 3d to 2d
    capsulated_new_frames=encapsulate(frames,has_bot=True)
    dcm_file.PixelData=capsulated_new_frames
    dcm_file.save_as('New_original.dcm',write_like_original=False)
    del dcm_file

    return DataFrame.astype(np.uint8),SampleSize,predictions_1,predictions_2








def DownSampleHeatMap(DataFrame,Factor,FrameSize):
    shape=DataFrame.shape
    #DataFrame2WriteResult=da.from_array(np.zeros((int(shape[0]/(Factor*FrameSize)),int(shape[1]/(Factor*FrameSize))),dtype=np.uint8),chunks=(None,None))
    DataFrame2WriteResult = np.zeros((int(shape[0] / (Factor * FrameSize)), int(shape[1] / (Factor * FrameSize))), dtype=np.float64)
    x_nodes=torch.range(0,shape[1],Factor*FrameSize)
    y_nodes=torch.range(0,shape[0],Factor*FrameSize)
    y_counter=0

    for FramesInY in tqdm(y_nodes[:-1]):
        x_counter = 0
        for FramesInX in x_nodes[:-1]:

            ImageForInference=DataFrame[FramesInY.item():(FramesInY.item()+(Factor*FrameSize)),FramesInX.item():(FramesInX.item()+(FrameSize*Factor))]#channel where label is stored
            InferenceArray=np.asarray(ImageForInference)
            Sum=np.sum(InferenceArray)
            if Sum!=0:
                Sum=Sum/((2*FrameSize)**2)
            else:
                Sum=0

            DataFrame2WriteResult[y_counter, x_counter] = int(Sum)
            """
            try:
                DataFrame2WriteResult[y_counter-1,x_counter]=int(Sum/4)
                DataFrame2WriteResult[y_counter + 1, x_counter] = int(Sum / 4)
                DataFrame2WriteResult[y_counter, x_counter-1] = int(Sum / 4)
                DataFrame2WriteResult[y_counter, x_counter+1] = int(Sum / 4)
            except Exception:
                pass
            """


            x_counter+=1
        y_counter += 1
    return DataFrame2WriteResult

def writeHeatMapToDICOM(map,out_path,reference_DCM):

    created_profile=ImageCms.createProfile('sRGB')
    prf=ImageCms.ImageCmsProfile(created_profile)
    ICC_Profil=prf.tobytes()
    pat_name='heatMap'
    pat_name=str(pat_name)



    image_dims=map.shape

    #mpp=float(MPP_value)
    volume_width=reference_DCM.ImagedVolumeWidth#((image_dims[1])*mpp)/1000
    volume_height = reference_DCM.ImagedVolumeHeight#((image_dims[0]) * mpp) / 1000
    OriginalPixelSize=volume_width/image_dims[0]
    date_time=str(datetime.now())
    date=date_time[0:10].replace('-','')
    time=date_time[10:].replace(':','')
    rows=image_dims[0]
    columns=image_dims[1]
    numbr_of_frames=1
    SOPinstanceUID = generate_uid()
    file_name = 'new_heat_map.dcm'
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.6'  # VL Whole Slide Microscopy Image Storage
    file_meta.MediaStorageSOPInstanceUID = SOPinstanceUID
    file_meta.ImplementationClassUID = '1.3.6.1.4.1.5962.99.2'
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.ImplementationVersionName = 'MaxTest'
    file_meta.SourceApplicationEntityTitle = 'MaxTitle'
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2.4.50'  # JPEG baseline
    file_meta.FileMetaInformationGroupLength = len(file_meta)
    dcm_file = FileDataset(file_name, {}, preamble=b"\0" * 128, file_meta=file_meta, is_implicit_VR=False,is_little_endian=True)
    dcm_file.ImageType = ['DERIVED', 'PRIMARY', 'VOLUME', 'RESAMPLED']
    dcm_file.SOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.6'  # VL Whole Slide Microscopy Image Storage
    dcm_file.SOPInstanceUID = SOPinstanceUID
    dcm_file.ContentDate = date
    tag = pydicom.tag.Tag('AcquisitionDateTime')
    pd_ele = DataElement(tag, 'TM', time)
    dcm_file.add(pd_ele)

    tag = pydicom.tag.Tag('StudyTime')
    pd_ele = DataElement(tag, 'TM', time)
    dcm_file.add(pd_ele)

    tag = pydicom.tag.Tag('ContentTime')
    pd_ele = DataElement(tag, 'TM', time)
    dcm_file.add(pd_ele)
    dcm_file.AccessionNumber = 'A20210527083404'
    dcm_file.Modality = 'SM'
    dcm_file.Manufacturer = 'MyManufacturer'
    dcm_file.ReferringPhysicianName = 'SOME^PHYSICIAN'
    ##########################################
    dcm_file_Coding_Scheme_Identific_1 = Dataset()
    dcm_file_Coding_Scheme_Identific_1.CodingSchemeDesignator = "DCM"
    dcm_file_Coding_Scheme_Identific_1.CodingSchemeUID = "DICOM Controlled Terminology"
    dcm_file_Coding_Scheme_Identific_1.CodingSchemeRegistry = "HL7"
    dcm_file_Coding_Scheme_Identific_1.CodingSchemeName = "DICOM Controlled Terminology"
    dcm_file_Coding_Scheme_Identific_2 = Dataset()
    dcm_file_Coding_Scheme_Identific_2.CodingSchemeDesignator = "SCT"
    dcm_file_Coding_Scheme_Identific_2.CodingSchemeUID = "2.16.840.1.113883.6.96"
    dcm_file_Coding_Scheme_Identific_2.CodingSchemeRegistry = "HL7"
    dcm_file_Coding_Scheme_Identific_2.CodingSchemeName = "SNOMED-CT using SNOMED-CT style values"
    dcm_file.CodingSchemeIdentificationSequence = Sequence([dcm_file_Coding_Scheme_Identific_1, dcm_file_Coding_Scheme_Identific_2])
    dcm_file.TimezoneOffsetFromUTC = '+0200'
    dcm_file.StudyDescription = ''
    dcm_file.ManufacturerModelName = 'MyModel'
    dcm_file.VolumetricProperties = 'VOLUME'
    dcm_file.PatientName = pat_name
    dcm_file.PatientID = pat_name
    dcm_file.PatientBirthDate = '2021-10-23'  # '19700101'
    dcm_file.PatientSex = 'M'
    dcm_file.DeviceSerialNumber = 'MySerialNumber'
    dcm_file.SoftwareVersions = 'MyVersion'
    dcm_file.AcquisitionDuration = 80
    ContributingEquipment = Dataset()
    ContributingEquipment.Manufacturer = 'Manu'
    ContributingEquipment.InstitutionName = 'Instui'
    ContributingEquipment.InstitutionAddress = 'Add'
    ContributingEquipment.InstitutionalDepartmentName = 'Develop'
    ContributingEquipment.ManufacturerModelName = 'Decription'
    ContributingEquipment.SoftwareVersions = 'wsi2dcm'
    ContributingEquipment.ContributionDateTime = '20210103165006.573-000'
    ContributingEquipment.ContributionDescription = 'Description'
    PurposeOfReferenceCodeSequence = Dataset()
    PurposeOfReferenceCodeSequence.CodeValue = "109103"
    PurposeOfReferenceCodeSequence.CodingSchemeDesignator = "DCM"
    PurposeOfReferenceCodeSequence.CodeMeaning = "Modifying Equipment"
    ContributingEquipment.PurposeOfReferenceCodeSequence = Sequence([PurposeOfReferenceCodeSequence])
    dcm_file.ContributingEquipmentSequence = Sequence([ContributingEquipment])
    dcm_file.StudyInstanceUID = reference_DCM.StudyInstanceUID#study_instance_uid#---------------------------adaptions
    dcm_file.SeriesInstanceUID = generate_uid()#series_instance_uid#----------------------------adaptions
    dcm_file.StudyID = reference_DCM.StudyID#study_id#----------------------------------------------------------------------------------------------------adaptions
    dcm_file.SeriesNumber = ''
    dcm_file.InstanceNumber = '2'#'10'
    dcm_file.FrameOfReferenceUID = reference_DCM.FrameOfReferenceUID#frame_of_reference#-----------------------------------adaptions
    dcm_file.PositionReferenceIndicator = 'SLIDE_CORNER'
    dcm_file.ImageComments = 'http://openslide.cs.cmu.edu/download/openslide-testdata/Aperio/'
    dcm_file_DimensionOrganization = Dataset()
    dcm_file_DimensionOrganization.DimensionOrganizationUID = generate_uid()
    dcm_file.DimensionOrganizationSequence = Sequence([dcm_file_DimensionOrganization])
    dcm_file.DimensionOrganizationType = 'TILED_FULL'
    dcm_file.SamplesPerPixel = 3
    dcm_file.PhotometricInterpretation = 'YBR_FULL_422'
    dcm_file.PlanarConfiguration = 0
    dcm_file.NumberOfFrames = int(numbr_of_frames)
    dcm_file.Rows = rows
    dcm_file.Columns = columns
    dcm_file.BitsAllocated = 8
    dcm_file.BitsStored = 8
    dcm_file.HighBit = 7
    dcm_file.PixelRepresentation = 0
    dcm_file.BurnedInAnnotation = 'NO'
    dcm_file.LossyImageCompression = '01'
    dcm_file.LossyImageCompressionRatio = [25.0, 25.0]  # [24.91,24.91]
    dcm_file.LossyImageCompressionMethod = ['ISO_10918_1', 'ISO_10918_1']
    dcm_file.ContainerIdentifier = '888'  # '1000'+str(path[42:])
    dcm_file.IssuerOfTheContainerIdentifierSequence = []
    dcm_file_ContainerTypeCodeSequence = Dataset()
    dcm_file_ContainerTypeCodeSequence.CodeValue = '433466003'
    dcm_file_ContainerTypeCodeSequence.CodingSchemeDesignator = 'SCT'
    dcm_file_ContainerTypeCodeSequence.CodeMeaning = 'Microscope slide'
    dcm_file.ContainerTypeCodeSequence = Sequence([dcm_file_ContainerTypeCodeSequence])
    dcm_file.AcquisitionContextSequence = []
    dcm_file.ColorSpace = 'sRGB'
    Specimen_Description_Sequence = Dataset()
    Primary_Anatomic_Structure_Sequence = Dataset()
    Primary_Anatomic_Structure_Sequence.CodeValue = '32849002'
    Primary_Anatomic_Structure_Sequence.CodingSchemeDesignator = 'SCT'
    Primary_Anatomic_Structure_Sequence.CodeMeaning = 'lymph node'
    Specimen_Description_Sequence.PrimaryAnatomicStructureSequence = Sequence([Primary_Anatomic_Structure_Sequence])
    Specimen_Description_Sequence.SpecimenIdentifier = 'Running Identifier(may be provided in YAML)'  # 'Unknown_0_20210527083404'
    Specimen_Description_Sequence.SpecimenUID = generate_uid()
    Specimen_Description_Sequence.IssuerOfTheSpecimenIdentifierSequence = []
    Specimen_Description_Sequence.SpecimenShortDescription = 'lymph node,sec'
    Specimen_Description_Sequence.SpecimenDetailedDescription = ''
    ######################################################
    ##########################################################
    ########################################################
    specimen_preparation_sequence1 = Dataset()
    a = Dataset()
    a.ValueType = 'TEXT'
    a_ConceptNameCodeSequence = Dataset()
    a_ConceptNameCodeSequence.CodeValue = '121041'
    a_ConceptNameCodeSequence.CodingSchemeDesignator = 'DCM'
    a_ConceptNameCodeSequence.CodeMeaning = 'Specimen Identifier'
    a.ConceptNameCodeSequence = Sequence([a_ConceptNameCodeSequence])
    a.TextValue = 'Array'  ##########################ARR Checken
    b = Dataset()
    b.ValueType = 'CODE'
    b_ConceptNameCodeSequence = Dataset()
    b_ConceptNameCodeSequence.CodeValue = '111701'
    b_ConceptNameCodeSequence.CodingSchemeDesignator = 'DCM'
    b_ConceptNameCodeSequence.CodeMeaning = 'Processing type'
    b.ConceptNameCodeSequence = Sequence([b_ConceptNameCodeSequence])
    b_Concept_Code_Sequence = Dataset()
    b_Concept_Code_Sequence.CodeValue = '9265001'
    b_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    b_Concept_Code_Sequence.CodeMeaning = 'Specimen processing'
    b.ConceptCodeSequence = Sequence([b_Concept_Code_Sequence])
    c = Dataset()
    c.ValueType = 'CODE'
    c_ConceptNameCodeSequence = Dataset()
    c_ConceptNameCodeSequence.CodeValue = '430864009'
    c_ConceptNameCodeSequence.CodingSchemeDesignator = 'SCT'
    c_ConceptNameCodeSequence.CodeMeaning = 'Tissue Fixative'
    c.ConceptNameCodeSequence = Sequence([c_ConceptNameCodeSequence])
    c_Concept_Code_Sequence = Dataset()
    c_Concept_Code_Sequence.CodeValue = '431510009'
    c_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    c_Concept_Code_Sequence.CodeMeaning = 'some description'
    c.ConceptCodeSequence = Sequence([c_Concept_Code_Sequence])
    specimen_preparation_sequence1.SpecimenPreparationStepContentItemSequence = Sequence([a, b, c])
    ################################################################
    specimen_preparation_sequence2 = Dataset()
    a2 = Dataset()
    a2.ValueType = 'TEXT'
    a2_ConceptNameCodeSequence = Dataset()
    a2_ConceptNameCodeSequence.CodeValue = '121041'
    a2_ConceptNameCodeSequence.CodingSchemeDesignator = 'DCM'
    a2_ConceptNameCodeSequence.CodeMeaning = 'Specimen Identifier'
    a2.ConceptNameCodeSequence = Sequence([a2_ConceptNameCodeSequence])
    a2.TextValue = 'Array'  ##########################ARR Checken
    b2 = Dataset()
    b2.ValueType = 'CODE'
    b2_ConceptNameCodeSequence = Dataset()
    b2_ConceptNameCodeSequence.CodeValue = '111701'
    b2_ConceptNameCodeSequence.CodingSchemeDesignator = 'DCM'
    b2_ConceptNameCodeSequence.CodeMeaning = 'Processing type'
    b2.ConceptNameCodeSequence = Sequence([b2_ConceptNameCodeSequence])
    b2_Concept_Code_Sequence = Dataset()
    b2_Concept_Code_Sequence.CodeValue = '9265001'
    b2_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    b2_Concept_Code_Sequence.CodeMeaning = 'Specimen processing'
    b2.ConceptCodeSequence = Sequence([b2_Concept_Code_Sequence])
    c2 = Dataset()
    c2.ValueType = 'CODE'
    c2_ConceptNameCodeSequence = Dataset()
    c2_ConceptNameCodeSequence.CodeValue = '430863003'
    c2_ConceptNameCodeSequence.CodingSchemeDesignator = 'SCT'
    c2_ConceptNameCodeSequence.CodeMeaning = 'Embedding medium'
    c2.ConceptNameCodeSequence = Sequence([c2_ConceptNameCodeSequence])
    c2_Concept_Code_Sequence = Dataset()
    c2_Concept_Code_Sequence.CodeValue = '311731000'
    c2_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    c2_Concept_Code_Sequence.CodeMeaning = 'some medium'  # 'Paraffin wax'
    c2.ConceptCodeSequence = Sequence([c2_Concept_Code_Sequence])
    specimen_preparation_sequence2.SpecimenPreparationStepContentItemSequence = Sequence([a2, b2, c2])
    #########################################################
    #############################################################
    specimen_preparation_sequence3 = Dataset()
    a3 = Dataset()
    a3.ValueType = 'TEXT'
    a3_ConceptNameCodeSequence = Dataset()
    a3_ConceptNameCodeSequence.CodeValue = '121041'
    a3_ConceptNameCodeSequence.CodingSchemeDesignator = 'DCM'
    a3_ConceptNameCodeSequence.CodeMeaning = 'Specimen Identifier'
    a3.ConceptNameCodeSequence = Sequence([a3_ConceptNameCodeSequence])
    a3.TextValue = 'Array'  ##########################ARR Checken
    b3 = Dataset()
    b3.ValueType = 'CODE'
    b3_ConceptNameCodeSequence = Dataset()
    b3_ConceptNameCodeSequence.CodeValue = '111701'
    b3_ConceptNameCodeSequence.CodingSchemeDesignator = 'DCM'
    b3_ConceptNameCodeSequence.CodeMeaning = 'Processing type'
    b3.ConceptNameCodeSequence = Sequence([b3_ConceptNameCodeSequence])
    b3_Concept_Code_Sequence = Dataset()
    b3_Concept_Code_Sequence.CodeValue = '127790008'
    b3_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    b3_Concept_Code_Sequence.CodeMeaning = 'Staining'
    b3.ConceptCodeSequence = Sequence([b3_Concept_Code_Sequence])
    c3 = Dataset()
    c3.ValueType = 'CODE'
    c3_ConceptNameCodeSequence = Dataset()
    c3_ConceptNameCodeSequence.CodeValue = '424361007'
    c3_ConceptNameCodeSequence.CodingSchemeDesignator = 'SCT'
    c3_ConceptNameCodeSequence.CodeMeaning = 'Using substance'
    c3.ConceptNameCodeSequence = Sequence([c3_ConceptNameCodeSequence])
    c3_Concept_Code_Sequence = Dataset()
    c3_Concept_Code_Sequence.CodeValue = '12710003'
    c3_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    c3_Concept_Code_Sequence.CodeMeaning = 'H&E or IHC'
    c3.ConceptCodeSequence = Sequence([c3_Concept_Code_Sequence])
    d3 = Dataset()
    d3.ValueType = 'CODE'
    d3_Concept_Name_Code_Sequence = Dataset()
    d3_Concept_Name_Code_Sequence.CodeValue = '424361007'
    d3_Concept_Name_Code_Sequence.CodingSchemeDesignator = 'SCT'
    d3_Concept_Name_Code_Sequence.CodeMeaning = 'Using substance'
    d3.ConceptNameCodeSequence = Sequence([d3_Concept_Name_Code_Sequence])
    d3_Concept_Code_Sequence = Dataset()
    d3_Concept_Code_Sequence.CodeValue = '36879007'
    d3_Concept_Code_Sequence.CodingSchemeDesignator = 'SCT'
    d3_Concept_Code_Sequence.CodeMeaning = 'further description'
    d3.ConceptCodeSequence = Sequence([d3_Concept_Code_Sequence])
    specimen_preparation_sequence3.SpecimenPreparationStepContentItemSequence = Sequence([a3, b3, c3, d3])
    #######################
    Specimen_Description_Sequence.SpecimenPreparationSequence = Sequence([specimen_preparation_sequence1, specimen_preparation_sequence2, specimen_preparation_sequence3])
    dcm_file.SpecimenDescriptionSequence = Sequence([Specimen_Description_Sequence])
    ######################################
    #####################################
    dcm_file.ImagedVolumeWidth = volume_width
    dcm_file.ImagedVolumeHeight = volume_height
    dcm_file.ImagedVolumeDepth = 1.0
    dcm_file.TotalPixelMatrixColumns = int(columns)
    dcm_file.TotalPixelMatrixRows = int(rows)
    dcm_file_Total_Pixel_Matrix_Origin_Sequence = Dataset()
    dcm_file_Total_Pixel_Matrix_Origin_Sequence.XOffsetInSlideCoordinateSystem = 0.0  # float(linearxoffset)#float(slide.properties.get('aperio.LineAreaXOffset'))#float(slide.properties.get('aperio.Left'))
    dcm_file_Total_Pixel_Matrix_Origin_Sequence.YOffsetInSlideCoordinateSystem = 0.0  # float(linearyoffset)#float(slide.properties.get('aperio.LineAreaYOffset'))#float(slide.properties.get('aperio.Left'))
    dcm_file.TotalPixelMatrixOriginSequence = Sequence([dcm_file_Total_Pixel_Matrix_Origin_Sequence])
    ####################################################
    ####################################################
    dcm_file.SpecimenLabelInImage = 'NO'
    dcm_file.FocusMethod = 'AUTO'
    dcm_file.ExtendedDepthOfField = 'NO'
    dcm_file.ImageOrientationSlide = ['-1', '0', '0', '0', '-1', '0']
    dcm_file_Optical_Path_Sequence = Dataset()
    dcm_file_Illumination_Type_Code_Sequence1 = Dataset()
    dcm_file_Illumination_Type_Code_Sequence1.CodeValue = '111744'
    dcm_file_Illumination_Type_Code_Sequence1.CodingSchemeDesignator = 'DCM'
    dcm_file_Illumination_Type_Code_Sequence1.CodeMeaning = 'Brightfield illumination'
    dcm_file_Optical_Path_Sequence.IlluminationTypeCodeSequence = Sequence([dcm_file_Illumination_Type_Code_Sequence1])
    dcm_file_Optical_Path_Sequence.ICCProfile = ICC_Profil
    dcm_file_Optical_Path_Sequence.OpticalPathIdentifier = '0'
    dcm_file_Illumination_Type_Code_Sequence2 = Dataset()
    dcm_file_Illumination_Type_Code_Sequence2.CodeValue = '414298005'
    dcm_file_Illumination_Type_Code_Sequence2.CodingSchemeDesignator = 'SCT'
    dcm_file_Illumination_Type_Code_Sequence2.CodeMeaning = 'Full Spectrum'
    dcm_file_Optical_Path_Sequence.IlluminationColorCodeSequence = Sequence([dcm_file_Illumination_Type_Code_Sequence2])
    dcm_file.OpticalPathSequence = Sequence([dcm_file_Optical_Path_Sequence])
    dcm_file.NumberOfOpticalPaths = 1
    dcm_file.TotalPixelMatrixFocalPlanes = 1
    dcm_file_Shared_Functional_Groups = Dataset()
    dcm_file_Pixel_Measures_Sequence = Dataset()
    dcm_file_Pixel_Measures_Sequence.SliceThickness = '0.001'  # '0.0010000002384'
    dcm_file_Pixel_Measures_Sequence.SpacingBetweenSlices = '0.006'  # '0.0006'
    print(OriginalPixelSize)
    dcm_file_Pixel_Measures_Sequence.PixelSpacing = [str(OriginalPixelSize), str(OriginalPixelSize)]
    dcm_file_Shared_Functional_Groups.PixelMeasuresSequence = Sequence([dcm_file_Pixel_Measures_Sequence])
    dcm_file_Whole_Slide_Microscopy_Image_Frame_Type_Sequence = Dataset()

    dcm_file_Whole_Slide_Microscopy_Image_Frame_Type_Sequence.FrameType = ['DERIVED', 'PRIMARY', 'VOLUME','RESAMPLED']
    dcm_file_Shared_Functional_Groups.WholeSlideMicroscopyImageFrameTypeSequence = Sequence([dcm_file_Whole_Slide_Microscopy_Image_Frame_Type_Sequence])
    dcm_file_Optical_Path_Identification_Sequence = Dataset()
    dcm_file_Optical_Path_Identification_Sequence.OpticalPathIdentifier = '0'
    dcm_file_Shared_Functional_Groups.OpticalPathIdentificationSequence = Sequence([dcm_file_Optical_Path_Identification_Sequence])
    dcm_file.SharedFunctionalGroupsSequence = Sequence([dcm_file_Shared_Functional_Groups])
    encoded_frames = []
    instance_byte_string_buffer = io.BytesIO()
    image = Image.fromarray(map)
    profile = image.info.get('icc_profile')
    image.save(instance_byte_string_buffer, "JPEG", quality=95, icc_profile=profile, progressive=False)

    t = instance_byte_string_buffer.getvalue()
    encoded_frames.append(t)
    capsulated = encapsulate(encoded_frames, has_bot=True)
    pixeL_data = capsulated
    data_elem_tag = pydicom.tag.TupleTag((0x7FE0, 0x0010))
    pd_ele = DataElement(data_elem_tag, 'OB', pixeL_data, is_undefined_length=True)
    dcm_file.add(pd_ele)
    store_path = out_path + file_name
    dcm_file.save_as(store_path, write_like_original=False)
    return 0


def visualizeHeatMap(downsampled_result):
    Original=downsampled_result
    eroded=ndimage.binary_erosion(downsampled_result).astype(np.uint8)
    eroded = ndimage.binary_erosion(eroded).astype(np.uint8)
    DilatedImage=ndimage.binary_dilation(eroded).astype(np.uint8)
    DilatedImage = ndimage.binary_dilation(DilatedImage).astype(np.uint8)
    DilatedImage = ndimage.binary_dilation(DilatedImage).astype(np.uint8)
    shape=downsampled_result.shape
    Dummy=np.zeros((shape[0],shape[1],1))
    DilatedImage=DilatedImage*255
    Original=np.expand_dims(Original,axis=2)
    DilatedImage=np.expand_dims(DilatedImage,axis=2)
    result=np.concatenate((Original,DilatedImage,Dummy),axis=2).astype(np.uint8)
    return result


mpp=0.24309399999999998

dcm_file=pydicom.dcmread('/home/m813r/PycharmProjects/patho_daten/dicoms/Full_but_too_large_Tumor/new-0-tiles.dcm')
DataFrame,SampleSize,pred1,pred2=reconstruct_image(dcm_file)

down_sampled_image=DownSampleHeatMap(DataFrame,2,SampleSize)
heat_map=visualizeHeatMap(down_sampled_image)

writeHeatMapToDICOM(heat_map,'./',dcm_file)


file=h5py.File('DataFrame_tumor_044.hdf5')
output=file.create_dataset('output',shape=DataFrame.shape,dtype=np.uint8)
da.store(DataFrame,output)



frame_generator=generate_pixel_data_frame(dcm_file.PixelData)
image_list=[]
for i in tqdm(range(3000)):
    frame=next(frame_generator)
    test_image=Image.open(io.BytesIO(frame))####1
    array=np.asarray(test_image)
    #if array.mean()>50 and array.mean()<240:
    image_list.append((array,i))





dictionary = dict()#{}
dictionary['general_info'] = dcm_file.NumberOfFrames, dcm_file.TotalPixelMatrixRows, dcm_file.TotalPixelMatrixColumns, dcm_file.Rows
dictionary['frame_information']={}
dictionary
TotalPixelMatrixColumns=dcm_file.TotalPixelMatrixColumns
TotalPixelMatrixRows=dcm_file.TotalPixelMatrixRows
FrameSize=dcm_file.Rows
NumberOfFrames=dcm_file.NumberOfFrames
FramesInX=TotalPixelMatrixColumns/FrameSize
FramesInY=TotalPixelMatrixRows/FrameSize
label=0
for i in range (len(image_list)):
    frame=image_list[i][0]
    ####sampling the subframe
    sub_batchsize=256
    sub_frames_per_frame=(frame.shape[0]*frame.shape[1])/sub_batchsize**2
    sub_frame_counter=0
    for y in range(0,frame.shape[1],sub_batchsize):
        for x in range(0,frame.shape[0],sub_batchsize):
            sub_frame_counter=sub_frame_counter+1
            sub_frame=frame[y:y+sub_batchsize,x:x+sub_batchsize,:]
            if i>150 and i<180:

                #frame_coding=int(i*sub_frames_per_frame+sub_frame_counter)
                xPosOfFrame=i%FramesInX
                yPosOfFrame=(i-xPosOfFrame)/FramesInX
                XCoordinate=int(xPosOfFrame*FrameSize+x)
                YCoordinate=int(yPosOfFrame*FrameSize+y)
                code=str(YCoordinate)+','+str(XCoordinate)
                print(code)
                dictionary['frame_information'][str(label)]=code
                label = label + 1
    dictionary['sub_batch_size']=str(sub_batchsize)

def remainder(a,b):
    result=[int(a/b),a%b]
    return result

def create_heat(dictionary,dcm_file):
    HigherFrames=dcm_file.pixel_array
    for entries in tqdm(range(len(dictionary['frame_information']))):
        sleep(3)
        coordinates=dictionary['frame_information'][str(entries)]
        y_coordinate=int(coordinates.rpartition(',')[0])
        x_coordinate = int(coordinates.rpartition(',')[2])
        ########compute downsampling_factor
        NumberOfRowsOriginal=dictionary['general_info'][1]
        NumberOfColumnsOriginal = dictionary['general_info'][2]
        FactorY=NumberOfRowsOriginal/dcm_file.TotalPixelMatrixRows
        FactorX=NumberOfColumnsOriginal/dcm_file.TotalPixelMatrixColumns
        ####################
        StartXHigherFrame=x_coordinate/FactorX
        StartYHigherFrame=y_coordinate/FactorY
        EndXHigherFrame=(x_coordinate+int(dictionary['sub_batch_size']))/FactorX
        EndYHigherFrame = (y_coordinate + int(dictionary['sub_batch_size'])) / FactorY
        #########Compute Corresponding Frame
        FramesInX=dcm_file.TotalPixelMatrixColumns/dcm_file.Rows
        FramesInY = dcm_file.TotalPixelMatrixRows / dcm_file.Rows
        XFrames=StartXHigherFrame%dcm_file.Rows
        YFrames = StartYHigherFrame % dcm_file.Rows
        FrameNumber=XFrames*YFrames
        XStartCoordInFrame=StartXHigherFrame-(XFrames*dcm_file.Rows)
        YStartCoordInFrame=StartYHigherFrame-(YFrames*dcm_file.Rows)
        XEndCoordInFrame=EndXHigherFrame-(XFrames*dcm_file.Rows)
        YEndCoordInFrame=EndYHigherFrame-(YFrames*dcm_file.Rows)
        Frames=dcm_file.pixel_array
        try:
            Frames[FrameNumber,YStartCoordInFrame:YEndCoordInFrame,XStartCoordInFrame:XEndCoordInFrame,:]=255
        except:
            print('Coordinates exceed FrameSize')
        manipulated_frames=[]
        for frame in range(Frames.shape[0]):
            instance_byte_string_buffer=io.BytesIO()
            image=Image.fromarray(Frames[frame,:,:,:])
            image.save(instance_byte_string_buffer,"JPEG",quality=75,icc_profile=image.info.get('icc_profile'),progressive=False)
            t=instance_byte_string_buffer.getvalue()
            manipulated_frames.append(t)
        EncapsulatedManipulatedFrames=encapsulate(manipulated_frames,has_bot=True)
        dcm_file.PixelData = EncapsulatedManipulatedFrames
        dcm_file.save_as('New.dcm', write_like_original=False)









