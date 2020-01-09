import SimpleITK as sitk
import tqdm
import numpy as np
import os 
import GeodisTK as dt
import matplotlib.pyplot as plt 
import time


def distance_transform(img_path, label_path, filename, lamb=0.5, iter_num=2):

    img = sitk.ReadImage(os.path.join(img_path, filename))
    volume = sitk.GetArrayFromImage(img)
    img = sitk.ReadImage(os.path.join(label_path, filename))
    label = sitk.GetArrayFromImage(img)
    volume = volume.astype(np.float32)
    label = label.astype(np.uint8)

    dis = dt.geodesic3d_raster_scan(volume, label, lamb, iter_num)
    dis /= dis.max()
    dis = dis.astype(volume.dtype)

    print(dis.dtype)
    return dis


def main():
    img_path = './abus_roi/image'
    label_path = './abus_roi/label'
    save_path = './abus_roi/dis_map'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filenames = os.listdir(img_path)
    filenames.sort()

    for filename in tqdm.tqdm(filenames):
        dis = distance_transform(img_path, label_path, filename, lamb=0.5, iter_num=8)
        img = sitk.GetImageFromArray(dis) 
        sitk.WriteImage(img, os.path.join(save_path, filename[:-3]+'nii.gz'))
        
    #dis_slice = dis[:, 20, :]
    #volume_slice = volume[:, 20, :]
    #plt.figure()
    #plt.imshow(volume_slice, 'gray')
    #plt.figure()
    #plt.imshow(dis_slice, 'hot')

    #plt.show()


if __name__ == '__main__':
    main()
