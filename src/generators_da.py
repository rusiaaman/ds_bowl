#Including data augmentation to generators
import numpy as np
from skimage.transform import resize,rotate
from skimage.io import imread,imsave
def img_gen(ids,root,batch_size=1):
    idx=np.random.permutation(len(ids))	
    loaded_imgs=np.zeros((len(ids),360,360,3))
    loaded_masks=np.zeros((len(ids),360,360,1))
    for i in range(len(ids)):
        filepath = root+ids[idx[i]]+'/images/'+ids[idx[i]]+'.png'
        img_arr=imread(filepath)[:,:,:3]
        loaded_imgs[i]=resize(img_arr,(360,360),mode='constant')
        img_arr=imread(root+ids[idx[i]]+'/masks/added/added.png')
        loaded_masks[i,:,:,0]=img_arr/255
    while(1==1):
        for i in range(0,len(ids),batch_size):
            batch=np.zeros((batch_size,360,360,3))
            batch_masks=np.zeros((batch_size,360,360,1))
            for j in range(batch_size):
                if(i+j==len(ids)):
                    break
                batch[j,:,:,:],batch_masks[j,:,:,:] = data_augment(loaded_imgs[i+j],loaded_masks[i+j])

            yield batch,batch_masks

def data_augment(img,mask):
    #angle=np.random.randint(360)
    #img_rot=rotate(img,angle=angle,mode='constant',cval=0,preserve_range=False)
    #mask_rot=rotate(mask,angle=angle,mode='constant',cval=0,preserve_range=False)
    img_rot=img
    mask_rot=mask
    prob_flip=np.random.random(1)
    if(prob_flip>0.3):
        img_rot=img_rot[:,::-1,:]
        mask_rot=mask_rot[:,::-1,:]
    prob_flip=np.random.random(1)
    if(prob_flip>0.3):
        img_rot=img_rot[::-1,:,:]
        mask_rot=mask_rot[::-1,:,:]
    prob_flip=np.random.random(1)
    #if(prob_flip>0.5):
        #img_rot=1-img_rot
    return img_rot,mask_rot
    
