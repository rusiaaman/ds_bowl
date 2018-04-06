import numpy as np
from skimage.transform import resize
from skimage.io import imread,imsave
def img_gen(ids,root,batch_size=1):
    idx=np.random.permutation(len(ids))
    while(1==1):
        for i in range(0,len(ids),batch_size):
            batch=np.zeros((batch_size,360,360,3))
            batch_masks=np.zeros((batch_size,360,360,1))
            for j in range(batch_size):
                if(i+j==len(ids)):
                    break
                filepath = root+ids[idx[i+j]]+'/images/'+ids[idx[i+j]]+'.png'
                img_arr=imread(filepath)[:,:,:3]
                batch[j,:,:,:] = resize(img_arr,(360,360),mode='constant')

                img_arr=imread(root+ids[idx[i+j]]+'/masks/added/added.png')
                batch_masks[j,:,:,0]=img_arr/255

            yield batch,batch_masks
