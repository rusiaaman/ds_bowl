from skimage.transform import resize
from skimage.io import imread,imsave
def add_masks(shape):
#Preprocessing to add all the masks in the directory
	for ids in train_ids:
	    filepath = './train/'+ids+'/masks/'
	    mask=np.zeros(shape)
	    for masks in os.listdir(filepath):
		img_path=filepath+masks
		try:
		    curr_mask=imread(img_path)[:,:]
		    mask=np.maximum(mask,resize(curr_mask,shape,mode='constant',preserve_range=True)).astype('int')
		except:
		    continue
	    assert(np.max(mask)<=255 and np.min(mask)>=0)
	    imsave(filepath+'/added/added.png',mask)
	    print(ids)

