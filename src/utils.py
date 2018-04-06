import matplotlib.pyplot as plt
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        if(ims.shape[-1]==1):
            plt.imshow(ims[i,:,:,0], interpolation=None if interp else 'none', cmap='gray')
        else:
            plt.imshow(ims[i], interpolation=None if interp else 'none')
