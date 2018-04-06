import numpy as np
from keras.callbacks import ModelCheckpoint
def fit(model,train_ids,valid_ids,img_gen_train,img_gen_valid,batch_size,epochs,best_fname):
	#Defining fit generator using img_gen defined above

	saveWeightsCallBack=ModelCheckpoint(best_fname, monitor='val_loss', save_best_only=True)

	model.fit_generator(img_gen_train(train_ids,'./train/',batch_size),epochs=epochs, steps_per_epoch=np.ceil(len(train_ids)/batch_size),
			   validation_data=img_gen_valid(valid_ids,'./train/',batch_size),validation_steps=np.ceil(len(valid_ids)/batch_size)
			   ,callbacks=[saveWeightsCallBack])
