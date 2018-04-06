from keras.models import Model
from keras.layers import Input, Dense, Conv2D,Conv2DTranspose,Dropout,MaxPooling2D,BatchNormalization,concatenate
def get_model():
    
    Input_layer = Input(shape=(360,360,3))
    activation='relu'
    c1 = Conv2D(32, (3, 3), strides=1, activation=activation, kernel_initializer='he_normal', padding='same') (Input_layer)
    c1 = Conv2D(32, (3, 3), strides=1, activation=activation, kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = BatchNormalization()(p1)
    c2 = Conv2D(64, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c2)
    c2 = Conv2D(64, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c2)
    c2 = Conv2D(64, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = BatchNormalization()(p2)
    c3 = Conv2D(128, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c3)
    c3 = Conv2D(128, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c3)
    c3 = Conv2D(128, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = BatchNormalization()(p3)
    c4 = Conv2D(256, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c4)
    c4 = Conv2D(256, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c4)
    c4 = Conv2D(256, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D((2, 2)) (c4)

    c5 = BatchNormalization()(p4)
    c5 = Conv2D(512, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c5)
    c5 = Conv2D(512, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c5)
    c5 = Conv2D(512, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c5)
    p5 = MaxPooling2D((2, 2)) (c5)
    
    c55 = BatchNormalization()(p5)
    c55 = Conv2D(1024, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c55)
    c55 = Conv2D(1024, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c55)
    
    u65 = Conv2DTranspose(512, (2,2), strides=(2, 2), padding='same') (c55)
    u65 = BatchNormalization()(u65)
    u65 = concatenate([u65, c5])
    c65 = Conv2D(512, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (u65)
    c65 = Conv2D(512, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c65)
    c65 = Conv2D(512, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c65)

    u6 = Conv2DTranspose(256, (3,3), strides=(2, 2), padding='valid') (c65)
    u6 = BatchNormalization()(u6)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (u6)
    c6 = Conv2D(256, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c6)
    c6 = Conv2D(256, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(128, (2,2), strides=(2, 2), padding='same') (c6)
    u7 = BatchNormalization()(u7)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (u7)
    c7 = Conv2D(128, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c7)
    c7 = Conv2D(128, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = BatchNormalization()(u8)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (u8)
    c8 = Conv2D(64, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c8)
    c8 = Conv2D(64, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = BatchNormalization()(u9)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (u9)
    c9 = Conv2D(32, (3, 3), strides=1,activation=activation, kernel_initializer='he_normal', padding='same') (c9)

    output_layer = Conv2D(1, (1, 1),strides=1, activation='sigmoid') (c9)

    model = Model(inputs=[Input_layer], outputs=[output_layer])
    return model
best_model_fname=None
#best_model_fname='weights_bn.13-0.05.hdf5'
