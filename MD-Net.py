from keras.models import *
from keras.layers import *
import keras.backend as K
import keras

import tensorflow as tf
IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1
def get_flops_params():
    sess = tf.compat.v1.Session()
    graph = sess.graph
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    params = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)
 
    return flops.total_float_ops

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1),block_id=1):

    channel_axis = 1 if IMAGE_ORDERING == 'channels_first' else -1
    filters = int(filters * alpha)
    x = ZeroPadding2D(padding=(1, 1), data_format=IMAGE_ORDERING ,name='conv_pad_%d' % block_id )(inputs)
    x = Conv2D(filters, kernel , data_format=IMAGE_ORDERING  ,
                    padding='valid',
                    use_bias=False,
                    strides=strides,
                    name='conv_%d' % block_id )(x)
    x = BatchNormalization(axis=channel_axis, name='conv_bn_%d' % block_id)(x)
    return Activation('relu', name='conv_relu_%d' % block_id)(x)
def SE(inputs,ratio=4,block_id=1):
    ex = GlobalAveragePooling2D(data_format=IMAGE_ORDERING)(inputs)
    outdim = inputs.get_shape().as_list()[-1]
    full1 = Dense(int(outdim/ratio),use_bias=True,name='full1_%d' % block_id)(ex)
    full1 = Activation('relu',name='full1_relu_%d' % block_id)(full1)
    full2 = Dense(outdim,use_bias=True,name='full2_%d' % block_id)(full1)
    full2 = Activation('sigmoid',name='full2_relu_%d' % block_id)(full2)
    scale = Reshape((1,1,outdim))(full2)
    return Multiply()([inputs, scale])
def conv_bn_relu_1_1(inputs, filters, kernel = (1, 1), strides = (1, 1), block_id = 1):
    channel_axis = -1 if IMAGE_ORDERING == 'channels_last' else 1
    x = Conv2D(filters, kernel, strides=strides, padding="same",
               data_format=IMAGE_ORDERING, use_bias=False, name='conv1_%d' % block_id )(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn_%d' % block_id)(x)
    return Activation('relu', name='conv1_relu_%d' % block_id)(x)
def resblock(inputs, strides = (1,1),dilation_rate=(1,1),block_id='1'):
    filters = inputs.get_shape().as_list()[-1]
    channel_axis = -1 if IMAGE_ORDERING == 'channels_last' else 1

    x = Conv2D(filters=int(filters/2),kernel_size=(1,1),strides = strides,padding='same',
               data_format=IMAGE_ORDERING, use_bias=False, name='resblock_conv1_%s' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='resblcok_bn1_%s' % block_id)(x)
    x = Activation('relu', name='resblock_relu1_%s' % block_id)(x)

    x = Conv2D(filters=int(filters/2), kernel_size=(3,3), strides=strides, padding="same",
               data_format=IMAGE_ORDERING, dilation_rate=dilation_rate, use_bias=False, name='resblock_conv2_%s' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='resblcok_bn2_%s' % block_id)(x)
    x = Activation('relu', name='resblock_relu2_%s' % block_id)(x)


    x = Conv2D(filters=filters,kernel_size=(1,1),strides = strides,padding='same',
               data_format=IMAGE_ORDERING, use_bias=False, name='resblock_conv3_%s' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='resblcok_bn3_%s' % block_id)(x)
    x = Activation('relu', name='resblock_relu3_%s' % block_id)(x)
    return Add()([x,inputs])
def Res_ASPP(inputs, filters, strides = (1,1), block_id='1'):
    channel_axis = -1 if IMAGE_ORDERING == 'channels_last' else 1
    x = resblock(inputs,  strides = strides,dilation_rate=(2,2), block_id=block_id+'_1')
    x2 = resblock(inputs, strides = strides,dilation_rate=(4,4), block_id=block_id+'_2')
    x3 = resblock(inputs, strides = strides,dilation_rate=(8,8) ,block_id=block_id+'_3')
    x4 = resblock(inputs, strides = strides,dilation_rate=(16,16), block_id=block_id+'_4')
    o = ( concatenate([ x ,x2,x3,x4],axis=-1 )  )
    o = Conv2D(filters=filters,kernel_size=(1,1),strides = strides,padding='same',
               data_format=IMAGE_ORDERING, use_bias=False, name='convr1_%s' % block_id)(o)
    o = BatchNormalization(axis=channel_axis, name='r1_bn_%s' % block_id)(o)
    o = Activation('relu', name='r1_relu_%s' % block_id)(o)
    return o

def ASPP(inputs, filters, kernel = (3, 3), strides = (1, 1), block_id = 1):
    channel_axis = -1 if IMAGE_ORDERING == 'channels_last' else 1
    x = Conv2D(filters, kernel, strides=strides, padding="same",
               data_format=IMAGE_ORDERING, dilation_rate=(2,2), use_bias=False, name='convd1_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='convd1_bn_%d' % block_id)(x)
    x = Activation('relu', name='convd1_relu_%d' % block_id)(x)
    
    x2 = Conv2D(filters, kernel, strides=strides, padding="same",
               data_format=IMAGE_ORDERING, dilation_rate=(4,4), use_bias=False, name='convd2_%d' % block_id)(inputs)
    x2 = BatchNormalization(axis=channel_axis, name='convd2_bn_%d' % block_id)(x2)
    x2 = Activation('relu', name='convd2_relu_%d' % block_id)(x2)
  
    x3 = Conv2D(filters, kernel, strides=strides, padding="same",
               data_format=IMAGE_ORDERING, dilation_rate=(8,8), use_bias=False, name='convd3_%d' % block_id)(inputs)
    x3 = BatchNormalization(axis=channel_axis, name='convd3_bn_%d' % block_id)(x3)
    x3 = Activation('relu', name='convd3_relu_%d' % block_id)(x3)
    
    x4 = Conv2D(filters, kernel, strides=strides, padding="same",
               data_format=IMAGE_ORDERING, dilation_rate=(16,16), use_bias=False, name='convd4_%d' % block_id)(inputs)
    x4 = BatchNormalization(axis=channel_axis, name='convd4_bn_%d' % block_id)(x4)
    x4 = Activation('relu', name='convd4_relu_%d' % block_id)(x4)
    o = ( concatenate([ x ,x2,x3,x4],axis=-1 )  )
    o = conv_bn_relu_1_1(o, filters=filters, kernel = (1, 1), strides = (1, 1), block_id = block_id)
    return o
    

def unet( input_height=48 ,  imput_weight=48 , pretrained='imagenet' ):

    alpha=1.0
    depth_multiplier=1
    dropout=1e-3


    img_input = Input(shape=(input_height,imput_weight , 1 ))


    x = _conv_block(img_input, 32, alpha, strides=(1, 1),block_id=1)   
    x = _conv_block(x, 64, alpha, strides=(1, 1),block_id=2)
    x = Res_ASPP(x, 64, strides=(1, 1), block_id='1')


    f1 = x 
    f11 = MaxPool2D(pool_size=(2,2), strides=(2,2), name='f1/2')(f1) 
    f12 = MaxPool2D(pool_size=(4,4), strides=(4,4), name='f1/4')(f1) 
    
    
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), name='f1')(x) 
    x = _conv_block(x, 128, alpha, strides=(1, 1),block_id=3)
    x = _conv_block(x, 128, alpha, strides=(1, 1),block_id=4)
    x = Res_ASPP(x, 128, strides = (1, 1), block_id = '2')

    f2 = x 
    f21 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',name='convtran_f2-2')(f2)
    f22 = MaxPool2D(pool_size=(2,2),strides=(2,2), name='f2/2')(f2)
    
    
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), name='f2')(x) 
    x = _conv_block(x, 256, alpha, strides=(1, 1),block_id=5)
    x = _conv_block(x, 256, alpha, strides=(1, 1),block_id=6)
    x = Res_ASPP(x, 256, strides = (1, 1), block_id = '3')

    f3 = x 
    f31 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same',name='convtran_f3-2')(f3)
    f32 = Conv2DTranspose(256, (4, 4), strides=(4, 4), padding='same',name='convtran_f3-4')(f3)
    
    
    #decoder--------------------------------------------------------------------------------------------------
    
    c4 = ( concatenate([f3, f22, f12],axis=MERGE_AXIS )  )
    c4 = conv_bn_relu_1_1(c4, 256,kernel=(1,1), block_id=13)
    c4 = SE(c4, block_id=1)
    c4 = _conv_block(c4, 256, alpha, strides=(1, 1),block_id=7)
    c4 = _conv_block(c4, 256, alpha, strides=(1, 1),block_id=8)

    c5=Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',name='convtran_1')(c4)
    c5 = ( concatenate([c5 ,f2,f11,f31],axis=MERGE_AXIS )  )
    c5 = conv_bn_relu_1_1(c5, 128,kernel=(1,1), block_id=14)
    c5 = SE(c5, block_id=2)
    c5 = _conv_block(c5, 128, alpha, strides=(1, 1),block_id=9)
    c5 = _conv_block(c5, 128, alpha, strides=(1, 1),block_id=10)
    c6=Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same',name='convtran_2')(c5)  

    o = ( concatenate([c6 ,f1,f21,f32],axis=MERGE_AXIS )  )
    o = conv_bn_relu_1_1(o, 64,kernel=(1,1), block_id=15)
    o = SE(o, block_id=3)
    o = _conv_block(o, 64, alpha, strides=(1, 1),block_id=11)
    o = _conv_block(o, 32, alpha, strides=(1, 1),block_id=12) 

    o =  Conv2D( 1 , (1, 1) ,activation="sigmoid", padding='same', data_format=IMAGE_ORDERING )( o )
    model = Model(img_input,o)

    return model

def DRU_net(input_height=48,imput_weight=48):
    model=unet(input_height=input_height,imput_weight=imput_weight)
    model.model_name = 'DRU_net'
    return model
model=DRU_net()
# print(model.summary())
#
# print(get_flops(model))
# print(get_flops_params())
