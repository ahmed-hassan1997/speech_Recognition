from keras import backend as K
from keras.models import Model,Sequential
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM , RepeatVector ,Bidirectional )





def simple_rnn_model_spectogram(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    rnn=LSTM(64, return_sequences=True)(input_data)
    rnn2 =LSTM(128,return_sequences=True)(rnn)
    rnn3 =LSTM(256 ,return_sequences=True)(rnn2)
    logits = TimeDistributed(Dense(output_dim))(rnn3)
    
    model = Model(input_data, Activation('softmax')(logits))
    model.output_length = lambda x: x
    print(model.summary())
    return model

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    rnn=GRU(64, return_sequences=True)(input_data)
    rnn2 =GRU(128,return_sequences=True)(rnn)
    rnn3 =GRU(256 ,return_sequences=True)(rnn2)
    logits = TimeDistributed(Dense(output_dim))(rnn3)
    
    model = Model(input_data, Activation('softmax')(logits))
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name = 'batch_normalization1')(simp_rnn)
    rnn2 = GRU(256, activation=activation , return_sequences=True , name='rnn2' ,dropout=0.2)(bn_rnn)
    bn_rnn2 = BatchNormalization(name = 'batch_normalization2')(rnn2)
    rnn3 = GRU(512, activation=activation , return_sequences=True , name='rnn3' ,dropout=0.5)(bn_rnn2)

    bn_rnn3 = BatchNormalization(name = 'batch_normalization3')(rnn3)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn3)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
#      # Add convolutional layer
#     conv_2d = Conv1D(filters, 5, 
#                      strides=conv_stride, 
#                      padding=conv_border_mode,
#                      activation='relu',
#                      name='conv2d')(bn_cnn)
#     bn_cnn1 = BatchNormalization(name='bn_conv_2d')(conv_2d)

    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
      # TODO: Add batch normalization
    bn_rnn =BatchNormalization(name='bn_simple_rnn1')(simp_rnn)
    simp_rnn2 = GRU(256, activation='relu',
        return_sequences=True, implementation=2, name='rnn2' , dropout = 0.2)(bn_rnn)
    # TODO: Add batch normalization
    bn_rnn2 =BatchNormalization(name='bn_simple_rnn2')(simp_rnn2)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn2)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    if recur_layers == 1:
        layer1 = LSTM(units, return_sequences=True, activation='relu')(input_data)
        layer2 = BatchNormalization(name='bt_rnn_1')(layer1)
    else:
        
        layer1 = LSTM(units, return_sequences=True, activation='relu', dropout=0.2)(input_data)
        layer2 = BatchNormalization(name='bt_rnn_1')(layer1)
        layer2 = LSTM(units, return_sequences=True, activation='relu',dropout=0.2)(layer2)
        layer2 = BatchNormalization(name='bt_rnn_2')(layer2)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(layer2)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    rnn1 = LSTM(units, return_sequences=True, dropout=0.1)
#     rnn2 = LSTM(units, return_sequences=True, dropout=0.1)
    # TODO: Add bidirectional recurrent layer
    bidirectional=Bidirectional(rnn1, input_shape = (None, input_dim))(input_data)
#     bidirectional1=Bidirectional(rnn2)(bidirectional)
  
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidirectional)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
      # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
   
  
    bn_cnn1 = BatchNormalization(name='bn_conv_1d')(conv_1d)

    # TODO: Specify the layers in your network
    rnn1 = LSTM(units, return_sequences=True, dropout=0.4)
    rnn2 = LSTM(units, return_sequences=True, dropout=0.5)
    bidirectional=Bidirectional(rnn1)(bn_cnn1)
    bidirectional1=Bidirectional(rnn2)(bidirectional)
    
     # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidirectional1)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model
    