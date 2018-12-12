from sklearn.model_selection import StratifiedKFold
from keras import losses, models, optimizers
from keras.activations import relu, softmax, tanh, sigmoid, linear, exponential
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.utils import Sequence, to_categorical
from keras.layers import (Convolution1D, Dense, Dropout, GlobalAveragePooling1D,
                          GlobalMaxPool1D, Input, MaxPool1D, concatenate)
class Sound_Models:
#
#     def get_model(self):
#         return [self.add, self.mul, self.divide]
#     def add(self,num1, num2):
#         return num1+num2
#     def mul(self,num1,num2):
#         return num1*num2
#     def divide(self,num1,num2):
#         return num1/num2
#
# # sm = Sound_Models()
#
# models = Sound_Models.get_models()
# for i in models:
#     print(i(2,3))
    def get_models(self):
        models = []
        models.append(self.predictions_1d_conv_output_1)

        # models.append(self.predictions_1d_conv_decrease)
        # models.append(self.predictions_1d_conv_double)
        # models.append(self.predictions_1d_conv_double_dropout)
        # models.append(self.predictions_1d_conv_activ_tanh)
        # models.append(self.predictions_1d_conv_activ_sig)
        # models.append(self.predictions_1d_conv_activ_linear)
        # models.append(self.predictions_1d_conv_activ_exp)
        # models.append(self.predictions_1d_conv_double_deep_unif)
        # models.append(self.predictions_1d_conv_double_deep_mixed)

        return models
    def get_names(self):
        names = []
        names.append('predictions_1d_conv_output_1')

        # names.append('predictions_1d_conv_decrease')
        # names.append('predictions_1d_conv_double')
        # names.append('predictions_1d_conv_double_dropout')
        # names.append('predictions_1d_conv_activ_tanh')
        # names.append('predictions_1d_conv_activ_sig')
        # names.append('predictions_1d_conv_activ_linear')
        # names.append('predictions_1d_conv_activ_exp')
        # names.append('predictions_1d_conv_double_deep_unif')
        # names.append('predictions_1d_conv_double_deep_mixed')


        return names

    #
    # def predictions_1d_conv_base(self,config):
    #
    #     nclass = config.n_classes
    #     input_length = config.audio_length
    #
    #     inp = Input(shape=(input_length,1))
    #     x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
    #     x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
    #     x = MaxPool1D(16)(x)
    #     x = Dropout(rate=0.1)(x)
    #
    #     x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    #     x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    #     x = MaxPool1D(4)(x)
    #     x = Dropout(rate=0.1)(x)
    #
    #     x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    #     x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
    #     x = MaxPool1D(4)(x)
    #     x = Dropout(rate=0.1)(x)
    #
    #     x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    #     x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
    #     x = GlobalMaxPool1D()(x)
    #     x = Dropout(rate=0.2)(x)
    #     x = Dense(64, activation=relu)(x)
    #     x = Dense(1028, activation=relu)(x)
    #     out = Dense(nclass, activation=softmax)(x)
    #
    #     model = models.Model(inputs=inp, outputs=out)
    #     opt = optimizers.Adam(config.learning_rate)
    #
    #     model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    #     return model

    def predictions_1d_conv_output_1(self,config):

        nclass = config.n_classes
        input_length = config.audio_length

        inp = Input(shape=(input_length,1))
        x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = MaxPool1D(16)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(64, activation=relu)(x)
        x = Dense(1028, activation=relu)(x)
        out = Dense(2, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(config.learning_rate)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model

    def predictions_1d_conv_double(self,config):

        nclass = config.n_classes
        input_length = config.audio_length

        inp = Input(shape=(input_length,1))
        x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)


        x = MaxPool1D(16)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(64, activation=relu)(x)
        x = Dense(1028, activation=relu)(x)
        out = Dense(nclass, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(config.learning_rate)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model

    def predictions_1d_conv_double_deep_unif(self,config):

        nclass = config.n_classes
        input_length = config.audio_length

        inp = Input(shape=(input_length,1))
        x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = MaxPool1D(16)(x)
        x = Dropout(rate=0.1)(x)






        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)



        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)


        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(rate=0.2)(x)

        x = Dense(64, activation=relu)(x)
        x = Dense(1028, activation=relu)(x)
        out = Dense(nclass, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(config.learning_rate)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model

    def predictions_1d_conv_double_deep_mixed(self,config):

        nclass = config.n_classes
        input_length = config.audio_length

        inp = Input(shape=(input_length,1))
        x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(16)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(64, activation=relu)(x)
        x = Dense(1028, activation=relu)(x)
        out = Dense(nclass, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(config.learning_rate)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model

    def predictions_1d_conv_decrease(self,config):

        nclass = config.n_classes
        input_length = config.audio_length

        inp = Input(shape=(input_length,1))
        x = Convolution1D(256, 9, activation=relu, padding="valid")(inp)
        x = Convolution1D(256, 9, activation=relu, padding="valid")(x)
        x = MaxPool1D(16)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)

        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(16, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(16, 3, activation=relu, padding="valid")(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(64, activation=relu)(x)
        x = Dense(1028, activation=relu)(x)
        out = Dense(nclass, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(config.learning_rate)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model

    def predictions_1d_conv_double_dropout(self,config):

        nclass = config.n_classes
        input_length = config.audio_length

        inp = Input(shape=(input_length,1))
        x = Convolution1D(16, 9, activation=relu, padding="valid")(inp)
        x = Convolution1D(16, 9, activation=relu, padding="valid")(x)
        x = MaxPool1D(16)(x)
        x = Dropout(rate=0.2)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.2)(x)

        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(32, 3, activation=relu, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.2)(x)

        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = Convolution1D(256, 3, activation=relu, padding="valid")(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(rate=0.4)(x)
        x = Dense(64, activation=relu)(x)
        x = Dense(1028, activation=relu)(x)
        out = Dense(nclass, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(config.learning_rate)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model

    def predictions_1d_conv_activ_tanh(self,config):

        nclass = config.n_classes
        input_length = config.audio_length

        inp = Input(shape=(input_length,1))
        x = Convolution1D(16, 9, activation=tanh, padding="valid")(inp)
        x = Convolution1D(16, 9, activation=tanh, padding="valid")(x)
        x = MaxPool1D(16)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=tanh, padding="valid")(x)
        x = Convolution1D(32, 3, activation=tanh, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=tanh, padding="valid")(x)
        x = Convolution1D(32, 3, activation=tanh, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(256, 3, activation=tanh, padding="valid")(x)
        x = Convolution1D(256, 3, activation=tanh, padding="valid")(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(64, activation=tanh)(x)
        x = Dense(1028, activation=tanh)(x)
        out = Dense(nclass, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(config.learning_rate)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model
    def predictions_1d_conv_activ_sig(self,config):

        nclass = config.n_classes
        input_length = config.audio_length

        inp = Input(shape=(input_length,1))
        x = Convolution1D(16, 9, activation=sigmoid, padding="valid")(inp)
        x = Convolution1D(16, 9, activation=sigmoid, padding="valid")(x)
        x = MaxPool1D(16)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=sigmoid, padding="valid")(x)
        x = Convolution1D(32, 3, activation=sigmoid, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=sigmoid, padding="valid")(x)
        x = Convolution1D(32, 3, activation=sigmoid, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(256, 3, activation=sigmoid, padding="valid")(x)
        x = Convolution1D(256, 3, activation=sigmoid, padding="valid")(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(64, activation=sigmoid)(x)
        x = Dense(1028, activation=sigmoid)(x)
        out = Dense(nclass, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(config.learning_rate)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model
    def predictions_1d_conv_activ_linear(self,config):

        nclass = config.n_classes
        input_length = config.audio_length

        inp = Input(shape=(input_length,1))
        x = Convolution1D(16, 9, activation=linear, padding="valid")(inp)
        x = Convolution1D(16, 9, activation=linear, padding="valid")(x)
        x = MaxPool1D(16)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=linear, padding="valid")(x)
        x = Convolution1D(32, 3, activation=linear, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=linear, padding="valid")(x)
        x = Convolution1D(32, 3, activation=linear, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(256, 3, activation=linear, padding="valid")(x)
        x = Convolution1D(256, 3, activation=linear, padding="valid")(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(64, activation=linear)(x)
        x = Dense(1028, activation=linear)(x)
        out = Dense(nclass, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(config.learning_rate)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model

    def predictions_1d_conv_activ_exp(self,config):

        nclass = config.n_classes
        input_length = config.audio_length

        inp = Input(shape=(input_length,1))
        x = Convolution1D(16, 9, activation=exponential, padding="valid")(inp)
        x = Convolution1D(16, 9, activation=exponential, padding="valid")(x)
        x = MaxPool1D(16)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=exponential, padding="valid")(x)
        x = Convolution1D(32, 3, activation=exponential, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(32, 3, activation=exponential, padding="valid")(x)
        x = Convolution1D(32, 3, activation=exponential, padding="valid")(x)
        x = MaxPool1D(4)(x)
        x = Dropout(rate=0.1)(x)

        x = Convolution1D(256, 3, activation=exponential, padding="valid")(x)
        x = Convolution1D(256, 3, activation=exponential, padding="valid")(x)
        x = GlobalMaxPool1D()(x)
        x = Dropout(rate=0.2)(x)
        x = Dense(64, activation=exponential)(x)
        x = Dense(1028, activation=exponential)(x)
        out = Dense(nclass, activation=softmax)(x)

        model = models.Model(inputs=inp, outputs=out)
        opt = optimizers.Adam(config.learning_rate)

        model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
        return model