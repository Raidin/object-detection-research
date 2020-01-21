
import keras
from keras.applications.vgg16 import VGG16
from keras import Model, metrics
from keras.optimizers import Adam
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

class RCNN():
    def __init__(self, config, data, class_num=2):
        self.cls_trn_img = data['cls_trn_img']
        self.cls_trn_lb = data['cls_trn_lb']
        self.reg_trn_img = data['reg_trn_img']
        self.reg_trn_delta = data['reg_trn_delta']
        self.class_num = class_num
        self.config = config

    def RegionProposal(self):
        print('RegionProposal!!!')

    def DataAugmentation(self, val_split=0.2):

        self.cls_trn_lb = to_categorical(self.cls_trn_lb, self.class_num)

        # Data Augmentation
        self.train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, rotation_range=90, validation_split=val_split)
        self.train_datagen.fit(self.cls_trn_img)

        # train-data
        self.train_generator = self.train_datagen.flow(self.cls_trn_img, self.cls_trn_lb, batch_size=128, shuffle=True, subset='training')
        # val-data
        self.validation_generator = self.train_datagen.flow(self.cls_trn_img, self.cls_trn_lb, batch_size=128, shuffle=True, subset='validation')

    def CreateModel(self, mode='cls'):
        pretrained_model = VGG16(weights='imagenet', include_top=True)

        for layers in (pretrained_model.layers)[:15]:
            layers.trainable = False

        X = pretrained_model.layers[-2].output

        if mode == 'cls':
            predictions = Dense(2, activation="softmax")(X)
        elif mode == 'reg':
            predictions = Dense(4, activation="linear")(X)

        self.model = Model(input=pretrained_model.input, output=predictions)

        # Multi GPU
        # self.model = multi_gpu_model(self.model, gpus=2)

        opt = Adam(lr=0.0001)

        if mode == 'cls':
            self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])
        elif mode == 'reg':
            self.model.compile(loss=keras.losses.mean_squared_error, optimizer=opt, metrics=[metrics.mse])

    def Train(self, mode='cls'):
        log = CSVLogger('{}/log.csv'.format(self.config['jog-dir']), append=True, separator=';')
        checkpoint = ModelCheckpoint(filepath='{}/vgg16-airplane_{epoch:02d}_{val_loss:.4f}.h5'.format(self.config['jog-dir']), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

        callback_list = [log, checkpoint, early]

        self.history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=int((self.cls_trn_img.shape[0] * (1.0 - 0.2)) / 128),
            epochs=1000,
            validation_data=self.validation_generator,
            validation_steps=int((self.cls_trn_img.shape[0] * 0.2) / 128),
            verbose=1,
            callbacks=callback_list)

    def SaveModel(self):
        # Model Save
        model_json = self.model.to_json()
        with open('network_model.json', 'w') as json_file:
            json_file.write(model_json)
        # To Text File
        with open('network_model.txt', 'w') as model_file:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(print_fn=lambda x: model_file.write(x + '\n'))

        # To Model Visualization
        # plot_model(model_final, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    def Test():
        print('RegionProposal!!!')

    def PredictObject():
        print('RegionProposal!!!')

    def BoundingBoxRegression():
        print('RegionProposal!!!')

    def RefineBoundingBox():
        print('RegionProposal!!!')

    def NonMaximumSuppression():
        print('RegionProposal!!!')
