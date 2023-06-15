# this module creates objects of the different models

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    VGG16, VGG19, Xception, ResNet50, ResNet50V2, ResNet101,
    ResNet101V2, ResNet152, ResNet152V2, EfficientNetB0,
    EfficientNetB4,EfficientNetV2B0,EfficientNetV2B3,
    MobileNet,MobileNetV3Large,MobileNetV3Small,MobileNetV2, 
    DenseNet201, ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase
)


def custom_block(model, filters):
    model.add(layers.Conv2D(filters, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.BatchNormalization())


def build_cnn_model(
    model, model_name, num_classes, dimx,
    batch_size, transfer_learning
):
    if transfer_learning:
        weights = "imagenet"
        trainable = False
        include_top = False
    else:
        weights = None
        trainable = True
        include_top = True

    if model_name=="custom":
        custom_block(model, 64)  # block 1
        custom_block(model, 128) # block 2
        custom_block(model, 256) # block 3
        custom_block(model, 256) # block 4
        custom_block(model, 512) # block 5
    elif model_name=="vgg16":
        model.add(VGG16(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="vgg19":
        model.add(VGG19(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="xception":
        model.add(Xception(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="resnet50":
        model.add(ResNet50(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="resnet50v2":
        model.add(ResNet50V2(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="resnet101":
        model.add(ResNet101(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="resnet101v2":
        model.add(ResNet101V2(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="resnet152":
        model.add(ResNet152(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="resnet152v2":
        model.add(ResNet152V2(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="efficientnetb0":
        model.add(EfficientNetB0(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="efficientnetb4":
        model.add(EfficientNetB4(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="efficientnetb0v2":
        model.add(EfficientNetV2B0(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes,
                    include_preprocessing=False))
    elif model_name=="efficientnetb3v2":
        model.add(EfficientNetV2B3(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes,
                    include_preprocessing=False))
    elif model_name=="mobilenet":
        model.add(MobileNet(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="mobilenetv3":
        model.add(MobileNetV3Large(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes,
                    include_preprocessing=True))
    elif model_name=="densenet":
        model.add(DenseNet201(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="convnexttiny":
        model.add(ConvNeXtTiny(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="convnextsmall":
        model.add(ConvNeXtSmall(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))
    elif model_name=="convnextbase":
        model.add(ConvNeXtBase(
                    weights=weights, include_top=include_top, 
                    input_shape=(dimx,dimx,3),classes=num_classes))

    model.layers[-1].trainable=trainable

    return model

        
def create_model(
    model_name, num_classes, dimx, batch_size=32, 
    data_augmentation=False, transfer_learning=False
):
    # base model
    model = models.Sequential()
    model.add(layers.Input(shape=(dimx,dimx,3)))
    model.add(layers.Rescaling(1./127.5,offset=-1))

    # data augmentation
    if data_augmentation:
        model.add(layers.RandomFlip("horizontal"))
        model.add(layers.RandomRotation(0.1, fill_mode='constant',fill_value=-1.0))
        model.add(layers.RandomZoom((-0.2,0.2), fill_mode='constant',fill_value=-1.0))

    model=build_cnn_model(
        model, model_name, num_classes, dimx, 
        batch_size, transfer_learning
    )
        
    # MLP on top
    if transfer_learning or model_name=="custom":
        model.add(layers.Flatten())
        if model_name=='custom':
            model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        loss="categorical_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy"],
    )

    return model
