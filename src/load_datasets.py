import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(dataset_dir, image_size, batch_size):

    seed = 7891 #1337 #2162 #6543 #9513
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir+"/training",
        validation_split=0.1112,
        subset="training",
        seed=seed,
        labels="inferred",
        label_mode="categorical",
        interpolation="bilinear",
        image_size=image_size,
        batch_size=batch_size,
        color_mode='rgb'
    )
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir+"/training",
        validation_split=0.1112,
        subset="validation",
        seed=seed,
        labels="inferred",
        label_mode="categorical",
        interpolation="bilinear",
        image_size=image_size,
        batch_size=batch_size,
        color_mode='rgb'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir+"/test",
        seed=seed,
        labels="inferred",
        label_mode="categorical",
        interpolation="bilinear",
        image_size=image_size,
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=False
    )

    return train_ds, val_ds, test_ds 


def load_figshare(directory, image_size, batch_size):
    class_names = ["Glioma", "Meningioma", "Pituitary tumor"]
    train_ds, val_ds, test_ds=load_dataset(
        directory+"figshare_dataset", image_size, batch_size
    )
    return train_ds, val_ds, test_ds, class_names


def load_kaggle(directory, image_size, batch_size):
    class_names = ["Glioma", "Meningioma", "No tumor", "Pituitary tumor"]
    train_ds, val_ds, test_ds=load_dataset(
        directory+"kaggle_dataset", image_size, batch_size
    )
    return train_ds, val_ds, test_ds, class_names


def load(dataset, directory, image_size, batch_size, verbose=False):
    # load one of the datasets
            
    if dataset=="figshare":
        train_ds, val_ds, test_ds, class_names = load_figshare(
            directory, image_size, batch_size
        )
    else:
        train_ds, val_ds, test_ds, class_names = load_kaggle(
            directory, image_size, batch_size
        )


    if verbose:
        train=np.concatenate([np.argmax(l,axis=1) for x,l in train_ds], axis=0)
        val=np.concatenate([np.argmax(l,axis=1) for x,l in val_ds], axis=0)
        test=np.concatenate([np.argmax(l,axis=1) for x,l in test_ds], axis=0)

        print()
        print("Summary:")
        print("--------------------------------")
        print()
        print("Dataset Classes: ", class_names)
        print()
        print("Dataset split:")
        print("--------------------------------")
        print("Using ", len(train), " files for training")
        print("Using ", len(val), " files for validation")
        print("Using ", len(test), " files for testing")
        print()
        print("Images per label in each set:")
        print("--------------------------------")

        print("Training: ")    
        for i,c in enumerate(class_names):
            print("   ", np.count_nonzero(train==i), c)
            
        print("Validation: ")
        for i,c in enumerate(class_names):
            print("   ", np.count_nonzero(val==i), c)
            
        print("Test: ")
        for i,c in enumerate(class_names):
            print("   ", np.count_nonzero(test==i), c)

        # show sample images
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[np.argmax(labels[i])])
                plt.axis("off")

    return train_ds, val_ds, test_ds, class_names
