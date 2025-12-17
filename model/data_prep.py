import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

image_size = (64, 64)
batch_size = 32
image_path = './data/'


def load_data(data_dir=image_path):
    train_dataset = image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        image_size=image_size,
        interpolation='nearest',
        batch_size=batch_size,
        validation_split=0.3,
        subset='training',
        seed=42
    )

    validation_dataset = image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        image_size=image_size,
        interpolation='nearest',
        batch_size=batch_size,
        validation_split=0.3,
        subset='validation',
        seed=42
    )

    # Split validation dataset into validation and test datasets
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 2)
    validation_dataset = validation_dataset.skip(val_batches // 2)

    return train_dataset, validation_dataset, test_dataset, train_dataset.class_names
