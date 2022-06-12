import datetime
import tensorflow as tf
import tensorflow_datasets as tfds ##Import dataset provided by Tensorflow
import matplotlib.pyplot as plt #Plot images

##DATASETS
dataset_mnist = "mnist"
dataset_fashion_mnist = "fashion_mnist"
dataset_food101 = "food101"
##ASSIGNE ONE OF ABOVE CHOOSEN DATASET VARIABLES TO THE used_dataset VARIABLE BELOW
used_dataset = dataset_mnist
s_batch_size = 32

if used_dataset == dataset_mnist:
    test_dataset = "test"
    used_input_shape = (224, 224, 1)
    tensorboard_logs_path = "sequential_dataset_mnist"
    model_checkpoint_path = "model_checkpoints_dataset_mnist/cp.ckpt"
    model_saving_path = "saved_model/sequential_model_mnist"
    nr_of_epochs = 3
elif used_dataset == dataset_fashion_mnist:
    test_dataset = "test"
    used_input_shape = (224, 224, 1)
    tensorboard_logs_path = "sequential_dataset_fashion_mnist"
    model_checkpoint_path = "model_checkpoints_dataset_fashion_mnist/cp.ckpt"
    model_saving_path = "saved_model/sequential_model_fashion_mnist"
    nr_of_epochs = 3
elif used_dataset == dataset_food101:
    test_dataset = "validation"
    used_input_shape = (224, 224, 3)
    tensorboard_logs_path = "sequential_dataset_food101"
    model_checkpoint_path = "model_checkpoints_dataset_food101/cp.ckpt"
    model_saving_path = "saved_model/sequential_model_food101"
    nr_of_epochs = 8
else:
    print(used_dataset, " is not a valid dataset. Please select and assign one of the given datasets in the top of the code to the used_dataset variable.")

##USE THIS TO RUN ON CPU, IT WILL SHOW MORE ACCURATE ERRORS IF RUN ON CPU INSTEAD OF GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
##---------

##Control if GPU is found and allow memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
#print(physical_devices)
if physical_devices:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)

##List all availabe datasets
datasets_list = tfds.list_builders()
print("mnist" in datasets_list)
print("fashion_mnist" in datasets_list)
print("food101" in datasets_list)

##Load in the chosen dataset named (it wont re-download again)
(train_data, test_data), ds_info = tfds.load(name=used_dataset,
                                             split=["train", test_dataset],
                                             shuffle_files=True,
                                             as_supervised=True,
                                             with_info=True)

##Print features of Food101
print(ds_info.features)

##Get class names
class_names = ds_info.features["label"].names
print(class_names[:20])

##Get one sample of training data, for testing
train_one_sample = train_data.take(1)
print(train_one_sample)

##Output info about out training sample that we will test
for image, label in train_one_sample:
  print(f"""
  Image shape: {image.shape}
  Image datatype: {image.dtype}
  Target class from Food101 (tensor form): {label}
  Class name (str form): {class_names[label.numpy()]}
  """)

##Print to see what the tensor looks like:
print(image)

##Plot the image
#plt.imshow(image)
#plt.title(class_names[label.numpy()])
#plt.axis(False)
#plt.show()

##Preprocessing functions for our chosen data
def preprocess_img(image, label, img_shape=224):
    """
    Converts image datatype from 'uint8' to 'float32' and reshapes image to [img_shape, img_shape, colour_channels] dimensions
    :param image:
    :param label:
    :param img_shape:
    :return:
    """
    image = tf.image.resize(image, [img_shape, img_shape]) # #Resize tensors to be of same size
    return tf.cast(image, tf.float32), label # return (float32_image, label) tuple  Change type of tensor to float32

#preprocessed_img = preprocess_img(image, label)[0]
#print(f"Image before preprocessing:\n {image[:2]}..., \nShape: {image.shape},\nDatatype:: {image.dtype}\n")
#print(f"Image after preprocessing:\n {preprocessed_img[:2]}..., \nShape: {preprocessed_img.shape},\nDatatype:: {preprocessed_img.dtype}\n")

##BATCH & PREPARE DATASET
train_data = train_data.map(map_func = preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)

##Shuffle train_data and turn it into batches and prefetch it (load it faster)
#We shuffle 1000 elements
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=s_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#buffer_size is how many elements to shuffle at single time, we limit this because of limited memory (ram)
#We batch tensors into size of 32 instead of having a massiv array of tensor, we divide then into smaler batches

##Map preprocessing function to test data
test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=s_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

##CREATE MODELLING CALLBACKS
##Create tensorboard:
def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a Tesorboard callback instance to store log files.
    Stores log files with the filepath:
        "dir_name/experiment_name/current_datetime/"
    :param dir_name: Target directory to store Tensorboard lo files
    :param experiment_name: Name of the experiment directory (e.g. effecientnet_model_1)
    :return:
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    print(f"Saving Tensorboard log files to: {log_dir}")
    return tensorboard_callback

# Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=2)

#Create modelcheckpoint callback to save a model's progress during training, saving weights requires ".ckpt" extension
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path,
                                                      montior="val_accuracy",
                                                      save_best_only=True,
                                                      save_weights_only=True,
                                                      verbose=1)

# Creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                 factor=0.2,
                                                 patience=2,
                                                 verbose=1,
                                                 min_lr=1e-7)

num_classes = len(class_names)

model = tf.keras.models.Sequential([
  tf.keras.layers.Rescaling(1./255, input_shape=used_input_shape),
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

#Compile for both models
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(model.summary())

history_101_food_classes_feature_extract = model.fit(train_data,
                                                     epochs=nr_of_epochs,
                                                     verbose=1,
                                                     batch_size=s_batch_size,
                                                     steps_per_epoch=len(train_data),
                                                     validation_data=test_data,
                                                     validation_steps=int(0.15 * len(test_data)),
                                                     callbacks=[create_tensorboard_callback("training_logs",
                                                                                            tensorboard_logs_path),
                                                                model_checkpoint])#,
                                                                #early_stopping,
                                                                #reduce_lr])

print("Training completed")

#Save model
#print("Saving model and trained weights...")
#model.save(model_saving_path)
#print("Model was saved successfully")

#Load model
#print("Loading model...")
#loaded_saved_model = tf.keras.models.load_model(model_saving_path)
#print("Model was loaded successfully")

#Evaluate model
print("Evaluating model...")
results_model_evaluation = model.evaluate(test_data)
print("Model evaluations completed")