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
    tensorboard_logs_path = "efficientnetb0_dataset_mnist"
    model_checkpoint_path = "model_checkpoints_dataset_mnist/cp.ckpt"
    model_saving_path = "saved_model/efficientnetb0_model_mnist"
    nr_of_epochs = 3
elif used_dataset == dataset_fashion_mnist:
    test_dataset = "test"
    used_input_shape = (224, 224, 1)
    tensorboard_logs_path = "efficientnetb0_dataset_fashion_mnist"
    model_checkpoint_path = "model_checkpoints_dataset_fashion_mnist/cp.ckpt"
    model_saving_path = "saved_model/efficientnetb0_model_fashion_mnist"
    nr_of_epochs = 3
elif used_dataset == dataset_food101:
    test_dataset = "validation"
    used_input_shape = (224, 224, 3)
    tensorboard_logs_path = "efficientnetb0_dataset_food101"
    model_checkpoint_path = "model_checkpoints_dataset_food101/cp.ckpt"
    model_saving_path = "saved_model/efficientnetb0_model_food101"
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
datasets_list = tfds.list_builders() #Get all available datasets in TFDS
print("mnist" in datasets_list) #Control if we have our dataset that we want, mnist
print("fashion_mnist" in datasets_list) #Control if we have our dataset that we want, fashion mnist
print("food101" in datasets_list) #Control if we have our dataset that we want, food101

##Load in the dataset named "food101" (once loaded it wont re-download again)
(train_data, test_data), ds_info = tfds.load(name=used_dataset,  #ÄNDRA MANUELLT 1/3
                                             split=["train", test_dataset], #Get training model and validation/test model
                                             shuffle_files=True, #Shuffle the tensors
                                             as_supervised=True, #Get lable and data
                                             with_info=True) #Get metadata aswell which will be stored in ds_info

##Print features of Food101
print(ds_info.features)

##THE FOLLOWING WILL BE NEEDED:
#1)Class names
#2)The shape of our input data (image tensors)
#3)The datatype of our input tensors
#4)What the lables look like (are they one-hot-encoded or are they lable encoded) THEY ARE NOT ONE-HOT-ENCODED
#5)Do the lables match up with the class names

##Get class names
class_names = ds_info.features["label"].names
print(class_names[:20])

##Get one sample of training data
train_one_sample = train_data.take(1) #samples are in format (image_tensor, lable)
print(train_one_sample)

##Output info about out training sample
for image, label in train_one_sample:
  print(f"""
  Image shape: {image.shape}
  Image datatype: {image.dtype}
  Target class from Food101 (tensor form): {label}
  Class name (str form): {class_names[label.numpy()]}
  """)

##What does our image tensors from Food101 look like:
print(image)

##Plot the image
#plt.imshow(image)
#plt.title(class_names[label.numpy()]) ##Add title associated with image to plot
#plt.axis(False)
#plt.show()

##Create preprocessing functions for our data (neural network works when data is in a certain way)
#This preprocessing will be done to make our tensors in the shape that works best for neuralnetworks (is fastests)
##What models like:
#1) Models like data in float32 type (for mixed precision float 16)
#2) For batches, Tensorflow like all of tensors within a batch
#3) Scaled (normalized, meaning numburs between 1 and 0) tensors, they perform better.
#4) Tensors that are resized to same size
def preprocess_img(image, label, img_shape=224): #ÄNDRA MANUELLT 2/3
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
#With map we see to it that all elements of the tensor goes through our preprocessing method and not just one of them,
#because we want all of our elements in the tensor to be preprocessed.
#tf.data.AUTOTUNE is used so that we use multiple threads to handle alla the batching instead of just using one thread, it
#divides the labour of batching to multiple threads, so they work parallell with eachother and it goes faster this way.

##Shuffle train_data and turn it into batches and prefetch it (load it faster)
#We shuffle the tensors again so that the neural network dosent learn by the order of the numbers, we shuffle 1000 elements
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=s_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
#buffer_size is how many elements to shuffle at single time, we limit this because of limited memory (ram)
#We batch tensors into size of 32 instead of having a massiv array of tensor, we divide then into smaler batches

##Map preprocessing function to test data
test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size=s_batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

##CREATE MODELLING CALLBACKS
#Modelling callbacks (tensorboard etc) are important to track the progress and also save the progress, set checkpoints, so that we dont need to
#redo everything if anything goes wrong during training of a modell or inference

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
                                                  patience=2) # if val loss decreases for 2 epochs in a row, stop training

#Create modelcheckpoint callback to save a model's progress during training, saving weights requires ".ckpt" extension
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path,
                                                      montior="val_accuracy", # save the model weights with best validation accuracy
                                                      save_best_only=True, # only save the best weights
                                                      save_weights_only=True, # only save model weights (not whole model)
                                                      verbose=1) # don't print out whether or not model is being saved

# Creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=2,
                                                 verbose=1, # print out when learning rate goes down
                                                 min_lr=1e-7)

#Download base model and freez underlying layers
base_model = tf.keras.applications.EfficientNetB0(include_top=False) #weights='ImageNet' (pre-training on ImageNet network) weights=None (pre-training on dataset)
base_model.trainable = False # freeze base model layers

#Create functional model
inputs = tf.keras.layers.Input(shape=used_input_shape, name="input_layer", dtype=tf.float16) #What shape our tensors are Innana: shape=input_shape_mnist
#x = base_model(inputs, training=False) # set base_model to inference mode only
#x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = base_model(inputs, training=False) # set base_model to inference mode only
x = tf.keras.layers.GlobalAveragePooling2D(name="pooling_layer")(x)
x = tf.keras.layers.Dense(len(class_names))(x) # want one output neuron per class
outputs = tf.keras.layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)
#outputs = tf.keras.layers.Dense(128, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

#Compile for both models
model.compile(loss="sparse_categorical_crossentropy", # Use sparse_categorical_crossentropy when labels are *not* one-hot
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

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

##Save model
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