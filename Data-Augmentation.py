import os, shutil

base_dir = './datasets/Dog_v_Cat_small'
checkpoint_dir = './checkpoint'

# Directory for Training, Validation, Test Set
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.utils import plot_model

inputs = layers.Input(shape=(150,150,3))

# representation layers
x = layers.Conv2D(32, (3,3), activation='relu')(inputs)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(128, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(128, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2,2))(x)

# fully connected layers
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x) # add a dropout layer
x = layers.Dence(512, activation='relu')(x)
x = layers.Dence(2, activation='softmax')(x)

model = keras.Model(inputs, x)
model.sumary()

from keras import optimizers

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

#data preprocessing
from keras.preprocessing.image import ImageDataGenerator

# data augmentation for training dataset
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir, # target directory
    target_size=(150,150), # input size to rescale
    batch_size=20, # (samples * epoch) / steps
    class_mode='categorical'
)

validation_generator = test_datagen.flow_directory(
    validation_dir, # target directory
    target_size=20, # input size to rescale
    class_mode='categorical' # change batch_size : 20
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch = range(1, len(acc)+1)

# Draw Graph
plt.plot(epochs, acc, 'bo', label='Training_acc')
plt.plot(epochs, val_acc, 'r', label='Validation_acc')
plt.xlabel('# of epoch')
plt.ylabel('accuracy')
plt.tittle('Training and validation accuracy')
plt.legend() # display label name

plt.figure()

plt.plot(epochs, loss, 'b--', label='Train_loss')
plt.plot(epochs, val_loss, 'r', label='Validation_loss')
plt.xlabel('# of epoch')
plt.ylabel('loss')
plt.tittle('Training and validation loss')
plt.legend() # display label name

plt.show()

##

latent_dim = 2

encorder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Flatten()(encorder_inputs)
x = layers.Dense(512, activation='relu')(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
x = Sampling()([z_mean, z_log_var])
encorder = keras.Model(encorder_inputs, [z_mean, z_log_var, z], name="encorder")
encorder.summary()
plot_model(encorder, to_file='vae_mlp_encorder.jpg', show_shape=True)

