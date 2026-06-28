import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
train_data=train_datagen.flow_from_directory(
    "Kagglecatsanddogs_5340/PetImages",
    target_size=(150,150),
    batch_size=32,
    class_mode="binary",
    subset="training"
)
validation_data=train_datagen.flow_from_directory(
    "Kagglecatsanddogs_5340/PetImages",
    target_size=(150,150),
    batch_size=32,
    class_mode="binary",
    subset="validation"
)
print("Dataset Loaded Successfully")
print(train_data.class_indices)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
model=Sequential()
model.add(Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.summary()
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
history=model.fit(train_data,validation_data=validation_data,epochs=5)
model.save("cat_dog_model.keras")
print("model saved successfully")