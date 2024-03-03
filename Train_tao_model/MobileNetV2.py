# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

INIT_LR = 1e-4
EPOCHS = 2
BS = 32

# Data generators
train_datagen= ImageDataGenerator(rescale=1./255)
# Data flow
train_generator = train_datagen.flow_from_directory(
                            directory="data/train/",
                            target_size=(224, 224),
                            color_mode="rgb",
                            batch_size=BS,
                            class_mode="categorical",
                            shuffle=True,
                            seed=42
                        )

valid_datagen = ImageDataGenerator(rescale=1./255)
valid_generator = valid_datagen.flow_from_directory(
                            directory="data/valid/",
                            target_size=(224, 224),
                            color_mode="rgb",
                            batch_size=BS,
                            class_mode="categorical",
                            shuffle=True,
                            seed=42
                        )

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
                            directory="data/test/",
                            target_size=(224, 224),
                            color_mode="rgb",
                            batch_size=1,
                            class_mode=None,
                            shuffle=False,
                            seed=42
                            )
n_train_steps = train_generator.n//train_generator.batch_size
n_valid_steps = valid_generator.n//valid_generator.batch_size

model = MobileNetV2(weights=None, include_top=True, classes=3, classifier_activation="softmax", input_tensor=Input(shape=(224, 224, 3)))

# compile model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"]) # 3 lớp


# train the head of the network
print("[INFO] training head...")
H = model.fit(train_generator,
            steps_per_epoch=n_train_steps,
            validation_data=valid_generator,
            validation_steps=n_valid_steps,
            epochs=EPOCHS)

model.summary()
model.save("MobileNetV2.h5")
# make predictions on the testing set
print("[INFO] evaluating network...")

n_test_steps = test_generator.n
test_generator.reset()
y_pred = model.predict(test_generator,steps=n_test_steps,verbose=1)
y_pred = np.argmax(y_pred,axis=1)
print(classification_report(test_generator.classes,y_pred,target_names=["Correct_mask", "InCorrect_mask","Without_mask"]))

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("plotMobileNetV2.png")

