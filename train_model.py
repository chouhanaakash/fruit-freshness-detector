import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

train_path = 'dataset/train'
test_path = 'dataset/test'

classes = os.listdir(train_path)
print(" Classes found:", classes)

count_data = {fruit_class: len(os.listdir(os.path.join(train_path, fruit_class))) for fruit_class in classes}
eda_df = pd.DataFrame(list(count_data.items()), columns=['Fruit Type', 'Image Count'])

plt.figure(figsize=(8,5))
sns.barplot(x='Fruit Type', y='Image Count', data=eda_df, palette='viridis')
plt.title(" Image Distribution per Class (Training Data)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_set = train_datagen.flow_from_directory(
    train_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    test_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6)

history = model.fit(
    train_set,
    epochs=20,
    validation_data=val_set,
    callbacks=[early_stop, reduce_lr]
)


plt.figure(figsize=(12,5))


plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='lime')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss', color='cyan')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


test_loss, test_acc = model.evaluate(test_set)
print(f"\n Test Accuracy: {test_acc*100:.2f}%")
print(f" Test Loss: {test_loss:.4f}")

Y_pred = model.predict(test_set)
y_pred = np.argmax(Y_pred, axis=1)

cm = confusion_matrix(test_set.classes, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='crest', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

report = classification_report(test_set.classes, y_pred, target_names=classes, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\nClassification Report:\n")
print(pd.DataFrame(report).transpose())

plt.figure(figsize=(8,4))
sns.barplot(x=report_df.index[:-3], y=report_df['f1-score'][:-3], palette='cool')
plt.title("F1-Score per Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



os.makedirs('model', exist_ok=True)
model.save('model/fruit_freshness_model.h5')
print("\n Model saved successfully!")
