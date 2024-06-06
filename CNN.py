#Installing dependencies and libraries for Data preprocessing and data collection
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
dataset_path = "/Volumes/Development/Kidneyy/dataset"
categories = ["cyst", "stone", "normal", "tumor"]


# In[2]:


#Preprocessing of data(Resizing images, normalizing it)
images = []
labels = []
img_paths = []

for category_id, category in enumerate(categories):
    folder_path = os.path.join(dataset_path, category)
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (224, 224))  # Resize to a common size
        img = img / 255.0  # Normalize pixel values to [0, 1]
        images.append(img)
        img_paths.append(img_path) 
        labels.append(category_id)
        print(f"Image path: {img_path}, Category: {category}")

images = np.array(images)
labels = np.array(labels)


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


# In[7]:


plt.figure(figsize=(10, 10))
for i in range(9):    
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(categories[labels[i]])
    plt.axis("off")
plt.show()


# In[5]:


np.savez("preprocessed_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


# In[15]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# In[16]:


# Define CNN model architecture
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(categories), activation='softmax')
])


# In[17]:


# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[18]:


#Training the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# In[19]:


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


# In[20]:


# Predictions
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)


# In[ ]:





# In[23]:


import seaborn as sns

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# In[24]:


# Calculate normalized confusion matrix
conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

# Calculate correlation matrix
correlation_matrix = np.corrcoef(conf_matrix_norm)

# Plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", xticklabels=categories, yticklabels=categories)
plt.xlabel("Classes")
plt.ylabel("Classes")
plt.title("Correlation Matrix of Normalized Confusion Matrix")
plt.show()


# In[25]:


# Classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=categories))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[26]:


# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
