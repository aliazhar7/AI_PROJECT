# AI FASHION DSIGNER

Introducing the AI Fashion Designer project, a groundbreaking innovation poised to revolutionize the way we explore and discover fashion. Harnessing the power of cutting-edge technology, this project employs a Convolutional Neural Network (CNN) as its core engine, meticulously trained on a rich dataset teeming with diverse and captivating clothing designs. Its primary objective is to provide personalized recommendations by generating visually similar images to those already possessed by an individual. By leveraging the impressive capabilities of this AI-powered fashion guru, users can delve into an unparalleled realm of sartorial exploration, discovering new styles, and expanding their fashion horizons. Seamlessly bridging the gap between creativity and technology, the AI Fashion Designer project transcends traditional fashion paradigms, making it a compelling and indispensable tool for fashion enthusiasts, professionals, and those seeking to infuse their wardrobes with fresh inspiration. Get ready to unlock a world of style possibilities with this remarkable foray into the future of fashion design.

## Authors
- ALI AZHAR

## Features
- Utilizes a Convolutional Neural Network (CNN) for image analysis and generation.
- Trained on a comprehensive dataset of diverse and captivating clothing designs.
- Recommends visually similar clothing designs based on the user's existing wardrobe.
- Provides personalized fashion recommendations to inspire and expand users' style horizons.
- Offers a unique and innovative approach to infusing fresh inspiration into individual wardrobes.
- Opens up a world of style possibilities, revolutionizing the way we explore and discover fashion trends.

## Table of Contents
- Libraries
- Data
- URL to Image Function
- Data Pre-processing
- Image Processing
- Formatting Data for Model
- Model
- Image Prediction
- Predicting Similar Images
- Testing

## Libraries
```python
import pandas as pd
import numpy as np
import gdown
from matplotlib import pyplot as plt
import urllib 
import cv2   
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from sklearn.metrics import classification_report
}
```
### Installation
Install libraries using !pip install 'library name' e.g. 
```python
pip install gdown
```

## Data
```python
data=pd.read_csv("dress.csv",error_bad_lines=False, engine="python")
data
}
```
### Output:
![Screenshot 2023-05-08 103959](https://user-images.githubusercontent.com/132945205/236915632-5df2ab26-5158-44ee-baf9-1be4106e5ca3.jpg)

## URL to Image Function
```python
def show_image_from_url(image_url):
    # Open the image URL and read its content
    with urllib.request.urlopen(image_url) as url:
        s = url.read()
    
    # Convert the content to a NumPy array of uint8 data type
    arr = np.asarray(bytearray(s), dtype=np.uint8)
    
    # Decode the image from the NumPy array using OpenCV
    image_bgr = cv2.imdecode(arr, -1)
    
    # Convert the color space from BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Display the image using matplotlib
    plt.imshow(image_rgb)
    
    # Remove the axis labels
    plt.axis('off')
    
    # Show the plot
    plt.show()

# Call the function with the image URL from the 'image_url' column of the 'data' DataFrame
show_image_from_url(data['image_url'].loc[666])
}
```
### Output:
![Screenshot 2023-05-08 053612](https://user-images.githubusercontent.com/132945205/236916058-04113d23-05b0-4d6f-b112-efacc1d03c99.jpg)

## Data Pre-processing
### Unique Value Count in 'category' Column
```python
# Assuming your dataframe is called 'data' and the column you want to plot is called 'category'
column_data = data['category']
category_counts = column_data.value_counts()

# Plotting the frequency graph
plt.figure(figsize=(6, 4))
category_counts.plot(kind='bar')
plt.xlabel('Categories')
plt.ylabel('Count')
plt.title('Frequency of Unique Categories')
plt.show()
}
```
### Output:
![Screenshot 2023-05-08 053348](https://user-images.githubusercontent.com/132945205/236916789-bbfb7bd5-211c-4fa5-ba2c-c083912d18b6.jpg)
### Statistics of Quantitative Data
```python
data.describe()
}
```
### Output:
![Screenshot 2023-05-08 123810](https://user-images.githubusercontent.com/132945205/236917138-bf47dee7-0c87-4fb8-8211-49ea3ef2f17c.jpg)
### Removing Other Variable
```python
data = data.loc[(data['category'] != 'OTHER')].reset_index(drop=True)
data['category'].value_counts()
}
```
### Output:
![Screenshot 2023-05-08 123907](https://user-images.githubusercontent.com/132945205/236917283-1d67f3a5-2843-4e3e-9537-429e518ce4aa.jpg)
### Selecting Categories with 100 Percent Accuracy
```python
data=data[data['category:confidence']==1]
data['category'].value_counts()
}
```
### Output:
![Screenshot 2023-05-08 124040](https://user-images.githubusercontent.com/132945205/236917621-15e25d7d-603a-46e8-8639-248c0846fe9a.jpg)
### Selecting Data
Data is selected in this section
```python
# Group the rows by category and limit the count of each group to 100
data = data.groupby('category').apply(lambda x: x.head(100))

# Reset the index of the resulting DataFrame and drop the old index
data = data.reset_index(drop=True)

# Print the counts of the unique values in the 'category' column
data['category'].value_counts()
}
```
```python
# Count the number of occurrences of each category
category_counts = data['category'].value_counts()

# Create a mask for rows with a category count of less than 10
mask = data['category'].isin(category_counts[category_counts >= 10].index)

# Filter the DataFrame using the mask
data = data[mask]
data['category'].value_counts()
}
```
### Shuffling + Dropping Columns
```python
data=data.sample(frac=1,random_state=42) # shuffling
data=data.reset_index()
data.drop(columns=['index'	,'_unit_id', 'category:confidence'],inplace=True)
data
}
```

## Image Pre-processing
```python
def image_processing(image_url):

    # Download from image URL and import it as a NumPy array
    response = urllib.request.urlopen(image_url)
    image = np.asarray(bytearray(response.read()), dtype="uint8")                         

    # Read the NumPy array as a color image in OpenCV
    image_bgr = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Convert the image to HSV color space for creating a mask
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Convert the image to grayscale that will be used for training
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Create a mask that detects the red rectangular tags present in each image
    mask = cv2.inRange(image_hsv, (0,255,255), (0,255,255))

    # Get the coordinates of the red rectangle in the image,
    # but take the entire image if the mask fails to detect the red rectangle
    if len(np.where(mask != 0)[0]) != 0:
        y1 = min(np.where(mask != 0)[0])
        y2 = max(np.where(mask != 0)[0])
    else:
        y1 = 0                                     
        y2 = len(mask)
    
    if len(np.where(mask != 0)[1]) != 0:
        x1 = min(np.where(mask != 0)[1])
        x2 = max(np.where(mask != 0)[1])
    else:
        x1 = 0
        x2 = len(mask[0])

    # Crop the grayscale image along those coordinates
    image_cropped = image_gray[y1:y2, x1:x2]

    # Resize the image to 100x100 pixels size
    image_100x100 = cv2.resize(image_cropped, (100, 100))

    # Save the image as an array of size 10000x1
    image_arr = image_100x100.flatten()
    return image_arr
}
```
```python
np.random.seed(17)
#displaying 5 images after preprocessing
for i in np.random.randint(0, len(X), 5):
  plt.figure()
  plt.imshow(X[i].reshape(100, 100)), plt.axis('off')
}
```
### Output:
![Screenshot 2023-05-08 124741](https://user-images.githubusercontent.com/132945205/236919384-739b2c22-7fdb-465a-b2b1-45d0f4d09589.jpg)

## Formatting Data for Model
#### Encoding
```python
#Fetch the categories column from the dataframe, and tranform into to numerical labels
encoder = preprocessing.LabelEncoder()
Targets = encoder.fit_transform(data['category'])
Targets
}
```
```python
#One-hot encoding of the Target vector
n_classes=16
Y = to_categorical(Targets, num_classes = n_classes)
}
```
### Splitting Data
```python
#Segregation of a test set for testing on the trained model
X_test = X[400:,]
Y_test = Y[400:,]
#Seperation of a validation set from the remaing training set (required for validation while training)
X_train, X_val, Y_train, Y_val = train_test_split(X[:400,], Y[:400,], test_size=0.10, random_state=13)
}
```
### 3 Dimensional
```python
#Reshape the input matrices such that each sample is three-dimensional
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 1)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)
}
```

## Model
### CNN Model
```python
# Create an instance of the Sequential model
model = Sequential()

# Add a 2D convolutional layer with 16 filters, a 3x3 kernel, and 'relu' activation
# Set the input shape to the specified 'input_shape'
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

# Add a BatchNormalization layer to normalize the activations of the previous layer
model.add(BatchNormalization())

# Add another 2D convolutional layer with 16 filters, a 3x3 kernel, and 'relu' activation
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

# Add another BatchNormalization layer
model.add(BatchNormalization())

# Add a MaxPooling layer with a stride of (2, 2) to downsample the spatial dimensions
model.add(MaxPool2D(strides=(2, 2)))

# Add a Dropout layer to randomly set a fraction (0.25) of the input units to 0 at each update during training
model.add(Dropout(0.25))

# Add another set of convolutional, normalization, pooling, and dropout layers similar to the previous ones
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(strides=(2, 2)))
model.add(Dropout(0.25))

# Flatten the output from the previous layers to a 1D vector
model.add(Flatten())

# Add a fully connected Dense layer with 512 units and 'relu' activation
model.add(Dense(512, activation='relu'))

# Add a Dropout layer
model.add(Dropout(0.25))

# Add another fully connected Dense layer with 1024 units and 'relu' activation
model.add(Dense(1024, activation='relu'))

# Add a Dropout layer
model.add(Dropout(0.4))

# Add the final output layer with 'n_classes' units and 'softmax' activation
model.add(Dense(n_classes, activation='softmax'))

# Set the learning rate for the Adam optimizer
learning_rate = 0.001

# Compile the model by specifying the loss function, optimizer, and metrics to evaluate during training
model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate), metrics=['accuracy'])
}
```
```python
# Summary
model.summary()
}
```
###Output:
![Screenshot 2023-05-08 125728](https://user-images.githubusercontent.com/132945205/236921344-3443e89b-cb59-4c96-99d9-20edb0343950.jpg)
### Training Model
```python
# Define the checkpoint to save the best model
save_at = "/content/model.hdf5"
save_best = ModelCheckpoint(save_at, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='max')
}
```
```python
# Training the CNN
history = model.fit( X_train, Y_train, epochs = 10, batch_size = 20,  callbacks=[save_best], verbose=1, validation_data = (X_val, Y_val))
# Plot the training history (Training accuracy & Validation accuracy)
}
```
### Training and Validation Accuracy
```python
# Access the accuracy values from the history object
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

# Plot the training and validation accuracy
epochs = range(1, len(training_accuracy) + 1)

plt.plot(epochs, training_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, validation_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
}
```
### Output: 
![Screenshot 2023-05-08 052046](https://user-images.githubusercontent.com/132945205/236921696-96d6194c-954c-4344-80f5-984181a3cd2f.jpg)




