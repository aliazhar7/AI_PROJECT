# AI FASHION DSIGNER

Introducing the AI Fashion Designer project, a groundbreaking innovation poised to revolutionize the way we explore and discover fashion. Harnessing the power of cutting-edge technology, this project employs a Convolutional Neural Network (CNN) as its core engine, meticulously trained on a rich dataset teeming with diverse and captivating clothing designs. Its primary objective is to provide personalized recommendations by generating visually similar images to those already possessed by an individual. By leveraging the impressive capabilities of this AI-powered fashion guru, users can delve into an unparalleled realm of sartorial exploration, discovering new styles, and expanding their fashion horizons. Seamlessly bridging the gap between creativity and technology, the AI Fashion Designer project transcends traditional fashion paradigms, making it a compelling and indispensable tool for fashion enthusiasts, professionals, and those seeking to infuse their wardrobes with fresh inspiration. Get ready to unlock a world of style possibilities with this remarkable foray into the future of fashion design.

## Author
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
### Data Source 🔗 Link:
Kaggle, https://www.kaggle.com/datasets/zalando-research/fashionmnist
[![kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)](https://www.kaggle.com/)
### Creating a DataFrame
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

My CNN model can only work with numbers. What I have are Image URLs and their categorical labels. I will deal with the labels later. Now I need to convert the images into something the CNN can work with.

As I have seen earlier, the images are colored. An image is just a matrix of pixels. Every pixel in a colored image consists of three-dimensional information (amount of red, green, and blue respectively). The pixels of grayscale (black and white) images, however, only need one-dimensional data. It varies from 0 to 255. The lower corresponds to absolute white and the upper end corresponds to absolute black. Everything in between can represent any shade of grayscale possible.

I will use urllib to download the images from the URLs and use numpy to convert the images into nice clean ndarrays. I will use OpenCV to read those ndarrays in color form (The order in OpenCV is Blue-Green-Red). I will convert it to another format 'HSV' where I can create a mask that will detect those red rectangles.

Using OpenCV, I can either read the ndarray directly in grayscale format or convert the BGR format to grayscale.

Then I will fetch the pixel coordinates of the corners of the detected rectangles and using them crop the grayscale versions. I also need uniformity in my data. The original images have different shapes. So, I will reshape every cropped grayscale image into 100x100 pixels.

Every OpenCV image is stored as array data. So all these ndarrays can be flattened into a single-dimensional array of length 10000x1 (I can reshape arrays into any desirable dimensions as long as the total number of elements in the original array and the transformed array remain the same.

Note that reshaping images on OpenCV though is different since there is no loss of information. I can compromise with the image resolution there.

Also note that I am only using OpenCV to read and edit images. I will use Pyplot to view the images, and that requires RGB (Red-Green-Blue) format. I can use OpenCV to convert from BGR to RGB.
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
### Output:
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

## Image Prediction
```python
# Predict the labels for the test data
Y_pred = np.round(model.predict(X_test))

# Set the random seed for reproducibility
np.random.seed(87)

# Display a random sample of 5 test images along with their predicted labels
for rand_num in np.random.randint(0, len(Y_test), 5):
  # Create a new figure for each image
  plt.figure()
  
  # Display the image
  plt.imshow(X_test[rand_num].reshape(100, 100))
  
  # Turn off the axis
  plt.axis('off')
  
  # Check if the predicted label matches the true label
  if np.where(Y_pred[rand_num] == 1)[0].sum() == np.where(Y_test[rand_num] == 1)[0].sum():
    # If the predicted label does not match the true label, set the title as the predicted label in red color
    plt.title(encoder.classes_[np.where(Y_pred[rand_num] == 1)[0].sum()], color='r')
  else:
    # If the predicted label matches the true label, set the title as the predicted label in green color
    plt.title(encoder.classes_[np.where(Y_pred[rand_num] == 1)[0].sum()], color='g')
}
```
### Output:

## Predicting Similar Images
```python
def find_similar_images(image_url, no_of_images):

  # Preprocess the query image
  X_query = image_processing(image_url)
  X_query = X_query/255
  X_query = X_query.reshape(1, 100, 100, 1)

  # Make a prediction on the query image
  Y_query = np.round(model.predict(X_query))

  # Get the predicted dress category index
  i = np.where(Y_query == 1)[0][1]

  # Print the detected dress category
  print('Type detected by model:', encoder.classes_[i].upper())

  # Filter the dataset to retrieve similar images of the detected dress category
  data_req = data.loc[data['category'] == encoder.classes_[i]]
  data_req = data_req.reset_index(drop=True)

  # Check if the number of requested images is more than the available similar images in the dataset
  if no_of_images > len(data_req):
    return print('number of images needed is more than similar images in the dataset')

  # Plot the query image
  plt.figure()
  show_image_from_url(image_url)
  plt.title('Query Image')

  # Plot the similar images
  c = 1
  np.random.seed(13)
  for j in np.random.randint(0, len(data_req), no_of_images):
    #plt.figure()
    url = data_req['image_url'].iloc[j]
    show_image_from_url(url)
    plt.title('Similar Image {}'.format(c))
    c += 1
}
```
### Output:


## Testing
To run tests, run the following command:
```bash
  find_similar_images('https://i.dailymail.co.uk/1s/2018/11/06/23/5855600-6360713-Ashley_James_stuns_in_emerald_green_animal_print_dress_at_glitzy-a-123_1541546195058.jpg', 2)
}
```
### Output:
![Screenshot 2023-05-08 131514](https://user-images.githubusercontent.com/132945205/236926111-9bc6dcd3-ca5b-419e-838e-7f68dad6bd34.jpg)
### Similar Images: 
![Screenshot 2023-05-08 131732](https://user-images.githubusercontent.com/132945205/236925653-d0cacc8d-09a3-4b3a-8c05-d705cc27bda2.jpg)
![Screenshot 2023-05-08 131713](https://user-images.githubusercontent.com/132945205/236925660-ca89f4e0-dd3f-4799-bbc3-b2132c1194b4.jpg)



