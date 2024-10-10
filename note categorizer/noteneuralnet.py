import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("G:\\neural netwok\\notes_dataset.csv") #change the path to the path of the dataset


notes = df['Note'].values
categories = df['Category'].values
subcategories = df['Subcategory'].values
subcategories2 = df['Subcategory2'].values


vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(notes).toarray()


category_encoder = LabelEncoder()
subcategory_encoder = LabelEncoder()
subcategory2_encoder = LabelEncoder()
y_category = category_encoder.fit_transform(categories)
y_subcategory = subcategory_encoder.fit_transform(subcategories)
y_subcategory2 = subcategory2_encoder.fit_transform(subcategories2)


y = np.vstack((y_category, y_subcategory, y_subcategory2)).T


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


input_layer = layers.Input(shape=(X_train.shape[1],))

# hidden layers(10 layers with 50 neurons each) (you can change the number of layers and neurons as you like)
x = input_layer
for _ in range(10):
    x = layers.Dense(50, activation='relu')(x)

# Output layers
category_output = layers.Dense(len(np.unique(y_category)), activation='softmax', name='category_output')(x)
subcategory_output = layers.Dense(len(np.unique(y_subcategory)), activation='softmax', name='subcategory_output')(x)
subcategory2_output = layers.Dense(len(np.unique(y_subcategory2)), activation='softmax', name='subcategory2_output')(x)


model = models.Model(inputs=input_layer, outputs={'category_output': category_output, 'subcategory_output': subcategory_output, 'subcategory2_output': subcategory2_output})


model.compile(optimizer='adam', 
              loss={'category_output': 'sparse_categorical_crossentropy', 'subcategory_output': 'sparse_categorical_crossentropy', 'subcategory2_output': 'sparse_categorical_crossentropy'}, 
              metrics={'category_output': 'accuracy', 'subcategory_output': 'accuracy', 'subcategory2_output': 'accuracy'})


history = model.fit(X_train, {'category_output': y_train[:, 0], 'subcategory_output': y_train[:, 1], 'subcategory2_output': y_train[:, 2]}, 
                    epochs=50, batch_size=3, validation_data=(X_test, {'category_output': y_test[:, 0], 'subcategory_output': y_test[:, 1], 'subcategory2_output': y_test[:, 2]}))


def plot_performance(history):
    plt.figure(figsize=(12, 4))


    plt.subplot(1, 2, 1)
    plt.plot(history.history['category_output_accuracy'], label='Category Training Accuracy')
    plt.plot(history.history['val_category_output_accuracy'], label='Category Validation Accuracy')
    plt.plot(history.history['subcategory_output_accuracy'], label='Subcategory Training Accuracy')
    plt.plot(history.history['val_subcategory_output_accuracy'], label='Subcategory Validation Accuracy')
    plt.plot(history.history['subcategory2_output_accuracy'], label='Subcategory2 Training Accuracy')
    plt.plot(history.history['val_subcategory2_output_accuracy'], label='Subcategory2 Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(history.history['category_output_loss'], label='Category Training Loss')
    plt.plot(history.history['val_category_output_loss'], label='Category Validation Loss')
    plt.plot(history.history['subcategory_output_loss'], label='Subcategory Training Loss')
    plt.plot(history.history['val_subcategory_output_loss'], label='Subcategory Validation Loss')
    plt.plot(history.history['subcategory2_output_loss'], label='Subcategory2 Training Loss')
    plt.plot(history.history['val_subcategory2_output_loss'], label='Subcategory2 Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


plot_performance(history)

plt.show()


def predict_note_category(note):
    note_features = vectorizer.transform([note]).toarray()
    prediction = model.predict(note_features)
    predicted_category = category_encoder.inverse_transform([np.argmax(prediction['category_output'])])
    predicted_subcategory = subcategory_encoder.inverse_transform([np.argmax(prediction['subcategory_output'])])
    predicted_subcategory2 = subcategory2_encoder.inverse_transform([np.argmax(prediction['subcategory2_output'])])
    return predicted_category[0], predicted_subcategory[0], predicted_subcategory2[0]

#example prediction
new_note = "baire brishti porche,coffee khete icche korche khub, tahsan er notun gaaan ta shundor"
predicted_category, predicted_subcategory, predicted_subcategory2 = predict_note_category(new_note)
print(f"The predicted category for the note is: {predicted_category}")
print(f"The predicted subcategory for the note is: {predicted_subcategory}")
print(f"The predicted subcategory2 for the note is: {predicted_subcategory2}")
