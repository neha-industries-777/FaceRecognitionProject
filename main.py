import os
import cv2
import numpy as np

# Path to dataset
dataset_path =r"C:\Users\sheza\OneDrive\Desktop\FaceRecognitionProject\dataset\dataset\faces"

faces = []
labels = []
image_paths = []
names = {}
label_id = 0

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    
    if not os.path.isdir(person_path):
        continue
    
    names[label_id] = person_name
    
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
        
        img = cv2.resize(img, (100, 100))
        img_flat = img.flatten()
        
        faces.append(img_flat)
        labels.append(label_id)
        image_paths.append(image_path)
    
    label_id += 1

# Conversion to numpy arrays
faces = np.array(faces)
labels = np.array(labels)

print("Total faces loaded:", faces.shape)
print("Total labels:", labels.shape)
print(len(faces), len(labels), len(image_paths))
from sklearn.decomposition import PCA

# components
k = 80

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
faces = scaler.fit_transform(faces)

# Applying PCA
pca = PCA(n_components=120,svd_solver='randomized', whiten=True)
faces_pca = pca.fit_transform(faces)

print("Original shape:", faces.shape)
print("Reduced shape after PCA:", faces_pca.shape)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Splitting of data
X_train, X_test, y_train, y_test = train_test_split(
    faces_pca, labels, test_size=0.4, random_state=42
)

# ANN model
model = MLPClassifier(hidden_layer_sizes=(200,100),activation='relu',solver='adam', max_iter=700)

# Train model
model.fit(X_train, y_train)

# Prediction on test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nHow would you like to detect?")
print("1. Search by Image Path")
print("2. Search by Name")

choice = input("Enter choice (1/2): ")

if choice == "1":
    image_path = input("Enter image path: ")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("FAAHHHHH!!!  bro...that's not in my dataset...")
    else:
        img = cv2.resize(img, (100, 100))
        img_flat = img.flatten()

        # Applying transformations
        img_flat = scaler.transform([img_flat])
        img_pca = pca.transform(img_flat)

        # Prediction
        prediction = model.predict(img_pca)
        predicted_name = names[prediction[0]]

        # Distance checking
        distances = np.linalg.norm(X_train - img_pca, axis=1)
        min_distance = np.min(distances)

        print("Confidence distance:", min_distance)

        # conditions
        if min_distance > 3000:
            print("Face not found in dataset")
        else:
            print("Predicted Person:", predicted_name)

elif choice == "2":
    name_input = input("Enter name: ")

    found = False
    person_id = None

    # Find person ID
    for key, value in names.items():
        if value.lower() == name_input.lower():
            found = True
            person_id = key
            break

    if not found:
        print("Name not found in dataset")

    else:
        print("\nHow would you like result?")
        print("1. Negative Image")
        print("2. Colour Image")
        print("3. Show Image Path")

        option = input("Enter option (1/2/3): ")

        # Find an image of that person safely
        selected_path = None
        for i in range(len(labels)):
            if labels[i] == person_id:
                selected_path = image_paths[i]
                break

        if selected_path is None:
            print("Error: No image found for this person")
        else:
            img = cv2.imread(selected_path)

            if option == "1":
                import matplotlib.pyplot as plt

                negative = 255 - img
                negative_rgb = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)

                plt.imshow(negative_rgb)
                plt.title("Negative Image")
                plt.axis('off')
                plt.show()

            elif option == "2":
                import matplotlib.pyplot as plt

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img_rgb)
                plt.title("Colour Image")
                plt.axis('off')
                plt.show()

            elif option == "3":
                print("Image Path:", selected_path)

            else:
                print("Invalid option selected")

else:
    print("Invalid option selected")