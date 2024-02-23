from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QLabel, QPushButton, QComboBox, QSpinBox,QFileDialog
from PyQt6.QtGui import QPixmap
import sys
import random
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import Xception, ResNet50, VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from sklearn.model_selection import train_test_split


class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(71, 71, 3), activation='relu'))

        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)

        self.train_button = QPushButton('Train Model', self)
        self.train_button.clicked.connect(self.train_model)

        self.validation_button = QPushButton('Validation', self)
        self.validation_button.clicked.connect(self.validation_process)

        self.test_button = QPushButton('Test Model', self)
        self.test_button.clicked.connect(self.test_model)

        self.listbox = QComboBox(self)

        self.history = None

        self.data_augmentation_button = QPushButton('Data Augmentation', self)
        self.data_augmentation_button.clicked.connect(self.data_augmentation)

        self.data_folder = "/Users/bilalcg/Desktop/dataset"

        self.load_folders()

        self.image_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.train_button)
        layout.addWidget(self.validation_button)
        layout.addWidget(self.test_button)
        layout.addWidget(self.data_augmentation_button)
        layout.addWidget(self.listbox)
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_text)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.listbox.currentIndexChanged.connect(self.on_listbox_item_click)


        self.model = self.create_model()

        # Model eğitim işlemlerinin değişkenleri
        self.epoch_value = 10
        self.batch_size_value = 32
        self.early_stopping_value = 5

        self.epoch_label = QLabel(f'Epoch: {self.epoch_value}', self)
        self.batch_size_label = QLabel(f'Batch Size: {self.batch_size_value}', self)
        self.early_stopping_label = QLabel(f'Early Stopping: {self.early_stopping_value}', self)

        self.epoch_input = QSpinBox(self)
        self.epoch_input.setValue(10)
        self.epoch_input.setRange(1, 100)
        self.epoch_input.valueChanged.connect(self.update_epoch_label)

        self.batch_size_input = QSpinBox(self)
        self.batch_size_input.setValue(32)
        self.batch_size_input.setRange(1, 100)
        self.batch_size_input.valueChanged.connect(self.update_batch_size_label)

        self.early_stopping_input = QSpinBox(self)
        self.early_stopping_input.setValue(5)
        self.early_stopping_input.setRange(1, 20)
        self.early_stopping_input.valueChanged.connect(self.update_early_stopping_label)

        input_layout = QVBoxLayout()
        input_layout.addWidget(self.epoch_label)
        input_layout.addWidget(self.epoch_input)
        input_layout.addWidget(self.batch_size_label)
        input_layout.addWidget(self.batch_size_input)
        input_layout.addWidget(self.early_stopping_label)
        input_layout.addWidget(self.early_stopping_input)


        self.model_checkpoint_button = QPushButton('Model Checkpoint', self)
        self.model_checkpoint_button.clicked.connect(self.model_checkpoint)

        # model eğitim butonu
        self.train_results_button = QPushButton('Training Results', self)
        self.train_results_button.clicked.connect(self.training_results)


        self.confusion_matrix_button = QPushButton('Confusion Matrix', self)
        self.confusion_matrix_button.clicked.connect(self.confusion_matrix)

        # test butonu tanımlamaca
        self.test_system_button = QPushButton('Test System', self)
        self.test_system_button.clicked.connect(self.test_system)

        self.model_label = QLabel('Select Model:', self)
        self.model_combobox = QComboBox(self)
        self.model_combobox.addItems(['Initial Model', 'ResNetV2', 'ResNet50', 'VGG16'])
        self.model_combobox.currentIndexChanged.connect(self.on_model_selection_change)

        # layout
        layout.addWidget(self.model_checkpoint_button)
        layout.addWidget(self.train_results_button)
        layout.addWidget(self.confusion_matrix_button)
        layout.addWidget(self.test_system_button)
        layout.addLayout(input_layout)
        layout.addWidget(self.model_label)
        layout.addWidget(self.model_combobox)

        self.selected_model = self.create_model()

    #rasnetv2 modelke
    def create_resnet152v2_model(self):
            base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
            model = Sequential()
            model.add(base_model)
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            for layer in base_model.layers:
                layer.trainable = False

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

            return model

    #resnet50 modelke
    def create_resnet50_model(self):
               base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
               model = Sequential()
               model.add(base_model)
               model.add(Flatten())
               model.add(Dense(128, activation='relu'))
               model.add(Dense(1, activation='sigmoid'))

               for layer in base_model.layers:
                   layer.trainable = False

               model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

               return model
    #vgg16 modelke
    def create_vgg16_model(self):
               base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
               model = Sequential()
               model.add(base_model)
               model.add(Flatten())
               model.add(Dense(128, activation='relu'))
               model.add(Dense(1, activation='sigmoid'))

               for layer in base_model.layers:
                   layer.trainable = False

               model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

               return model

    def update_epoch_label(self, value):
        self.epoch_value = value
        self.epoch_label.setText(f'Epoch: {self.epoch_value}')

    def update_batch_size_label(self, value):
        self.batch_size_value = value
        self.batch_size_label.setText(f'Batch Size: {self.batch_size_value}')

    def update_early_stopping_label(self, value):
        self.early_stopping_value = value
        self.early_stopping_label.setText(f'Early Stopping: {self.early_stopping_value}')

    def load_folders(self):
        subfolders = [f.name for f in os.scandir(self.data_folder) if f.is_dir()]
        self.listbox.addItems(subfolders)

    def on_listbox_item_click(self, index):
        selected_folder = self.listbox.itemText(index)
        folder_path = os.path.join(self.data_folder, selected_folder)
        if folder_path:
            images = [f.path for f in os.scandir(folder_path) if f.is_file() and f.name.endswith(('.png', '.jpg', '.jpeg'))]
            if images:
                self.load_image(images[0])

    def load_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        print("Train Model Button Clicked")
        # Eğitim için kullanılacak verileri hazırlayın
        x_train, x_test, y_train, y_test = self.prepare_training_data()

        # Modeli eğitin
        self.history = self.model.fit(x_train, y_train, epochs=self.epoch_value, validation_data=(x_test, y_test))

    def prepare_training_data(self):
            print("Preparing Training Data")
            # Eğitim verilerini hazırla
            categories = ['with_mask', 'without_mask']
            training_data = []

            for category in categories:
                path = os.path.join("/Users/bilalcg/Desktop/dataset", category)
                label = categories.index(category)
                count = 0

                for file in os.listdir(path):
                    if count < 1000:
                        img_path = os.path.join(path, file)
                        img = cv2.imread(img_path)

                        if img is not None:  # Görüntü boş değilse devam et
                            img = cv2.resize(img, (64, 64))
                            training_data.append([img, label])
                            count += 1

            random.shuffle(training_data)
            X = []
            Y = []
            for image, label in training_data:
                X.append(image)
                Y.append(label)

            # Verileri normalize etmeden önce boyutları düzelt
            X = np.array(X)
            X = X / 255.0  # Normalize et

            Y = np.array(Y)

            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

            return x_train, x_test, y_train, y_test


    def validation_process(self):
        print("Validation Button Clicked")

        # Modelin performansını değerlendir
        x_test, y_test = self.prepare_test_data()
        loss, accuracy = self.model.evaluate(x_test, y_test)

        print(f"Validation Loss: {loss}")
        print(f"Validation Accuracy: {accuracy}")

    def test_model(self):
        print("Test Model Button Clicked")

        # Test verilerini hazırla
        x_test, y_test = self.prepare_test_data()

        # Model üzerinde tahmin yap
        predictions = self.model.predict(x_test)

        # Tahminleri sınıflara çevir
        predicted_labels = (predictions > 0.5).astype(int)

        # Confusion Matrix'i oluştur
        confusion_matrix = np.zeros((2, 2))
        for true_label, predicted_label in zip(y_test, predicted_labels):
            confusion_matrix[true_label][predicted_label] += 1

        # Tahmin sonuçlarını ekrana yazdır
        print("Confusion Matrix:")
        print(confusion_matrix)

        accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
        print(f"Accuracy: {accuracy}")

        precision = confusion_matrix[1][1] / np.sum(confusion_matrix[:, 1])
        print(f"Precision: {precision}")

        recall = confusion_matrix[1][1] / np.sum(confusion_matrix[1, :])
        print(f"Recall: {recall}")

    def prepare_test_data(self):
        print("Preparing Test Data")
        # Test verilerini hazırla
        categories = ['with_mask', 'without_mask']
        test_data = []

        for category in categories:
            path = os.path.join("/Users/bilalcg/Desktop/dataset", category)
            label = categories.index(category)
            count = 0

            for file in os.listdir(path):
                if count < 200:  # 200 test örneği alabilirsiniz, isteğe bağlı
                    img_path = os.path.join(path, file)
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (64, 64))
                    test_data.append([img, label])
                    count += 1

        random.shuffle(test_data)
        X_test = []
        Y_test = []
        for image, label in test_data:
            X_test.append(image)
            Y_test.append(label)

        # Verileri normalize etmeden önce boyutları düzelt
        X_test = np.array(X_test)
        X_test = X_test / 255.0  # Normalize et

        Y_test = np.array(Y_test)

        return X_test, Y_test

    def data_augmentation(self):
        print("Data Augmentation Button Clicked")

        # Eğitim için kullanılacak verileri hazırlayın
        categories = ['with_mask', 'without_mask']
        data = []

        for category in categories:
            path = os.path.join("/Users/bilalcg/Desktop/dataset", category)
            label = categories.index(category)
            count = 0

            for file in os.listdir(path):
                if count < 1000:
                    img_path = os.path.join(path, file)
                    img = cv2.imread(img_path)

                    if img is not None:  # Görüntü boş değilse devam et
                        img = cv2.resize(img, (64, 64))

                        # Veri artırma işlemleri
                        augmented_images = self.apply_data_augmentation(img)

                        for augmented_image in augmented_images:
                            data.append([augmented_image, label])

                        count += 1

        random.shuffle(data)
        X = []
        Y = []
        for image, label in data:
            X.append(image)
            Y.append(label)

        # Verileri normalize etmeden önce boyutları düzelt
        X = np.array(X)
        X = X / 255.0  # Normalize et

        Y = np.array(Y)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        # Modeli eğitin
        self.model.fit(x_train, y_train, epochs=self.epoch_value, validation_data=(x_test, y_test))

    def apply_data_augmentation(self, image):
        # Veri artırma işlemlerini uygula
        augmented_images = []

        if image is not None:  # Görüntü boş değilse devam et
            # Örneğin, resmi döndürme işlemi burada saat yönüne doğru 90 derece döndürülüyor
            rotated_image_1 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rotated_image_2 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            augmented_images.extend([rotated_image_1, rotated_image_2])

        return augmented_images


    def model_checkpoint(self):
        print("Model Checkpoint Button Clicked")

        # Kontrol noktası dosya yolu ve adı
        checkpoint_path = '/Users/bilalcg/Desktop/model_checkpoint.h5'  # Değiştirin

        # Model kontrol noktası oluşturucusu
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',  # izlenecek metrik (örneğin 'val_loss' veya 'val_accuracy')
            save_best_only=True,  # Sadece en iyi modeli kaydet
            mode='max',  # 'max' için doğrulama metriği üzerinden maksimize etme
            verbose=1
        )

        # Eğitim verilerini hazırla
        x_train, x_test, y_train, y_test = self.prepare_training_data()

        # Modeli eğit
        self.model.fit(
            x_train, y_train,
            epochs=self.epoch_value,
            validation_data=(x_test, y_test),
            callbacks=[checkpoint_callback]  # Kontrol noktası gerçekleştirmek için callback ekleyin
        )

    def training_results(self):
            print("Training Results Button Clicked")

            # Eğitim sırasında kaydedilen metrik değerleri al
            history = self.history.history

            # Doğruluk oranları
            train_accuracy = history.get('accuracy', [])  # Güncellenmiş anahtar 'accuracy'
            val_accuracy = history.get('val_accuracy', [])  # Güncellenmiş anahtar 'val_accuracy'

            # Kayıplar
            train_loss = history.get('loss', [])
            val_loss = history.get('val_loss', [])

            if not train_accuracy or not val_accuracy or not train_loss or not val_loss:
                print("Eğitim geçmişi bulunamadı.")
                return

            # Eğitim sonuçlarını görselleştir
            epochs = range(1, len(train_accuracy) + 1)

            plt.figure(figsize=(12, 6))

            # Doğruluk oranları
            plt.subplot(1, 2, 1)
            plt.plot(epochs, train_accuracy, 'bo-', label='Eğitim Doğruluğu')
            plt.plot(epochs, val_accuracy, 'ro-', label='Doğrulama Doğruluğu')
            plt.title('Eğitim ve Doğrulama Doğruluğu')
            plt.xlabel('Epoklar')
            plt.ylabel('Doğruluk')
            plt.legend()

            # Kayıplar
            plt.subplot(1, 2, 2)
            plt.plot(epochs, train_loss, 'bo-', label='Eğitim Kaybı')
            plt.plot(epochs, val_loss, 'ro-', label='Doğrulama Kaybı')
            plt.title('Eğitim ve Doğrulama Kaybı')
            plt.xlabel('Epoklar')
            plt.ylabel('Kayıp')
            plt.legend()

            plt.tight_layout()
            plt.show()





    def confusion_matrix(self):
            print("Confusion Matrix Button Clicked")

            # Test verilerini hazırla
            x_test, y_test = self.prepare_test_data()

            # Model üzerinde tahmin yap
            predictions = self.model.predict(x_test)

            # Tahminleri sınıflara çevir
            predicted_labels = (predictions > 0.5).astype(int)

            # Confusion Matrix'i oluştur
            cm = confusion_matrix(y_test, predicted_labels)

            # Confusion Matrix'i görselleştir
            plt.figure(figsize=(len(np.unique(y_test)), len(np.unique(y_test))))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.show()

    def test_system(self):
                print("Test System Button Clicked")

                # Bilgisayardan fotoğraf seçme iletişim kutusu
                file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")

                if file_paths:
                    # Seçilen ilk fotoğrafı yükle
                    image_path = file_paths[0]
                    self.load_image(image_path)

                    # Test görüntüsünü 64x64 boyutuna ayarla
                    img = image.load_img(image_path, target_size=(64, 64))
                    img_array_64x64 = image.img_to_array(img)
                    img_array_64x64 = np.expand_dims(img_array_64x64, axis=0)

                    if self.selected_model == self.create_resnet152v2_model():
                        img_array_64x64 = resnetv2_preprocess_input(img_array_64x64)
                    elif self.selected_model == self.create_resnet50_model():
                        img_array_64x64 = resnet50_preprocess_input(img_array_64x64)
                    elif self.selected_model == self.create_vgg16_model():
                        img_array_64x64 = vgg16_preprocess_input(img_array_64x64)

                    # Seçilen modelin bir örneğini oluştur
                    model_instance = self.selected_model

                    # Model örneği üzerinde tahmin yap
                    prediction = model_instance.predict(img_array_64x64)

                    # Tahmin sonucunu ekrana yazdır
                    result = "With Mask" if prediction[0][0] < 0.5 else "Without Mask"
                    self.result_text.setPlainText(f"Sonuc: {result}")
                    print(result)




    def on_model_selection_change(self, index):
                        model_options = {
                            0: self.create_model,
                            1: self.create_resnet152v2_model,
                            2: self.create_resnet50_model,
                            3: self.create_vgg16_model,
                        }
                        self.selected_model = model_options.get(index, self.create_model)()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec())
