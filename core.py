import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QPushButton, QLabel,
    QVBoxLayout, QWidget, QMessageBox, QTabWidget, QComboBox, QLineEdit
)

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KMeans and SVM Classifier")
        self.setGeometry(100, 100, 1200, 900)
        self.setStyleSheet("background-color: #f0f0f0;")

        self.data = None

        self.initUI()

    def initUI(self):
        # Tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tabs
        self.dataset_tab = QWidget()
        self.plots_tab = QWidget()
        self.clustering_tab = QWidget()
        self.ml_tab = QWidget()

        # Add tabs to QTabWidget
        self.tabs.addTab(self.dataset_tab, "Dataset")
        self.tabs.addTab(self.plots_tab, "Plots")
        self.tabs.addTab(self.clustering_tab, "Clustering")
        self.tabs.addTab(self.ml_tab, "Machine Learning")

        # Setup tabs
        self.init_dataset_tab()
        self.init_plots_tab()
        self.init_clustering_tab()
        self.init_ml_tab()

        # Set default tab
        self.tabs.setCurrentIndex(0)

    def init_dataset_tab(self):
        layout = QVBoxLayout()

        header_label = QLabel("Dataset Operations")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        self.load_data_button = QPushButton("Load Dataset")
        self.load_data_button.clicked.connect(self.load_dataset)
        self.style_button(self.load_data_button)
        layout.addWidget(self.load_data_button)

        self.dataset_tab.setLayout(layout)

    def init_plots_tab(self):
        layout = QVBoxLayout()

        header_label = QLabel("Plots")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        self.elbow_button = QPushButton("Generate Elbow Plot")
        self.elbow_button.clicked.connect(self.generate_elbow_plot)
        self.style_button(self.elbow_button)
        layout.addWidget(self.elbow_button)

        self.plot_label = QLabel("Visualization Area")
        self.plot_label.setStyleSheet("font-size: 22px; font-family: 'Times New Roman';")
        self.plot_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.plot_label)

        self.plots_tab.setLayout(layout)

    def init_clustering_tab(self):
        layout = QVBoxLayout()

        header_label = QLabel("Clustering")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        self.cluster_button = QPushButton("Perform Clustering")
        self.cluster_button.clicked.connect(self.perform_clustering)
        self.style_button(self.cluster_button)
        layout.addWidget(self.cluster_button)

        self.clustering_tab.setLayout(layout)

    def init_ml_tab(self):
        layout = QVBoxLayout()

        header_label = QLabel("Machine Learning")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        self.svm_button = QPushButton("Train and Evaluate SVM")
        self.svm_button.clicked.connect(self.train_evaluate_svm)
        self.style_button(self.svm_button)
        layout.addWidget(self.svm_button)

        self.ml_tab.setLayout(layout)

    def style_button(self, button):
        button.setFixedSize(200, 50)
        button.setStyleSheet(
            """
            QPushButton {
                background-color: #00b8b8;
                color: black;
                font-size: 20px;
                border-radius: 10px;
                font-family: 'Times New Roman';
            }
            QPushButton:hover {
                background-color: #00008B;
                color: white;
            }
            """
        )

    def load_dataset(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                QMessageBox.information(self, "Success", "Dataset loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load dataset: {e}")

    def generate_elbow_plot(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        try:
            features = self.data[['Porosity', 'Air Permability (md), Kair@323 (K)']]
            wcss = []
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(features)
                wcss.append(kmeans.inertia_)

            # Plot elbow method
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(range(1, 11), wcss, marker='o')
            ax.set_title('Elbow Method')
            ax.set_xlabel('Number of clusters (k)')
            ax.set_ylabel('WCSS')
            ax.grid()

            self.show_plot(fig)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate elbow plot: {e}")

    def perform_clustering(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        try:
            features = self.data[['Porosity', 'Air Permability (md), Kair@323 (K)']]
            optimal_k = 3  # Based on elbow plot analysis
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            self.data['Cluster'] = kmeans.fit_predict(features)

            # Plot clusters
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(features['Porosity'], features['Air Permability (md), Kair@323 (K)'], 
                                  c=self.data['Cluster'], cmap='viridis', marker='o')
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
            ax.set_title('KMeans Clustering')
            ax.set_xlabel('Porosity')
            ax.set_ylabel('Air Permability (md), Kair@323 (K)')
            fig.colorbar(scatter, label='Cluster')
            ax.grid()

            self.show_plot(fig)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to perform clustering: {e}")

    def train_evaluate_svm(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        try:
            features = self.data[['Porosity', 'Air Permability (md), Kair@323 (K)']]
            target = self.data['Rock Type']

            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

            svm_classifier = SVC(kernel='linear', random_state=42)
            svm_classifier.fit(X_train, y_train)

            accuracy = svm_classifier.score(X_test, y_test)
            QMessageBox.information(self, "SVM Results", f"SVM Accuracy: {accuracy:.2f}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to train and evaluate SVM: {e}")

    def show_plot(self, fig):
        if hasattr(self, 'canvas') and self.canvas:
            self.plots_tab.layout().removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None

        self.canvas = FigureCanvas(fig)
        self.plots_tab.layout().addWidget(self.canvas)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
