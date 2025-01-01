import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QVBoxLayout, QWidget, QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem, QMenu
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QClipboard
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut
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

        self.table = QTableWidget(10, 2)  # Create a table with 10 rows and 2 columns
        self.table.setHorizontalHeaderLabels(["Porosity", "Absolute Permeability (md)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setStyleSheet("font-size: 18px; font-family: 'Times New Roman';")
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.itemChanged.connect(self.validate_cell)

        # Add shortcut for pasting
        paste_shortcut = QShortcut(QKeySequence("Ctrl+V"), self)
        paste_shortcut.activated.connect(self.handle_paste)

        layout.addWidget(self.table)

        # Add a button to clear the table
        clear_table_button = QPushButton("Clear Table")
        clear_table_button.clicked.connect(self.clear_table)
        clear_table_button.setStyleSheet(
            """
            QPushButton {
                background-color: #FF6347;
                color: white;
                font-size: 18px;
                border-radius: 8px;
                font-family: 'Times New Roman';
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #FF4500;
            }
            """
        )
        layout.addWidget(clear_table_button)

        self.dataset_tab.setLayout(layout)

    def clear_table(self):
        for row in range(self.table.rowCount()):
            for column in range(self.table.columnCount()):
                self.table.setItem(row, column, QTableWidgetItem(""))

    def show_context_menu(self, position):
        menu = QMenu()
        paste_action = menu.addAction("Paste from Clipboard")
        delete_action = menu.addAction("Delete")
        delete_all_action = menu.addAction("Delete All")

        paste_action.triggered.connect(self.handle_paste)
        delete_action.triggered.connect(self.handle_delete)
        delete_all_action.triggered.connect(self.handle_delete_all)

        menu.exec_(self.table.viewport().mapToGlobal(position))

    def handle_paste(self):
        clipboard = QApplication.clipboard()
        data = clipboard.text()

        if not data:
            QMessageBox.warning(self, "Warning", "Clipboard is empty.")
            return

        selected = self.table.selectedRanges()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a cell to paste data.")
            return

        start_row = selected[0].topRow()
        start_column = selected[0].leftColumn()

        rows = data.splitlines()
        for i, line in enumerate(rows):
            values = line.split('\t')
            for j, value in enumerate(values):
                row = start_row + i
                column = start_column + j

                if row >= self.table.rowCount():
                    self.table.insertRow(self.table.rowCount())
                self.table.setItem(row, column, QTableWidgetItem(value))

    def handle_delete(self):
        for item in self.table.selectedItems():
            self.table.setItem(item.row(), item.column(), QTableWidgetItem(""))

    def handle_delete_all(self):
        selected = self.table.selectedRanges()
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select a column to delete all values.")
            return

        for column in range(selected[0].leftColumn(), selected[0].rightColumn() + 1):
            for row in range(self.table.rowCount()):
                self.table.setItem(row, column, QTableWidgetItem(""))

    def validate_cell(self, item):
        self.table.blockSignals(True)  # Prevent recursive signal triggering
        try:
            float(item.text())
        except ValueError:
            item.setText("")  # Clear invalid input without a prompt
        self.table.blockSignals(False)

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

    def generate_elbow_plot(self):
        if self.data is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        try:
            features = self.data[['Porosity', 'Absolute Permeability (md)']]
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
            features = self.data[['Porosity', 'Absolute Permeability (md)']]
            optimal_k = 3  # Based on elbow plot analysis
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            self.data['Cluster'] = kmeans.fit_predict(features)

            # Plot clusters
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(features['Porosity'], features['Absolute Permeability (md)'], 
                                  c=self.data['Cluster'], cmap='viridis', marker='o')
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
            ax.set_title('KMeans Clustering')
            ax.set_xlabel('Porosity')
            ax.set_ylabel('Absolute Permeability (md)')
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
            features = self.data[['Porosity', 'Absolute Permeability (md)']]
            target = self.data['Target']

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
