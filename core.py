import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.widgets import Button, TextBox
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem, QMenu,QLineEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence , QIntValidator
from PyQt5.QtWidgets import QShortcut
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas , NavigationToolbar2QT as NavigationToolbar
)

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KMeans and SVM Classifier")
        self.setGeometry(100, 100, 1200, 900)
        self.setStyleSheet("background-color: #f0f0f0;")
        self.batch_processing = False  # Flag to control batch updates
        self.data = None
        self.initUI()

    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()

        # Horizontal layout for header
        top_layout = QHBoxLayout()

        # Header Label
        header_label = QLabel("Main Application")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignLeft)
        top_layout.addWidget(header_label)

        # Add top layout to the main layout
        main_layout.addLayout(top_layout)

        # Initialize tabs
        self.tabs = QTabWidget()  # Initialize QTabWidget
        self.dataset_tab = QWidget()
        self.plots_tab = QWidget()
        self.clustering_tab = QWidget()
        self.ml_tab = QWidget()

        self.tabs.addTab(self.dataset_tab, "Dataset")
        self.tabs.addTab(self.plots_tab, "Plots")
        self.tabs.addTab(self.clustering_tab, "Clustering")
        self.tabs.addTab(self.ml_tab, "Machine Learning")

        # Add Tabs to the main layout
        main_layout.addWidget(self.tabs)

        # Set central widget layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Initialize individual tabs
        self.init_dataset_tab()
        self.init_plots_tab()
        self.init_clustering_tab()
        self.init_ml_tab()
    
    def init_dataset_tab(self):
        layout = QVBoxLayout()

        header_label = QLabel("Dataset Operations")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        self.table = QTableWidget(10, 5)  # Create a table with 10 rows and 5 columns
        self.table.setHorizontalHeaderLabels([
            "Porosity", "Absolute Permeability (md)", "RQI", "Phi z", "FZI"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setStyleSheet("font-size: 18px; font-family: 'Times New Roman';")
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.itemChanged.connect(self.validate_and_calculate)

        # Initialize all cells in RQI, Phi z, and FZI columns as non-editable
        for row in range(self.table.rowCount()):
            for col in range(2, 5):
                item = QTableWidgetItem()
                item.setFlags(Qt.ItemIsEnabled)  # Make cells non-editable
                self.table.setItem(row, col, item)

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
                    # Initialize non-editable cells for new rows
                    for col in range(2, 5):
                        item = QTableWidgetItem()
                        item.setFlags(Qt.ItemIsEnabled)
                        self.table.setItem(row, col, item)

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

    def validate_and_calculate(self, item):
        # Temporarily block signals to prevent recursion
        self.table.blockSignals(True)

        try:
            # Debugging output to track changes
            #print(f"Cell ({item.row()}, {item.column()}) changed to: {item.text()}")

            # Validate and format the changed value
            try:
                value = float(item.text())
                item.setText(f"{value}")
            except ValueError:
                return

            # Perform calculations if Porosity (column 0) and Permeability (column 1) are valid
            try:
                porosity = float(self.table.item(item.row(), 0).text()) if self.table.item(item.row(), 0) else None
                permeability = float(self.table.item(item.row(), 1).text()) if self.table.item(item.row(), 1) else None

                if porosity is not None and permeability is not None:
                    # Calculate RQI, Phi z, and FZI
                    rqi = 0.0314 * (permeability / porosity) ** 0.5
                    phi_z = porosity / (1 - porosity)
                    fzi = rqi / phi_z

                    # Update calculated values in non-editable cells
                    if not self.table.item(item.row(), 2):
                        self.table.setItem(item.row(), 2, QTableWidgetItem())
                    self.table.item(item.row(), 2).setText(f"{rqi:.5f}")

                    if not self.table.item(item.row(), 3):
                        self.table.setItem(item.row(), 3, QTableWidgetItem())
                    self.table.item(item.row(), 3).setText(f"{phi_z:.5f}")

                    if not self.table.item(item.row(), 4):
                        self.table.setItem(item.row(), 4, QTableWidgetItem())
                    self.table.item(item.row(), 4).setText(f"{fzi:.5f}")
            except (ValueError, ZeroDivisionError):
                # Clear calculated columns if errors occur
                if self.table.item(item.row(), 2):
                    self.table.item(item.row(), 2).setText("")
                if self.table.item(item.row(), 3):
                    self.table.item(item.row(), 3).setText("")
                if self.table.item(item.row(), 4):
                    self.table.item(item.row(), 4).setText("")
        finally:
            # Re-enable signals after processing
            self.table.blockSignals(False)
    
    def update_plots(self):
        # Extract data from the table
        porosity = []
        permeability = []
        rqi = []
        phi_z = []

        for row in range(self.table.rowCount()):
            try:
                if self.table.item(row, 0) and self.table.item(row, 0).text():
                    p = float(self.table.item(row, 0).text())
                else:
                    p = None

                if self.table.item(row, 1) and self.table.item(row, 1).text():
                    k = float(self.table.item(row, 1).text())
                else:
                    k = None

                if self.table.item(row, 2) and self.table.item(row, 2).text():
                    r = float(self.table.item(row, 2).text())
                else:
                    r = None

                if self.table.item(row, 3) and self.table.item(row, 3).text():
                    z = float(self.table.item(row, 3).text())
                else:
                    z = None

                if p is not None and k is not None:
                    porosity.append(p)
                    permeability.append(k)

                if r is not None and z is not None:
                    rqi.append(r)
                    phi_z.append(z)

            except ValueError as e:
                print(f"Error parsing row {row}: {e}")
                continue

        # Clear the existing figure
        self.plot_canvas.figure.clear()

        # Create a 1x2 grid for two plots
        axes = self.plot_canvas.figure.subplots(1, 2)
        self.plot_canvas.figure.tight_layout(pad=5.0)

        # Plot 1: Absolute Permeability vs Porosity
        scatter1 = axes[0].scatter(porosity, permeability, color='blue', picker=True)
        axes[0].set_title("Absolute Permeability (md) vs Porosity")
        axes[0].set_xlabel("Porosity")
        axes[0].set_ylabel("Absolute Permeability (md)")

        # Plot 2: log(RQI) vs log(Phi z)
        import numpy as np
        log_rqi = np.log(rqi) if rqi else []
        log_phi_z = np.log(phi_z) if phi_z else []
        scatter2 = axes[1].scatter(log_phi_z, log_rqi, color='red', picker=True)
        axes[1].set_title("log(RQI) vs log(Phi z)")
        axes[1].set_xlabel("log(Phi z)")
        axes[1].set_ylabel("log(RQI)")

            # Connect hover events for both plots using a unified event handler
        self.tooltip = None  # To store the active tooltip
        self.plot_data = [
            {"scatter": scatter1, "x_data": porosity, "y_data": permeability, "axis": axes[0]},
            {"scatter": scatter2, "x_data": log_phi_z, "y_data": log_rqi, "axis": axes[1]},]

        self.plot_canvas.mpl_connect('motion_notify_event', self.handle_hover_event)
        # Redraw the canvas
        self.plot_canvas.draw()

    def show_tooltip(self, event, scatter, x_data, y_data, axis):
        if event.inaxes != axis:
            # Remove tooltip if the mouse moves outside the plot
            if self.tooltip:
                self.tooltip.remove()
                self.tooltip = None
                self.plot_canvas.draw_idle()
            return

        # Check if hovering over a point
        cont, ind = scatter.contains(event)
        if cont:
            index = ind["ind"][0]
            x = x_data[index]
            y = y_data[index]
            tooltip_text = f"({x:.2f}, {y:.2f})"

            # Remove the old tooltip
            if self.tooltip:
                self.tooltip.remove()

            # Add a new tooltip at the hovered point
            self.tooltip = axis.annotate(
                tooltip_text, 
                (x, y), 
                textcoords="offset points", 
                xytext=(10, 10), 
                ha='center',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow"),
                fontsize=10
            )
            self.plot_canvas.draw_idle()
        else:
            # Remove the tooltip if not hovering over any point
            if self.tooltip:
                self.tooltip.remove()
                self.tooltip = None
                self.plot_canvas.draw_idle()

    def handle_hover_event(self, event):
        for plot in self.plot_data:
            scatter = plot["scatter"]
            x_data = plot["x_data"]
            y_data = plot["y_data"]
            axis = plot["axis"]

            if event.inaxes == axis:
                # Show tooltip for the active plot
                cont, ind = scatter.contains(event)
                if cont:
                    index = ind["ind"][0]
                    x = x_data[index]
                    y = y_data[index]
                    tooltip_text = f"({x:.2f}, {y:.2f})"

                    # Remove the old tooltip
                    if self.tooltip:
                        self.tooltip.remove()

                    # Add a new tooltip at the hovered point
                    self.tooltip = axis.annotate(
                        tooltip_text,
                        (x, y),
                        textcoords="offset points",
                        xytext=(10, 10),
                        ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow"),
                        fontsize=10
                    )
                    self.plot_canvas.draw_idle()
                    return  # Exit after handling the first matching plot

        # Remove tooltip if the mouse is not hovering over any plot
        if self.tooltip:
            self.tooltip.remove()
            self.tooltip = None
            self.plot_canvas.draw_idle()

    def init_plots_tab(self):
        layout = QVBoxLayout()

        header_label = QLabel("Plots")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Create a 1x2 grid of subplots (two plots)
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))  # <== Modified: 1 row, 2 columns
        fig.tight_layout(pad=5.0)
        
        # Set titles for empty plots
        axes[0].set_title("Empty Plot 1")  # Empty placeholder
        axes[0].axis('off')               # Turn off axes for clarity

        axes[1].set_title("Empty Plot 2")  # Empty placeholder
        axes[1].axis('off')               # Turn off axes for clarity

        # Add the figure to the plots tab
        self.plot_canvas = FigureCanvas(fig)
        layout.addWidget(self.plot_canvas)

        # Add a "Plot Data" button
        plot_button = QPushButton("Plot Data")
        plot_button.clicked.connect(self.update_plots)
        plot_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d7;
                color: white;
                font-size: 18px;
                border-radius: 8px;
                font-family: 'Times New Roman';
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            """
        )
        layout.addWidget(plot_button)

        self.plots_tab.setLayout(layout)  

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

    def update_plot(self, item):
        # Extract data from the table
        porosity = []
        permeability = []

        for row in range(self.table.rowCount()):
            try:
                p = float(self.table.item(row, 0).text()) if self.table.item(row, 0) else None
                k = float(self.table.item(row, 1).text()) if self.table.item(row, 1) else None
                if p is not None and k is not None:
                    porosity.append(p)
                    permeability.append(k)
            except ValueError:
                continue

        # Plot data
        if hasattr(self, 'plot_canvas') and self.plot_canvas:
            self.plots_tab.layout().removeWidget(self.plot_canvas)
            self.plot_canvas.deleteLater()

        if porosity and permeability:
            fig, ax = plt.subplots()
            ax.scatter(porosity, permeability, color='blue', label="Data Points")
            ax.set_title("Porosity vs Absolute Permeability")
            ax.set_xlabel("Porosity")
            ax.set_ylabel("Absolute Permeability (md)")
            ax.legend()
            ax.grid()

            self.plot_canvas = FigureCanvas(fig)
            self.plots_tab.layout().addWidget(self.plot_canvas)

    def init_clustering_tab(self):
        layout = QVBoxLayout()

        header_label = QLabel("Clustering")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Add the Elbow Method button
        self.elbow_button = QPushButton("Generate Elbow Plot")
        self.elbow_button.clicked.connect(self.generate_elbow_plot)
        self.style_button(self.elbow_button)
        layout.addWidget(self.elbow_button)

        # Placeholder for cluster number input and button layout
        self.cluster_layout = None

        # Set layout for the clustering tab
        self.clustering_tab.setLayout(layout)
    
    def generate_elbow_plot(self):
        # Extract data from the first two columns of the table
        porosity = []
        permeability = []

        for row in range(self.table.rowCount()):
            try:
                if self.table.item(row, 0) and self.table.item(row, 0).text():
                    porosity.append(float(self.table.item(row, 0).text()))
                if self.table.item(row, 1) and self.table.item(row, 1).text():
                    permeability.append(float(self.table.item(row, 1).text()))
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", f"Non-numeric value in row {row + 1}. Skipping.")
                continue

        if not porosity or not permeability:
            QMessageBox.warning(self, "Insufficient Data", "Please enter valid data in both columns before generating the Elbow Plot.")
            return

        # Combine porosity and permeability into a single array
        X = np.array(list(zip(porosity, permeability)))

        # Perform the Elbow Method
        wcss = []
        for k in range(1, 11):  # Try k values from 1 to 10
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        # Example optimal k value
        optimal_k = 3

        # Create the figure and position the subplot using gridspec
        fig = plt.figure(figsize=(10, 8))  # Adjust size as needed
        gs = fig.add_gridspec(2, 2)  # Create a 2x2 grid layout
        ax = fig.add_subplot(gs[0, 1])  # Use the top-right quadrant (50% width and height)

        # Plot the elbow method
        ax.plot(range(1, 11), wcss, marker='o', linestyle='-', color='blue')
        ax.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('WCSS', fontsize=12)
        ax.grid(True)

        # Embed the figure in the Clustering tab
        if hasattr(self, 'elbow_canvas') and self.elbow_canvas:
            self.clustering_tab.layout().removeWidget(self.elbow_canvas)
            self.elbow_canvas.deleteLater()
            self.elbow_canvas = None

        self.elbow_canvas = FigureCanvas(fig)
        self.clustering_tab.layout().addWidget(self.elbow_canvas)

        # Create a container for the button and textbox
        container = QWidget()
        container_layout = QHBoxLayout()

        # Create a textbox for cluster input
        cluster_textbox = QLineEdit()
        cluster_textbox.setPlaceholderText("Optimal k")
        cluster_textbox.setText(str(optimal_k))
        cluster_textbox.setValidator(QIntValidator(1, 10))  # Allow only numbers between 1 and 10
        cluster_textbox.setStyleSheet(
            "font-size: 14px; padding: 5px; border: 1px solid gray; border-radius: 5px;"
        )
        container_layout.addWidget(cluster_textbox)

        # Create a button to assign the cluster
        assign_button = QPushButton("Assign Cluster")
        assign_button.setStyleSheet(
            """
            QPushButton {
                background-color: #0078d7;
                color: white;
                font-size: 14px;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            """
        )

        # Connect the button to an action
        def on_assign():
            QMessageBox.information(self, "Cluster Assignment", f"Cluster number {cluster_textbox.text()} assigned.")

        assign_button.clicked.connect(on_assign)
        container_layout.addWidget(assign_button)

        # Add the container to the clustering tab
        container.setLayout(container_layout)
        self.clustering_tab.layout().addWidget(container)

        self.elbow_canvas.draw()
  
    def add_button_and_textbox(self, optimal_k):
        # Create a button and text box overlay
        if not hasattr(self, 'assign_cluster_button'):
            # Button to assign cluster number
            self.assign_cluster_button = QPushButton("Assign Cluster Number", self)
            self.assign_cluster_button.setGeometry(50, 50, 150, 30)  # Adjust position and size
            self.assign_cluster_button.clicked.connect(self.assign_cluster_number)

        if not hasattr(self, 'cluster_input'):
            # Text box to display the optimal number of clusters
            self.cluster_input = QLineEdit(self)
            self.cluster_input.setGeometry(220, 50, 50, 30)  # Adjust position and size
            self.cluster_input.setText(str(optimal_k))
            self.cluster_input.setValidator(QIntValidator(1, 10))  # Allow only numbers between 1 and 10

        self.assign_cluster_button.show()
        self.cluster_input.show()
   
    def assign_cluster_number(self):
        # Handle the assignment of the cluster number from the text box
        cluster_number = int(self.cluster_input.text())
        QMessageBox.information(self, "Cluster Assignment", f"Cluster number {cluster_number} assigned.")
    
    def assign_cluster_number(self):
        # Handle the assignment of the cluster number from the text box
        cluster_number = int(self.cluster_input.text())
        QMessageBox.information(self, "Cluster Assignment", f"Cluster number {cluster_number} assigned.")
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())