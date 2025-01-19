import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.widgets import Button, TextBox
from matplotlib.backend_bases import cursors
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem, QMenu,QLineEdit ,
    QSizePolicy
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence , QIntValidator , QCursor 
from PyQt5.QtWidgets import QShortcut
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas , NavigationToolbar2QT as NavigationToolbar)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist  

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rock Typing Application")
        self.setGeometry(100, 100, 1200, 900)
        self.setStyleSheet("background-color: #f0f0f0;")
        self.batch_processing = False  # Flag to control batch updates
        self.data = None
        self.tooltip = None  # Initialize tooltip
        self.initUI()

    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()

        # Horizontal layout for header
        top_layout = QHBoxLayout()

        # Header Label
        header_label = QLabel("Rock Typing Application")
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
        self.rock_type_tab = QWidget()

        self.tabs.addTab(self.dataset_tab, "Dataset")
        self.tabs.addTab(self.plots_tab, "Plots")
        self.tabs.addTab(self.clustering_tab, "Clustering")
        self.tabs.addTab(self.rock_type_tab, "Rock Type")
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
        self.init_rock_type_tab()
   
    def init_rock_type_tab(self):
        layout = QVBoxLayout()

        # Header
        header_label = QLabel("Rock Type Visualization")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        fig.tight_layout(pad=5.0)

        # Placeholders for axes
        axes[0].set_title("Empty Plot 1")
        axes[0].axis('off')
        axes[1].set_title("Empty Plot 2")
        axes[1].axis('off')

        # Add canvas to layout
        self.rock_type_canvas = FigureCanvas(fig)
        layout.addWidget(self.rock_type_canvas)
        self.rock_type_canvas.mpl_connect('button_press_event', self.handle_plot_click)

        # Spacer for alignment
        layout.addStretch()

        # Button for plotting, placed at the bottom
        button_layout = QHBoxLayout()
        plot_button = QPushButton("Plot Rock Type Data")
        plot_button.clicked.connect(self.update_rock_type_tab)
        self.style_button(plot_button)  # Reuse button styling
        button_layout.addWidget(plot_button)

        layout.addLayout(button_layout)

        self.rock_type_tab.setLayout(layout)
    
    def update_rock_type_tab(self):
        # Extract data from the table
        porosity = []
        permeability = []
        rqi = []
        phi_z = []

        for row in range(self.table.rowCount()):
            try:
                if self.table.item(row, 0) and self.table.item(row, 0).text():
                    porosity.append(float(self.table.item(row, 0).text()))
                if self.table.item(row, 1) and self.table.item(row, 1).text():
                    permeability.append(float(self.table.item(row, 1).text()))
                if self.table.item(row, 2) and self.table.item(row, 2).text():
                    rqi.append(float(self.table.item(row, 2).text()))
                if self.table.item(row, 3) and self.table.item(row, 3).text():
                    phi_z.append(float(self.table.item(row, 3).text()))
            except ValueError:
                continue  # Skip rows with invalid or missing data

        if not porosity or not permeability or not rqi or not phi_z:
            QMessageBox.warning(self, "Warning", "Insufficient data to plot. Please enter valid data.")
            return

        # Prepare data for clustering
        X = np.array(list(zip(porosity, permeability)))

        # Perform clustering (default to 3 clusters if no input)
        try:
            n_clusters = int(self.selected_K_textbox.text())
            if n_clusters <= 0:
                raise ValueError
        except ValueError:
            n_clusters = min(3, len(X))  # Default to 3 clusters if input is invalid

        if n_clusters > len(X):
            QMessageBox.warning(self, "Error", f"Number of clusters ({n_clusters}) exceeds the number of samples ({len(X)}).")
            return

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)

        # Filter valid data for logarithmic plots
        valid_indices = np.where((np.array(rqi) > 0) & (np.array(phi_z) > 0))[0]
        log_rqi = np.log(np.array(rqi)[valid_indices])
        log_phi_z = np.log(np.array(phi_z)[valid_indices])
        filtered_clusters = clusters[valid_indices]

        # Assign cluster colors
        cluster_colors = plt.cm.tab10.colors

        # Clear the previous plots
        self.rock_type_canvas.figure.clear()

        # Create a 1x2 grid for the subplots
        axes = self.rock_type_canvas.figure.subplots(1, 2)
        self.rock_type_canvas.figure.tight_layout(pad=5.0)

        # Plot 1: Absolute Permeability vs Porosity
        scatter1 = axes[0].scatter(
            porosity, permeability, c=clusters, cmap='tab10', alpha=0.7
        )
        axes[0].set_title("Absolute Permeability (md) vs Porosity")
        axes[0].set_xlabel("Porosity")
        axes[0].set_ylabel("Absolute Permeability (md)")
        axes[0].grid(True)

        # Plot 2: log(RQI) vs log(Phi z)
        scatter2 = axes[1].scatter(
            log_phi_z, log_rqi, c=filtered_clusters, cmap='tab10', alpha=0.7
        )
        axes[1].set_title("log(RQI) vs log(Phi z)")
        axes[1].set_xlabel("log(Phi z)")
        axes[1].set_ylabel("log(RQI)")
        axes[1].grid(True)

        # Save plot data for export
        self.current_plot_data = {
            "points1": list(zip(porosity, permeability)),
            "clusters1": clusters,
            "points2": list(zip(log_phi_z.tolist(), log_rqi.tolist())),
            "clusters2": filtered_clusters,
        }

        # Add hover functionality for tooltips
        self.rock_type_tooltip = None
        self.rock_type_plot_data = [
            {"scatter": scatter1, "x_data": porosity, "y_data": permeability, "axis": axes[0]},
            {"scatter": scatter2, "x_data": log_phi_z.tolist(), "y_data": log_rqi.tolist(), "axis": axes[1]}
        ]

        self.rock_type_canvas.mpl_connect('motion_notify_event', self.handle_rock_type_hover_event)
        self.rock_type_canvas.mpl_connect('button_press_event', self.handle_plot_click)

        # Update the canvas
        self.rock_type_canvas.draw()
    
    def handle_rock_type_hover_event(self, event):
        for plot in self.rock_type_plot_data:
            scatter = plot["scatter"]
            x_data = plot["x_data"]
            y_data = plot["y_data"]
            axis = plot["axis"]

            if event.inaxes == axis:
                cont, ind = scatter.contains(event)
                if cont:
                    index = ind["ind"][0]
                    x = x_data[index]
                    y = y_data[index]
                    tooltip_text = f"({x:.2f}, {y:.2f})"

                    # Remove previous tooltip
                    if self.rock_type_tooltip:
                        self.rock_type_tooltip.remove()

                    # Create new tooltip
                    self.rock_type_tooltip = axis.annotate(
                        tooltip_text,
                        (x, y),
                        textcoords="offset points",
                        xytext=(10, 10),
                        ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow"),
                        fontsize=10
                    )
                    self.rock_type_canvas.draw_idle()
                    return

        # Remove tooltip if not hovering over any point
        if self.rock_type_tooltip:
            self.rock_type_tooltip.remove()
            self.rock_type_tooltip = None
            self.rock_type_canvas.draw_idle()
    
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

    def show_context_menu(self, event):
        if isinstance(event, MouseEvent):  # Check if it's a Matplotlib mouse event
            # Convert the Matplotlib event position to a QPoint
            position = self.mapToGlobal(QPoint(int(event.x), int(event.y)))
        else:
            position = event

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #cccccc;
            }
            QMenu::item {
                padding: 8px 20px;
            }
            QMenu::item:selected {
                background-color: #0078d7;
                color: #ffffff;
            }
        """)

        save_plot_action = menu.addAction("Save Plot As...")
        save_plot_action.triggered.connect(lambda: self.save_plot(self.plot_canvas))

        if hasattr(self, "current_plot_data") and self.current_plot_data:
            export_csv_action = menu.addAction("Export Data As CSV")
            export_csv_action.triggered.connect(self.export_plot_data_to_csv)

        menu.exec_(position)  # Show the menu at the converted position
    
    def export_to_csv(self):
        # Open a file dialog to get the file path
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_path:  # If a valid file path is selected
            try:
                # Prepare data for CSV export
                data = []
                for row in range(self.table.rowCount()):
                    row_data = []
                    for column in range(self.table.columnCount()):
                        item = self.table.item(row, column)
                        row_data.append(item.text() if item else '')  # Get text from table item
                    data.append(row_data)

                # Write data to CSV file
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False, header=[self.table.horizontalHeaderItem(i).text() for i in range(self.table.columnCount())])

                QMessageBox.information(self, "Success", "Data has been exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {e}")

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
                cont, ind = scatter.contains(event)
                if cont:
                    index = ind["ind"][0]
                    x = x_data[index]
                    y = y_data[index]
                    tooltip_text = f"({x:.2f}, {y:.2f})"

                    if self.tooltip:
                        self.tooltip.remove()

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
                    return

        # Check if hovering over the red circle
        if event.inaxes == axis:
            for circle in axis.collections:
                cont, ind = circle.contains(event)
                if cont:
                    tooltip_text = "Recommended k"
                    if self.tooltip:
                        self.tooltip.remove()

                    self.tooltip = axis.annotate(
                        tooltip_text,
                        event,
                        textcoords="offset points",
                        xytext=(10, 10),
                        ha='center',
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow"),
                        fontsize=10
                    )
                    self.plot_canvas.draw_idle()
                    return

        # Remove tooltip if not hovering over any point
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
        self.plot_canvas.mpl_connect('button_press_event', self.handle_plot_click)

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

        # Header at the top
        header_label = QLabel("Machine Learning")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Create a placeholder plot
        fig, ax = plt.subplots()
        ax.set_title("SVM Results Placeholder")
        ax.axis('off')  # Placeholder, no axes visible

        # Add the figure canvas
        self.ml_canvas = FigureCanvas(fig)
        layout.addWidget(self.ml_canvas)

        # Add a spacer to push the button to the bottom
        layout.addStretch()

        # Add the button at the bottom
        self.svm_button = QPushButton("Train and Evaluate SVM")
        self.svm_button.clicked.connect(self.train_evaluate_svm)
        self.style_button(self.svm_button)  # Style the button
        layout.addWidget(self.svm_button)

        # Set the layout for the ML tab
        self.ml_tab.setLayout(layout)
    
    def train_evaluate_svm(self):
        # Extract features and target from the table
        features = []
        target = []
        original_features = []  # To store original feature values

        for row in range(self.table.rowCount()):
            try:
                # Extract porosity and permeability as features
                porosity = float(self.table.item(row, 0).text()) if self.table.item(row, 0) else None
                permeability = float(self.table.item(row, 1).text()) if self.table.item(row, 1) else None

                if porosity is not None and permeability is not None:
                    # Limit porosity to 1.0 and include only positive values
                    if 0 < porosity <= 1.0 and permeability > 0:
                        features.append([porosity, permeability])  # Use raw values for original_features
                        original_features.append([porosity, permeability])  # Keep original values
                        # Assuming rock types are integers (you may want to replace this logic):
                        rock_type = self.determine_rock_type(porosity, permeability)  # Replace with your own logic
                        target.append(rock_type)
            except ValueError:
                continue

        if not features or not target:
            QMessageBox.warning(self, "Warning", "Insufficient data for SVM training.")
            return

        features = np.array(features)
        target = np.array(target)

        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

        # Train SVM
        svm_classifier = SVC(kernel='linear', random_state=42)
        svm_classifier.fit(X_train, y_train)

        # Predictions
        y_pred = svm_classifier.predict(X_test)

        # Generate the classification report (silent handling)
        report = classification_report(y_test, y_pred)

        # Plot results
        self.plot_svm_results(svm_classifier, scaler, original_features, target)
      
    def determine_rock_type(self, porosity, permeability):
        # Replace this with the actual logic to determine rock type based on porosity and permeability
        if porosity < 0.15:
            return 0  # For example, clay
        elif 0.15 <= porosity < 0.25:
            return 1  # For example, sand
        else:
            return 2  # For example, limestone
    
    def style_button(self, button):
        button.setStyleSheet(
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
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Make it full width
    
    def perform_clustering(self):
        # Extract data from the table to self.data
        self.extract_table_data()
    
        if self.data is None or self.data.empty:
            QMessageBox.warning(self, "Warning", "Please load a dataset first.")
            return

        try:
            # Extract features for clustering
            features = self.data[['Porosity', 'Absolute Permeability (md)']].dropna()
            X = features.values

            # Get the number of clusters from the textbox
            try:
                n_clusters = int(self.max_clusters_textbox.text())
            except ValueError:
                QMessageBox.warning(self, "Warning", "Please specify a valid number of clusters.")
                return

            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)

            # Assign cluster labels to the original DataFrame
            self.data['Cluster'] = np.nan  # Initialize with NaN
            self.data.loc[features.index, 'Cluster'] = clusters

            QMessageBox.information(self, "Success", f"Clustering performed with {n_clusters} clusters.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Clustering failed: {e}")
    
    def plot_svm_results(self, classifier, scaler, original_features, target):
        # Create a mesh grid for decision boundary visualization
        h = 0.02  # Step size in mesh
        x_min, x_max = max(0, np.array(original_features)[:, 0].min() - 1), np.array(original_features)[:, 0].max() + 1
        y_min, y_max = max(0, np.array(original_features)[:, 1].min() - 1), np.array(original_features)[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict on the grid points using the scaled features
        Z = classifier.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)

        # Plot data points using original features
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.contour(xx, yy, Z, levels=[0.5, 1.5], colors="black", linewidths=2, linestyles="dashed")

        scatter = ax.scatter(
            np.array(original_features)[:, 0],
            np.array(original_features)[:, 1],
            c=target,
            cmap=plt.cm.coolwarm,
            edgecolors="k",
            s=100,
            marker="o",
        )

        # Set titles, labels, and limits
        ax.set_title("SVM Classification of Rock Types (with Decision Boundary Lines)")
        ax.set_xlabel("Porosity")
        ax.set_ylabel("Permeability")
        ax.legend(*scatter.legend_elements(), title="Rock Types", loc="upper right")
        
        # Limit X-axis from 0 to 1.0
        ax.set_xlim(0, 1.0)

        # Save plot data for CSV export
        self.current_svm_data = {
            "porosity": np.array(original_features)[:, 0],
            "permeability": np.array(original_features)[:, 1],
            "rock_type": target
        }

        # Attach context menu for export
        fig.canvas.mpl_connect('button_press_event', self.show_context_menu)

        # Check if a canvas already exists and remove it
        if hasattr(self, 'ml_canvas') and self.ml_canvas:
            self.ml_tab.layout().removeWidget(self.ml_canvas)
            self.ml_canvas.deleteLater()

        # Add the new canvas to the layout
        self.ml_canvas = FigureCanvas(fig)
        self.ml_tab.layout().insertWidget(1, self.ml_canvas)  # Insert just after the header label
        self.ml_canvas.draw()
    
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

        # Header
        header_label = QLabel("Clustering")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Distortion Plot Layout
        self.distortion_plot_layout = QVBoxLayout()
        layout.addLayout(self.distortion_plot_layout)

        # Add inputs for max clusters
        max_clusters_layout = QHBoxLayout()
        self.max_clusters_textbox = QLineEdit()
        self.max_clusters_textbox.setPlaceholderText("Max Clusters (e.g., 10)")
        self.max_clusters_textbox.setValidator(QIntValidator(1, 50))
        max_clusters_layout.addWidget(QLabel("Max Clusters:"))
        max_clusters_layout.addWidget(self.max_clusters_textbox)
        layout.addLayout(max_clusters_layout)

        # Add Recommended K inputs
        recommended_k_layout = QHBoxLayout()
        self.selected_K_textbox = QLineEdit()
        self.selected_K_textbox.setPlaceholderText("Recommended K")
        recommended_k_layout.addWidget(QLabel("Recommended K:"))
        recommended_k_layout.addWidget(self.selected_K_textbox)
        layout.addLayout(recommended_k_layout)

        # Spacer for alignment
        layout.addStretch()

        # Button for clustering, placed at the bottom
        button_layout = QHBoxLayout()
        cluster_button = QPushButton("Generate Distortion Plot")
        cluster_button.clicked.connect(self.generate_distortion_plot)
        self.style_button(cluster_button)  # Reuse button styling
        button_layout.addWidget(cluster_button)

        layout.addLayout(button_layout)

        self.clustering_tab.setLayout(layout)
    
    def generate_distortion_plot(self):
        porosity = []
        permeability = []

        # Extract data from the table
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
            QMessageBox.warning(self, "Insufficient Data", "Please enter valid data in both columns before generating the Distortion Plot.")
            return

        # Prepare data for clustering
        X = np.array(list(zip(porosity, permeability)))

        # Get the maximum number of clusters
        max_clusters_text = self.max_clusters_textbox.text()
        try:
            max_clusters = int(max_clusters_text)
            if max_clusters <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number of clusters.")
            return

        # Generate distortion plot
        distortions = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            distortion = sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
            distortions.append(distortion)

        # Create the plot
        fig, ax = plt.subplots(figsize=(5, 10))
        scatter = ax.scatter(
            range(1, max_clusters + 1), distortions, color='blue', s=100, label='Distortion Points'
        )
        ax.plot(range(1, max_clusters + 1), distortions, marker='o', color='blue', linestyle='-', label='Distortion Curve')

        # Highlight optimal K with a red circle
        optimal_k = self.find_optimal_k(distortions)
        selected_circle = ax.scatter(
            optimal_k,
            distortions[optimal_k - 1],
            facecolors='none',
            edgecolors='red',
            s=500,
            linewidth=2,
            label='Recommended k'
        )

        # Set plot labels and title
        ax.set_title("Distortion Method for Optimal K", fontsize=14, fontweight='bold')
        ax.set_xlabel("K", fontsize=12)
        ax.set_ylabel("Distortion", fontsize=12)

        # Add legend to the plot
        ax.legend(loc='best', fontsize=10, title="Legend")

        # Set aspect ratio to "equal"
        ax.set_aspect('equal', adjustable='datalim')  # Ensure circles are not distorted

        # Attach hover and click events
        self.hover_circle = None  # To store the circle artist for hover effect
        fig.canvas.mpl_connect('motion_notify_event', lambda event: self.on_hover_distortion_plot(event, scatter, ax))
        fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click_distortion_plot(event, distortions))

        # Replace or update the canvas
        if hasattr(self, 'distortion_canvas') and self.distortion_canvas:
            self.distortion_plot_layout.removeWidget(self.distortion_canvas)
            self.distortion_canvas.deleteLater()
            self.distortion_canvas = None

        self.distortion_canvas = FigureCanvas(fig)
        self.distortion_plot_layout.addWidget(self.distortion_canvas)
        self.distortion_canvas.draw()
   
    def find_optimal_k(self, distortions):
        if len(distortions) == 2:
            # If distortions length is less than 3, we can't calculate a second derivative properly
            return max(2, len(distortions))  # Return at least 2 or the number of data points
        if len(distortions) == 1:
            # If distortions length is less than 2, we can't calculate a second derivative properly
            return max(1, len(distortions))  # Return at least 2 or the number of data points

        diff = np.diff(distortions)
        second_diff = np.diff(diff)

        # Add a check to ensure second_diff has enough elements for argmax
        if len(second_diff) < 1:
            return 2  # Default to 2 clusters if we can't compute a meaningful second derivative

        optimal_k = np.argmax(second_diff) + 2  # +2 because np.diff reduces the length twice
        return optimal_k
    
    def on_hover_distortion_plot(self, event, scatter, ax):
        if event.inaxes:
            # Check if hovering over a point
            cont, ind = scatter.contains(event)
            if cont:
                index = ind["ind"][0]
                x, y = scatter.get_offsets()[index]
                
                # Remove existing circle
                if self.hover_circle:
                    self.hover_circle.remove()
                
                # Add a new circle around the hovered point
                self.hover_circle = plt.Circle((x, y), radius=0.2, color='red', fill=False, linewidth=2)
                ax.add_artist(self.hover_circle)
                self.distortion_canvas.draw_idle()  # Redraw the canvas
            else:
                # Remove the circle if not hovering over any point
                if self.hover_circle:
                    self.hover_circle.remove()
                    self.hover_circle = None
                    self.distortion_canvas.draw_idle()
    
    def on_click_distortion_plot(self, event, distortions):
        cont, ind = event.inaxes.collections[0].contains(event)
        if cont:
            index = ind["ind"][0]
            chosen_k = index + 1
            self.selected_K_textbox.setText(str(chosen_k))
            QMessageBox.information(self, "Chosen k", f"You have chosen k = {chosen_k}")
    
    def find_selected_K(self, wcss):
        """
        A more sophisticated method to find the "distortion" point using the Kneedle algorithm.

        Parameters:
        - wcss: List of WCSS values for different k values.

        Returns:
        - Recommended number of clusters (k) based on the distortion method.
        """
        if len(wcss) < 2:
            return 1  # if there's not enough data, default to 1 cluster

        # Calculate the first derivative (how much WCSS is dropping)
        first_derivative = np.diff(wcss)
        
        # Calculate second derivative (acceleration)
        second_derivative = np.diff(first_derivative)

        # Find the index of the maximum value of the first derivative (indicating the distortion)
        distortion_index = np.argmax(first_derivative)  # This locates the point of maximum drop

        # Find the best k - this is the index of the distortion point + 1 due to np.diff reducing length by 1
        selected_K = distortion_index + 1

        # Ensure the recommended k is within the range of available k values
        return min(selected_K, len(wcss))  
    
    def custom_clustering(self):
        # Handling custom clustering with user-defined K
        user_k = self.max_clusters_textbox.text()
        if not user_k:
            QMessageBox.warning(self, "Warning", "Please enter a valid number for K.")
            return

        optimal_k = int(user_k)
        self.perform_clustering(optimal_k)
    
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

    def extract_table_data(self):
        data = []
        for row in range(self.table.rowCount()):
            row_data = []
            for col in range(self.table.columnCount()):
                item = self.table.item(row, col)
                row_data.append(float(item.text()) if item and item.text() else None)
            data.append(row_data)

        columns = ["Porosity", "Absolute Permeability (md)", "RQI", "Phi z", "FZI"]
        self.data = pd.DataFrame(data, columns=columns)

    def show_plot_context_menu(self, event, plot_type):
        if event.button == 3:  # Right-click
            menu = QMenu(self)
            menu.setStyleSheet("""
                QMenu {
                    background-color: #ffffff;
                    color: #000000;
                    border: 1px solid #cccccc;
                }
                QMenu::item {
                    padding: 8px 20px;
                }
                QMenu::item:selected {
                    background-color: #0078d7;
                    color: #ffffff;
                }
            """)

            if plot_type == "distortion" and "distortion" in self.current_clustering_data:
                export_csv_action = menu.addAction("Export Distortion Data As CSV")
                export_csv_action.triggered.connect(self.export_distortion_to_csv)

            menu.exec_(QCursor.pos())
    
    def export_distortion_to_csv(self):
        if "distortion" not in self.current_clustering_data:
            QMessageBox.warning(self, "No Data", "No distortion data available for export.")
            return

        # Open a file dialog to save the CSV
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options
        )

        if file_path:
            try:
                # Extract distortion data
                data = self.current_clustering_data["distortion"]
                df = pd.DataFrame({"Number of Clusters (k)": data["k"], "Distortion": data["values"]})
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", "Distortion data exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {e}")
    
    def save_plot(self, canvas):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "Images (*.png *.jpg *.jpeg *.bmp *.svg);;All Files (*)", options=options
        )
        if file_path:
            canvas.figure.savefig(file_path)

    def handle_plot_click(self, event):
        if event.button == 3:  # Right-click
            menu = QMenu(self)
            menu.setStyleSheet("""
                QMenu {
                    background-color: #ffffff;
                    color: #000000;
                    border: 1px solid #cccccc;
                }
                QMenu::item {
                    padding: 8px 20px;
                }
                QMenu::item:selected {
                    background-color: #0078d7;
                    color: #ffffff;
                }
            """)

            save_plot_action = menu.addAction("Save Plot As...")
            save_plot_action.triggered.connect(lambda: self.save_plot(self.rock_type_canvas))

            if hasattr(self, "current_plot_data") and self.current_plot_data:
                export_csv_action = menu.addAction("Export Data as CSV")
                export_csv_action.triggered.connect(self.export_plot_data_to_csv)

            menu.exec_(QCursor.pos())

    
    def export_plot_data_to_csv(self):
        if not hasattr(self, "current_plot_data") or not self.current_plot_data:
            QMessageBox.warning(self, "No Data", "No plot data available for export.")
            return

        # Open a file dialog to save the CSV
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options
        )

        if file_path:
            try:
                # Prepare data for the CSV
                # Extract data from the points1 and points2 along with their respective clusters
                points1_data = [
                    {"Porosity": pt[0], "Permeability": pt[1], "Cluster": cl}
                    for pt, cl in zip(self.current_plot_data["points1"], self.current_plot_data["clusters1"])
                ]
                points2_data = [
                    {"log(Phi z)": pt[0], "log(RQI)": pt[1], "Cluster": cl}
                    for pt, cl in zip(self.current_plot_data["points2"], self.current_plot_data["clusters2"])
                ]

                # Create DataFrames for both sets of plot data
                df1 = pd.DataFrame(points1_data)
                df2 = pd.DataFrame(points2_data)

                # Merge the two DataFrames on the Cluster column
                merged_df = pd.merge(df1, df2, on='Cluster', how='outer', suffixes=('_plot1', '_plot2'))

                # Save the merged DataFrame into a single CSV file
                merged_df.to_csv(file_path, index=False)

                QMessageBox.information(self, "Success", "Plot data exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {e}")
    
    def export_svm_data_to_csv(self):
        if not hasattr(self, "current_svm_data") or not self.current_svm_data:
            QMessageBox.warning(self, "No Data", "No SVM data available for export.")
            return

        # Open a file dialog to save the CSV
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options
        )

        if file_path:
            try:
                # Extract SVM data
                data = self.current_svm_data
                df = pd.DataFrame({
                    "Porosity": data["porosity"],
                    "Permeability (md)": data["permeability"],
                    "Rock Type": data["rock_type"]
                })
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", "SVM data exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())