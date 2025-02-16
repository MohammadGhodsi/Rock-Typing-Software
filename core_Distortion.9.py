

##### Importing libraries ########

import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.backend_bases import cursors
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import QPropertyAnimation, QRect , QEvent , QEasingCurve
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QMessageBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QMenu, QLineEdit,
    QSizePolicy, QSplashScreen, QShortcut)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QIntValidator, QCursor, QPixmap, QIcon

class MainApp(QMainWindow):
    
#### Initializing application #########
    
    def __init__(self):
        super().__init__()
        self.plot_data = []  # Initialize plot_data as an empty list
        self.setWindowTitle("Rock Typing Application")
        self.setGeometry(100, 100, 1200, 900)
        self.setStyleSheet("background-color: #f0f0f0;")
        self.setWindowIcon(QIcon("Axone_logo.png"))  # Set the window icon
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
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman'; color: #333;")
        header_label.setAlignment(Qt.AlignLeft)
        top_layout.addWidget(header_label)


        # Add top layout to the main layout
        main_layout.addLayout(top_layout)


        # Initialize tabs
        self.tabs = QTabWidget()  # Initialize QTabWidget
        self.tabs.setStyleSheet("""

            QTabWidget::pane { 

                border: 1px solid #0078d7; 

                background-color: #f0f0f0; 

            }

            QTabBar::tab {

                padding: 10px; 

                font-size: 14px; 

                font-weight: bold; 

                border: 1px solid #32a1a2; 

                border-bottom: none; 
                
                background-color: #f0f0f0;  /* Default background */

            }

            QTabBar::tab:selected {

                background: #32a1a2; 

                color: white; 

            }

            QTabBar::tab:hover {

                background:rgb(60, 203, 205); 
                color: white;  /* Text color on hover */

            }

        """)

        # Add tabs to the QTabWidget
        self.dataset_tab = QWidget()
        self.plots_tab = QWidget()
        self.distortion_clustering_tab = QWidget()
        self.rock_type_tab = QWidget()
        self.ml_tab = QWidget()
        self.distance_clustering_tab = QWidget()  # New tab for Distance Clustering
        
        # New tabs for Inertia Clustering and Inertia Rock Type
        self.inertia_clustering_tab = QWidget()

        self.inertia_rock_type_tab = QWidget()

        self.tabs.addTab(self.dataset_tab, QIcon("dataset_icon.png"), "Dataset")  # Add icons for tabs

        self.tabs.addTab(self.plots_tab, QIcon("plots_icon.png"), "Plots")

        self.tabs.addTab(self.distortion_clustering_tab, QIcon("clustering_icon.png"), "Distortion Clustering")
        
        self.tabs.addTab(self.rock_type_tab, QIcon("rock_type_icon.png"), "Distortion Rock Type")
        
        self.tabs.addTab(self.inertia_clustering_tab, QIcon("inertia_clustering_icon.png"), "Inertia Clustering")
        
        self.tabs.addTab(self.inertia_rock_type_tab, QIcon("inertia_rock_type_icon.png"), "Inertia Rock Type")

        self.tabs.addTab(self.distance_clustering_tab, QIcon("distance_clustering_icon.png"), "Distance Clustering")  # Add new tab


        # Add Tabs to the main layout
        main_layout.addWidget(self.tabs)


        # Set central widget layout
        central_widget = QWidget()
        
        central_widget.setLayout(main_layout)
        
        self.setCentralWidget(central_widget)

        # Initialize individual tabs
        self.init_dataset_tab()

        self.init_plots_tab()

        self.init_distortion_clustering_tab()
        
        self.init_inertia_clustering_tab()
        
        self.init_inertia_rock_type_tab()
        
        self.init_distortion_rock_type_tab()
          
        self.init_distance_clustering_tab() 
          
        # Connect mouse events for the tabs
        self.tabs.tabBar().installEventFilter(self)  

####### Simple Plot  #############

    def update_plots(self):

        # Extract data from the table

        porosity = []

        permeability = []

        rqi = []

        phi_z = []


        for row in range(self.table.rowCount()):

            try:

                p = float(self.table.item(row, 0).text()) if self.table.item(row, 0) else None

                k = float(self.table.item(row, 1).text()) if self.table.item(row, 1) else None

                r = float(self.table.item(row, 2).text()) if self.table.item(row, 2) else None

                z = float(self.table.item(row, 3).text()) if self.table.item(row, 3) else None


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

        scatter1 = axes[0].scatter(

            porosity, 

            permeability, 

            color='royalblue', 

            s=100,  # Size of the dots

            alpha=0.6,  # Transparency

            edgecolor='black',  # Edge color for better visibility

            linewidth=0.5,  # Edge width

            marker='o',  # Marker style

            label="Data Points"

        )


        axes[0].set_title("Absolute Permeability (md) vs Porosity", fontsize=16, fontweight='bold')

        axes[0].set_xlabel("Porosity", fontsize=14)

        axes[0].set_ylabel("Absolute Permeability (md)", fontsize=14)

        axes[0].grid(True, linestyle='--', alpha=0.7)  # Add grid lines

        axes[0].legend()


        # Calculate power law fit

        if len(porosity) > 0 and len(permeability) > 0:

            log_porosity = np.log(porosity)
            log_permeability = np.log(permeability)

            # Perform linear regression on log-log scale
            coeffs = np.polyfit(log_porosity, log_permeability, 1)
            a = np.exp(coeffs[1])  # Intercept
            b = coeffs[0]  # Slope

            # Generate trend line data
            porosity_fit = np.linspace(min(porosity), max(porosity), 100)
            permeability_fit = a * (porosity_fit ** b)

            # Plot the trend line
            axes[0].plot(porosity_fit, permeability_fit, color='orange', label=f'Trend Line: y = {a:.2f}x^{b:.2f}')
            axes[0].legend()

        # Plot 2: log(RQI) vs log(Phi z)

        log_rqi = np.log(rqi) if rqi else []
        
        log_phi_z = np.log(phi_z) if phi_z else []

        if log_phi_z.size > 0 and log_rqi.size > 0:  # Updated condition
            scatter2 = axes[1].scatter(
                log_phi_z, 
                log_rqi, 
                color='tomato', 
                s=100,  # Size of the dots
                alpha=0.6,  # Transparency
                edgecolor='black',  # Edge color for better visibility
                linewidth=0.5,  # Edge width
                marker='o',  # Marker style
                label="Data Points"

            )


            axes[1].set_title("log(RQI) vs log(Phi z)", fontsize=16, fontweight='bold')

            axes[1].set_xlabel("log(Phi z)", fontsize=14)

            axes[1].set_ylabel("log(RQI)", fontsize=14)

            axes[1].grid(True, linestyle='--', alpha=0.7)  # Add grid lines

            axes[1].legend()


            # Synchronize X and Y axis limits

            min_limit = min(min(log_phi_z), min(log_rqi))

            max_limit = max(max(log_phi_z), max(log_rqi))

            axes[1].set_xlim(min_limit, max_limit)

            axes[1].set_ylim(min_limit, max_limit)


        # Update the canvas

        self.plot_canvas.draw()


        # Reconnect tooltip functionality

        self.plot_canvas.mpl_connect('motion_notify_event', lambda event: self.handle_hover_event(event))
        
        self.plot_data = [

        {"scatter": scatter1, "x_data": porosity, "y_data": permeability, "axis": axes[0]},

        {"scatter": scatter2, "x_data": log_phi_z.tolist(), "y_data": log_rqi.tolist(), "axis": axes[1]}

        ]
    
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
    
######## Inertia Section ###############
    
    def init_inertia_clustering_tab(self):
        layout = QVBoxLayout()
    
        # Header
        header_label = QLabel("Inertia Clustering")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Inertia Plot Layout
        self.inertia_plot_layout = QVBoxLayout()
        layout.addLayout(self.inertia_plot_layout)

        # Add inputs for max clusters
        max_clusters_layout = QHBoxLayout()
        self.max_clusters_textbox_inertia = QLineEdit()
        self.max_clusters_textbox_inertia.setPlaceholderText("Maximum Number of Clusters (e.g., 10)")
        self.max_clusters_textbox_inertia.setValidator(QIntValidator(1, 50))
        max_clusters_layout.addWidget(QLabel("Maximum Number of Clusters:"))
        max_clusters_layout.addWidget(self.max_clusters_textbox_inertia)
        layout.addLayout(max_clusters_layout)

        # Add Recommended K inputs
        recommended_k_layout = QHBoxLayout()
        self.inertia_selected_K_textbox = QLineEdit()
        self.inertia_selected_K_textbox.setPlaceholderText("Recommended Number of Clusters")
        recommended_k_layout.addWidget(QLabel("Recommended Number of Clusters:"))
        recommended_k_layout.addWidget(self.inertia_selected_K_textbox)
        layout.addLayout(recommended_k_layout)

        # Spacer for alignment
        layout.addStretch()

        # Button for clustering, placed at the bottom
        button_layout = QHBoxLayout()
        cluster_button = QPushButton("Generate Inertia Plot")
        cluster_button.clicked.connect(self.generate_inertia_plot)
        self.style_button(cluster_button)  # Reuse button styling
        button_layout.addWidget(cluster_button)

        layout.addLayout(button_layout)

        self.inertia_clustering_tab.setLayout(layout)
         
    def init_inertia_rock_type_tab(self):
        layout = QVBoxLayout()
        
        # Header
        header_label = QLabel("Inertia Rock Type Visualization")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        fig.tight_layout(pad=5.0)

        # Placeholders for axes
        axes[0].set_title("Empty Plot 1")
        axes[0].axis('off')
        axes[1].set_title("Empty Plot 2")
        axes[1].axis('off')
        
        # Add canvas to layout
        self.rock_type_canvas_inertia = FigureCanvas(fig)
        layout.addWidget(self.rock_type_canvas_inertia)
        self.rock_type_canvas_inertia.mpl_connect('button_press_event', self.handle_plot_inertia_click)
        
        # Button for plotting inertia rock type
        plot_button_inertia = QPushButton("Plot Inertia Rock Type Data")
        plot_button_inertia.clicked.connect(self.update_inertia_rock_type)
        self.style_button(plot_button_inertia)  # Reuse button styling

        # Spacer for alignment
        layout.addStretch()

        # Button for plotting, placed at the bottom
        button_layout = QHBoxLayout()
        plot_button_inertia = QPushButton("Plot Inertia Rock Type Data")  # Updated button text
        plot_button_inertia.clicked.connect(self.update_inertia_rock_type)
        self.style_button(plot_button_inertia)  # Reuse button styling
        button_layout.addWidget(plot_button_inertia)
        
        #layout.addWidget(plot_button_inertia)
        # Create a figure with subplots

        layout.addLayout(button_layout)
        
        self.inertia_rock_type_tab.setLayout(layout)
    
    def update_inertia_rock_type(self):
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

        # Check if data is sufficient
        if not porosity or not permeability or not rqi or not phi_z:
            QMessageBox.warning(self, "Warning", "Insufficient data to plot. Please enter valid data.")
            return

        # Prepare data for clustering
        X = np.array(list(zip(porosity, permeability)))

        # Initialize n_clusters
        n_clusters = None
        
        try:
            n_clusters = int(self.inertia_selected_K_textbox.text())
            if n_clusters <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid number of clusters.")
            return

        if n_clusters > len(X):
            QMessageBox.warning(self, "Error", f"Number of clusters ({n_clusters}) exceeds the number of samples ({len(X)}).")
            return

        # Perform KMeans clustering
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
        self.rock_type_canvas_inertia.figure.clear()

        # Create a 1x2 grid for the subplots
        axes = self.rock_type_canvas_inertia.figure.subplots(1, 2)
        self.rock_type_canvas_inertia.figure.tight_layout(pad=5.0)

        # Plot 1: Absolute Permeability vs Porosity
        scatter1 = axes[0].scatter(porosity, permeability, c=clusters, cmap='tab10', alpha=0.6, s=150, edgecolor='black', marker='o')
        axes[0].set_title("Absolute Permeability (md) vs Porosity")
        axes[0].set_xlabel("Porosity")
        axes[0].set_ylabel("Absolute Permeability (md)")
        axes[0].grid(True)
        legend1 = axes[0].legend(*scatter1.legend_elements(), title="Cluster")
        axes[0].add_artist(legend1)

        # Plot 2: log(RQI) vs log(Phi z)
        #valid_indices = np.where((np.array(rqi) > 0) & (np.array(phi_z) > 0))[0]
        #log_rqi = np.log(np.array(rqi)[valid_indices])
        #log_phi_z = np.log(np.array(phi_z)[valid_indices])
        #filtered_clusters = clusters[valid_indices]
        
        # Plot 2: log(RQI) vs log(Phi z)
        scatter2 = axes[1].scatter(log_phi_z, log_rqi, c=filtered_clusters, cmap='tab10', alpha=0.6, s=150, edgecolor='black', marker='o')
        axes[1].set_title("log(RQI) vs log(Phi z)")
        axes[1].set_xlabel("log(Phi z)")
        axes[1].set_ylabel("log(RQI)")
        axes[1].grid(True)
        legend2 = axes[1].legend(*scatter2.legend_elements(), title="Cluster")
        axes[1].add_artist(legend2)

        # Synchronize X and Y axis limits
        
        min_limit = min(min(log_phi_z), min(log_rqi))
        max_limit = max(max(log_phi_z), max(log_rqi))
        axes[1].set_xlim(min_limit, max_limit)
        axes[1].set_ylim(min_limit, max_limit)

        self.rock_type_canvas_inertia.mpl_connect('motion_notify_event', self.handle_rock_type_hover_event_inertia)

        self.rock_type_canvas_inertia.mpl_connect('button_press_event', self.handle_plot_inertia_click)

        # Update the canvas
        self.rock_type_canvas_inertia.draw()
    
    def generate_inertia_plot(self):
        rqi = []
        phi_z = []

        # Extract RQI and Phi z data from the table
        for row in range(self.table.rowCount()):
            try:
                if self.table.item(row, 2) and self.table.item(row, 2).text():
                    rqi_value = float(self.table.item(row, 2).text())
                    if rqi_value > 0:  # Ensure we only log positive values
                        rqi.append(rqi_value)

                if self.table.item(row, 3) and self.table.item(row, 3).text():
                    phi_z_value = float(self.table.item(row, 3).text())
                    if phi_z_value > 0:  # Ensure we only log positive values
                        phi_z.append(phi_z_value)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", f"Non-numeric value in row {row + 1}. Skipping the row.")
                continue

        if not rqi or not phi_z:
            QMessageBox.warning(self, "Insufficient Data", "Please enter valid data in RQI and Phi z columns before generating the Inertia Plot.")
            return

        # Prepare data for clustering using log values
        log_rqi = np.log(np.array(rqi))
        log_phi_z = np.log(np.array(phi_z))
        X = np.array(list(zip(log_rqi, log_phi_z)))

        # Get the maximum number of clusters for inertia plot
        inertia_max_clusters_text = self.max_clusters_textbox_inertia.text()
        try:
            max_clusters = int(inertia_max_clusters_text)
            if max_clusters <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number of clusters.")
            return

        # Generate inertia plot
        inertias = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # Create the plot
        fig, ax = plt.subplots(figsize=(5, 10))
        
        scatter = ax.scatter(
            range(1, max_clusters + 1), inertias, color='blue', s=100, label='Inertia Points'
        )
        
        ax.plot(range(1, max_clusters + 1), inertias, marker='o', color='blue', linestyle='-', label='Inertia Curve')
        
        ax.set_xlim(1, max_clusters)  # Ensure X-axis ranges from 1 to max_clusters
        
        # Highlight the optimal K with a red circle
        optimal_k = self.find_optimal_k(inertias)
        selected_circle = ax.scatter(
            optimal_k,
            inertias[optimal_k - 1],
            facecolors='none',
            s=500,
            linewidth=2,
            label='Recommended k',
            edgecolor='red'
        )
        
        # Set plot labels and title
        ax.set_title("Inertia Method to Find the Optimal Number of Clusters", fontsize=14, fontweight='bold')
        ax.set_xlabel("Number of Clusters", fontsize=12)
        ax.set_ylabel("Inertia", fontsize=12)
        
        # Add legend to the plot
        ax.legend(loc='best', fontsize=10, title="Legend")

        # Attach hover and click events
        self.hover_circle = None  # To store the circle artist for hover effect
        
        fig.canvas.mpl_connect('motion_notify_event', lambda event: self.on_hover_inertia_plot(event, scatter, ax))
        
        fig.canvas.mpl_connect('button_press_event', lambda event: self.on_click_inertia_plot(event, inertias))
        
        # Replace or update the canvas
        if hasattr(self, 'inertia_canvas') and self.inertia_canvas:
            
            self.inertia_clustering_tab.layout().removeWidget(self.inertia_canvas)
            self.inertia_canvas.deleteLater()
            self.inertia_canvas = None

        self.inertia_canvas = FigureCanvas(fig)
        self.inertia_clustering_tab.layout().addWidget(self.inertia_canvas)
        self.inertia_canvas.draw()

    def handle_rock_type_hover_event_inertia(self, event):
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
                    self.rock_type_canvas_inertia.draw_idle()
                    return

        # Remove tooltip if not hovering over any point
        if self.rock_type_tooltip:
            self.rock_type_tooltip.remove()
            self.rock_type_tooltip = None
            self.rock_type_canvas_inertia.draw_idle()
    
    def on_hover_inertia_plot(self, event, scatter, ax):
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
                self.inertia_canvas.draw_idle()  # Redraw the canvas
            else:
                # Remove the circle if not hovering over any point
                if self.hover_circle:
                    self.hover_circle.remove()
                    self.hover_circle = None
                    self.inertia_canvas.draw_idle()
    
    def on_click_inertia_plot(self, event, inertias):

        cont, ind = event.inaxes.collections[0].contains(event)

        if cont:

            index = ind["ind"][0]

            chosen_k = index + 1

            self.inertia_selected_K_textbox.setText(str(chosen_k))

            QMessageBox.information(self, "Chosen k", f"You have chosen k = {chosen_k}")
    
    def handle_plot_inertia_click(self, event):
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
            save_plot_action.triggered.connect(lambda: self.save_plot(self.rock_type_canvas_inertia))

            if hasattr(self, "current_plot_data") and self.current_plot_data:
                export_csv_action = menu.addAction("Export Data as CSV")
                export_csv_action.triggered.connect(self.export_plot_data_to_csv)

            menu.exec_(QCursor.pos())

######## Distortion Section ###############

    def init_distortion_clustering_tab(self):
        layout = QVBoxLayout()
    
        # Header
        header_label = QLabel("Distortion Clustering")
        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Distortion Plot Layout
        self.distortion_plot_layout = QVBoxLayout()
        layout.addLayout(self.distortion_plot_layout)

        # Add inputs for max clusters
        max_clusters_layout = QHBoxLayout()
        self.max_clusters_textbox_distortion = QLineEdit()
        self.max_clusters_textbox_distortion.setPlaceholderText("Maximum Number of Clusters (e.g., 10)")
        self.max_clusters_textbox_distortion.setValidator(QIntValidator(1, 50))
        max_clusters_layout.addWidget(QLabel("Maximum Number of Clusters:"))
        max_clusters_layout.addWidget(self.max_clusters_textbox_distortion)
        layout.addLayout(max_clusters_layout)

        # Add Recommended K inputs
        recommended_k_layout = QHBoxLayout()
        self.distortion_selected_K_textbox = QLineEdit()
        self.distortion_selected_K_textbox.setPlaceholderText("Recommended Number of Clusters")
        recommended_k_layout.addWidget(QLabel("Recommended Number of Clusters:"))
        recommended_k_layout.addWidget(self.distortion_selected_K_textbox)
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

        self.distortion_clustering_tab.setLayout(layout)
    
    def init_distortion_rock_type_tab(self):
        layout = QVBoxLayout()

        # Header
        header_label = QLabel("Distortion Rock Type Visualization")  # Updated header
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
        self.rock_type_canvas_distortion = FigureCanvas(fig)
        layout.addWidget(self.rock_type_canvas_distortion)
        self.rock_type_canvas_distortion.mpl_connect('button_press_event', self.handle_plot_distortion_click)

        # Spacer for alignment
        layout.addStretch()

        # Button for plotting, placed at the bottom
        button_layout = QHBoxLayout()
        plot_button_distortion = QPushButton("Plot Distortion Rock Type Data")  # Updated button text
        plot_button_distortion.clicked.connect(self.update_distortion_rock_type)
        self.style_button(plot_button_distortion)  # Reuse button styling
        button_layout.addWidget(plot_button_distortion)

        layout.addLayout(button_layout)

        self.rock_type_tab.setLayout(layout)
    
    def update_distortion_rock_type(self):

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
        
        # Initialize n_clusters
        n_clusters = None
        
        try:
            n_clusters = int(self.distortion_selected_K_textbox.text())
            if n_clusters <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Error", "Please enter a valid number of clusters.")
            return  # Exit the function if the input is invalid

        if n_clusters > len(X):
            QMessageBox.warning(self, "Error", f"Number of clusters ({n_clusters}) exceeds the number of samples ({len(X)}).")
            return
        
        # Perform KMeans clustering
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
        self.rock_type_canvas_distortion.figure.clear()

        # Create a 1x2 grid for the subplots
        axes = self.rock_type_canvas_distortion.figure.subplots(1, 2)
        self.rock_type_canvas_distortion.figure.tight_layout(pad=5.0)

        # Plot 1: Absolute Permeability vs Porosity
        scatter1 = axes[0].scatter(porosity, permeability, c=clusters, cmap='tab10', alpha=0.6, s=150, edgecolor='black', marker='o')  # Change size and marker style
        axes[0].set_title("Absolute Permeability (md) vs Porosity")
        axes[0].set_xlabel("Porosity")
        axes[0].set_ylabel("Absolute Permeability (md)")
        axes[0].grid(True)
        legend1 = axes[0].legend(*scatter1.legend_elements(), title="Cluster")
        axes[0].add_artist(legend1)

        # Plot 2: log(RQI) vs log(Phi z)
        scatter2 = axes[1].scatter(log_phi_z, log_rqi, c=clusters, cmap='tab10', alpha=0.6, s=150, edgecolor='black', marker='o')  # Change size and marker style
        axes[1].set_title("log(RQI) vs log(Phi z)")
        axes[1].set_xlabel("log(Phi z)")
        axes[1].set_ylabel("log(RQI)")
        axes[1].grid(True)
        legend2 = axes[1].legend(*scatter2.legend_elements(), title="Cluster")
        axes[1].add_artist(legend2)


        # Synchronize X and Y axis limits

        min_limit = min(min(log_phi_z), min(log_rqi))
        max_limit = max(max(log_phi_z), max(log_rqi))
        axes[1].set_xlim(min_limit, max_limit)
        axes[1].set_ylim(min_limit, max_limit)

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


        self.rock_type_canvas_distortion.mpl_connect('motion_notify_event', self.handle_rock_type_hover_event_distortion)

        self.rock_type_canvas_distortion.mpl_connect('button_press_event', self.handle_plot_distortion_click)


        # Update the canvas

        self.rock_type_canvas_distortion.draw()
    
    def generate_distortion_plot(self):
        rqi = []
        phi_z = []

        # Extract RQI and Phi z data from the table
        for row in range(self.table.rowCount()):
            try:
                if self.table.item(row, 2) and self.table.item(row, 2).text():
                    rqi_value = float(self.table.item(row, 2).text())
                    if rqi_value > 0:  # Ensure we only log positive values
                        rqi.append(rqi_value)

                if self.table.item(row, 3) and self.table.item(row, 3).text():
                    phi_z_value = float(self.table.item(row, 3).text())
                    if phi_z_value > 0:  # Ensure we only log positive values
                        phi_z.append(phi_z_value)
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", f"Non-numeric value in row {row + 1}. Skipping the row.")
                continue

        if not rqi or not phi_z:
            QMessageBox.warning(self, "Insufficient Data", "Please enter valid data in RQI and Phi z columns before generating the Distortion Plot.")
            return

        # Prepare data for clustering using log values
        log_rqi = np.log(np.array(rqi))
        log_phi_z = np.log(np.array(phi_z))
        X = np.array(list(zip(log_rqi, log_phi_z)))

        # Get the maximum number of clusters for distortion plot
        max_clusters_text_distortion = self.max_clusters_textbox_distortion.text()
        try:
            max_clusters = int(max_clusters_text_distortion)
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

        # Highlight the optimal K with a red circle
        optimal_k = self.find_optimal_k(distortions)
        selected_circle = ax.scatter(
            optimal_k,
            distortions[optimal_k - 1],
            facecolors='none',
            edgecolors='black',
            s=100,
            linewidth=0.5,
            label='Recommended k',
            alpha=0.6,  # Transparency
        )

        # Set plot labels and title
        ax.set_title("Distortion Method to Find the Optimal Number of Clusters", fontsize=14, fontweight='bold')
        ax.set_xlabel("Number of Clusters", fontsize=12)
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
            self.distortion_selected_K_textbox.setText(str(chosen_k))
            QMessageBox.information(self, "Chosen k", f"You have chosen k = {chosen_k}")
    
    def handle_plot_distortion_click(self, event):
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
            save_plot_action.triggered.connect(lambda: self.save_plot(self.rock_type_canvas_distortion))

            if hasattr(self, "current_plot_data") and self.current_plot_data:
                export_csv_action = menu.addAction("Export Data as CSV")
                export_csv_action.triggered.connect(self.export_plot_data_to_csv)

            menu.exec_(QCursor.pos())

    def handle_rock_type_hover_event_distortion(self, event):
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
                    self.rock_type_canvas_distortion.draw_idle()
                  
                    return

        # Remove tooltip if not hovering over any point
        if self.rock_type_tooltip:
            self.rock_type_tooltip.remove()
            self.rock_type_tooltip = None
            self.rock_type_canvas_distortion.draw_idle()
      
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
        
######## Distance Section ###############
    
    def init_distance_clustering_tab(self):

        layout = QVBoxLayout()

        # Header
        header_label = QLabel("Distance Clustering")

        header_label.setStyleSheet("font-size: 35px; font-weight: bold; font-family: 'Times New Roman';")

        header_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(header_label)


        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))

        fig.tight_layout(pad=5.0)


        # Set titles for empty plots
        axes[0].set_title("Porosity vs Permeability")

        axes[0].set_xlabel("Porosity")

        axes[0].set_ylabel("Permeability (md)")

        axes[0].grid(True)


        axes[1].set_title("Log(RQI) vs Log(Phi z)")

        axes[1].set_xlabel("Log(Phi z)")

        axes[1].set_ylabel("Log(RQI)")

        axes[1].grid(True)


        # Add canvas to layout
        self.distance_clustering_canvas = FigureCanvas(fig)

        layout.addWidget(self.distance_clustering_canvas)


        # Add input fields for distance clustering parameters
        self.distance_input = QLineEdit()

        self.distance_input.setPlaceholderText("Enter distance threshold")

        layout.addWidget(self.distance_input)


        # Button to perform distance clustering
        cluster_button = QPushButton("Perform Distance Clustering")

        cluster_button.clicked.connect(self.perform_distance_clustering)

        self.style_button(cluster_button)  # Reuse button styling

        layout.addWidget(cluster_button)


        self.distance_clustering_tab.setLayout(layout)


        # Connect hover event for tooltips
        self.distance_clustering_canvas.mpl_connect('motion_notify_event', self.handle_distance_clustering_hover_event)
        
        self.distance_clustering_canvas.mpl_connect('button_press_event', self.show_distance_clustering_context_menu)
    
    def update_distance_clustering_tab(self):

        # Extract data from the table for plotting
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

        # Clear the previous plots
        self.distance_clustering_canvas.figure.clear()
        


        # Create a 1x2 grid for the subplots
        axes = self.distance_clustering_canvas.figure.subplots(1, 2)

        self.distance_clustering_canvas.figure.tight_layout(pad=5.0)


        # Plot 1: Porosity vs Permeability
        axes[0].scatter(porosity, permeability, color='blue', alpha=0.6, s=100)

        axes[0].set_title("Porosity vs Permeability")

        axes[0].set_xlabel("Porosity")

        axes[0].set_ylabel("Permeability (md)")

        axes[0].grid(True)


        # Plot 2: Log(RQI) vs Log(Phi z)
        log_rqi = np.log(np.array(rqi))

        log_phi_z = np.log(np.array(phi_z))

        axes[1].scatter(log_phi_z, log_rqi, color='orange', alpha=0.6, s=100)

        axes[1].set_title("Log(RQI) vs Log(Phi z)")

        axes[1].set_xlabel("Log(Phi z)")

        axes[1].set_ylabel("Log(RQI)")

        axes[1].grid(True)


        # Update the canvas
        self.distance_clustering_canvas.draw()
        
        self.distance_clustering_canvas.mpl_connect('motion_notify_event', self.handle_distance_clustering_hover_event)
    
    def show_distance_clustering_context_menu(self, event):

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

            save_plot_action.triggered.connect(lambda: self.save_plot(self.distance_clustering_canvas))


            export_data_action = menu.addAction("Export Merged Data as CSV")

            export_data_action.triggered.connect(self.export_merged_distance_clustering_data_to_csv)


            menu.exec_(QCursor.pos())
    
    def export_merged_distance_clustering_data_to_csv(self):

        if not hasattr(self, "current_distance_clustering_data") or not self.current_distance_clustering_data:

            QMessageBox.warning(self, "No Data", "No distance clustering data available for export.")

            return


        # Extract data for merging

        log_rqi = np.array(self.current_distance_clustering_data["log_rqi"])

        log_phi_z = np.array(self.current_distance_clustering_data["log_phi_z"])

        clusters = self.current_distance_clustering_data["clusters"]


        # Prepare data for DataFrame

        data = []

        for cluster_index, cluster in enumerate(clusters):

            for idx in cluster:

                # Append the required data to the list

                data.append({

                    "Porosity": self.table.item(idx, 0).text() if self.table.item(idx, 0) else None,

                    "Permeability": self.table.item(idx, 1).text() if self.table.item(idx, 1) else None,

                    "Log(RQI)": log_rqi[idx],

                    "Log(Phi z)": log_phi_z[idx],

                    "Cluster": cluster_index

                })


        # Create DataFrame

        df = pd.DataFrame(data)


        # Open a file dialog to save the CSV

        options = QFileDialog.Options()

        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)


        if file_path:

            try:

                df.to_csv(file_path, index=False)

                QMessageBox.information(self, "Success", "Merged distance clustering data exported successfully.")

            except Exception as e:

                QMessageBox.critical(self, "Error", f"Failed to export data: {e}")

    def perform_distance_clustering(self):

        # Extract data from the table for clustering

        log_rqi = []

        log_phi_z = []

        porosity = []

        permeability = []

        rqi = []

        phi_z = []


        for row in range(self.table.rowCount()):

            try:

                if self.table.item(row, 2) and self.table.item(row, 2).text():

                    log_rqi.append(np.log(float(self.table.item(row, 2).text())))

                    rqi.append(float(self.table.item(row, 2).text()))


                if self.table.item(row, 3) and self.table.item(row, 3).text():

                    log_phi_z.append(np.log(float(self.table.item(row, 3).text())))

                    phi_z.append(float(self.table.item(row, 3).text()))


                if self.table.item(row, 0) and self.table.item(row, 0).text():

                    porosity.append(float(self.table.item(row, 0).text()))


                if self.table.item(row, 1) and self.table.item(row, 1).text():

                    permeability.append(float(self.table.item(row, 1).text()))


            except ValueError:

                continue  # Skip rows with invalid or missing data


        if not log_rqi or not log_phi_z:

            QMessageBox.warning(self, "Warning", "Insufficient data to perform clustering.")

            return


        # Convert lists to numpy arrays for distance calculations

        log_rqi = np.array(log_rqi)

        log_phi_z = np.array(log_phi_z)

        points = np.column_stack((log_rqi, log_phi_z))


        # Get the distance threshold from the input

        distance_threshold = self.distance_input.text()

        try:

            distance_threshold = float(distance_threshold)

        except ValueError:

            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for the distance threshold.")

            return


        # Perform clustering based on distance threshold

        clusters = self.cluster_points(points, distance_threshold)
        
        # After clustering, store the data

        self.current_distance_clustering_data = {

            "log_rqi": log_rqi,

            "log_phi_z": log_phi_z,

            "clusters": clusters,

            # Add any other relevant data you want to export

        }
        
            # Debugging output

        print("Distance clustering data stored:", self.current_distance_clustering_data)


        # Plot the results

        self.plot_distance_clustering(points, clusters, porosity, permeability, rqi, phi_z)
           
    def plot_distance_clustering(self, points, clusters, porosity, permeability, rqi, phi_z):

        # Clear the previous plots

        self.distance_clustering_canvas.figure.clear()
    

        axes = self.distance_clustering_canvas.figure.subplots(1, 2)  # Create 1x2 subplots

        
        

        # Assign colors for clusters

        colors = plt.get_cmap('tab10', len(clusters))  # Get a colormap with enough colors


        # Plot 1: Porosity vs Permeability

        for cluster_index, cluster in enumerate(clusters):

            cluster_points = [(porosity[i], permeability[i]) for i in cluster]

            cluster_porosity, cluster_permeability = zip(*cluster_points)

            axes[0].scatter(cluster_porosity, cluster_permeability, color=colors(cluster_index), alpha=0.6, s=150, linewidth=0.1, label=f'Cluster {cluster_index + 1}')


        axes[0].set_title("Porosity vs Permeability")

        axes[0].set_xlabel("Porosity")

        axes[0].set_ylabel("Permeability (md)")

        axes[0].grid(True)

        axes[0].legend()


        # Plot 2: Log(RQI) vs Log(Phi z)

        for cluster_index, cluster in enumerate(clusters):

            cluster_points = [(np.log(phi_z[i]), np.log(rqi[i])) for i in cluster if rqi[i] > 0 and phi_z[i] > 0]

            if cluster_points:  # Check if there are points in the cluster

                cluster_log_phi_z, cluster_log_rqi = zip(*cluster_points)

                axes[1].scatter(cluster_log_phi_z, cluster_log_rqi, color=colors(cluster_index), alpha=0.6, s=150, linewidth=0.1,  label=f'Cluster {cluster_index + 1}')


        axes[1].set_title("Log(RQI) vs Log(Phi z)")

        axes[1].set_xlabel("Log(Phi z)")

        axes[1].set_ylabel("Log(RQI)")

        axes[1].grid(True)

        axes[1].legend()


        
        
        # Update the canvas

        self.distance_clustering_canvas.draw()
    
    def handle_distance_clustering_hover_event(self, event):

        if event.inaxes is not None:  # Check if the event is within the axes

            for ax in self.distance_clustering_canvas.figure.axes:

                if event.inaxes == ax:

                    for scatter in ax.collections:  # Iterate through scatter plots

                        cont, ind = scatter.contains(event)

                        if cont:

                            index = ind["ind"][0]

                            x = scatter.get_offsets()[index][0]

                            y = scatter.get_offsets()[index][1]

                            tooltip_text = f"({x:.2f}, {y:.2f})"


                            # Remove previous tooltip

                            if self.tooltip:

                                self.tooltip.remove()


                            # Create new tooltip

                            self.tooltip = ax.annotate(

                                tooltip_text,

                                (x, y),

                                textcoords="offset points",

                                xytext=(10, 10),

                                ha='center',

                                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow"),

                                fontsize=10

                            )


                            self.distance_clustering_canvas.draw_idle()

                            return


            # Remove tooltip if not hovering over any point

            if self.tooltip:

                self.tooltip.remove()

                self.tooltip = None

                self.distance_clustering_canvas.draw_idle()
    
    def export_distance_clustering_data_to_csv(self):
        if not hasattr(self, "current_distance_clustering_data") or not self.current_distance_clustering_data:
            QMessageBox.warning(self, "No Data", "No distance clustering data available for export.")
            return

        # Open a file dialog to save the CSV
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_path:
            try:
                # Extract and filter data
                log_rqi = np.array(self.current_distance_clustering_data["log_rqi"])  # Convert to NumPy array
                log_phi_z = np.array(self.current_distance_clustering_data["log_phi_z"])  # Convert to NumPy array
                clusters = self.current_distance_clustering_data["clusters"]  # This should be a list of lists

                # Prepare data for DataFrame
                data = []
                for cluster_index, cluster in enumerate(clusters):
                    for idx in cluster:
                        data.append({
                            "log_RQI": log_rqi[idx],
                            "log_Phi_z": log_phi_z[idx],
                            "Cluster": cluster_index
                        })

                # Create DataFrame
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)

                QMessageBox.information(self, "Success", "Distance clustering data exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export data: {e}")
     
###### Date entry section ############
             
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
        # Map the position to global coordinates for the context menu
        global_position = self.table.viewport().mapToGlobal(position)

        # Create the context menu
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

        # Add "Delete" action to clear the value of the selected cell
        delete_action = menu.addAction("Delete")
        delete_action.triggered.connect(self.handle_delete)

        # Add "Delete All" action to clear the entire column
        delete_all_action = menu.addAction("Delete All")
        delete_all_action.triggered.connect(self.handle_delete_all)

        # Add "Save as CSV" action to export the table as CSV
        save_csv_action = menu.addAction("Save as CSV")
        save_csv_action.triggered.connect(self.export_to_csv)

        # Show the context menu at the global position
        menu.exec_(global_position)
    
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

#######  Miscellaneous section ##########

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


        # Remove tooltip if not hovering over any point

        if self.tooltip:

            self.tooltip.remove()

            self.tooltip = None

            self.plot_canvas.draw_idle()
    
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
                n_clusters = int(self.max_clusters_texbox_distortion.text())
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
    
    def custom_clustering(self):
        # Handling custom clustering with user-defined K
        user_k = self.max_clusters_textbox.text()
        if not user_k:
            QMessageBox.warning(self, "Warning", "Please enter a valid number for K.")
            return

        optimal_k = int(user_k)
        self.perform_clustering(optimal_k)
    
    def show_plot(self, fig):
        if hasattr(self, 'canvas') and self.canvas:
            self.plots_tab.layout().removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.canvas = None

        self.canvas = FigureCanvas(fig)
        self.plots_tab.layout().addWidget(self.canvas)

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
            
            save_plot_action.triggered.connect(lambda: self.save_plot(self.plot_canvas))

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
     
    def cluster_points(self, points, threshold):

        from scipy.spatial.distance import cdist


        # Calculate the distance matrix

        distance_matrix = cdist(points, points)


        # Initialize clusters

        clusters = []

        visited = set()


        for i in range(len(points)):

            if i in visited:

                continue


            # Start a new cluster

            current_cluster = [i]

            visited.add(i)


            # Find all points within the threshold distance

            for j in range(len(points)):

                if j != i and distance_matrix[i][j] <= threshold:

                    current_cluster.append(j)

                    visited.add(j)


            clusters.append(current_cluster)


        return clusters
        
########  Executing the application ############
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Create and display the splash screen
    splash_pix = QPixmap("Axone_logo.png")  # Replace with your image path
    splash = QSplashScreen(splash_pix)
    splash.setMask(splash_pix.mask())
    splash.show()
    
    # Simulate loading time (optional)
    time.sleep(2)  # Adjust the time as needed
    
    # Initialize the main application
    main_app = MainApp()
    main_app.show()
    
    # Close the splash screen
    splash.finish(main_app)
    sys.exit(app.exec_())