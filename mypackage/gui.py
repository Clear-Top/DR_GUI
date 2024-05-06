from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap
import pandas as pd
import os
import pacmap                       # PaCMAP
import umap                         # UMAP
import trimap                       # TriMAP
from sklearn.manifold import TSNE   # t-SNE
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from threading import Thread
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class GraphWindow(QMainWindow):
    def __init__(self, file_path, drName):
        super().__init__()
        
        self.file_path = file_path
        self.file_path_labels = file_path.split('.npy')[0] + '_labels.npy'
        self.file_name = file_path.split('/')[-1]
        
        self.setWindowTitle(f"Visualization with 3D-plot ({drName})")
        self.setGeometry(910, 100, 800, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.label = QLabel("Click the button to plot the graph")
        self.layout.addWidget(self.label)

        self.button = QPushButton("Plot Graph")
        self.button.clicked.connect(self.plot_graph)
        self.layout.addWidget(self.button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.layout.addWidget(self.progress_bar)
        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.savePlot)
        self.layout.addWidget(self.save_button)
        
        self.label = QLabel("Execution Time: 0 sec", self)
        self.layout.addWidget(self.label)
        
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        
        self.timer = QTimer()    # Revised
        self.timer_value = 0
        self.timer.timeout.connect(self.update_timer)
    
    def update_timer(self):
        self.timer_value += 0.001
        self.label.setText(f'Execution Time: {self.timer_value: .3f} sec')
    
    def savePlot(self):
        # 레이아웃을 캡처하여 QPixmap으로 변환
        pixmap = QPixmap(self.central_widget.size())
        self.central_widget.render(pixmap)
        
        # 파일 대화 상자 열기
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Layout", "", "PNG (*.png);;All Files (*)")

        if file_path:
            # 그래프 저장
            pixmap.save(file_path)
            print("Plot saved as:", file_path)
            
            try:
                QMessageBox.information(self, "Success", "성공적으로 저장되었습니다.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"저장 중에 오류가 발생했습니다: {str(e)}")
                
        
class PaCMAP(GraphWindow):
    def plot_graph(self):  
        def update_progress(progress):
            self.progress_bar.setValue(progress)

        def finish_plotting():
            self.button.setEnabled(True)  # Re-enable the button
            self.progress_bar.setValue(100)  # Set progress to 100% when 
            # finished
        
        # Function to plot the graph
        def plot():
            total_iterations = 100  # Total number of iterations for plotting
            X = np.load(self.file_path, allow_pickle=True)
            X = X.reshape(X.shape[0], -1)
            y = np.load(self.file_path_labels, allow_pickle=True)
            self.embedding = pacmap.PaCMAP(n_components=3, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=111) 
            X_transformed = self.embedding.fit_transform(X, init="pca")

            for i in range(total_iterations):
                progress = int((i + 1) / total_iterations * 100)
                
                update_progress(progress)
                time.sleep(0.01)

            finish_plotting()
            
            self.figure.clear() 
            ax = self.figure.add_subplot(111, projection='3d')
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2],cmap="Spectral", c=y, s=0.6)
            ax.set_title(self.file_name)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            self.canvas.draw()
            
            self.timer.stop()   # Revised

        self.timer_value = 0
        self.timer.start(1)     # Revised (occur signal by 1 milisec)
        self.button.setEnabled(False)  # Disable the button during plotting
        self.progress_bar.setValue(0)  # Reset progress bar
        
        # Plotting in a separate thread
        thread = Thread(target=plot)
        thread.start()

class UMAP(GraphWindow):
    def plot_graph(self):  
        def update_progress(progress):
            self.progress_bar.setValue(progress)

        def finish_plotting():
            self.button.setEnabled(True)  # Re-enable the button
            self.progress_bar.setValue(100)  # Set progress to 100% when 
            # finished
        
        # Function to plot the graph
        def plot():
            total_iterations = 100  # Total number of iterations for plotting
            X = np.load(self.file_path, allow_pickle=True)
            X = X.reshape(X.shape[0], -1)
            y = np.load(self.file_path_labels, allow_pickle=True)
            self.embedding = umap.UMAP(n_components=3, init='pca', random_state=111)
            X_transformed = self.embedding.fit_transform(X)

            for i in range(total_iterations):
                progress = int((i + 1) / total_iterations * 100)
                
                update_progress(progress)
                time.sleep(0.01)

            finish_plotting()
            
            self.figure.clear() 
            ax = self.figure.add_subplot(111, projection='3d')
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2],cmap="Spectral", c=y, s=0.6)
            ax.set_title(self.file_name)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            self.canvas.draw()
            
            self.timer.stop()   # Revised

        self.timer_value = 0
        self.timer.start(1)     # Revised (occur signal by 1 milisec)
        self.button.setEnabled(False)  # Disable the button during plotting
        self.progress_bar.setValue(0)  # Reset progress bar
        
        # Plotting in a separate thread
        thread = Thread(target=plot)
        thread.start()
        
class TriMAP(GraphWindow):
    def plot_graph(self):  
        def update_progress(progress):
            self.progress_bar.setValue(progress)

        def finish_plotting():
            self.button.setEnabled(True)  # Re-enable the button
            self.progress_bar.setValue(100)  # Set progress to 100% when 
            # finished
        
        # Function to plot the graph
        def plot():
            total_iterations = 100  # Total number of iterations for plotting
            X = np.load(self.file_path, allow_pickle=True)
            X = X.reshape(X.shape[0], -1)
            y = np.load(self.file_path_labels, allow_pickle=True)
            self.embedding = trimap.TRIMAP(n_dims=3, n_random=111)
            X_transformed = self.embedding.fit_transform(X, init='pca')

            for i in range(total_iterations):
                progress = int((i + 1) / total_iterations * 100)
                
                update_progress(progress)
                time.sleep(0.01)

            finish_plotting()
            
            self.figure.clear() 
            ax = self.figure.add_subplot(111, projection='3d')
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2],cmap="Spectral", c=y, s=0.6)
            ax.set_title(self.file_name)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            self.canvas.draw()
            
            self.timer.stop()   # Revised

        self.timer_value = 0
        self.timer.start(1)     # Revised (occur signal by 1 milisec)
        self.button.setEnabled(False)  # Disable the button during plotting
        self.progress_bar.setValue(0)  # Reset progress bar
        
        # Plotting in a separate thread
        thread = Thread(target=plot)
        thread.start()

class tSNE(GraphWindow):
    def plot_graph(self):  
        def update_progress(progress):
            self.progress_bar.setValue(progress)

        def finish_plotting():
            self.button.setEnabled(True)  # Re-enable the button
            self.progress_bar.setValue(100)  # Set progress to 100% when 
            # finished
        
        # Function to plot the graph
        def plot():
            total_iterations = 100  # Total number of iterations for plotting
            X = np.load(self.file_path, allow_pickle=True)
            X = X.reshape(X.shape[0], -1)
            y = np.load(self.file_path_labels, allow_pickle=True)
            self.embedding = TSNE(n_components=3, init='pca', random_state=111)
            X_transformed = self.embedding.fit_transform(X)

            for i in range(total_iterations):
                progress = int((i + 1) / total_iterations * 100)
                
                update_progress(progress)
                time.sleep(0.01)

            finish_plotting()
            
            self.figure.clear() 
            ax = self.figure.add_subplot(111, projection='3d')
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2],cmap="Spectral", c=y, s=0.6)
            ax.set_title(self.file_name)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            self.canvas.draw()
            
            self.timer.stop()   # Revised

        self.timer_value = 0
        self.timer.start(1)     # Revised (occur signal by 1 milisec)
        self.button.setEnabled(False)  # Disable the button during plotting
        self.progress_bar.setValue(0)  # Reset progress bar
        
        # Plotting in a separate thread
        thread = Thread(target=plot)
        thread.start()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("파일 탐색기")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel("파일 내용이 여기에 표시됩니다.")
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        self.local_path = os.getcwd()
        self.file_system_model = QFileSystemModel()
        self.file_system_model.setRootPath(self.local_path)  # 현재 작업 디렉토리를 루트 경로로 설정
        
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_system_model)
        self.tree_view.setRootIndex(self.file_system_model.index(self.local_path))
        self.tree_view.clicked.connect(self.load_file)

        self.search_button = QPushButton("데이터프레임 조회")
        self.search_button.clicked.connect(self.display_dataframe)

        self.load_button = QPushButton("데이터프레임 초기화")
        self.load_button.clicked.connect(self.clear_dataframe)

        self.PaCMAP_button = QPushButton("3D 그래프 표시 (PaCMAP)")
        self.PaCMAP_button.clicked.connect(self.show_PaCMAP)
        
        self.UMAP_button = QPushButton("3D 그래프 표시 (UMAP)")
        self.UMAP_button.clicked.connect(self.show_UMAP)
        
        self.TriMAP_button = QPushButton("3D 그래프 표시 (TriMAP)")
        self.TriMAP_button.clicked.connect(self.show_TriMAP)
        
        self.tSNE_button = QPushButton("3D 그래프 표시 (tSNE)")
        self.tSNE_button.clicked.connect(self.show_tSNE)

        self.exit_button = QPushButton("종료")
        self.exit_button.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.tree_view)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.load_button)
        layout.addWidget(self.search_button)
        layout.addWidget(self.PaCMAP_button)
        layout.addWidget(self.UMAP_button)
        layout.addWidget(self.TriMAP_button)
        layout.addWidget(self.tSNE_button)
        layout.addWidget(self.exit_button)

        central_widget = QWidget()  # Revised
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_file(self, index):
        file_path = self.file_system_model.filePath(index)
        if not self.file_system_model.isDir(index):
            if file_path.endswith(".csv") or file_path.endswith(".npy"):
                self.label.setText("선택한 파일: " + file_path)
                # df = pd.read_csv(file_path)
                # self.display_dataframe(df)
            else:
                self.label.setText("선택한 파일은 csv/npy 이 아닙니다.")
        else:
            self.label.setText("선택한 항목은 파일이 아닙니다.")

    def display_dataframe(self):
        file_path = self.label.text().replace("선택한 파일: ", "")
        if file_path.endswith(".csv"):
            self.df = pd.read_csv(file_path)
            self.text_edit.clear()
            self.text_edit.append(str(self.df))
            self.label.setText("불러온 데이터프레임:")
        else:
            QMessageBox.warning(self, "경고", "선택한 파일은 CSV 파일이 아닙니다.")

    def clear_dataframe(self):
        self.text_edit.clear()
        self.label.setText("파일 내용이 여기에 표시됩니다.")
    
    def show_PaCMAP(self):
        file_path = self.label.text().replace("선택한 파일: ", "")
        print(file_path)
        if file_path.endswith(".csv") or file_path.endswith(".npy"):
            self.PaCMAP_window = PaCMAP(file_path, "PaCMAP")  # 상속 클래스
            self.PaCMAP_window.show()
        else:
            QMessageBox.warning(self, "경고", "선택한 파일은 CSV/npy 파일이 아닙니다.")
            
    def show_UMAP(self):
        file_path = self.label.text().replace("선택한 파일: ", "")
        print(file_path)
        if file_path.endswith(".csv") or file_path.endswith(".npy"):
            self.UMAP_window = UMAP(file_path, "UMAP")  # 상속 클래스
            self.UMAP_window.show()
        else:
            QMessageBox.warning(self, "경고", "선택한 파일은 CSV/npy 파일이 아닙니다.")

    def show_TriMAP(self):
        file_path = self.label.text().replace("선택한 파일: ", "")
        print(file_path)
        if file_path.endswith(".csv") or file_path.endswith(".npy"):
            self.TriMAP_window = TriMAP(file_path, "TriMAP")  # 상속 클래스
            self.TriMAP_window.show()
        else:
            QMessageBox.warning(self, "경고", "선택한 파일은 CSV/npy 파일이 아닙니다.")
            
    def show_tSNE(self):
        file_path = self.label.text().replace("선택한 파일: ", "")
        print(file_path)
        if file_path.endswith(".csv") or file_path.endswith(".npy"):
            self.tSNE_window = tSNE(file_path, "t-SNE")  # 상속 클래스
            self.tSNE_window.show()
        else:
            QMessageBox.warning(self, "경고", "선택한 파일은 CSV/npy 파일이 아닙니다.")        

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '프로그램 종료', 
            "프로그램을 종료하시겠습니까?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            QMessageBox.information(self, '종료', '프로그램이 정상적으로 종료되었습니다.')
            event.accept()
        else:
            event.ignore()

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
