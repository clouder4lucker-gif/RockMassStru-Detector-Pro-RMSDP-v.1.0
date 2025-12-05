## 其他模块
import sys,threading,io,tensorflow,json # 系统模块
import os,traceback,tempfile,shutil,subprocess # 系统模块

#重置Open3D环境
def setup_open3d_environment():
    # 确认我们在打包环境中
    if getattr(sys, 'frozen', False):
        # 1. 创建一个独立的 open3d 临时目录
        temp_dir = os.path.join(tempfile.gettempdir(), "open3d_runtime")
        os.makedirs(temp_dir, exist_ok=True)

        # 2. 设置环境变量，让 Open3D 使用它
        os.environ["O3D_TEMP_DIRECTORY"] = temp_dir
        os.environ["TEMP"] = temp_dir
        os.environ["TMP"] = temp_dir
        os.environ["TMPDIR"] = temp_dir
        os.chdir(temp_dir)

        # 3. 如果 open3d 在 _internal 目录下（即 PyInstaller 模式）
        internal_path = os.path.join(sys._MEIPASS, "open3d") if hasattr(sys, "_MEIPASS") else None
        if internal_path and os.path.exists(internal_path):
            # 拷贝必须的渲染动态库（否则 Visualizer 会因找不到 DLL 报错）
            dll_dir = os.path.join(internal_path, "resources")
            if os.path.exists(dll_dir):
                dest_dll_dir = os.path.join(temp_dir, "open3d_resources")
                if not os.path.exists(dest_dll_dir):
                    shutil.copytree(dll_dir, dest_dll_dir)
                os.environ["OPEN3D_RESOURCES_PATH"] = dest_dll_dir

        print("[Open3D Runtime Initialized]")
        print("TEMP DIR:", temp_dir)
        print("OPEN3D PATH:", internal_path or "not found")

setup_open3d_environment()

import open3d as o3d # 显示点云模块
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


import vtk # 显示点云模块
import numpy as np # 数据及操作模块
from sklearn.neighbors import BallTree # 智慧树搜寻点模块
import time #导入时间计算模块
import pickle # pkl保存模块
import math # 绘制模块
import pandas as pd # 保存成excel模块
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

## Qt页面操作模块
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSpinBox,
                             QFileDialog,QInputDialog,QMessageBox,
                             QFrame, QPushButton,QListWidgetItem,
                             QListWidget,QStyle,QDialog,QSizePolicy,
                             QVBoxLayout,QLineEdit,QLabel,QRadioButton)
from PyQt5.QtCore import QThread, pyqtSignal,Qt,QSize,QFileSystemWatcher,QObject,QTimer
from PyQt5.QtGui import QFont, QPalette, QColor, QIntValidator, QDoubleValidator

## 多线程模块
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderWindow
from threading import Lock,Event

## 导入人工神经网络模块
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dropout

## 导入绘图模块
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import rcParams

## 导入其他界面
import ui_Main_Window # 窗口处理UI界面
import ui_Login_2  # 导入界面设计函数

# 设置字体为 Times New Roman
rcParams['font.family'] = 'serif'  # serif 字体族中包括 Times New Roman
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12  # 全局字体大小设置
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

## 解决路径问题
if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))
else:
    os.chdir(os.path.dirname(__file__))


## 计算点曲率 类函数（1）-----主要负责开辟新线程计算
class Curvature_Calculation_Thread(QThread):
    result_signal = pyqtSignal(str)  # 用于返回计算完成的消息
    progress_signal = pyqtSignal(int)  # 用于传递进度信息
    data_signal = pyqtSignal(np.ndarray)  # 用于返回数据
    paused_signal = pyqtSignal(bool)  # 暂停信号
    time_signal = pyqtSignal(int)  # 用于运行时间

    def __init__(self, pointcloud,Number):
        super().__init__()
        self.pointcloud_CC = pointcloud
        self.Number = Number
        self.lock = Lock()
        self.paused_event = Event()  # 创建暂停事件
        self.paused_event.set()  # 默认是启动状态
        self.stop_flag = False  # 停止标志\

    def run(self):
       with self.lock:
            try:
                start_time = time.time()  # 记录开始时间
                Point_Locations = np.asarray(self.pointcloud_CC.points)
                Point_Length = len(Point_Locations)
                Point_Curvature = np.zeros((Point_Length, 3))
                tree = BallTree(Point_Locations)
                update_interval = 500000  # 定义更新间隔，调整进度条更新频率

                for i in range(Point_Length):
                    # 暂停控制模块
                    if self.stop_flag:
                        self.result_signal.emit("Calculation terminated")
                        return  # 如果停止标志为真，则终止线程
                    self.paused_event.wait()  # 如果是暂停状态，则阻塞在这里

                    # 点曲率计算模块
                    Query_Point = Point_Locations[i].reshape(1, -1)
                    _, indices = tree.query(Query_Point, self.Number)
                    Nearest_points = Point_Locations[indices][0]
                    Coordinate_Nearest = np.array(Nearest_points)
                    Coordinate_Average = np.mean(Coordinate_Nearest, axis=0)
                    Calculate_Variable = (Coordinate_Nearest - np.tile(Coordinate_Average, (self.Number, 1)))
                    Character_Array = Calculate_Variable.T @ Calculate_Variable
                    Eigenvalues, _ = np.linalg.eig(Character_Array)
                    Eigenvalues_min = min(Eigenvalues)
                    Point_Curvature[i, 2] = Eigenvalues_min / sum(Eigenvalues) * 100

                    # 更新进度条
                    if i % update_interval == 0:
                        progress = int(((i + 1) / len(Point_Locations)) * 100)
                        self.progress_signal.emit(progress)  # 发出进度信号
                    elif i == Point_Length - 1:
                        self.progress_signal.emit(100)  # 发出进度信号

                end_time = time.time()  # 记录结束时间
                total_time = end_time-start_time
                self.time_signal.emit(int(total_time))
                # 计算完成后，将结果传递到主线程
                self.data_signal.emit(Point_Curvature)  # 返回曲率数据和PcData
                self.result_signal.emit("Calculation complete")  # 发出完成信号


            except Exception as e:
                self.result_signal.emit(f"Miscalculation: {str(e)}")

    # 清除事件，线程进入等待状态
    def pause(self):
        self.paused_event.clear()

    # 事件设置，线程恢复
    def resume(self):
        self.paused_event.set()

    # 事件设置，线程停止
    def stop(self):
        self.stop_flag = True  # 设置停止标志
        self.paused_event.set()  # 确保线程退出时可以停止阻塞

## 计算点法向量 类函数（1）-----主要负责开辟新线程计算
class Normal_Calculation_Thread(QThread):
    result_signal = pyqtSignal(str)  # 用于返回计算完成的消息
    data_signal = pyqtSignal(object)  # 用于返回数据
    time_signal = pyqtSignal(int)  # 用于运行时间

    def __init__(self, pointcloud,Number):
        super().__init__()
        self.pointcloud_NN = pointcloud
        self.Number = Number
        self.lock = Lock()

    def run(self):
        with self.lock:
            start_time = time.time()  # 记录开始时间
            self.result_signal.emit("Begin to calculate")
            pcd_copy = o3d.geometry.PointCloud()
            pcd_copy.points = o3d.utility.Vector3dVector(np.asarray(self.pointcloud_NN.points).copy())

            # 如果原点云有颜色，也一并复制
            if self.pointcloud_NN.has_colors():
                pcd_copy.colors = o3d.utility.Vector3dVector(np.asarray(self.pointcloud_NN.colors).copy())
            pcd_copy.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.Number))
            self.data_signal.emit(pcd_copy)
            self.result_signal.emit("Normal calculation complete")
            end_time = time.time()  # 记录结束时间
            total_time = end_time - start_time
            self.time_signal.emit(int(total_time))

## 模块二预测类函数（1）-----主要负责开辟新线程进行预测
class PredictionThread(QThread):
    # 定义两个信号
    progress_signal = pyqtSignal(str) # 用于传递进度消息
    result_signal = pyqtSignal(np.ndarray) # 用于传递预测结果
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data

    def run(self):
        try:
            tensorflow.get_logger().setLevel('ERROR')
            # 触发 "开始预测" 信号

            self.progress_signal.emit("---Start making predictions---")
            # ✅ 确保 NumPy 数据是独立副本（防止引用跨线程崩溃）
            data_copy = np.array(self.data, copy=True)
            # ✅ 禁用 GPU 或设置单线程（可选，如果你用的是 CPU）
            tensorflow.config.set_visible_devices([], 'GPU')
            with tensorflow.device('/CPU:0'):
                predictions = self.model.predict(data_copy, verbose=0)
            # 处理并返回预测结果（这里只是示例，通常需要根据具体情况格式化结果）
            self.result_signal.emit(predictions)  # 将预测结果传递给主线程
            # 触发 "预测结束" 信号
            self.progress_signal.emit("---Prediction completed---")
        except Exception as e:
            self.progress_signal.emit(f"Prediction error: {e}")

## 模块三DBSCAN线程聚类
class DBSCAN_Cluster_Thread(QThread):
    result_signal = pyqtSignal(str)  # 用于返回计算完成的消息
    progress_signal = pyqtSignal(int)  # 用于传递进度信息
    data_signal = pyqtSignal(np.ndarray)  # 用于返回计算完成的消息
    paused_signal = pyqtSignal(bool)  # 暂停信号

    def __init__(self, U_Location,Min_radio,Min_Number):
        super().__init__()
        self.pointcloud = U_Location
        self.radius = Min_radio
        self.Number = Min_Number
        self.lock = Lock()
        self.paused_event = Event()  # 创建暂停事件
        self.paused_event.set()  # 默认是启动状态
        self.stop_flag = False  # 停止标志

    def run(self):
       with self.lock:
            try:
                n, d = self.pointcloud.shape
                minpts = d + 1  # 我们取最小点的数量为d+1
                T = np.zeros((n, 1))
                maxcluster = 1  # 第一个cluster为1（第1类cluster）
                update_interval = 50000  # 定义更新间隔，调整进度条更新频率
                Tree = BallTree(self.pointcloud)
                indices = []
                for j in range(len(self.pointcloud)):
                    # 暂停控制模块
                    if self.stop_flag:
                        self.result_signal.emit("Cluster exit")
                        return  # 如果停止标志为真，则终止线程
                    self.paused_event.wait()  # 如果是暂停状态，则阻塞在这里
                    Query_Point = self.pointcloud[j].reshape(1, -1)
                    index = Tree.query_radius(Query_Point, r=self.radius)
                    indices.append(index[0])
                    # 更新进度条
                    if j % update_interval == 0:
                        progress = int(((j + 1) / len(self.pointcloud)) * 100)
                        self.progress_signal.emit(progress)  # 发出进度信号
                    elif j == len(self.pointcloud) - 1:
                        self.progress_signal.emit(100)  # 发出进度信号
                        self.result_signal.emit("Parameters have been successfully established")  # 发出完成信号
                        self.result_signal.emit("Start clustering")  # 发出完成信号

                update_interval = 100000
                self.progress_signal.emit(0)
                ## 对索引值进行聚类
                for i in range(n):
                    if self.stop_flag:
                        self.result_signal.emit("Cluster exit")
                        return  # 如果停止标志为真，则终止线程
                    self.paused_event.wait()  # 如果是暂停状态，则阻塞在这里

                    NeighborPts = np.array(indices[i])
                    if len(NeighborPts) > minpts:
                        cv = T[NeighborPts]
                        mincv = np.min(cv)
                        _, Acolumn = (cv[cv > 0].reshape(1, -1)).shape
                        if Acolumn == 0:
                            mincv2 = 0
                        else:
                            mincv2 = np.min(cv[cv > 0])
                        maxcv = np.max(cv)
                        if maxcv == 0:
                            caso = 0
                        else:
                            if maxcv == mincv2:
                                caso = 1
                            else:
                                caso = 2

                        if caso == 0:
                            T[NeighborPts] = maxcluster
                            maxcluster += 1
                        elif caso == 1:
                            if mincv == 0:
                                _, Bcolumn = (NeighborPts[np.where(cv == 0)[0]].reshape(-1, 1)).shape
                                if Bcolumn == 0:
                                    pass
                                else:
                                    T[NeighborPts[np.where(cv == 0)[0]]][0] = mincv2
                        elif caso == 2:
                            _, Ccolumn = (NeighborPts[np.where(cv == 0)[0]].reshape(-1, 1)).shape
                            if Ccolumn == 0:
                                pass
                            else:
                                T[NeighborPts[np.where(cv == 0)[0]]] = mincv2
                            b = cv[cv > mincv2].reshape(-1, 1)
                            n1, _ = b.shape
                            aux = 0
                            for i in range(n1):
                                if b[i] != aux:
                                    T[T == b[i]] = mincv2

                    if i% update_interval == 0:
                        progress = int(((i + 1) / n) * 100)
                        self.progress_signal.emit(progress)  # 发出进度信号


                ## 去除点数小于最小数量的聚类
                if np.sum(T) == 0:
                    pass
                else:
                    T2 = T.copy()
                    DBcluster = np.unique(T2)
                    DBcluster = DBcluster[DBcluster > 0].reshape(-1, 1)
                    NumCluster, _ = DBcluster.shape
                    Tempo_A = np.zeros((2, NumCluster))
                    A = np.zeros((2, NumCluster))
                    Numeroclusters = np.zeros((1, NumCluster))
                    for ik in range(NumCluster):
                        Numeroclusters[0, ik] = len(np.where(T2 == DBcluster[ik])[0])
                    Tempo_A[1, :] = DBcluster[:, 0]
                    Tempo_A[0, :] = Numeroclusters[0, :]
                    IX = np.argsort(Tempo_A[0, :].flatten())[::-1]
                    for ids in range(len(IX)):
                        A[:, ids] = Tempo_A[:, IX[ids]]
                    Ithan = np.where(A[0, :] > self.Number)[0]
                    Jless = np.where(A[0, :] <= self.Number)[0]
                    for ee in range(len(Jless)):
                        T[np.where(T2 == A[1, Jless[ee]])[0]] = 0
                    for ff in range(len(Ithan)):
                        T[np.where(T2 == A[1, Ithan[ff]])[0]] = ff + 1
                # 计算完成后，将结果传递到主线程
                self.progress_signal.emit(100)  # 发出进度信号
                self.data_signal.emit(T)  # 返回曲率数据和PcData
                self.result_signal.emit("Clustering completed")  # 发出完成信号
            except Exception as e:
                self.result_signal.emit(f"Cluster failed: {str(e)}")

    # 清除事件，线程进入等待状态
    def pause(self):
        self.paused_event.clear()

    # 事件设置，线程恢复
    def resume(self):
        self.paused_event.set()

    # 事件设置，线程停止
    def stop(self):
        self.stop_flag = True  # 设置停止标志
        self.paused_event.set()  # 确保线程退出时可以停止阻塞

## OPen3D_Ui panel
class PointCloudViewer:
    def __init__(self, pcd, ui=None):
        self.pcd = pcd
        self.scale = 1.0
        self.ui = ui

        # ✅ 修复资源路径（必须在 initialize() 前调用）
        self._set_open3d_resource_path()

        # 初始化 Open3D GUI 应用
        self.app = gui.Application.instance
        self.app.initialize()

    def _set_open3d_resource_path(self):
        """确保 Open3D 能找到资源目录（适用于 v0.15.1）"""
        try:
            if hasattr(sys, "_MEIPASS"):  # ✅ PyInstaller 打包路径
                resource_path = os.path.join(sys._MEIPASS, "open3d", "resources")
            else:  # ✅ 普通开发环境
                resource_path = os.path.join(os.path.dirname(o3d.__file__), "resources")

            if os.path.exists(resource_path):
                # ✅ 调用 EngineInstance（旧版 Open3D 必需）
                rendering.EngineInstance.set_resource_path(resource_path)
                print(f"✅ [Open3D] Resource path set: {resource_path}")
            else:
                print(f"⚠️ [Open3D] Resource directory not found: {resource_path}")
        except Exception as e:
            print(f"⚠️ [Open3D] Failed to set resource path: {e}")

    def show(self):
        """显示点云窗口"""
        # 创建窗口
        self.window = self.app.create_window("Open3D Point Cloud Visualization", 1024, 768)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)

        # 背景白色
        self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.scene_widget.scene.view.set_post_processing(False)

        # 判断点云是否为空
        if self.pcd.is_empty():
            if self.ui:
                self.ui.textEdit_2.append('<span style="color: red;"> * Error: The point cloud data set is empty </span>')
            else:
                print("❌ Error: The point cloud data set is empty.")
            return

        # 设置材质并加载点云
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        self.scene_widget.scene.add_geometry("pcd", self.pcd, mat)

        # 相机视角
        bounds = self.pcd.get_axis_aligned_bounding_box()
        center = bounds.get_center()
        self.center = center
        self.radius = bounds.get_extent().max()
        self.scene_widget.setup_camera(60, bounds, center)

        # 键盘事件兼容不同版本
        if hasattr(self.window, "set_on_key"):
            self.window.set_on_key(self.on_key_press)
        else:
            self.scene_widget.set_on_key(self.on_key_press)

        self.window.add_child(self.scene_widget)

        # 尝试加载控制面板（如果存在）
        try:
            self.control_dialog = CameraControlDialog(self, uiqt=self.ui)
            self.control_dialog.show()
        except Exception as e:
            print(f"⚠️ 控制对话框未加载: {e}")

        self.app.run()

    def set_camera_view(self, azimuth_deg, elevation_deg):
        """设置相机角度"""
        az_rad = math.radians(azimuth_deg)
        el_rad = math.radians(elevation_deg)
        r = self.radius

        x = r * math.cos(el_rad) * math.sin(az_rad)
        y = r * math.sin(el_rad)
        z = r * math.cos(el_rad) * math.cos(az_rad)

        eye = self.center + o3d.utility.Vector3dVector([[x, y, z]])[0]
        up = [0, 1, 0] if abs(elevation_deg) < 89 else [0, 0, 1]

        self.scene_widget.scene.camera.look_at(self.center, eye, up)
        self.scene_widget.force_redraw()

    def on_key_press(self, event):
        """键盘事件"""
        if event.key == gui.KeyName.L:
            self.save_screenshot()
            return 1  # HANDLED
        return 0  # IGNORED

    def save_screenshot(self):
        """保存当前窗口截图"""
        filename = getattr(self, 'screenshot_path', None)
        if not filename:
            if self.ui:
                self.ui.textEdit_2.append('<span style="color: red;"> * Error: No save path specified </span>')
            return

        # 白底截图
        self.scene_widget.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.scene_widget.scene.view.set_post_processing(False)

        width = int(self.window.content_rect.width * self.scale)
        height = int(self.window.content_rect.height * self.scale)

        self.scene_widget.force_redraw()
        img = self.app.render_to_image(self.scene_widget.scene, width, height)

        if img is not None:
            o3d.io.write_image(filename, img)
            msg = f"* The image has been saved to {filename} ({width}×{height})"
            if self.ui:
                self.ui.textEdit_2.append(f'<span style="color: black;"> {msg} </span>')
            print(f"✅ {msg}")
        else:
            if self.ui:
                self.ui.textEdit_2.append('<span style="color: red;"> * Error: Screenshot failed </span>')
            print("❌ Error: Screenshot failed.")

## Open3d_View_Panel
class CameraControlDialog(QDialog):
    def __init__(self, viewer,uiqt = None):
        super().__init__()
        self.viewer = viewer
        self.ui = uiqt
        self.setWindowTitle("Camera & Screenshot Settings")
        self.setFixedSize(480, 420)

        layout = QVBoxLayout()
        layout.setSpacing(18)

        # 标题标签 - 统一英文和字体
        label = QLabel("Camera & Screenshot Settings")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-weight: 600;
                color: #1a237e;
                font-family: Arial, "Segoe UI", Tahoma, sans-serif;
            }
        """)
        layout.addWidget(label)

        # 分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: #90a4ae;")
        layout.addWidget(line)

        # 输入框标签 + 输入框，统一英文和字体
        self.pitch_input = QLineEdit("20")
        self.yaw_input = QLineEdit("45")
        self.scale_input = QLineEdit("2")

        def add_labeled_input(text, widget):
            lbl = QLabel(text)
            lbl.setStyleSheet("""
                font-size: 14px; 
                font-family: Arial, "Segoe UI", Tahoma, sans-serif;
                color: #1a237e; 
                font-weight: 500;
            """)
            layout.addWidget(lbl)
            widget.setFixedHeight(30)
            widget.setStyleSheet("""
                QLineEdit {
                    border: 1px solid #90a4ae;
                    border-radius: 5px;
                    padding-left: 8px;
                    font-size: 15px;
                    font-family: Arial, "Segoe UI", Tahoma, sans-serif;
                    color: #1a237e;
                }
                QLineEdit:focus {
                    border: 1.5px solid #294a70;
                }
            """)
            layout.addWidget(widget)

        add_labeled_input("Pitch (degrees):", self.pitch_input)
        add_labeled_input("Yaw (degrees):", self.yaw_input)
        add_labeled_input("Save Resolution Scale:", self.scale_input)

        # 应用按钮
        btn_apply = QPushButton("Apply View")
        btn_apply.setFixedHeight(40)
        btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #1e3a5f;
                color: white;
                border-radius: 6px;
                font-size: 17px;
                font-weight: 500;
                font-family: Arial, "Segoe UI", Tahoma, sans-serif;
            }
            QPushButton:hover {
                background-color: #294a70;
            }
            QPushButton:pressed {
                background-color: #162b45;
            }
        """)
        btn_apply.clicked.connect(self.apply_camera_settings)
        layout.addWidget(btn_apply)

        self.setLayout(layout)
        self.scale = 1.0

    def apply_camera_settings(self):
        try:
            pitch = float(self.pitch_input.text())
            yaw = float(self.yaw_input.text())
            scale = float(self.scale_input.text())
        except ValueError:
            self.ui.textEdit_2.append(
                f'<span style="color: red;"> * Error: Invalid input!</span>')
            return

        self.viewer.set_camera_view(pitch, yaw)
        self.viewer.scale = scale
        self.ui.textEdit_2.append(f"* Applied view Pitch = {pitch}, Yaw = {yaw}, Scale = {scale}")

## Cluster panel
class DBSCANWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Operation")
        self.setFixedSize(600, 600)
        self.parameters = {}
        self.init_ui()

    def init_ui(self):
        # 设置背景颜色为浅灰
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#f0f2f5"))
        self.setPalette(palette)

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(25)

        # 标题
        title = QLabel("DBSCAN")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        layout.addWidget(title)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # 第一行：整数输入
        self.label1 = QLabel("Rock joint set :")
        self.label1.setFont(QFont("Arial", 12))
        self.input1 = QLineEdit()
        self.input1.setValidator(QIntValidator())
        self.input1.setFont(QFont("Arial", 12))
        self.input1.setText("1")  # 默认值

        # 第二行：浮点数输入
        self.label2 = QLabel("Min radius")
        self.label2.setFont(QFont("Arial", 12))
        self.input2 = QLineEdit()
        double_validator2 = QDoubleValidator(0.0, 999999.0, 6)  # 最多6位小数
        double_validator2.setNotation(QDoubleValidator.StandardNotation)
        self.input2.setValidator(double_validator2)
        self.input2.setFont(QFont("Arial", 12))
        self.input2.setText("0.0001")  # 默认值

        # 第三行：整数输入
        self.label3 = QLabel("Min cluster number")
        self.label3.setFont(QFont("Arial", 12))
        self.input3 = QLineEdit()
        self.input3.setValidator(QIntValidator())
        self.input3.setFont(QFont("Arial", 12))
        self.input3.setText("1")  # 默认值

        # 添加输入控件到布局
        layout.addWidget(self.label1)
        layout.addWidget(self.input1)

        layout.addWidget(self.label2)
        layout.addWidget(self.input2)

        layout.addWidget(self.label3)
        layout.addWidget(self.input3)

        # 确认按钮
        self.ok_button = QPushButton("Confirm")
        self.ok_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.ok_button.setMinimumHeight(50)
        self.ok_button.setMinimumWidth(200)
        self.ok_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        self.ok_button.clicked.connect(self.perform_action)

        layout.addStretch(1)
        layout.addWidget(self.ok_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def perform_action(self):
        int_text = self.input1.text()
        float_text1 = self.input2.text()
        float_text2 = self.input3.text()

        if not int_text or not float_text1 or not float_text2:
            QMessageBox.warning(self, "Warning", "Please fill in all the fields before proceeding.")
            return

        try:
            self.int_text = int_text
            self.text1 = float_text1
            self.text2 = float_text2
            self.accept()
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid input. Please ensure values are in correct format.")

## Parameter_choose
class ParameterSelectionWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Operation")
        self.setFixedSize(600, 360)  # 调整宽高，更大气
        self.parameter = None
        self.init_ui()

    def init_ui(self):
        # 设置背景颜色为浅灰
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#f0f2f5"))
        self.setPalette(palette)

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(25)

        # 标题：支持自动换行
        title = QLabel("Training Parameter Selection")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)  # 允许自动换行
        layout.addWidget(title)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # 参数选项
        self.option1 = QRadioButton("Point Normal")
        self.option2 = QRadioButton("Point Curvature")
        self.option3 = QRadioButton("Point Normal + Point Curvature")

        for option in [self.option1, self.option2, self.option3]:
            option.setFont(QFont("Arial", 12))
            layout.addWidget(option)

        # 确认按钮
        self.ok_button = QPushButton("Confirm")
        self.ok_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.ok_button.setMinimumHeight(50)
        self.ok_button.setMinimumWidth(200)
        self.ok_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
        """)
        self.ok_button.clicked.connect(self.perform_action)

        layout.addStretch(1)
        layout.addWidget(self.ok_button, alignment=Qt.AlignCenter)
        self.setLayout(layout)

    def perform_action(self):
        if self.option1.isChecked():
            self.parameter = 1
        elif self.option2.isChecked():
            self.parameter = 2
        elif self.option3.isChecked():
            self.parameter = 3
        else:
            QMessageBox.warning(self, "Warning", "Please select a parameter type before proceeding.")
            return
        self.accept()

## Control_process
class PauseDialog(QDialog):
    continue_signal = pyqtSignal()
    pause_signal = pyqtSignal()
    stop_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computing Control Panel")
        self.setFixedSize(360, 300)

        layout = QVBoxLayout()
        layout.setSpacing(18)

        # 标题标签
        self.label = QLabel("Processing...")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                font-size: 22px;
                font-weight: 600;
                color: #1a237e;
                font-family: "Arial";
            }
        """)
        layout.addWidget(self.label)

        # 添加一条横线分隔标题和按钮
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: #90a4ae;")
        layout.addWidget(line)

        # 控制按钮
        self.continue_button = QPushButton("▶ Continue")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.on_continue)

        self.pause_button = QPushButton("⏸ Pause")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.on_pause)

        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.clicked.connect(self.on_stop)

        # 设置按钮统一样式（工业蓝灰风）
        button_style = """
            QPushButton {
                background-color: #1e3a5f;
                color: white;
                border-radius: 6px;
                padding: 12px;
                font-size: 17px;
                font-weight: 500;
                font-family: "Arial";
            }
            QPushButton:hover {
                background-color: #294a70;
            }
            QPushButton:disabled {
                background-color: #a0aab4;
                color: #e0e0e0;
            }
        """
        self.continue_button.setStyleSheet(button_style)
        self.pause_button.setStyleSheet(button_style)
        self.stop_button.setStyleSheet(button_style)

        layout.addWidget(self.continue_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.stop_button)

        self.setLayout(layout)

        # 整体窗口样式
        self.setStyleSheet("""
            QDialog {
                background-color: #f2f4f7;
                border: 1px solid #b0bec5;
                border-radius: 10px;
            }
        """)

        self.is_calculating = False

    def on_continue(self):
        if self.is_calculating:
            self.continue_signal.emit()
            self.continue_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)

    def on_pause(self):
        if self.is_calculating:
            self.pause_signal.emit()
            self.continue_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(True)

    def on_stop(self):
        self.stop_signal.emit()
        self.close()

    def start_calculation(self):
        self.is_calculating = True
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)

## Login window
class Operate_Login(QMainWindow, ui_Login_2.Ui_MainWindow):
    def __init__(self):
        super().__init__()  # 初始化 QMainWindow
        self.setupUi(self)  # 调用 Ui_MainWindow 中的 setupUi 方法来设置 UI
        ZhangH =self.lineEdit_5.text()  # 获取账号输入框的文本
        Mima = self.lineEdit_2.text()  # 获取密码输入框的文本
        # 定制窗口外观
        self.setWindowFlags(Qt.FramelessWindowHint)  # 移除窗口边框
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明

        # 用于记下鼠标位置
        self.is_dragging = False
        self.drag_position = None

        # 使输入的密码被掩盖
        self.lineEdit_2.setEchoMode(QLineEdit.Password)

    ## 使窗口可以任意移动

    # 当鼠标按下时，记录鼠标当前的位置
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.drag_position = event.globalPos() - self.pos()  # 计算鼠标按下的相对位置
            event.accept()

    #当鼠标移动时，移动窗口
    def mouseMoveEvent(self, event):
        if self.is_dragging:
            # 计算新的窗口位置并设置
            new_pos = event.globalPos() - self.drag_position
            self.move(new_pos)
            event.accept()

    #当鼠标释放时，停止拖动。
    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        event.accept()

## Main function
class Operate_Dealing(QMainWindow, ui_Main_Window.Ui_MainWindow):

    def __init__(self):
        super().__init__()  # 初始化 QMainWindow
        self.setupUi(self)  # 调用 Ui_MainWindow 中的 setupUi 方法来设置 UI
        self.setWindowTitle('RMSDP')
        self.stackedWidget.setCurrentIndex(0) # 定义初始的功能页面
        self.form_layout = self.formLayout # 定义全局变量方便使显示窗口的loss等曲线图出现叠加的情况

       # 定义项目栏
        self.listWidget.setViewMode(QListWidget.IconMode)
        self.listWidget.setViewMode(QListWidget.ListMode)  # 关键设置
        self.listWidget.setFlow(QListWidget.TopToBottom)  # 从上到下排列
        self.listWidget.setWrapping(False)  # 禁止自动换行
        self.listWidget.setGridSize(QSize(120, 50))
        self.listWidget.setIconSize(QSize(64, 64))

        # 定义交互栏
        self.lineEdit.setPlaceholderText("Press the Enter key after entering the command")

        ## 按键的连接
        self.pushButton.clicked.connect(self.display_page1)
        self.pushButton_2.clicked.connect(self.display_page2)
        self.pushButton_3.clicked.connect(self.display_page3)
        self.pushButton_4.clicked.connect(self.display_page4)
        self.pushButton_5.clicked.connect(self.display_page5)
        self.pushButton_6.clicked.connect(self.display_page6)
        self.pushButton_23.clicked.connect(self.load_cloud_model) # 连接参数页面中使用vtk显示点云的函数
        self.pushButton_14.clicked.connect(self.display_3d_model)
        self.pushButton_17.clicked.connect(self.navigate_up) #返回上级目录
        self.pushButton_24.clicked.connect(self.Calculate_point_Normals) # 计算点法向量
        self.pushButton_20.clicked.connect(self.Calculate_point_Length)
        self.pushButton_25.clicked.connect(self.Calculate_Point_Curvature)
        self.pushButton_7.clicked.connect(self.Input_data)
        self.pushButton_8.clicked.connect(self.Choose_Train_Points)
        self.pushButton_9.clicked.connect(self.Build_Artificial_Neural_Network)
        self.pushButton_10.clicked.connect(self.Neural_Network_Using)
        self.pushButton_11.clicked.connect(self.load_and_display_point_cloud_model3)
        self.pushButton_12.clicked.connect(self.extraction_individual_joint)
        self.pushButton_13.clicked.connect(self.integrate_Model3)
        self.pushButton_15.clicked.connect(self.Input_Model4)
        self.pushButton_16.clicked.connect(self.Calculate_Model4)
        self.pushButton_22.clicked.connect(self.save_point_data)
        self.pushButton_19.clicked.connect(self.Showing_point_coordinate)
        self.pushButton_21.clicked.connect(self.Calculte_selective_area)
        self.pushButton_26.clicked.connect(self.Calculate_fit_plane_Ora)
        self.pushButton_42.clicked.connect(self.save_vtk_view_as_image)
        self.pushButton_43.clicked.connect(self.save_display_3d_model)
        self.pushButton_44.clicked.connect(self.save_matlab_image)
        # 功能栏按钮
        self.actionNew.triggered.connect(self.create_new_folder)
        self.actionOpen.triggered.connect(self.open_operate_file)
        self.actionVTK.triggered.connect(self.save_vtk_view_as_image)
        self.actionOpen3D.triggered.connect(self.save_display_3d_model)
        self.actionMatplob.triggered.connect(self.save_matlab_image)
        self.actionCoordinate.triggered.connect(self.Showing_point_coordinate)
        self.actionLength.triggered.connect(self.Calculate_point_Length)
        self.actionArea.triggered.connect(self.Calculte_selective_area)
        self.actionManualOR.triggered.connect(self.Calculate_fit_plane_Ora)
        self.actionExit.triggered.connect(self.close)
        # 文件栏按钮
        self.listWidget.itemDoubleClicked.connect(self.open_file_interface)
        self.lineEdit.returnPressed.connect(self.Parmeter_receive_Function)


        ## 定义默认的蓝色按钮的颜色
        self.default_style_Blue = '''
                QPushButton {
                background-color: rgb(43, 87, 154);
                color: #ffffff;
                border: None;
                padding: 8px 15px;
                
            }
            QPushButton:hover {
                background-color: #274f8b;
            }
            QPushButton:pressed {
                background-color: rgb(235, 235, 235);
                color: rgb(45, 92, 163);
            }
            '''
        self.buttons_Blue = [self.findChild(QPushButton, 'pushButton'),
                             self.findChild(QPushButton, 'pushButton_2'),
                             self.findChild(QPushButton, 'pushButton_3'),
                             self.findChild(QPushButton, 'pushButton_4'),
                             self.findChild(QPushButton, 'pushButton_5'),
                             self.findChild(QPushButton, 'pushButton_6') ]  # 所有按钮列表
        for button_1 in self.buttons_Blue:
            button_1.clicked.connect(self.on_button_clicked_Blue)
        ## 定义默认的灰色按钮的颜色
        self.default_style_Gray = '''
                QPushButton {
                background-color: rgb(235, 235, 235);
                color: rgb(0, 0, 0);
                border: None;
                padding: 8px 15px;
                font: 12pt "Noto Sans";
                }
                QPushButton:hover {
                background-color: rgb(190, 190, 190);
                }
                QPushButton:pressed {
                background-color: rgb(190, 190, 190);
                color: rgb(240, 240, 240);
                }
            '''
        self.buttons_Gray = [self.findChild(QPushButton, 'pushButton_23'),
                        self.findChild(QPushButton, 'pushButton_24'),
                        self.findChild(QPushButton, 'pushButton_25'),
                        self.findChild(QPushButton, 'pushButton_7'),
                        self.findChild(QPushButton, 'pushButton_8'),
                        self.findChild(QPushButton, 'pushButton_9'),
                        self.findChild(QPushButton, 'pushButton_10'),
                        self.findChild(QPushButton, 'pushButton_11'),
                        self.findChild(QPushButton, 'pushButton_12'),
                        self.findChild(QPushButton, 'pushButton_13'),
                        self.findChild(QPushButton, 'pushButton_15'),
                        self.findChild(QPushButton, 'pushButton_16'),
                        self.findChild(QPushButton, 'pushButton_19'),
                        self.findChild(QPushButton, 'pushButton_20'),
                        self.findChild(QPushButton, 'pushButton_21'),
                        self.findChild(QPushButton, 'pushButton_42'),
                        self.findChild(QPushButton, 'pushButton_43'),
                        self.findChild(QPushButton, 'pushButton_44')
                             ]  # 所有按钮列表
        for button_2 in self.buttons_Gray:
            button_2.clicked.connect(self.on_button_clicked_Gray)

        ## 创建 QSpinBox 控制VTK颜色通道
        self.spinBox_R = self.findChild(QSpinBox, 'spinBox')
        self.spinBox_G = self.findChild(QSpinBox, 'spinBox_2')
        self.spinBox_B = self.findChild(QSpinBox, 'spinBox_3')

        ## 连接信号与槽函数，QSpinBox 值变化时调用更新背景颜色的槽
        self.spinBox_R.valueChanged.connect(self.update_background_color)
        self.spinBox_G.valueChanged.connect(self.update_background_color)
        self.spinBox_B.valueChanged.connect(self.update_background_color)

        ## 设置DoubleSinBox的初始值、范围和精度
        self.doubleSpinBox.setRange(-180,360)
        self.doubleSpinBox_2.setRange(-90,90)
        self.doubleSpinBox.setSingleStep(1)
        self.doubleSpinBox_2.setSingleStep(1)

        ## 连接信号与槽函数，DoubleQSpinBox 值变化时调用更新照相机的槽
        self.doubleSpinBox.valueChanged.connect(self.update_camera_Azimuth)
        self.doubleSpinBox_2.valueChanged.connect(self.update_camera_Elevation)

    ## 当 DoubleQSpinBox 值变化时，实时更新照相机的位置
    def update_camera_Azimuth(self):
        azimuth_TLY = self.doubleSpinBox.value()
        self.camera_vtk.Azimuth(azimuth_TLY-self.default_azimuth)
        self.default_azimuth = azimuth_TLY
        # 更新渲染
        self.ren.GetRenderWindow().Render()

    def update_camera_Elevation(self):
        elevation_TLY = self.doubleSpinBox_2.value()
        self.camera_vtk.Elevation(elevation_TLY-self.default_elevation)
        self.default_elevation =  elevation_TLY
        # 更新渲染
        self.ren.GetRenderWindow().Render()

    ## 当 QSpinBox 值变化时，实时更新 VTK 渲染窗口的背景颜色
    def update_background_color(self):
        r = self.spinBox_R.value() / 255.0  # 获取红色通道值并标准化到 [0, 1]
        g = self.spinBox_G.value() / 255.0  # 获取绿色通道值并标准化到 [0, 1]
        b = self.spinBox_B.value() / 255.0  # 获取蓝色通道值并标准化到 [0, 1]
        # 更新 VTK 渲染器的背景颜色
        if hasattr(self, 'ren'):
            self.ren.SetBackground(r, g, b)
            self.vtkWidget.GetRenderWindow().Render()  # 更新渲染窗口

    ## 通过CheckBox来控制坐标轴的显示
    def toggle_axes_VTK(self,state):
        if state == Qt.Checked:
            self.ren.AddActor(self.axes)  # 显示坐标轴
        else:
            self.ren.RemoveActor(self.axes)  # 隐藏坐标轴
        self.vtkWidget.GetRenderWindow().Render()  # 更新渲染窗口

    ## 页面切换代码（点击后使页面发生切换）
    def display_page1(self):
        self.stackedWidget.setCurrentIndex(0)
    def display_page2(self):
        self.stackedWidget.setCurrentIndex(1)
    def display_page3(self):
        self.stackedWidget.setCurrentIndex(2)
    def display_page4(self):
        self.stackedWidget.setCurrentIndex(3)
    def display_page5(self):
        self.stackedWidget.setCurrentIndex(4)
    def display_page6(self):
        self.stackedWidget.setCurrentIndex(5)


    ## 切换窗口函数（点击后改变蓝色按键颜色）
    def on_button_clicked_Blue(self):
        # 获取被点击的按钮
        clicked_button = self.sender()
        # 恢复所有按钮的颜色
        self.reset_button_colors_Blue()
        # 修改被点击按钮的颜色为点击后的样式
        clicked_button.setStyleSheet(self.clicked_style_Blue())
    ## 切换窗口函数（点击后使其他蓝色按键的颜色重置）
    def reset_button_colors_Blue(self):
        # 恢复所有按钮为默认颜色
        for button in self.buttons_Blue:
            button.setStyleSheet(self.default_style_Blue)
    ## 切换窗口函数（定义点击后的蓝色按钮的颜色）
    def clicked_style_Blue(self):
        return '''
            background-color: rgb(235, 235, 235);
            color: rgb(45, 92, 163);
            border: None;
            padding: 8px 15px;
        '''

    ## 切换窗口函数（点击后改变灰色按键颜色）
    def on_button_clicked_Gray(self):
        # 获取被点击的按钮
        clicked_button = self.sender()
        # 恢复所有按钮的颜色
        self.reset_button_colors_Gray()
        # 修改被点击按钮的颜色为点击后的样式
        clicked_button.setStyleSheet(self.clicked_style_Gray())
    ## 切换窗口函数（点击后使其他灰色按键的颜色重置）
    def reset_button_colors_Gray(self):
        # 恢复所有按钮为默认颜色
        for button in self.buttons_Gray:
            button.setStyleSheet(self.default_style_Gray)
    ## 定切换窗口函数（义点击后的灰色按钮的颜色）
    def clicked_style_Gray(self):
        return '''
            background-color: rgb(190, 190, 190);
            color: rgb(240, 240, 240);
            border: None;
            padding: 8px 15px;
        '''

    ## 根据密码的正确性进行窗口切换 (需要详细密码见Main函数)
    def open(self):
        self.show()

    ## 以函数的形式(vtk)显示点云图形
    def Using_VTK_ShowPC(self,PointCloud):
        try:
            # 设置进度条初始值
            self.progressBar_2.setValue(0)
            self.progressBar_2.setMaximum(100)
            if hasattr(self, 'vtkWidget') and isinstance(self.vtkWidget, QObject):
                try:
                    self.iren.Disable()
                    self.vtkWidget.GetRenderWindow().Finalize()
                    self.vtkWidget.close()
                    self.vtkWidget.deleteLater()
                except RuntimeError:
                    pass
            # 清除之前的图形
            while self.form_layout.count():
                item = self.form_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            # 调用文本框
            log_text_edit = self.textEdit_2

            # 创建渲染窗口
            self.frame = QFrame()  # 创建一个QFrame小部件，用来容纳VTK组件
            self.vtkWidget = QVTKRenderWindowInteractor(
                self.frame)  # 创建VTK渲染窗口交互器组件（用于在Qt窗口中QVTKRenderWindowInteractor显示VTK内容）
            self.formLayout.addWidget(self.vtkWidget)
            self.ren = vtk.vtkRenderer()  # 创建一个VTK渲染器对象（用于渲染3D场景）
            self.ren.SetBackground(0, 0, 0)  # 设置背景颜色为黑色
            self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)  # 将渲染器添加到VTK的渲染窗口
            self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

            # 获取点云中的点坐标
            points = np.asarray(PointCloud.points)  # 获取点云的坐标（numpy 数组）
            colors = np.asarray(PointCloud.colors)  # 获取点云的颜色（numpy 数组，范围是 [0, 1]）
            vtk_points = vtk.vtkPoints()  # 创建一个vtkPoints对象，用来存储VTK需要的点坐标
            vertices = vtk.vtkCellArray()  # 创建一个vtkCellArray对象，包含每个点的连接

            num_points = len(points)
            update_interval = 1500000  # 定义更新间隔，这里表示每50000次循环更新一次进度条，可根据实际情况调整
            for i, point in enumerate(points):
                id = vtk_points.InsertNextPoint(point[0], point[1], point[2])  # 插入点坐标
                vertices.InsertNextCell(1)  # 插入一个点的连接
                vertices.InsertCellPoint(id)  # 连接该点
                if i % update_interval == 0:
                    progress = int(((i + 1) / num_points) * 100)  # 计算当前进度百分比
                    self.progressBar_2.setValue(progress)
                    QApplication.processEvents()  # 允许UI刷新进度条
                elif i == num_points - 1:
                    progress = int(100)  # 计算当前进度百分比
                    self.progressBar_2.setValue(progress)
                    QApplication.processEvents()  # 允许UI刷新进度条

            # 创建一个vtkPolyData对象，用来存储点数据
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(vtk_points)  # 设置点云数据
            polydata.SetVerts(vertices)  # 设置点的连接

            # 创建一个vtkPolyDataMapper对象，用来映射点数据到渲染管线
            dataMapper = vtk.vtkPolyDataMapper()
            dataMapper.SetInputData(polydata)

            # 创建一个vtkActor对象，用来将点云数据添加到渲染器中
            actor = vtk.vtkActor()
            actor.SetMapper(dataMapper)  # 将数据映射器应用到演员对象上

            # 创建一个vtkPointData对象，用来设置点的颜色
            pointData = polydata.GetPointData()

            # 设置颜色：Open3D中的颜色是 [0, 1] 范围，而VTK的颜色是 [0, 255]
            vtk_colors = vtk.vtkUnsignedCharArray()
            vtk_colors.SetName("Colors")  # 设置颜色数组的名称
            vtk_colors.SetNumberOfComponents(3)  # 颜色数组是RGB模式，所以每个点有3个通道

            # 为每个点设置颜色
            for color in colors:
                vtk_colors.InsertNextTuple3(int(color[0] * 255), int(color[1] * 255),
                                            int(color[2] * 255))  # 将 [0,1] 范围的颜色转换为 [0,255]
            pointData.SetScalars(vtk_colors)  # 将颜色信息添加到点数据中

            # 将颜色信息应用到数据映射器
            dataMapper.SetColorModeToDefault()  # 使用点数据中的颜色

            # 将actor添加到渲染器中
            self.ren.AddActor(actor)

            # 获取相机对象，设置视角
            self.camera_vtk = self.ren.GetActiveCamera()

            # 设置相机的方位角和俯仰角
            self.default_azimuth = 0  # 默认方位角
            self.default_elevation = 0  # 默认俯仰角
            self.camera_vtk.Azimuth(self.default_azimuth)  # 方位角：绕Z轴旋转 度
            self.camera_vtk.Elevation(self.default_elevation)  # 俯仰角：绕X轴旋转 260 度

            # 创建坐标轴但是不显示
            self.axes = vtk.vtkAxesActor()

            # 设置坐标轴标签的字体大小
            axis_label_prop = self.axes.GetXAxisCaptionActor2D().GetTextActor().GetTextProperty()
            axis_label_prop.SetFontSize(1)  # 设置X轴标签字体大小为20
            axis_label_prop = self.axes.GetYAxisCaptionActor2D().GetTextActor().GetTextProperty()
            axis_label_prop.SetFontSize(1)  # 设置Y轴标签字体大小为20
            axis_label_prop = self.axes.GetZAxisCaptionActor2D().GetTextActor().GetTextProperty()
            axis_label_prop.SetFontSize(1)  # 设置Z轴标签字体大小为20
            self.checkBox.stateChanged.connect(self.toggle_axes_VTK)

            # 重置相机
            self.ren.ResetCamera()

            # 显示窗口
            self.show()

            # 初始化渲染窗口的交互器
            self.iren.Initialize()

            # 在文本框内显示完成信息
            log_text_edit.append(
                f'<span style="color: black;"> * The data is displayed successfully in VTK </span>')
        except Exception as e:
            log_text_edit.append(
                f'<span style="color: red;"> * Error: The data is not displayed successfully in VTK\n {e} </span>')

        self.pointcloud = PointCloud

    ## 用Open3D来显示点云
    def display_3d_model(self):
        try:
            if self.pointcloud:
                self.textEdit_2.append(
                    f'<span style="color: black;"> * The data is displayed successfully in open3D </span>')
                vis = o3d.visualization.Visualizer()
                vis.create_window()  # 通过 Visualizer 创建窗口
                vis.add_geometry(self.pointcloud)  # 将点云添加到可视化器中
                opt = vis.get_render_option()
                opt.point_size = 2.0
                vis.run()
                vis.destroy_window()
        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: The point cloud data set is empty\n {e} </span>')

    ## 建立新的文件夹并且读取路径
    def create_new_folder(self):
        folder_path_gen = QFileDialog.getExistingDirectory(self, "Select folder location") # 打开文件夹选择对话框
        if folder_path_gen:
            folder_name, ok = QInputDialog.getText(self, "Enter folder name", "Enter a new folder name:")  # 获取文件夹名
            if ok and folder_name:
                self.new_folder_path = os.path.join(folder_path_gen, folder_name)
                try:
                    os.makedirs(self.new_folder_path) # 创建文件夹
                    self.current_folder_path = folder_path_gen
                    self.load_files(folder_path_gen)
                    self.textEdit_2.append(
                        f'<span style="color: black;"> * The new file is created successfully </span>')
                except FileExistsError:
                    self.textEdit_2.append(
                        '<span style="color: red;"> * Error: The new file has already existed </span>')
                except Exception as e:
                    self.textEdit_2.append(
                        '<span style="color: red;"> * Error: The new file is not created successfully </span>')
        else:
            self.textEdit_2.append(
                '<span style="color: red;"> * Error: The location of the folder is not selected </span>')

    ## 打开新的文件夹并读取路径
    def open_operate_file(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select folder", "")
        if folder_path:
            self.current_folder_path = folder_path
            self.file_system_watcher = QFileSystemWatcher(self)
            self.file_system_watcher.addPath(self.current_folder_path)  # 监控文件夹
            self.file_system_watcher.directoryChanged.connect(self.load_files)
            self.load_files(folder_path)
        else:
            self.textEdit_2.append(
                '<span style="color: red;"> * Error: No file is selected </span>')

    ## 读取项目文件
    def load_files(self,folder_path):
        self.listWidget.clear()
        folder_icon = self.style().standardIcon(QStyle.SP_DirIcon)
        file_icon = self.style().standardIcon(QStyle.SP_FileIcon)
        for item_name in sorted(os.listdir(folder_path)):
            item_path = os.path.join(folder_path, item_name)
            item = QListWidgetItem(item_name)

            if os.path.isdir(item_path):
                item.setIcon(folder_icon)
                item.setData(Qt.UserRole, "folder")  # 标记类型
                item.setForeground(QColor(0, 0, 0))  # 蓝色文件夹名
            else:
                item.setIcon(file_icon)
                item.setForeground(QColor(0, 0, 0))  # 黑色文件名
            self.listWidget.addItem(item)

    ## 点击文件时打开文件
    def open_file_interface(self,item):
        item_name = item.text()
        item_path = os.path.join(self.current_folder_path,item_name)
        if os.path.exists(item_path):
            try:
                if os.path.isfile(item_path):
                    # 打开文件
                    file_ext = os.path.splitext(item_path)[1].lower() # 获取文件扩展名字
                    if file_ext in ('.pcd', '.ply','.obj'):
                        self.pointcloud = o3d.io.read_point_cloud(item_path)
                        self.Using_VTK_ShowPC(self.pointcloud)
                        self.textEdit_2.append(
                            f'<span style="color: black;"> * File opened successfully: {item_name} </span>')
                    elif file_ext in ('.npy'):
                        self.PcData_TEXT = np.load(item_path)
                    else:
                        os.startfile(item_path)  # Windows平台
                        self.textEdit_2.append(
                            f'<span style="color: black;"> * File opened successfully: {item_name} </span>')
                else:
                    # 是文件夹，加载该文件夹内容
                    self.current_folder_path = item_path
                    self.load_files(item_path)
                    self.textEdit_2.append(
                        f'<span style="color: black;"> * Entered folder: {item_name} </span>')

            except Exception as e:
                self.textEdit_2.append(
                    f'<span style="color: red;"> * Error opening: {str(e)} </span>')

        else: self.textEdit_2.append(
            f'<span style="color: red;"> * Error: {item_name} does not exist </span>')

    ## 返回上级目录
    def navigate_up(self):
        if hasattr(self, 'current_folder_path'):
            parent_dir = os.path.dirname(self.current_folder_path)
            if os.path.exists(parent_dir):
                self.current_folder_path = parent_dir
                self.load_files(parent_dir)
            else:
                self.textEdit_2.append(
                    f'<span style="color: red;"> * Warning: It is the root directory </span>')
        else:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Warning: No directory is currently open </span>')

    ## 交互窗口数值接受函数
    def Parmeter_receive_Function(self):
        self.current_text = self.lineEdit.text()
        if self.current_text:
            self.textEdit_2.append(
                f'<span style="color: black;"> * {self.current_text} </span>')
            self.lineEdit.clear()

        if self.current_text == 'clear':
            self.textEdit_2.clear()

    ## 重写Ui界面关闭函数

    def closeEvent(self, event):
        try:
            if hasattr(self, 'iren') and self.iren is not None:
                self.iren.Disable()
                self.iren.TerminateApp()
                del self.iren
                self.iren = None
            if hasattr(self, 'vtkWidget') and self.vtkWidget is not None:
                rw = self.vtkWidget.GetRenderWindow()
                if rw is not None:
                    rw.ReleaseGraphicsResources(None)
                    rw.Finalize()
                self.vtkWidget.close()
                self.vtkWidget.deleteLater()
                self.vtkWidget = None
            if hasattr(self, 'ren') and self.ren is not None:
                self.ren.RemoveAllViewProps()
                del self.ren
                self.ren = None
        except Exception as e:
            print("Error occurred while releasing VTK resources:", e)
        event.accept()

#########################################************************ 模块一 ************************#########################################

## 用窗口内的VTK来显示点云(按键)
    def load_cloud_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select the point cloud file", "", "Point cloud file (*.pcd *.obj *.ply)")
        if not file_path:
            self.textEdit_2.append(
                '<span style="color: red;"> *  Warning: No point cloud file is selected </span>')
            return
        try:
            if file_path.endswith(".ply") or file_path.endswith(".obj") or file_path.endswith(".pcd"):
                reply_111 = QMessageBox.question(None, "Tips", "Whether to randomly sample the point cloud?",
                                               QMessageBox.Yes | QMessageBox.No)
                if reply_111 == QMessageBox.Yes:
                    self.pointcloud = o3d.io.read_point_cloud(file_path)
                    down_num, ok = QInputDialog.getInt(None, "Please enter the sparse multiple",
                                                          "Range (1-infinity):")

                    self.pointcloud = self.pointcloud.uniform_down_sample(every_k_points=down_num)
                    self.textEdit_2.append(
                        f'<span style="color: black;"> * Tip: The point cloud has been diluted by a factor of {down_num} </span>')
                else:
                    self.pointcloud = o3d.io.read_point_cloud(file_path)
                    self.pointcloud = self.pointcloud.uniform_down_sample(every_k_points=1)
            else:
                self.textEdit_2.append(
                    '<span style="color: red;"> * Warning: Unsupported file format </span>')
                return
        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: Failed to load the point cloud {e} </span>')

        self.Using_VTK_ShowPC(self.pointcloud)

    ## 计算点法向量
    def Calculate_point_Normals(self):
        try:
            if self.pointcloud:
                # 保存神经网络数据
                self.PcData = np.zeros((len(self.pointcloud.points), 7))
                self.PcData[:, 0:3] = self.pointcloud.points
                if self.pointcloud.has_normals() and not hasattr(self,'numnor'):
                    self.PcData[:, 3:6] = self.pointcloud.normals
                    Point_Locations = np.asarray(self.pointcloud.points)  # 导入点坐标信息
                    Point_Normals = np.asarray(self.pointcloud.normals)
                    Data_Normals = o3d.geometry.PointCloud()  # 创建一个空的Open3D 点云对象
                    Data_Normals.points = o3d.utility.Vector3dVector(Point_Locations)  # 设置点云的点坐标

                    normals_vis_color = (Point_Normals + 1.0) / 2.0
                    normals_vis_color = np.clip(normals_vis_color, 0, 1)

                    Data_Normals.colors = o3d.utility.Vector3dVector(normals_vis_color)  # 设置点云的颜色

                    if Data_Normals:
                        self.textEdit_2.append(
                            f'<span style="color: black;"> * The point normal is calculated successfully </span>')
                        QApplication.processEvents()
                    # 标志窗口是否关闭
                    self.o3d_window_closed = False

                    def show_window():
                        vis = o3d.visualization.Visualizer()
                        vis.create_window(window_name="PointCloud Preview")
                        vis.add_geometry(Data_Normals)
                        opt = vis.get_render_option()
                        opt.point_size = 2.0
                        vis.run()
                        vis.destroy_window()
                        # 标记窗口已关闭
                        self.o3d_window_closed = True

                    threading.Thread(target=show_window, daemon=True).start()
                    reply_m = QMessageBox.question(None, "Whether the processing result is reasonable?",
                                                   "Do you accept the results?",
                                                   QMessageBox.Yes | QMessageBox.No)

                    # 使用 QTimer 轮询等待（不会阻塞 GUI）
                    def wait_for_window_close():
                        if not self.o3d_window_closed:
                            QTimer.singleShot(500, wait_for_window_close)  # 0.5s 再检查一次
                            return
                        if reply_m == QMessageBox.Yes:
                            reply_3 = QMessageBox.question(None, "Tip", "Whether point cloud data is saved？",
                                                           QMessageBox.Yes | QMessageBox.No)
                            if reply_3 == QMessageBox.Yes:
                                file_types = "PLY Files (*.ply);;PCD Files (*.pcd);;All Files (*)"  # 限定数据保存类型
                                options = QFileDialog.Options()
                                file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save the point cloud file",
                                                                                         "",                                                                                                file_types,
                                                                                         options=options)
                                if file_path:
                                    try:
                                        folder = os.path.dirname(file_path)
                                        if folder and not os.path.exists(folder):
                                            os.makedirs(folder, exist_ok=True)
                                            self.textEdit_2.append(
                                                f'<span style="color: gray;">Created folder: {folder}</span>')
                                    except Exception as e:
                                        self.textEdit_2.append(
                                            f'<span style="color: red;"> * Failed to create folder: {e}</span>')

                                    # 根据选择的文件过滤器确定保存格式并保存文件
                                    if selected_filter == "PLY Files (*.ply)":
                                        try:
                                            o3d.io.write_point_cloud(file_path, Data_Normals)
                                            self.textEdit_2.append(
                                                f'<span style="color: black;"> * The point cloud set is saved successfully </span>')
                                        except Exception as e:
                                            self.textEdit_2.append(
                                                f'<span style="color: red;"> * Error: Failed to save point cloud file (PLY format): {str(e)} </span>')
                                    elif selected_filter == "PCD Files (*.pcd)":
                                        try:
                                            o3d.io.write_point_cloud(file_path, Data_Normals,
                                                                     write_ascii=True)  # PCD格式保存时可按需设置参数，这里以ASCII格式为例
                                            self.textEdit_2.append(
                                                f'<span style="color: black;"> * The point cloud set is saved successfully </span>')
                                        except Exception as e:
                                            self.textEdit_2.append(
                                                f'<span style="color: red;"> * Error: Failed to save point cloud file (PCD format):{str(e)} </span>')
                                    else:
                                        self.textEdit_2.append(
                                            '<span style="color: red;"> * Warning: Unsupported file format selection, and save operation cancelled </span>')
                                else:
                                    self.textEdit_2.append(
                                        f'<span style="color: black;"> * Tips: Cancel operation </span>')
                            else:
                                self.textEdit_2.append(
                                    f'<span style="color: black;"> * Tips: Canceled </span>')

                        else:
                            reply_n = QMessageBox.question(None, "Tip", "Whether to reprocess (Please input Yes/No)?",
                                                           QMessageBox.Yes | QMessageBox.No)
                            if reply_n == QMessageBox.Yes:
                                self.Calculate_point_Normals()
                            else:
                                self.textEdit_2.append(
                                    f'<span style="color: black;"> * Tips: Canceled </span>')
                    # 启动检查任务
                    QTimer.singleShot(500, wait_for_window_close)
                else:
                    self.numnor, ok = QInputDialog.getInt(None, "Enter the number of query points",
                                                  "Please input positive integer:")
                    self.Normal_thread = Normal_Calculation_Thread(self.pointcloud, self.numnor)
                    self.Normal_thread.result_signal.connect(self.show_result_message_model_N)  # 连接信息显示信号
                    self.Normal_thread.data_signal.connect(self.save_Normal_result_model_N)  # 连接结果显示信号
                    self.Normal_thread.time_signal.connect(self.output_Normal_time_N)  # 连接运行显示信号
                    self.Normal_thread.start()  # 启动线程
        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: The point data set is empty: {e} </span>')

    ## 点法向量计算副函数（1）------ 点法向量的计算情况反馈
    def show_result_message_model_N(self,message):
        self.textEdit_2.append(
            f'<span style="color: black;"> * {message} </span>')

    ## 点法向量计算副函数（2）------ 点法向量的计算结果的判别和保存
    def save_Normal_result_model_N(self,pointNor):
        self.pointcloud = pointNor
        self.PcData[:, 3:6] = self.pointcloud.normals
        Point_Locations = np.asarray(self.pointcloud.points)  # 导入点坐标信息
        Point_Normals = np.asarray(self.pointcloud.normals)

        normals_vis_color = (Point_Normals + 1.0) / 2.0
        normals_vis_color = np.clip(normals_vis_color, 0, 1)

        Data_Normals = o3d.geometry.PointCloud()  # 创建一个空的Open3D 点云对象
        Data_Normals.points = o3d.utility.Vector3dVector(Point_Locations)  # 设置点云的点坐标
        Data_Normals.colors = o3d.utility.Vector3dVector(normals_vis_color)  # 设置点云的颜色
        if Data_Normals:
            self.textEdit_2.append(
                f'<span style="color: black;"> * The point normal is calculated successfully </span>')
            QApplication.processEvents()

        # 标志窗口是否关闭
        self.o3d_window_closed = False

        def show_window():
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="PointCloud Preview")
            vis.add_geometry(Data_Normals)
            opt = vis.get_render_option()
            opt.point_size = 2.0
            vis.run()
            vis.destroy_window()
            # 标记窗口已关闭
            self.o3d_window_closed = True

        threading.Thread(target=show_window, daemon=True).start()

        # 使用 QTimer 轮询等待（不会阻塞 GUI）
        def wait_for_window_close():
            if not self.o3d_window_closed:
                QTimer.singleShot(500, wait_for_window_close)  # 0.5s 再检查一次
                return

            reply_m = QMessageBox.question(None, "Whether the processing result is reasonable?",
                                           "Do you accept the results?",
                                           QMessageBox.Yes | QMessageBox.No)
            if reply_m == QMessageBox.Yes:
                reply_3 = QMessageBox.question(None, "Tip", "Whether point cloud data is saved？",
                                               QMessageBox.Yes | QMessageBox.No)
                if reply_3 == QMessageBox.Yes:
                    file_types = "PLY Files (*.ply);;PCD Files (*.pcd);;All Files (*)"  # 限定数据保存类型
                    options = QFileDialog.Options()
                    file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save the point cloud file",
                                                                             "",
                                                                             file_types,
                                                                             options=options)

                    if file_path:

                        try:
                            folder = os.path.dirname(file_path)
                            if folder and not os.path.exists(folder):
                                os.makedirs(folder, exist_ok=True)
                                self.textEdit_2.append(f'<span style="color: gray;">Created folder: {folder}</span>')
                        except Exception as e:
                            self.textEdit_2.append(f'<span style="color: red;"> * Failed to create folder: {e}</span>')

                        # 根据选择的文件过滤器确定保存格式并保存文件
                        if selected_filter == "PLY Files (*.ply)":
                            try:
                                o3d.io.write_point_cloud(file_path, Data_Normals)
                                self.textEdit_2.append(
                                    f'<span style="color: black;"> * The point cloud set is saved successfully </span>')
                            except Exception as e:
                                self.textEdit_2.append(
                                    f'<span style="color: red;"> * Error: Failed to save point cloud file (PLY format): {str(e)} </span>')
                        elif selected_filter == "PCD Files (*.pcd)":
                            try:
                                o3d.io.write_point_cloud(file_path, Data_Normals,
                                                         write_ascii=True)  # PCD格式保存时可按需设置参数，这里以ASCII格式为例
                                self.textEdit_2.append(
                                    f'<span style="color: black;"> * The point cloud set is saved successfully </span>')
                            except Exception as e:
                                err_detail = traceback.format_exc()
                                self.textEdit_2.append(
                                    f'<span style="color: gray; font-size: 11px;">{err_detail}</span>')


                                self.textEdit_2.append(
                                    f'<span style="color: red;"> * Error: Failed to save point cloud file (PCD format):{str(e)} </span>')
                        else:
                            self.textEdit_2.append(
                                '<span style="color: red;"> * Warning: Unsupported file format selection, and save operation cancelled </span>')
                    else:
                        self.textEdit_2.append(
                            f'<span style="color: black;"> * Tips: Cancel operation </span>')
                else:
                    self.textEdit_2.append(
                        f'<span style="color: black;"> * Tips: Canceled </span>')
            else:
                reply_n = QMessageBox.question(None, "Tip", "Whether to reprocess (Please input Yes/No)?",
                                               QMessageBox.Yes | QMessageBox.No)
                if reply_n == QMessageBox.Yes:
                    self.Calculate_point_Normals()
                    if hasattr(self, 'numnor'):
                        del self.numnor
                else:
                    self.textEdit_2.append(
                        f'<span style="color: black;"> * Tips: Canceled </span>')

        # 启动检查任务
        QTimer.singleShot(500, wait_for_window_close)

    ## 点法向量计算副函数（3）------ 点法向量的计算运行时间函数
    def output_Normal_time_N(self,tish):
        self.textEdit_2.append(
            f'<span style="color: black;"> * The running time to calculate the point normal is {tish} </span>')

    ## 点曲率计算主函数
    def Calculate_Point_Curvature(self):
        self.progressBar_2.setValue(0)
        self.progressBar_2.setMaximum(100)
        try:
            if np.all(self.PcData[:,6] == 0):
                num, ok = QInputDialog.getInt(None, "Enter the number of query points",
                                              "Please input positive integer:")
                self.curvature_thread = Curvature_Calculation_Thread(self.pointcloud, num)
                self.curvature_thread.progress_signal.connect(self.update_progress_Curvature)  # 连接进度更新信号
                self.curvature_thread.result_signal.connect(self.show_result_message_model1)  # 连接信息显示信号
                self.curvature_thread.data_signal.connect(self.save_Curvature_result_model1)  # 连接结果显示信号
                self.curvature_thread.time_signal.connect(self.output_curvature_time)  # 连接运行显示信号
                self.curvature_thread.start()  # 启动线程

                # 弹出暂停/继续/结束的对话框
                self.pause_dialog = PauseDialog()
                self.pause_dialog.continue_signal.connect(self.on_continue_model1)
                self.pause_dialog.pause_signal.connect(self.on_pause_model1)
                self.pause_dialog.stop_signal.connect(self.on_stop_model1)
                self.pause_dialog.start_calculation()  # 开始计算，启用按钮
                self.pause_dialog.exec_()  # 阻塞，直到计算完成或点击结束
        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: The data set is empty\n{e} </span>')

    ## 点曲率计算副函数（2）------ 主要负责更新计算点曲率时的进度条
    def update_progress_Curvature(self, progress):
        self.progressBar_2.setValue(progress)
        QApplication.processEvents()  # 更新界面

    ## 点曲率计算副函数（3）------ 点曲率的计算情况反馈
    def show_result_message_model1(self, message):
        self.textEdit_2.append(
            f'<span style="color: black;"> * {message} </span>')

    ## 点曲率计算副函数（4）------ 点曲率计算结果的判别和保存
    def save_Curvature_result_model1(self, curvature):
        ## 以点曲率为颜色集创建一个点云数据集
        self.PcData[:,6] = curvature[:,2]
        Data_Curvature = o3d.geometry.PointCloud()  # 创建一个空的Open3D 点云对象
        Data_Curvature.points = o3d.utility.Vector3dVector(self.pointcloud.points)  # 设置点云的点坐标
        Curvature_vis_color = (curvature + 1.0) / 2.0
        Curvature_vis_color = np.clip(Curvature_vis_color, 0, 1)
        Data_Curvature.colors = o3d.utility.Vector3dVector(Curvature_vis_color)  # 设置点云的颜色
        self.Using_VTK_ShowPC(Data_Curvature)
        vis = o3d.visualization.Visualizer()
        vis.create_window()  # 通过 Visualizer 创建窗口
        vis.add_geometry(Data_Curvature)  # 将点云添加到可视化器中
        opt = vis.get_render_option()
        opt.point_size = 2.0
        vis.run()
        vis.clear_geometries()
        vis.destroy_window()
        del vis

        reply_1 = QMessageBox.question(None, "Whether the processing result is reasonable?", "Do you accept the results?",
                                       QMessageBox.Yes | QMessageBox.No)

        if reply_1 == QMessageBox.Yes:
            # 保存点法向量信息
            reply_4 = QMessageBox.question(None, "Tips", "Is point cloud data saved?",
                                           QMessageBox.Yes | QMessageBox.No)

            if reply_4 == QMessageBox.Yes:
                try:
                    if Data_Curvature.colors:
                        file_types = "PLY Files (*.ply);;PCD Files (*.pcd);;All Files (*)"  # 限定数据保存类型
                        options = QFileDialog.Options()
                        file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save the point cloud file", "", file_types,
                                                                                 options=options)
                        if file_path:
                            # 根据选择的文件过滤器确定保存格式并保存文件
                            if selected_filter == "PLY Files (*.ply)":
                                try:
                                    o3d.io.write_point_cloud(file_path, Data_Curvature)
                                    self.textEdit_2.append(
                                        f'<span style="color: black;"> * Tips: The point cloud file is saved successfully </span>')
                                except Exception as e:
                                    self.textEdit_2.append(
                                        f'<span style="color: red;"> * Error: Failed to save point cloud file (PLY format): {str(e)} </span>')
                            elif selected_filter == "PCD Files (*.pcd)":
                                try:
                                    o3d.io.write_point_cloud(file_path, Data_Curvature,
                                                             write_ascii=True)  # PCD格式保存时可按需设置参数，这里以ASCII格式为例
                                    self.textEdit_2.append(
                                        f'<span style="color: black;"> * Tips: The point cloud file is saved successfully </span>')
                                except Exception as e:
                                    self.textEdit_2.append(
                                        f'<span style="color: red;"> * Error: Failed to save point cloud file (PCD format): {str(e)} </span>')
                            else:
                                self.textEdit_2.append(
                                    f'<span style="color: red;"> * Warning: Unsupported file format selection, save operation cancelled </span>')
                        else:
                            self.textEdit_2.append(
                                f'<span style="color: black;"> * Tips: Save operation cancel </span>')

                except Exception as e:
                    self.textEdit_2.append(
                        f'<span style="color: red;"> * Warning: The point cloud data set is empty: {e} </span>')
            else:
                self.textEdit_2.append(
                    f'<span style="color: black;"> * Tips: Canceled </span>')


            # 保存神经网络训练点信息
            reply_5 = QMessageBox.question(None, "Tips", "Whether the training data is saved?",
                                           QMessageBox.Yes | QMessageBox.No)
            if reply_5 == QMessageBox.Yes:
                try:
                    if np.all(self.PcData[:,6] != 0):
                        file_types = "NPY Files (*.npy);;All Files (*)"  # 限定数据保存类型
                        options = QFileDialog.Options()
                        file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save data file", "",
                                                                                 file_types,
                                                                                 options=options)
                        if file_path:
                            # 根据选择的文件过滤器确定保存格式并保存文件
                            if selected_filter == "NPY Files (*.npy)":
                                try:
                                    np.save(file_path, self.PcData)  # 保存计算结果
                                    self.textEdit_2.append(
                                        f'<span style="color: black;"> * Tips: The data file is saved successfully</span>')
                                except Exception as e:
                                    self.textEdit_2.append(
                                        f'<span style="color: red;"> * Error: Failed to save data file (NPY format): {str(e)} </span>')
                            else:
                                self.textEdit_2.append(
                                    f'<span style="color: red;"> * Warning: Unsupported file format selection, save operation cancelled </span>')
                        else:
                            self.textEdit_2.append(
                                f'<span style="color: black;"> * Tips: Save operation cancel </span>')

                except Exception as e:
                    self.textEdit_2.append(
                        f'<span style="color: red;"> * Warning: The point cloud data set is empty: {e} </span>')
            else:
                self.textEdit_2.append(
                    f'<span style="color: black;"> * Tips: Canceled" </span>')

        else:
            reply_x = QMessageBox.question(None, "Tip", "Whether to reprocess (Please input Yes/No)?",
                                           QMessageBox.Yes | QMessageBox.No)
            if reply_x == QMessageBox.Yes:
                self.Calculate_Point_Curvature()
            else:
                self.textEdit_2.append(
                    f'<span style="color: black;"> * Tips: Canceled" </span>')

    ## 点曲率计算副函数（5）------ 点曲率计算过程继续函数
    def on_continue_model1(self):
        if self.curvature_thread:
            self.curvature_thread.resume()  # 恢复计算
            self.pause_dialog.continue_button.setEnabled(False)  # 继续后不能再继续
            self.pause_dialog.pause_button.setEnabled(True)  # 继续后可以暂停
            self.pause_dialog.stop_button.setEnabled(True)  # 继续后可以结束计算

    ## 点曲率计算副函数（6）------ 点曲率计算过程停止函数
    def on_pause_model1(self):
        if self.curvature_thread:
            self.curvature_thread.pause()  # 暂停计算
            self.pause_dialog.continue_button.setEnabled(True)  # 暂停后可以继续
            self.pause_dialog.pause_button.setEnabled(False)  # 暂停后不能再暂停
            self.pause_dialog.stop_button.setEnabled(True)  # 暂停

    ## 点曲率计算副函数（7）------ 点曲率计算过程结束函数
    def on_stop_model1(self):
        if self.curvature_thread:
            self.curvature_thread.stop()  # 结束计算
            self.pause_dialog.close()  # 关闭弹窗

    ## 点曲率计算副函数（8）------ 点曲率运行时间函数
    def output_curvature_time(self,operate_time):
        self.textEdit_2.append(
            f'<span style="color: black;"> * The running time to calculate the point curvature is {operate_time}" </span>')

#########################################************************ 模块2 ************************#########################################
    ## 按键 1 *******（主函数）-------导入数据信息
    def Input_data(self):
        # 读取点法向量信息
        file_path_cloud, _ = QFileDialog.getOpenFileName(self, "Please select the point cloud file", "", "Point cloud file (*.pcd *.obj *.ply)")
        if not file_path_cloud:
            self.textEdit_2.append(
                '<span style="color: red;"> *  Warning: No point cloud file is selected </span>')
            return
        try:
            if file_path_cloud.endswith(".ply") or file_path_cloud.endswith(".obj") or file_path_cloud.endswith(".pcd"):
                self.Normals_points_data = o3d.io.read_point_cloud(file_path_cloud)
                if  self.Normals_points_data:
                    self.textEdit_2.append(
                        f'<span style="color: black;"> * Tips: The point cloud has been successfully loaded </span>')
                    self.Using_VTK_ShowPC(self.Normals_points_data)
            else:
                self.textEdit_2.append(
                    '<span style="color: red;"> * Warning: Unsupported file format </span>')
                return
        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: Failed to load the point cloud \n {e} </span>')

        # 读取训练点信息
        file_path_train, _ = QFileDialog.getOpenFileName(self, "Please select the original training point file.", "", "Training data (*.npy)")
        if not file_path_train:
            self.textEdit_2.append(
                '<span style="color: red;"> * Warning: No selection of training point file! </span>')
            return
        try:
            if file_path_train.endswith(".npy"):
                self.PcData_lin = np.load(file_path_train)
            else:
                self.textEdit_2.append(
                    '<span style="color: red;"> * Warning: Unsupported file format </span>')
                return
        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: Failed to load the training point file \n {e} </span>')

        # 参数正交化
        PcData_max = np.max(self.PcData_lin, axis=0)
        PcData_min = np.min(self.PcData_lin, axis=0)
        self.PcDataNew = (self.PcData_lin - PcData_min) / (PcData_max - PcData_min)

        #输出导入数据成功
        if np.all(self.PcData_lin[:, 6] != 0) and self.Normals_points_data:
            self.textEdit_2.append(
                f'<span style="color: black;"> * Tips: The Data has been successfully loaded </span>')

    ## 按键 2 *******（主函数）-------选择学习样本
    def Choose_Train_Points(self):
        try:
            # ---------- 防止 stdout/stderr 为 None ----------
            class DummyIO:
                def write(self, *args, **kwargs): pass

                def flush(self): pass

            if sys.stdout is None:
                sys.stdout = DummyIO()
            if sys.stderr is None:
                sys.stderr = DummyIO()

            # ---------- 禁用 open3d 输出日志 ----------
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

            try:
                temp_dir = os.path.join(tempfile.gettempdir(), "open3d_temp")
                os.makedirs(temp_dir, exist_ok=True)
                os.chdir(temp_dir)

                # 设置环境变量（PyInstaller 下很关键）
                os.environ["TMPDIR"] = temp_dir
                os.environ["TEMP"] = temp_dir
                os.environ["TMP"] = temp_dir

                # 如果是打包环境，强制让 Open3D 从 _MEIPASS 加载资源
                if hasattr(sys, "_MEIPASS"):
                    open3d_path = os.path.join(sys._MEIPASS, "open3d")
                    if os.path.exists(open3d_path):
                        sys.path.append(open3d_path)

                # self.textEdit_2.append(f"<span style='color: gray;'>* 临时路径: {temp_dir}</span>")
                # self.textEdit_2.append(f"<span style='color: gray;'>* Open3D 模块路径: {o3d.__file__}</span>")
                # QApplication.processEvents()
            except Exception as e:
                self.textEdit_2.append(f"<span style='color: red;'>* 临时目录初始化失败: {e}</span>")

            # 临时工作目录（用户可写）
            temp_dir = os.path.join(tempfile.gettempdir(), "open3d_temp")
            os.makedirs(temp_dir, exist_ok=True)
            os.chdir(temp_dir)  # 强制切换到可写路径
            # self.textEdit_2.append(f"<span style='color: gray;'>* 临时路径: {temp_dir}</span>")
            # QApplication.processEvents()

            if np.all(self.PcData_lin[:, 6] != 0):
                # 样本点的选择
                GroupNum, ok = QInputDialog.getInt(None, "Please provide the total number of sets including both rock joints and non-rock joints", "Please enter an integer:")
                PcLearn = []
                # 开始选点
                for i in range(GroupNum):
                    var_name = f"Vision{i}"  # 动态生成变量名，例如 jk1, jk2, jk3...
                    # 在调用 VisualizerWithEditing 前加上
                    if sys.stdout is None:
                        sys.stdout = io.TextIOWrapper(open(sys.__stdout__.fileno(), 'wb', buffering=0),
                                                      write_through=True)
                    if sys.stderr is None:
                        sys.stderr = io.TextIOWrapper(open(sys.__stderr__.fileno(), 'wb', buffering=0),
                                                      write_through=True)

                    globals()[var_name] = o3d.visualization.VisualizerWithEditing()  # 为每个动态变量赋值
                    globals()[var_name].create_window()  # 通过 Visualizer 创建窗口
                    globals()[var_name].add_geometry(self.Normals_points_data)  # 将点云加入可视化器
                    OptN = globals()[var_name].get_render_option()
                    OptN.point_size = 2.0
                    globals()[var_name].run()  # 显示点云并等待用户点击以拾取多个点
                    Picked_Points_Idex = globals()[var_name].get_picked_points()  # 拾取多个点
                    globals()[var_name].destroy_window()  # 关闭窗口
                    PointsGet = np.zeros((len(Picked_Points_Idex), 8 + GroupNum))
                    self.Rock_number = GroupNum
                    PointsGet[:, 8 + i] = 1
                    PointsGet[:, 0:7] = self.PcDataNew[Picked_Points_Idex]
                    PointsGet[:, 7] = Picked_Points_Idex
                    PointsGet = list(PointsGet)
                    for Lin in PointsGet:
                        PcLearn.append(Lin)
                    # 提示选点信息
                    for idx in Picked_Points_Idex:
                        point_info = f" Point selection ({i + 1}) : {idx} ({self.PcData_lin[idx, 0]:.2f}, {self.PcData_lin[idx, 1]:.2f}, {self.PcData_lin[idx, 2]:.2f})"
                        self.textEdit_2.append(
                            f'<span style="color: black;"> * {point_info} </span>')
                    QApplication.processEvents()  # 刷新 UI
                    # 提示选点个数
                    self.textEdit_2.append(
                        f'<span style="color: black;"> * Tips: A total of {len(Picked_Points_Idex)} points were selected this time </span>')
                PcLearn = np.asarray(PcLearn)

                # 保存选点信息
                reply_6 = QMessageBox.question(None, "Tips", "Will the selected data be saved?",
                                               QMessageBox.Yes | QMessageBox.No)
                if reply_6 == QMessageBox.Yes:
                    try:
                        if np.all(PcLearn[:, 6] != 0):
                            file_types = "NPY Files (*.npy);;All Files (*)"  # 限定数据保存类型
                            options = QFileDialog.Options()
                            file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save point cloud file", "",
                                                                                     file_types,
                                                                                     options=options)
                            if file_path:
                                # 根据选择的文件过滤器确定保存格式并保存文件
                                if selected_filter == "NPY Files (*.npy)":
                                    try:
                                        np.save(file_path, PcLearn)  # 保存计算结果
                                        self.textEdit_2.append(f"* Tips: The Data has been successfully saved")
                                    except Exception as e:
                                        self.textEdit_2.append(
                                            f'<span style="color: red;"> * Error: Failed to save data (NPY format): {str(e)} </span>')
                                else:
                                    self.textEdit_2.append(
                                        f'<span style="color: red;"> * Warning: Unsupported file format selection, save operation cancelled </span>')
                            else:
                                self.textEdit_2.append(
                                    f"* Tips: Canceled")
                    except Exception as e:
                        self.textEdit_2.append(
                            f'<span style="color: red;"> * Error: The dataset is empty\n {e} </span>')
                else:
                    self.textEdit_2.append(
                        f"* Tips: Canceled")
        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: The dataset loading failed \n {e} </span>')

    ## 按键 3 *******（主函数）-------建立新的神经网络模型，并进行训练、显示和保存
    def Build_Artificial_Neural_Network(self):
        if sys.stdout is None:
            sys.stdout = io.StringIO()
        if sys.stderr is None:
            sys.stderr = io.StringIO()
            # --- 静默 TensorFlow 输出 ---
        tensorflow.get_logger().setLevel('ERROR')
        tensorflow.autograph.set_verbosity(0)

        log_text_edit = self.textEdit_2

        reply_7 = QMessageBox.question(None, "Tips", "Should a new network be established?", QMessageBox.Yes | QMessageBox.No)

        if reply_7 == QMessageBox.Yes:
            # 加载学习样本
            file_path_learn, _ = QFileDialog.getOpenFileName(self, "Please select the PcLearn file", "", "training data (*.npy)")
            if not file_path_learn:
                log_text_edit.append(
                    f'<span style="color: red;"> * Warning: No learning samples were selected!  </span>')
                return
            try:
                if file_path_learn.endswith(".npy"):
                    PcLearn = np.load(file_path_learn)
                else:
                    log_text_edit.append(
                        f'<span style="color: red;"> * Warning: Unsupported file format selection </span>')
                    return
            except Exception as e:
                log_text_edit.append(
                    f'<span style="color: red;"> * Error: The training data loading failed \n {e} </span>')

            # 选择输入数据类型
            ParameSelc = ParameterSelectionWindow()
            if ParameSelc.exec_() == QDialog.Accepted:
                reply8 = ParameSelc.parameter
                if reply8 == 1:
                    x = PcLearn[:, 3:6]
                elif reply8 == 2:
                    x = PcLearn[:, 6]
                elif reply8 == 3:
                    x = PcLearn[:, 3:7]
            try:
                if np.all(x):
                    if not hasattr(self, 'Rock_number'):
                        self.Rock_number, ok = QInputDialog.getInt(None, "Enter the number of classification sets",
                                                      "Please input positive integer:")
                    y = PcLearn[:, 8:8+self.Rock_number]  # 目标数据
                    Inputs_Parameter = x
                    Targets_Parameter = y
                    # 构建神经网络
                    Model_Front = Sequential()
                    # 添加输入层和第一个隐藏层
                    Model_Front.add(Dense(128, input_dim=Inputs_Parameter.shape[1], activation='relu'))  # 第1个隐藏层有64个节点
                    Model_Front.add(Dropout(0.2))  # 添加Dropout防过拟合
                    Model_Front.add(Dense(64, activation='relu'))  # 第二个隐藏层有32个节点
                    Model_Front.add(Dense(32, activation='relu'))  # 第二个隐藏层有32个节点
                    # 添加输出层，softmax 激活函数用于多分类问题
                    Model_Front.add(Dense(Targets_Parameter.shape[1], activation='softmax'))
                    # 编译模型，使用交叉熵损失函数和Adam优化器
                    Model_Front.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    # 训练模型
                    early_stopping = EarlyStopping(monitor='val_loss', patience=20,mode='max',restore_best_weights=True) #,mode='max',restore_best_weights=True #当20个epochs中损失率不出现变化时，训练自动停止

                    # 将训练信息实时反馈给UI界面
                    class TrainingCallback(Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            loss = logs.get('loss')
                            acc = logs.get('accuracy')
                            val_loss = logs.get('val_loss')
                            val_acc = logs.get('val_accuracy')
                            # 在 QTextEdit 中显示训练日志信息
                            log_info = f"Epoch {epoch + 1}/100\nloss: {loss:.4f} - accuracy: {acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}"
                            log_text_edit.append(
                                f'<span style="color: black;"> * {log_info} </span>')
                            QApplication.processEvents()  # 刷新 UI

                    # 训练模型并使用回调函数更新日志，同时启用EarlyStopping
                    # history = Model_Front.fit(Inputs_Parameter, Targets_Parameter, epochs=100,
                    #                           batch_size=5, validation_split=0.2,
                    #                           callbacks=[TrainingCallback(), early_stopping])  # 每次选取10样本更新模型，验证集的比例占20%
                    history = Model_Front.fit(Inputs_Parameter, Targets_Parameter, epochs=100,
                                              batch_size=10, validation_split=0.2,
                                              callbacks=[TrainingCallback()])  # 每次选取10样本更新模型，验证集的比例占20%

                    # 清除之前的图形
                    while self.form_layout.count():
                        item = self.form_layout.takeAt(0)
                        if item.widget():
                            item.widget().deleteLater()

                    # 创建一个Matplotlib图形
                    self.figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
                    # 绘制损失曲线
                    ax1.plot(history.history['loss'], label='Training Loss')
                    ax1.plot(history.history['val_loss'], label='Validation Loss')
                    ax1.set_title('Loss during training')
                    ax1.set_xlabel('Epochs')
                    ax1.set_ylabel('Loss')
                    ax1.legend()

                    # 绘制准确率曲线
                    ax2.plot(history.history['accuracy'], label='Training Accuracy')
                    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                    ax2.set_title('Accuracy during training')
                    ax2.set_xlabel('Epochs')
                    ax2.set_ylabel('Accuracy')
                    ax2.legend()

                    # 创建FigureCanvas并将其嵌入到QFormLayout中
                    self.canvas = FigureCanvas(self.figure)
                    self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                    self.canvas.setMinimumHeight(300)

                    # 将图形添加到QFormLayout，并使其居中
                    self.form_layout.addRow(self.canvas)
                    # 绘制图形
                    self.canvas.draw()

                    # 显示预测结果
                    if reply8 == 1:
                        x_input = self.PcDataNew[:, 3:6]
                    elif reply8 == 2:
                        x_input = self.PcDataNew[:, 6]
                    elif reply8 == 3:
                        x_input = self.PcDataNew[:, 3:7]

                    self.Showing_The_Predict_Result_New(Model_Front,x_input)

            except Exception as e:
                log_text_edit.append(
                    f'<span style="color: red;"> * Error: Input layer failed to load correctly\n {e} </span>')
        else:
            log_text_edit.append(
                f'<span style="color: black;"> * Tips: Canceled </span>')

    ## 按键 4 *******（主函数）-------选择已建立好的网络模型进行预测
    def Neural_Network_Using(self):
        reply_7_1 = QMessageBox.question(None, "Tips", "Would you like to activate the network that has been set up?", QMessageBox.Yes | QMessageBox.No)
        # 是否打开新的网络
        if reply_7_1 == QMessageBox.Yes:
            # 加载网络
            file_path_net, _ = QFileDialog.getOpenFileName(self, "Choose the existing network", "", "Network (*.keras)")
            if not file_path_net:
                self.textEdit_2.append(
                    '<span style="color: red;"> *  Warning: No Network is selected </span>')
                return
            try:
                if file_path_net.endswith(".keras"):
                    Model_Front = load_model(file_path_net)
                else:
                    self.textEdit_2.append(
                        '<span style="color: red;"> * Warning: Unsupported file format </span>')
                    return
            except Exception as e:
                self.textEdit_2.append(
                    f'<span style="color: red;"> * Error: Failed to load the Network \n {e} </span>')

            # 显示预测结果
            if Model_Front:
                ParameSelc = ParameterSelectionWindow()
                if ParameSelc.exec_() == QDialog.Accepted:
                    reply8_1 = ParameSelc.parameter
                    if reply8_1 == 1:
                        x_input = self.PcDataNew[:, 3:6]
                    elif reply8_1 == 2:
                        x_input = self.PcDataNew[:, 6]
                    elif reply8_1 == 3:
                        x_input = self.PcDataNew[:, 3:7]
                    x_input = np.atleast_2d(x_input)  # ✅ 强制二维安全输入
                    self.Showing_The_Predict_Result_Origin(Model_Front, x_input)
        else:
            self.textEdit_2.append(
                f'<span style="color: black;"> * Tips: Canceled </span>')

    ## 按键 3 *******（副函数1）-------开启副线程，通过新建立的网络对输入数据进行预测
    def Showing_The_Predict_Result_New(self,Netmodel,train_data):
        # 创建 PredictionThread 实例并连接信号
        self.prediction_thread = PredictionThread(Netmodel,train_data)
        self.prediction_thread.progress_signal.connect(self.update_progress_prediction)
        self.prediction_thread.result_signal.connect(self.temporary_print_New)
        self.net_Lin = Netmodel
        # 启动子线程
        self.prediction_thread.start()

    ## 按键 4 *******（副函数1）-------开启副线程，通过已有的神经网络对输入数据进行预测
    def Showing_The_Predict_Result_Origin(self,Netmodel,train_data):
        # 防止多线程重复启动崩溃
        if hasattr(self, 'prediction_thread') and self.prediction_thread.isRunning():
            self.textEdit_2.append('<span style="color:red;"> * Warning: Prediction already running </span>')
            return
        # 创建 PredictionThread 实例并连接信号
        self.prediction_thread = PredictionThread(Netmodel,train_data)
        self.prediction_thread.progress_signal.connect(self.update_progress_prediction)
        self.prediction_thread.result_signal.connect(self.temporary_print_Origin)
        self.net_Lin = Netmodel
        # 启动子线程
        self.prediction_thread.start()

    ## 按键 3和4 *******（副函数2）-------展示预测的开始和结束
    def update_progress_prediction(self, progress_text):
        # 更新 QTextEdit 显示进度信息
        self.textEdit_2.append(
            f'<span style="color: black;"> * {progress_text} </span>')

    ## 按键 3 *******（副函数3）-------展示使用新建立的神经网络对输入的数据集进行预测的结果
    def temporary_print_New(self,points_location):
        self.Prediction_Result = np.round(points_location)
        Outputs_color = np.zeros((len(self.Normals_points_data.points), 3))
        column_tes = self.Prediction_Result.shape[1]
        if column_tes == 6:
            Outputs_color[np.where(self.Prediction_Result[:, 0] == 1)[0], 0] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 1] == 1)[0], 2] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 2] == 1)[0], 1] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 3] == 1)[0], 1:3] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 4] == 1)[0], 0:2] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 5] == 1)[0], 0:3] = 1
            mask = np.all(self.Prediction_Result[:, 0:6] == 0, axis=1)
            Outputs_color[mask, 0:3] = 1
        elif column_tes == 5:
            Outputs_color[np.where(self.Prediction_Result[:, 0] == 1)[0], 0] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 1] == 1)[0], 2] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 2] == 1)[0], 1] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 3] == 1)[0], 1:3] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 4] == 1)[0], 0:3] = 1
            mask = np.all(self.Prediction_Result[:, 0:5] == 0, axis=1)
            Outputs_color[mask, 0:3] = 1
        elif column_tes == 4:
            Outputs_color[np.where(self.Prediction_Result[:, 0] == 1)[0], 0] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 1] == 1)[0], 2] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 2] == 1)[0], 1] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 3] == 1)[0], 0:3] = 1
            mask = np.all(self.Prediction_Result[:, 0:4] == 0, axis=1)
            Outputs_color[mask, 0:3] = 1
        elif column_tes == 3:
            Outputs_color[np.where(self.Prediction_Result[:, 0] == 1)[0], 0] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 1] == 1)[0], 2] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 3] == 1)[0], 0:3] = 1
            mask = np.all(self.Prediction_Result[:, 0:3] == 0, axis=1)
            Outputs_color[mask, 0:3] = 1
        elif column_tes == 2:
            Outputs_color[np.where(self.Prediction_Result[:, 0] == 1)[0], 0] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 3] == 1)[0], 0:3] = 1
            mask = np.all(self.Prediction_Result[:, 0:2] == 0, axis=1)
            Outputs_color[mask, 0:3] = 1

        ## 结构面预测结果的展示
        Canda_Predict = o3d.geometry.PointCloud()  # 创建一个空的Open3D 点云对象
        Canda_Predict.points = o3d.utility.Vector3dVector(self.Normals_points_data.points)  # 设置点云的点坐标
        Canda_Predict.colors = o3d.utility.Vector3dVector(Outputs_color)  # 设置点云的颜色
        self.Using_VTK_ShowPC(Canda_Predict)
        vis = o3d.visualization.Visualizer()
        vis.create_window()  # 通过 Visualizer 创建窗口
        vis.add_geometry(Canda_Predict)  # 将点云添加到可视化器中
        opt = vis.get_render_option()
        opt.point_size = 2.0
        vis.run()
        vis.clear_geometries()
        vis.destroy_window()
        del vis

        # 判断是否要保存结果
        self.Data_Saving_Model2_intergrated(Outputs_color)

    ## 按键 4 *******（副函数3）-------展示使用已有的神经网络对输入的数据集进行预测的结果
    def temporary_print_Origin(self, points_location):
        self.Prediction_Result = np.round(points_location)
        column_tes = self.Prediction_Result.shape[1]
        Outputs_color = np.zeros((len(self.Normals_points_data.points), 3))
        if column_tes == 6:
            Outputs_color[np.where(self.Prediction_Result[:, 0] == 1)[0], 0] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 1] == 1)[0], 2] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 2] == 1)[0], 1] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 3] == 1)[0], 1:3] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 4] == 1)[0], 0:2] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 5] == 1)[0], 0:3] = 1
            mask = np.all(self.Prediction_Result[:, 0:6] == 0, axis=1)
            Outputs_color[mask, 0:3] = 1
        elif column_tes == 5:
            Outputs_color[np.where(self.Prediction_Result[:, 0] == 1)[0], 0] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 1] == 1)[0], 2] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 2] == 1)[0], 1] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 3] == 1)[0], 1:3] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 4] == 1)[0], 0:3] = 1
            mask = np.all(self.Prediction_Result[:, 0:5] == 0, axis=1)
            Outputs_color[mask, 0:3] = 1
        elif column_tes == 4:
            Outputs_color[np.where(self.Prediction_Result[:, 0] == 1)[0], 0] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 1] == 1)[0], 2] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 2] == 1)[0], 1] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 3] == 1)[0], 0:3] = 1
            mask = np.all(self.Prediction_Result[:, 0:4] == 0, axis=1)
            Outputs_color[mask, 0:3] = 1
        elif column_tes == 3:
            Outputs_color[np.where(self.Prediction_Result[:, 0] == 1)[0], 0] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 1] == 1)[0], 2] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 3] == 1)[0], 0:3] = 1
            mask = np.all(self.Prediction_Result[:, 0:3] == 0, axis=1)
            Outputs_color[mask, 0:3] = 1
        elif column_tes == 2:
            Outputs_color[np.where(self.Prediction_Result[:, 0] == 1)[0], 0] = 1
            Outputs_color[np.where(self.Prediction_Result[:, 3] == 1)[0], 0:3] = 1
            mask = np.all(self.Prediction_Result[:, 0:2] == 0, axis=1)
            Outputs_color[mask, 0:3] = 1
        ## 结构面预测结果的展示
        Canda_Predict = o3d.geometry.PointCloud()  # 创建一个空的Open3D 点云对象
        Canda_Predict.points = o3d.utility.Vector3dVector(self.Normals_points_data.points)  # 设置点云的点坐标
        Canda_Predict.colors = o3d.utility.Vector3dVector(Outputs_color)  # 设置点云的颜色
        self.Using_VTK_ShowPC(Canda_Predict)
        vis = o3d.visualization.Visualizer()
        vis.create_window()  # 通过 Visualizer 创建窗口
        vis.add_geometry(Canda_Predict)  # 将点云添加到可视化器中
        opt = vis.get_render_option()
        opt.point_size = 2.0
        vis.run()
        vis.clear_geometries()
        vis.destroy_window()
        del vis
        # 是否要保存结构面识别结果
        self.Data_Saving_Model2_Predictresult(Outputs_color)

    ## 按键 3 *******（副函数4）-------判断通过新建立的神经网络进行预测的结果的好坏，并保存满意的网络模型和结构面识别的结果
    def Data_Saving_Model2_intergrated(self,color_Model2):
        # 是否接受预测结果
        reply_7_2 = QMessageBox.question(None, "Tips", "Do you accept the predicted result?",
                                         QMessageBox.Yes | QMessageBox.No)
        if reply_7_2 == QMessageBox.Yes:
            # 保存预测的模型
            reply_7_3 = QMessageBox.question(None, "Tips", "Should the network model be saved?",
                                             QMessageBox.Yes | QMessageBox.No)
            if reply_7_3 == QMessageBox.Yes:
                try:
                    if self.net_Lin:
                        file_types = "KERAS Files (*.keras);;All Files (*)"  # 限定数据保存类型
                        options = QFileDialog.Options()
                        file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save the network model", "",
                                                                                 file_types,
                                                                                 options=options)
                        if file_path:
                            # 根据选择的文件过滤器确定保存格式并保存文件
                            if selected_filter == "KERAS Files (*.keras)":
                                try:
                                    self.net_Lin.save(file_path)
                                    self.textEdit_2.append(
                                        f'<span style="color: black;"> * Tips: The network has saved successfully </span>')

                                    # 保存结构面识别结果
                                    self.Data_Saving_Model2_Predictresult(color_Model2)
                                except Exception as e:
                                    self.textEdit_2.append(
                                        f'<span style="color: red;"> * Error: Failed to save network file (KERAS format): {str(e)} </span>')
                            else:
                                self.textEdit_2.append(
                                    f'<span style="color: red;"> * Warning: Unsupported file format selection, save operation cancelled </span>')
                        else:
                            self.textEdit_2.append(
                                f'<span style="color: black;"> * Tips: Canceled </span>')
                except Exception as e:
                    self.textEdit_2.append(
                        f'<span style="color: red;"> * Error: Invalid network \n {e}</span>')
            else:
                self.textEdit_2.append(
                    f'<span style="color: black;"> * Tips: Canceled </span>')

        elif reply_7_2 == QMessageBox.No:
            reply_7_2_1 = QMessageBox.question(None, "Tips", "Is the model retrained?",
                                             QMessageBox.Yes | QMessageBox.No)
            if reply_7_2_1 == QMessageBox.Yes:
                self.Build_Artificial_Neural_Network()
            else:
                self.textEdit_2.append(
                    f'<span style="color: black;"> * Tips: Canceled </span>')
        else:
            self.textEdit_2.append(
                f'<span style="color: black;"> * Tips: Canceled </span>')

    ## 按键 3和4 *******（副函数4）-------保存新建立的或者现存的神经网络做出的结构面识别的结果和PcData-Final数据
    def Data_Saving_Model2_Predictresult(self,color_Model):
        Data_prediction = o3d.geometry.PointCloud()  # 创建一个空的Open3D 点云对象
        Data_prediction.points = o3d.utility.Vector3dVector(self.Normals_points_data.points)  # 设置点云的点坐标
        Data_prediction.colors = o3d.utility.Vector3dVector(color_Model)  # 设置点云的颜色
        reply_3_1 = QMessageBox.question(None, "Tips", "Is the recognition result saved?", QMessageBox.Yes | QMessageBox.No)
        if reply_3_1 == QMessageBox.Yes:
            file_types = "PLY Files (*.ply);;PCD Files (*.pcd);;All Files (*)"  # 限定数据保存类型
            options = QFileDialog.Options()
            file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save point cloud file", "", file_types,
                                                                     options=options)
            if file_path:
                # 根据选择的文件过滤器确定保存格式并保存文件
                if selected_filter == "PLY Files (*.ply)":
                    try:
                        o3d.io.write_point_cloud(file_path, Data_prediction)
                        self.textEdit_2.append(
                            f'<span style="color: black;"> * Tips: The point cloud file has been saved successfully </span>')
                    except Exception as e:
                        self.textEdit_2.append(
                            f'<span style="color: red;"> * Error: Failed to save point cloud file (PLY format): {str(e)} </span>')
                elif selected_filter == "PCD Files (*.pcd)":
                    try:
                        o3d.io.write_point_cloud(file_path, Data_prediction,
                                                 write_ascii=True)  # PCD格式保存时可按需设置参数，这里以ASCII格式为例
                        self.textEdit_2.append(
                            f'<span style="color: black;"> * Tips: The point cloud file is saved successfully </span>')
                    except Exception as e:
                        self.textEdit_2.append(
                            f'<span style="color: red;"> * Error: Failed to save point cloud file (PCD format): {str(e)} </span>')
                else:
                    self.textEdit_2.append(
                        f'<span style="color: red;"> * Warning: Unsupported file format selection, save operation cancelled </span>')
            else:
                self.textEdit_2.append(
                    f'<span style="color: black;"> * Tips: Canceled </span>')
        else:
            self.textEdit_2.append(
                f'<span style="color: black;"> * Tips: Canceled </span>')

        #保存PcData数据
        reply_3_4 = QMessageBox.question(None, "Tips", "Is the classification file saved?", QMessageBox.Yes | QMessageBox.No)
        if reply_3_4 == QMessageBox.Yes:
            PcData_Final = np.hstack((self.PcData_lin, self.Prediction_Result))
            file_types = "NPY Files (*.npy);;All Files (*)"  # 限定数据保存类型
            options = QFileDialog.Options()
            file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save point cloud file", "",
                                                                     file_types,
                                                                     options=options)
            if file_path:
                # 根据选择的文件过滤器确定保存格式并保存文件
                if selected_filter == "NPY Files (*.npy)":
                    try:
                        np.save(file_path, PcData_Final)  # 保存计算结果
                        self.textEdit_2.append(
                            f'<span style="color: black;"> * Tips: The data file is saved successfully </span>')
                    except Exception as e:
                        self.textEdit_2.append(
                            f'<span style="color: red;"> * Error: Failed to save data file (NPY format): {str(e)} </span>')
                else:
                    self.textEdit_2.append(
                        f'<span style="color: red;"> * Warning: Unsupported file format selection, save operation cancelled </span>')
            else:
                self.textEdit_2.append(
                    f'<span style="color: black;"> * Tips: Canceled </span>')
        else:
            self.textEdit_2.append(
                f'<span style="color: black;"> * Tips: Canceled </span>')

#########################################************************ 模块3 ************************#########################################
    ## 按键 1 *******（主函数）-------使用VTK显示点云图形
    def load_and_display_point_cloud_model3(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select the point cloud file", "",
                                                   "Point cloud file (*.pcd *.obj *.ply)")
        if not file_path:
            self.textEdit_2.append(
                '<span style="color: red;"> *  Warning: No point cloud file is selected </span>')
            return
        try:
            if file_path.endswith(".ply") or file_path.endswith(".obj") or file_path.endswith(".pcd"):
                self.pointcloud = o3d.io.read_point_cloud(file_path)
            else:
                self.textEdit_2.append(
                    '<span style="color: red;"> * Warning: Unsupported file format </span>')
                return
        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: Failed to load the point cloud\n{e} </span>')

        self.Using_VTK_ShowPC(self.pointcloud)

    ## 按键 2 *******（主函数）-------单个结构面提取主函数
    def extraction_individual_joint(self):
        # 读取处理点信息
        file_path_train, _ = QFileDialog.getOpenFileName(self, "Please input the data file.", "", "Data file (*.npy)")
        if not file_path_train:
            self.textEdit_2.append(
                '<span style="color: red;"> *  Warning: No file is selected </span>')
            return
        try:
            if file_path_train.endswith(".npy"):
                self.PcData_Final = np.load(file_path_train)
            else:
                self.textEdit_2.append(
                    '<span style="color: red;"> * Warning: Unsupported file format </span>')
                return
        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: Failed to load the file\n{e} </span>')

        # 输入想要处理的结构面的标签
        JUlei_AL = DBSCANWindow()
        if JUlei_AL.exec_() == QDialog.Accepted:
            self.cluster = self.PcData_Final[np.where(self.PcData_Final[:, int(JUlei_AL.int_text) + 6] == 1)[0], :]
            num1 = float(JUlei_AL.text1)
            num2 = int(JUlei_AL.text2)
        try:
            if num1 and num2:
                self.DBSCAN_thread = DBSCAN_Cluster_Thread(self.cluster[:,0:3],num1,num2)
                self.textEdit_2.append(
                    f'<span style="color: black;"> * Tip: The Minrad = {num1}, and the Minpts = {num2} </span>')
                self.DBSCAN_thread.progress_signal.connect(self.update_progress_DBSCAN)  # 聚类进度更新信号
                self.DBSCAN_thread.result_signal.connect(self.show_result_message_model3)  # 聚类结果显示信号
                self.DBSCAN_thread.data_signal.connect(self.save_DBSCAN_result)  # 聚类结果保存信号
                self.DBSCAN_thread.start()  # 启动线程

                # 弹出暂停/继续/结束的对话框
                self.pause_dialog = PauseDialog()
                self.pause_dialog.continue_signal.connect(self.on_continue_model3)
                self.pause_dialog.pause_signal.connect(self.on_pause_model3)
                self.pause_dialog.stop_signal.connect(self.on_stop_model3)
                self.pause_dialog.start_calculation()  # 开始计算，启用按钮
                self.pause_dialog.exec_()  # 阻塞，直到计算完成或点击结束
            else:
                self.textEdit_2.append(
                    f'<span style="color: black;"> * Tips: Canceled </span>')

        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: Wrong operation\n{e} </span>')

    ## 按键 2 *******（副函数1）-------更新进度条
    def update_progress_DBSCAN(self, progress):
        self.progressBar_2.setValue(progress)
        QApplication.processEvents()  # 更新界面

    ## 按键 2 *******（副函数2）-------显示聚类结果是否完成
    def show_result_message_model3(self, message):
        self.textEdit_2.append(
            f'<span style="color: black;"> * Tips: {message} </span>')

    ## 按键 2 *******（副函数3）-------判断聚类结果是否满足条件
    def save_DBSCAN_result(self, Label):
        Joint_1 = np.concatenate((self.cluster, Label), axis=1)  # 降输出结果于原数据集合并[n,12]
        locasaf  = Joint_1.shape[1]
        if np.sum(Joint_1[:,locasaf-1]) == 0:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: The figure for Min radius is too small </span>')
            return
        Disc_1 = Joint_1[Joint_1[:,locasaf-1] != 0, :]  # 将独属于想要提取的单个结构面的点提取出来，过滤噪点
        NumDisc_1 = int(np.max(Disc_1[:, locasaf-1]))  # 查看有多少个单个结构面
        self.textEdit_2.append(
            f'<span style="color: black;"> * There are {NumDisc_1} individual rock joints </span>')
        Color_Indi_1 = np.random.rand(NumDisc_1, 3)  # 设定单个结构面的颜色集
        Col_big_1 = np.ones((len(Label), 3))  # 定义所有点的颜色集
        JointData1 = []
        for iaf in range(NumDisc_1):
            Col_big_1[np.where(Label == iaf + 1), :] = Color_Indi_1[iaf, :]  # 对所有颜色集进行变量赋值
            JointData1.append(Joint_1[np.where(Joint_1[:, locasaf-1] == iaf + 1)[0], :])  # 将属于某一组单个结构面上的点进行归类
        PoinLoca = np.array(self.cluster[:,0:3])
        Canda_Individual = o3d.geometry.PointCloud()  # 创建一个空的Open3D 点云对象
        Canda_Individual.points = o3d.utility.Vector3dVector(PoinLoca)  # 设置点云的点坐标
        Canda_Individual.colors = o3d.utility.Vector3dVector(Col_big_1)  # 设置点云的颜色
        self.Using_VTK_ShowPC(Canda_Individual)
        vis = o3d.visualization.Visualizer()
        vis.create_window()  # 通过 Visualizer 创建窗口
        vis.add_geometry(Canda_Individual)  # 将点云添加到可视化器中
        opt = vis.get_render_option()
        opt.point_size = 2.0
        vis.run()
        vis.clear_geometries()
        vis.destroy_window()
        del vis
        reply_3_1_3 = QMessageBox.question(None, "Tips", "Is the clustering result satisfactory?", QMessageBox.Yes | QMessageBox.No)
        if reply_3_1_3 == QMessageBox.Yes:
            reply_3_1_4 = QMessageBox.question(None, "Tips", "Should the clustering results be saved?", QMessageBox.Yes | QMessageBox.No)
            if reply_3_1_4 == QMessageBox.Yes:
                # 保存聚类点云结果
                file_types = "PLY Files (*.ply);;PCD Files (*.pcd);;All Files (*)"  # 限定数据保存类型
                options = QFileDialog.Options()
                file_path, selected_filter = QFileDialog.getSaveFileName(self, "Please save point cloudy file", "", file_types,
                                                                         options=options)
                if file_path:
                    # 根据选择的文件过滤器确定保存格式并保存文件
                    if selected_filter == "PLY Files (*.ply)":
                        try:
                            o3d.io.write_point_cloud(file_path, Canda_Individual)
                            self.textEdit_2.append(
                                f'<span style="color: black;"> * The point cloud set is saved successfully </span>')
                        except Exception as e:
                            self.textEdit_2.append(
                                f'<span style="color: red;"> * Error: Failed to save point cloud file (PLY format): {str(e)} </span>')
                    elif selected_filter == "PCD Files (*.pcd)":
                        try:
                            o3d.io.write_point_cloud(file_path, Canda_Individual,
                                                     write_ascii=True)  # PCD格式保存时可按需设置参数，这里以ASCII格式为例
                            self.textEdit_2.append(
                                f'<span style="color: black;"> * The point cloud set is saved successfully </span>')
                        except Exception as e:
                            self.textEdit_2.append(
                                f'<span style="color: red;"> * Error: Failed to save point cloud file (PCD format):{str(e)} </span>')
                    else:
                        self.textEdit_2.append(
                            '<span style="color: red;"> * Warning: Unsupported file format selection, and save operation cancelled </span>')
                else:
                    self.textEdit_2.append(
                        f'<span style="color: black;"> * Tips: Cancel operation </span>')

                # 保存计算产状的过程结果
                file_types = "PKL Files (*.pkl);;All Files (*)"  # 限定数据保存类型
                options = QFileDialog.Options()
                file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save clustering file", "",
                                                                         file_types,
                                                                         options=options)
                if file_path:
                    # 根据选择的文件过滤器确定保存格式并保存文件
                    if selected_filter == "PKL Files (*.pkl)":
                        try:
                            with open(file_path, 'wb') as f:
                                  pickle.dump(JointData1, f)
                            self.textEdit_2.append(
                                f'<span style="color: black;"> * The data file is saved successfully </span>')
                        except Exception as e:
                            self.textEdit_2.append(
                                f'<span style="color: red;"> * Error: Failed to save data file (PKL format):{str(e)} </span>')
                    else:

                        self.textEdit_2.append(
                            '<span style="color: red;"> * Warning: Unsupported file format selection, and save operation cancelled </span>')
                else:
                    self.textEdit_2.append(
                        f'<span style="color: black;"> * Tips: Cancel operation </span>')
            else:
                self.textEdit_2.append(
                    f'<span style="color: black;"> * Tips: Cancel operation </span>')
        elif reply_3_1_3 == QMessageBox.No:
            reply_3_1_5 = QMessageBox.question(None, "Tips", "Should the system conduct a re-clustering?", QMessageBox.Yes | QMessageBox.No)
            if reply_3_1_5 == QMessageBox.Yes:
                self.extraction_individual_joint()
        else:
            self.textEdit_2.append(
                f'<span style="color: black;"> * Tips: Cancel operation </span>')

    ## 按键 2 *******（副函数4）-------聚类过程继续函数
    def on_continue_model3(self):
        if self.DBSCAN_thread:
            self.DBSCAN_thread.resume()  # 恢复计算
            self.pause_dialog.continue_button.setEnabled(False)  # 继续后不能再继续
            self.pause_dialog.pause_button.setEnabled(True)  # 继续后可以暂停
            self.pause_dialog.stop_button.setEnabled(True)  # 继续后可以结束计算

    ## 按键 2 *******（副函数5）-------聚类过程暂停函数
    def on_pause_model3(self):
        if self.DBSCAN_thread:
            self.DBSCAN_thread.pause()  # 暂停计算
            self.pause_dialog.continue_button.setEnabled(True)  # 暂停后可以继续
            self.pause_dialog.pause_button.setEnabled(False)  # 暂停后不能再暂停
            self.pause_dialog.stop_button.setEnabled(True)  # 暂停

    ## 按键 2 *******（副函数6）-------聚类过程结束函数
    def on_stop_model3(self):
        if self.DBSCAN_thread:
            self.DBSCAN_thread.stop()  # 结束计算
            self.pause_dialog.close()  # 关闭弹窗

    ## 按键 3 *******（主函数）-------结构面合并函数
    def integrate_Model3(self):
        file_path_cloud, _ = QFileDialog.getOpenFileNames(self, "Please slect point cloud file", "", "Point cloud file (*.pcd *.obj *.ply)")
        if not file_path_cloud:
            self.textEdit_2.append(
                '<span style="color: red;"> *  Warning: No point cloud file is selected </span>')
            return
        try:
            Individual_data_set = o3d.geometry.PointCloud()
            a_number = 0
            for file_path in file_path_cloud:
                if file_path.endswith(".ply") or file_path.endswith(".obj") or file_path.endswith(".pcd"):
                    current_point_cloud = o3d.io.read_point_cloud(file_path)
                    if not current_point_cloud.is_empty():
                        Individual_data_set += current_point_cloud
                        a_number += 1
                else:
                    self.textEdit_2.append(
                        '<span style="color: red;"> * Warning: Unsupported file format </span>')
                    return
                if a_number == len(file_path_cloud):
                    self.textEdit_2.append(
                        f'<span style="color: black;"> * Tips: Data merging is successful </span>')
        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: Fail to merge data \n {e}</span>')
        self.Using_VTK_ShowPC(Individual_data_set)
        vis = o3d.visualization.Visualizer()
        vis.create_window()  # 通过 Visualizer 创建窗口
        vis.add_geometry(Individual_data_set)  # 将点云添加到可视化器中
        opt = vis.get_render_option()
        opt.point_size = 2.0
        vis.run()
        vis.clear_geometries()
        vis.destroy_window()
        del vis
        reply_4_1_4 = QMessageBox.question(None, "Tips", "Do you want to save the results?", QMessageBox.Yes | QMessageBox.No)
        if reply_4_1_4 == QMessageBox.Yes:
            # 保存聚类点云结果
            file_types = "PLY Files (*.ply);;PCD Files (*.pcd);;All Files (*)"  # 限定数据保存类型
            options = QFileDialog.Options()
            file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save point cloud file", "", file_types,
                                                                     options=options)
            if file_path:
                # 根据选择的文件过滤器确定保存格式并保存文件
                if selected_filter == "PLY Files (*.ply)":
                    try:
                        o3d.io.write_point_cloud(file_path, Individual_data_set)
                        self.textEdit_2.append(
                            f'<span style="color: black;"> * The point cloud set is saved successfully </span>')
                    except Exception as e:
                        self.textEdit_2.append(
                            f'<span style="color: red;"> * Error: Failed to save point cloud file (PLY format): {str(e)} </span>')
                elif selected_filter == "PCD Files (*.pcd)":
                    try:
                        o3d.io.write_point_cloud(file_path, Individual_data_set,
                                                 write_ascii=True)  # PCD格式保存时可按需设置参数，这里以ASCII格式为例
                        self.textEdit_2.append(
                            f'<span style="color: black;"> * The point cloud set is saved successfully </span>')
                    except Exception as e:
                        self.textEdit_2.append(
                            f'<span style="color: red;"> * Error: Failed to save point cloud file (PCD format):{str(e)} </span>')
                else:
                    self.textEdit_2.append(
                        '<span style="color: red;"> * Warning: Unsupported file format selection, and save operation cancelled </span>')
            else:
                self.textEdit_2.append(
                    f'<span style="color: black;"> * Tips: Cancel operation </span>')
        else:
            self.textEdit_2.append(
                f'<span style="color: black;"> * Tips: Cancel operation </span>')

#########################################************************ 模块4 ************************#########################################
    ## 按键 1 *******（主函数）-------导入数据
    def Input_Model4(self):
        file_path_occurrence, _ = QFileDialog.getOpenFileNames(self, "Please select data", "", "calculated data (*.pkl)")
        self.number_model4 = len(file_path_occurrence)
        if not file_path_occurrence:
            self.textEdit_2.append(
                '<span style="color: red;"> *  Warning: No file is selected </span>')
            return
        try:
            number_model4 = 0
            self.J_Occurrence = {}
            for file_occ in range(self.number_model4):
                if file_path_occurrence[file_occ].endswith(".pkl"):
                    with open(file_path_occurrence[file_occ], 'rb') as file:
                        self.J_Occurrence[f'JointData{file_occ+1}'] = pickle.load(file)
                    number_model4 += 1
                else:
                    self.textEdit_2.append(
                        '<span style="color: red;"> * Warning: Unsupported file format </span>')
                    return
            if number_model4 == file_occ+1:
                self.textEdit_2.append(
                    f'<span style="color: black;"> * Tips: Load successfully </span>')
        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: Fail to load data\n{e} </span>')

    ## 按键 2 *******（主函数）-------计算产状
    def Calculate_Model4(self):
        # 更新日志
        log_text_edit_model4 = self.textEdit_2
        self.Rock_Joint_occurrence ={}
        for m in range(self.number_model4):
            self.Rock_Joint_occurrence[f'Dipdd{m + 1}'] = np.zeros((len(self.J_Occurrence[f'JointData{m+1}']),2))
        for i in range(self.number_model4):
            for j in range(len(self.J_Occurrence[f'JointData{i+1}'])):
                a = len(self.J_Occurrence[f'JointData{i + 1}'])
                JointNor = self.PointCloudVector(self.J_Occurrence[f'JointData{i+1}'][j][:,0:3])#self.J_Occurrence[f'JointData{i+1}'][j][:,0:3]
                self.Rock_Joint_occurrence[f'Dipdd{i + 1}'][j, 0], self.Rock_Joint_occurrence[f'Dipdd{i + 1}'][j, 1] = self.OrientationM(JointNor)
            b = 2
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.figure,ax_model4 = plt.subplots(2,self.number_model4,figsize=(12,6))
        for i in range(self.number_model4):
            ax_model4[0,i].hist(self.Rock_Joint_occurrence[f'Dipdd{i + 1}'][:,0],bins = 30,edgecolor = 'black')
            ax_model4[0,i].set_title(f"Orientation (Joint set {i + 1})",fontsize=16)  # 设置每个子图的标题
            ax_model4[0,i].set_xlabel("Dip(°)",fontsize=14)
            ax_model4[0,i].set_ylabel("Time",fontsize=14)
            mean_value1 = np.mean(self.Rock_Joint_occurrence[f'Dipdd{i + 1}'][:,0])
            max_value1= np.max(self.Rock_Joint_occurrence[f'Dipdd{i + 1}'][:,0])
            min_value1 = np.min(self.Rock_Joint_occurrence[f'Dipdd{i + 1}'][:,0])
            median_value1 = np.median(self.Rock_Joint_occurrence[f'Dipdd{i + 1}'][:,0])

            message_print_1 = (f"* The average dip (Joint set {i+1}) is {mean_value1:.2f}°\n* The median dip (Joint set {i+1}) is {median_value1:.2f}°\n"
                             f"* The maximum dip (Joint set {i+1}) is {max_value1:.2f}°\n"
                             f"* The minimum dip (Joint set {i+1}) is {min_value1:.2f}°")
            log_text_edit_model4.append(
                f'<span style="color: black;"> * {message_print_1} </span>')
            QApplication.processEvents()


        for j in range(self.number_model4):
            ax_model4[1,j].hist(self.Rock_Joint_occurrence[f'Dipdd{j + 1}'][:,1],bins = 30,edgecolor = 'black')
            ax_model4[1,j].set_title(f"Orientation (Joint set {j + 1})",fontsize=16)  # 设置每个子图的标题
            ax_model4[1,j].set_xlabel("Dip direction(°)",fontsize=14)
            ax_model4[1,j].set_ylabel("Time",fontsize=14)
            mean_value2 = np.mean(self.Rock_Joint_occurrence[f'Dipdd{j + 1}'][:,1])
            max_value2= np.max(self.Rock_Joint_occurrence[f'Dipdd{i + 1}'][:,1])
            min_value2 = np.min(self.Rock_Joint_occurrence[f'Dipdd{i + 1}'][:,1])
            median_value2 = np.median(self.Rock_Joint_occurrence[f'Dipdd{i + 1}'][:,1])

            message_print_2 = (f"* The average dip direction (Joint set {j+1}) is {mean_value2:.2f}°\n* The median dip of the {j+1} set of rock joints is {median_value2:.2f}°\n"
                             f"* The maximum dip direction (Joint set {j+1}) is {max_value2:.2f}°\n"
                             f"* The minimum dip direction (Joint set {j+1}) is {min_value2:.2f}°")
            log_text_edit_model4.append(
                f'<span style="color: black;"> * {message_print_2} </span>')
            QApplication.processEvents()

        plt.subplots_adjust(wspace=0.4, hspace=0.5)
        self.canvas_model4 = FigureCanvas(self.figure)
        self.canvas_model4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.form_layout.addRow(self.canvas_model4)
        self.canvas_model4.draw()

    ## 按键 2 *******（副函数1）-------计算产状
    def OrientationM(self,Normals):
        Xcor = Normals[:, 0]
        Ycor = Normals[:, 1]
        Zcor = Normals[:, 2]
        XY_Distance = np.sqrt(Xcor ** 2 + Ycor ** 2)
        if Zcor < 0:
            Dip = math.pi - math.acos(Zcor)
            if Xcor >= 0 and Ycor >= 0:
                Direction = math.asin(Xcor / XY_Distance) + math.pi
            elif Xcor < 0 and Ycor >= 0:
                Direction = 2 * math.pi - math.asin(-Xcor / XY_Distance) - math.pi
            elif Xcor >= 0 and Ycor < 0:
                Direction = math.pi - math.asin(Xcor / XY_Distance) + math.pi
            else:
                Direction = math.pi + math.asin(-Xcor / XY_Distance) - math.pi
        else:
            Dip = math.acos(Zcor)
            if Xcor >= 0 and Ycor >= 0:
                Direction = math.asin(Xcor / XY_Distance)
            elif Xcor < 0 and Ycor >= 0:
                Direction = 2 * math.pi - math.asin(-Xcor / XY_Distance)
            elif Xcor >= 0 and Ycor < 0:
                Direction = math.pi - math.asin(Xcor / XY_Distance)
            else:
                Direction = math.pi + math.asin(-Xcor / XY_Distance)
        Dip = math.degrees(Dip)
        Direction = math.degrees(Direction)
        return Dip, Direction

    ## 按键 2 *******（副函数2）-------计算点法向量
    def PointCloudVector(self,Coordinate):
        b = np.zeros((1, 3)) - np.sum(Coordinate, axis=0)
        a = Coordinate.T @ Coordinate
        N = np.linalg.solve(a, b.T)
        Vect_normal = (N / np.linalg.norm(N)).T
        return Vect_normal

    ## 按键 3 *******（主函数）-------保存数据
    def save_point_data(self):
        try:
            for itk in range(self.number_model4):
                if all(self.Rock_Joint_occurrence[f'Dipdd{itk + 1}'][:, 0] != 0):
                    file_types_save = "XLSX Files (*.xlsx);;Text Files (*.txt);;All Files (*)"  # 添加 TXT 格式
                    options = QFileDialog.Options()
                    file_path, selected_filter = QFileDialog.getSaveFileName(self, "Please save cluster file", "",
                                                                             file_types_save,
                                                                             options=options)
                    if file_path:
                        try:
                            Occurrence_Set = pd.DataFrame(self.Rock_Joint_occurrence[f'Dipdd{itk + 1}'])

                            # 根据选择的文件过滤器保存
                            if selected_filter == "XLSX Files (*.xlsx)":
                                Occurrence_Set.to_excel(file_path, index=False, header=False)
                            elif selected_filter == "Text Files (*.txt)":
                                Occurrence_Set.to_csv(file_path, index=False, header=False, sep='\t')  # 使用制表符分隔
                            else:
                                self.textEdit_2.append(
                                    '<span style="color: red;"> * Warning: Unsupported file format selection, and save operation cancelled </span>')
                        except Exception as e:
                            self.textEdit_2.append(
                                f'<span style="color: red;"> * Error: Fail to save data\n {e} </span>')
                    else:
                        self.textEdit_2.append(
                            f'<span style="color: black;"> * Tips: Cancel operation </span>')
        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: Fail to save data\n {e} </span>')

#########################################************************ 模块5 ************************#########################################
    # 显示点坐标
    def Showing_point_coordinate(self):
        try:
            # 防御式检查
            if not hasattr(self, 'pointcloud') or self.pointcloud is None:
                self.textEdit_2.append('<span style="color:red;"> * Error: No data has been imported </span>')
                return

            # ---------- 防止 stdout/stderr 丢失 ----------
            class DummyIO:
                def write(self, *args, **kwargs): pass

                def flush(self): pass

            import sys, tempfile, os
            import open3d as o3d, numpy as np

            if sys.stdout is None:
                sys.stdout = DummyIO()
            if sys.stderr is None:
                sys.stderr = DummyIO()

            # ---------- 设置 Open3D 环境 ----------
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

            temp_dir = os.path.join(tempfile.gettempdir(), "open3d_temp")
            os.makedirs(temp_dir, exist_ok=True)
            os.chdir(temp_dir)
            os.environ["TMPDIR"] = temp_dir
            os.environ["TEMP"] = temp_dir
            os.environ["TMP"] = temp_dir

            if hasattr(sys, "_MEIPASS"):
                open3d_path = os.path.join(sys._MEIPASS, "open3d")
                if os.path.exists(open3d_path):
                    sys.path.append(open3d_path)

            # ---------- 创建可视化窗口 ----------
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name="Point Coordinate Selection")
            vis.add_geometry(self.pointcloud)
            opt = vis.get_render_option()
            opt.point_size = 3.0
            vis.run()
            Picked_Points_Idex = vis.get_picked_points()
            vis.destroy_window()

            # ---------- 输出结果 ----------
            if not Picked_Points_Idex:
                self.textEdit_2.append('<span style="color:red;"> * No point selected </span>')
                return

            points = np.asarray(self.pointcloud.points)
            PointsGet = points[Picked_Points_Idex]

            for i, p in enumerate(PointsGet):
                info = f"* Point coordinate ({i + 1}): ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"
                self.textEdit_2.append(f'<span style="color:black;">{info}</span>')

            self.textEdit_2.append(
                f'<span style="color:black;"> * Total {len(Picked_Points_Idex)} points selected </span>'
            )

        except Exception as e:
            self.textEdit_2.append(f'<span style="color:red;"> * Error during visualization: {e}</span>')

    ## 测量两点之间的距离
    def Calculate_point_Length(self):
        try:
            # ---------- 基础检查 ----------
            if not hasattr(self, 'pointcloud') or self.pointcloud is None:
                self.textEdit_2.append('<span style="color:red;"> * Error: No data has been imported </span>')
                return
            # ---------- 防止 stdout/stderr 丢失 ----------
            class DummyIO:
                def write(self, *args, **kwargs): pass

                def flush(self): pass

            if sys.stdout is None:
                sys.stdout = DummyIO()
            if sys.stderr is None:
                sys.stderr = DummyIO()

            # ---------- 禁用 Open3D 日志 ----------
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

            # ---------- 设置临时路径和环境变量 ----------
            temp_dir = os.path.join(tempfile.gettempdir(), "open3d_temp")
            os.makedirs(temp_dir, exist_ok=True)
            os.chdir(temp_dir)
            os.environ["TMPDIR"] = temp_dir
            os.environ["TEMP"] = temp_dir
            os.environ["TMP"] = temp_dir

            # ---------- 若为 PyInstaller 打包环境 ----------
            if hasattr(sys, "_MEIPASS"):
                open3d_path = os.path.join(sys._MEIPASS, "open3d")
                if os.path.exists(open3d_path):
                    sys.path.append(open3d_path)

            # ---------- 创建可视化窗口 ----------
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name="Point Distance Measurement")
            vis.add_geometry(self.pointcloud)
            opt = vis.get_render_option()
            opt.point_size = 3.0

            # ---------- 启动交互 ----------
            vis.run()
            Picked_Points_Idex = vis.get_picked_points()
            vis.destroy_window()

            # ---------- 检查选点数量 ----------
            if len(Picked_Points_Idex) != 2:
                self.textEdit_2.append(
                    '<span style="color:red;"> * Error: Please select exactly two points </span>')
                return

            # ---------- 计算距离 ----------
            points = np.asarray(self.pointcloud.points)
            point1 = points[Picked_Points_Idex[0]]
            point2 = points[Picked_Points_Idex[1]]
            distance = np.linalg.norm(point2 - point1)

            # ---------- 输出结果 ----------
            self.textEdit_2.append(
                f'<span style="color:black;"> * Point 1: ({point1[0]:.2f}, {point1[1]:.2f}, {point1[2]:.2f})</span>')
            self.textEdit_2.append(
                f'<span style="color:black;"> * Point 2: ({point2[0]:.2f}, {point2[1]:.2f}, {point2[2]:.2f})</span>')
            self.textEdit_2.append(
                f'<span style="color:black;"> * Distance between points: {distance:.2f} m</span>')
            self.textEdit_2.append(
                f'<span style="color:gray;"> * (Shift + Left click to pick two points)</span>')

        except Exception as e:
            self.textEdit_2.append(f'<span style="color:red;"> * Error during measurement: {e}</span>')

    ## 计算选择点的面积
    def Calculte_selective_area(self):
        try:

            # ---------- 基础检查 ----------
            if not hasattr(self, 'pointcloud') or self.pointcloud is None:
                self.textEdit_2.append('<span style="color:red;"> * Error: No data has been imported </span>')
                return

            # ---------- 防止 stdout/stderr 丢失 ----------
            class DummyIO:
                def write(self, *args, **kwargs): pass

                def flush(self): pass

            if sys.stdout is None:
                sys.stdout = DummyIO()
            if sys.stderr is None:
                sys.stderr = DummyIO()

            # ---------- 禁用 Open3D 日志 ----------
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

            # ---------- 设置临时路径 ----------
            temp_dir = os.path.join(tempfile.gettempdir(), "open3d_temp")
            os.makedirs(temp_dir, exist_ok=True)
            os.chdir(temp_dir)
            os.environ["TMPDIR"] = temp_dir
            os.environ["TEMP"] = temp_dir
            os.environ["TMP"] = temp_dir

            # ---------- 若为 PyInstaller 打包环境 ----------
            if hasattr(sys, "_MEIPASS"):
                open3d_path = os.path.join(sys._MEIPASS, "open3d")
                if os.path.exists(open3d_path):
                    sys.path.append(open3d_path)

            # ---------- 创建可视化窗口 ----------
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name="Select Points for Area Calculation")
            vis.add_geometry(self.pointcloud)
            opt = vis.get_render_option()
            opt.point_size = 3.0

            # ---------- 启动交互 ----------
            vis.run()
            Picked_Points_Idex = vis.get_picked_points()
            vis.destroy_window()

            # ---------- 检查选点数量 ----------
            if len(Picked_Points_Idex) < 3:
                self.textEdit_2.append('<span style="color:red;"> * Error: Please select at least 3 points </span>')
                return

            # ---------- 获取选中点 ----------
            points = np.asarray(self.pointcloud.points)
            PointsGet = points[Picked_Points_Idex]

            # ---------- 计算平面投影并求面积 ----------
            centroid = np.mean(PointsGet, axis=0)
            centered = PointsGet - centroid
            pca = PCA(n_components=3)
            pca.fit(centered)

            # 平面内基向量
            plane_basis = pca.components_[0:2]
            projected_2d = centered @ plane_basis.T

            # 凸包面积（在2D中 hull.volume == 面积）
            hull = ConvexHull(projected_2d)
            area = hull.volume

            # ---------- 输出结果 ----------
            self.textEdit_2.append(
                f'<span style="color:black;"> * Selected points: {len(Picked_Points_Idex)}</span>')
            self.textEdit_2.append(
                f'<span style="color:black;"> * Calculated area: {area:.2f} m² </span>')
            self.textEdit_2.append(
                f'<span style="color:gray;"> * (Shift + Left click to pick ≥3 points)</span>')

        except Exception as e:
            self.textEdit_2.append(f'<span style="color:red;"> * Error during area calculation: {e}</span>')

    ## 计算选择点的拟合平面的产状
    def Calculate_fit_plane_Ora(self):
        try:
            # ---------- 检查点云 ----------
            if not hasattr(self, 'pointcloud') or self.pointcloud is None:
                self.textEdit_2.append('<span style="color:red;"> * Error: No data has been imported </span>')
                return

            # ---------- 防止 stdout/stderr 丢失 ----------
            class DummyIO:
                def write(self, *args, **kwargs): pass

                def flush(self): pass

            if sys.stdout is None:
                sys.stdout = DummyIO()
            if sys.stderr is None:
                sys.stderr = DummyIO()

            # ---------- 禁用 Open3D 日志 ----------
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

            # ---------- 设置临时环境路径 ----------
            temp_dir = os.path.join(tempfile.gettempdir(), "open3d_temp")
            os.makedirs(temp_dir, exist_ok=True)
            os.chdir(temp_dir)
            os.environ["TMPDIR"] = temp_dir
            os.environ["TEMP"] = temp_dir
            os.environ["TMP"] = temp_dir

            # ---------- 若为 PyInstaller 打包环境 ----------
            if hasattr(sys, "_MEIPASS"):
                open3d_path = os.path.join(sys._MEIPASS, "open3d")
                if os.path.exists(open3d_path):
                    sys.path.append(open3d_path)

            # ---------- 创建交互式窗口 ----------
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name="Select Points for Plane Fitting")
            vis.add_geometry(self.pointcloud)
            opt = vis.get_render_option()
            opt.point_size = 3.0
            vis.run()
            Picked_Points_Idex = vis.get_picked_points()
            vis.destroy_window()

            # ---------- 检查点数 ----------
            if len(Picked_Points_Idex) < 3:
                self.textEdit_2.append('<span style="color:red;"> * Error: Please select at least 3 points </span>')
                return

            # ---------- 提取选点坐标 ----------
            points = np.asarray(self.pointcloud.points)
            PointsGet = points[Picked_Points_Idex]

            # ---------- 拟合平面 ----------
            centroid = np.mean(PointsGet, axis=0)
            centered = PointsGet - centroid
            pca = PCA(n_components=3)
            pca.fit(centered)

            # 法向量（第3主成分）
            normal_vector = pca.components_[2]
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            Xcor, Ycor, Zcor = normal_vector
            XY_Distance = np.sqrt(Xcor ** 2 + Ycor ** 2)

            # ---------- 计算倾角 Dip、倾向 Direction ----------
            if Zcor < 0:
                Dip = math.pi - math.acos(Zcor)
                if Xcor >= 0 and Ycor >= 0:
                    Direction = math.asin(Xcor / XY_Distance) + math.pi
                elif Xcor < 0 and Ycor >= 0:
                    Direction = 2 * math.pi - math.asin(-Xcor / XY_Distance) - math.pi
                elif Xcor >= 0 and Ycor < 0:
                    Direction = math.pi - math.asin(Xcor / XY_Distance) + math.pi
                else:
                    Direction = math.pi + math.asin(-Xcor / XY_Distance) - math.pi
            else:
                Dip = math.acos(Zcor)
                if Xcor >= 0 and Ycor >= 0:
                    Direction = math.asin(Xcor / XY_Distance)
                elif Xcor < 0 and Ycor >= 0:
                    Direction = 2 * math.pi - math.asin(-Xcor / XY_Distance)
                elif Xcor >= 0 and Ycor < 0:
                    Direction = math.pi - math.asin(Xcor / XY_Distance)
                else:
                    Direction = math.pi + math.asin(-Xcor / XY_Distance)

            Dip = math.degrees(Dip)
            Direction = math.degrees(Direction)
            Direction = Direction % 360  # 归一化到 0-360°

            # ---------- 输出结果 ----------
            self.textEdit_2.append(f'<span style="color:black;"> * Selected points: {len(Picked_Points_Idex)}</span>')
            self.textEdit_2.append(
                f'<span style="color:black;"> * Dip angle: {Dip:.2f}°</span>')
            self.textEdit_2.append(
                f'<span style="color:black;"> * Dip direction: {Direction:.2f}°</span>')
            self.textEdit_2.append(
                f'<span style="color:gray;"> * (Shift + Left click to pick ≥3 points)</span>')

        except Exception as e:
            self.textEdit_2.append(f'<span style="color:red;"> * Error during plane fitting: {e}</span>')

#########################################************************ 模块6 ************************#########################################
    ## 保存VTK
    def save_vtk_view_as_image(self):
        # 弹出文件保存对话框，支持格式选择
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save the graphic",
            "output",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;TIFF Files (*.tif *.tiff)"
        )

        if not filename:
            self.textEdit_2.append(
                f'<span style="color: black;"> * Tips: Cancel operation </span>')
            return

        # 自动补充扩展名（如果用户没写）
        if '.' not in filename:
            if "PNG" in selected_filter:
                filename += ".png"
            elif "JPEG" in selected_filter:
                filename += ".jpg"
            elif "TIFF" in selected_filter:
                filename += ".tif"


        # 获取 VTK 渲染窗口和渲染器
        render_window = self.vtkWidget.GetRenderWindow()
        renderer = render_window.GetRenderers().GetFirstRenderer()
        renderer.ResetCamera()
        renderer.GetActiveCamera().Zoom(1.5)

        # 多次 Render 保证缓冲区更新
        render_window.Render()
        render_window.Render()

        # 设置窗口截图滤镜
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(render_window)
        window_to_image_filter.SetScale(3)  # 提高分辨率
        window_to_image_filter.SetInputBufferTypeToRGB()  # 避免使用 RGBA 导致图像变暗
        window_to_image_filter.ReadFrontBufferOff()  # 使用后缓冲区更可靠
        window_to_image_filter.Update()

        # 根据文件扩展名选择图像写入器
        extension = filename.split('.')[-1].lower()
        if extension == "png":
            writer = vtk.vtkPNGWriter()
        elif extension in ["tif", "tiff"]:
            writer = vtk.vtkTIFFWriter()
        elif extension in ["jpg", "jpeg"]:
            writer = vtk.vtkJPEGWriter()
        else:
            self.textEdit_2.append(
                '<span style="color: red;"> * Warning: Unsupported graph format, please use png, jpg, or tif </span>')
            return

        # 保存图像
        writer.SetFileName(filename)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()
        self.textEdit_2.append(
            f'<span style="color: black;"> * Success: Image saved to {filename} </span>')

    ## 保存Open3D
    def save_display_3d_model(self):
        try:
            if self.pointcloud:
                filename, _ = QFileDialog.getSaveFileName(
                    None,
                    "Save image",
                    "",
                    "PNG Files (*.png);;JPEG Files (*.jpg);;TIFF Files (*.tif);;All Files (*)"
                )
                if filename:
                    viewer = PointCloudViewer(self.pointcloud,ui = self)
                    viewer.screenshot_path = filename
                    viewer.show()

        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: The point cloud data set is empty\n {e} </span>')

    ## 保存Matplob
    def save_matlab_image(self):
        try:
            if self.figure:
                file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot", "",
                                                           "PNG Files (*.png);;JPG Files (*.jpg);;TIFF Files (*.tif)")
                try:
                    if file_path:
                        self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                        self.textEdit_2.append(
                            f'<span style="color: black;"> * The plot has been saved as: {file_path} </span>')
                except Exception as e:
                    self.textEdit_2.append(
                        f'<span style="color: red;"> * Error: The file_path is empty\n {e} </span>')
        except Exception as e:
            self.textEdit_2.append(
                f'<span style="color: red;"> * Error: The drawing window is empty\n {e} </span>')


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 启动应用
    win1 = Operate_Login()  # 创建 Operate 类实例，自动加载 UI
    win1.show()  # 显示窗口
    win2 = Operate_Dealing()
    def on_login_button_clicked():
        win2.open() #(win1.ZhangH,win1.Mima)  # 调用 Operate_Dealing 中的 open 方法
        win1.close()
    win1.pushButton_4.clicked.connect(on_login_button_clicked)
    sys.exit(app.exec_())  # 启动应用的事件循环
