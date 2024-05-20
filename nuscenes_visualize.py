"""
文件名: nuscenes_visualize.py
作者: EdvinYang
日期: 2024-04-08
描述: 该文件包含用于从NuScenes数据集加载和显示图像及其对应标注的功能。
功能: 
  - 加载NuScenes数据集的特定图片
  - 显示图片及其标注信息
  - 提供数据集内图片搜索功能
用途: 用于研究和展示NuScenes数据集中的各种传感器数据。
"""

from nuscenes.nuscenes import NuScenes
import os
import cv2
import numpy as np
# from nuscenes.utils.geometry_utils import transform_matrix, view_points, box_in_image, BoxVisibility
from pyquaternion import Quaternion
import yaml


class nuscenes_visualize:
    def __init__(self, nuscenes_path: str):
        # 初始化函数，用于初始化 NuScenes 数据集和所需的参数。
        """
        初始化函数，用于初始化 NuScenes 数据集和所需的参数。
    
        Args:
            nuscenes_path (str): NuScenes 数据集的路径。
    
        Returns:
            None
    
        """
        # 创建一个 NuScenes 对象，指定版本和数据集路径
        self.nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_path)

        # 定义检测类别集合
        self.detection_class = {'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian',
                   'traffic_cone', 'barrier'}

        # 定义 LIDAR 数据的最小和最大 X 坐标
        self.LIDAR_MIN_X = -20
        self.LIDAR_MAX_X = 30

        # 定义 LIDAR 数据的最小和最大 Y 坐标
        self.LIDAR_MIN_Y = -35
        self.LIDAR_MAX_Y = 50
    def get_obj3d_from_annotation(self, ann, ego_data, calib_data):
        """
        从注释中获取3D对象信息
        
        Args:
            ann (dict): 包含对象信息的注释字典
            ego_data (dict): 包含ego车辆信息的字典
            calib_data (dict): 包含传感器校准信息的字典
        
        Returns:
            dict: 包含对象3D框信息的字典，若对象类别不在检测类别中则返回None
        
        """
        obj_ann = dict()

        # 1. 类别
        # 获取注释中的类别名称，并过滤出存在于检测类别中的类别
        obj_type = set(ann['category_name'].split('.')).intersection(self.detection_class)
        if len(obj_type) == 0:
            return None
        else:
            obj_type = obj_type.pop()

        # 2. 3D框
        # global frame
        # 从注释中获取3D框的中心点和旋转量
        center = np.array(ann['translation'])
        orientation = np.array(ann['rotation'])

        # 从global frame转换到ego vehicle frame
        # 从全局坐标系转换到车辆坐标系
        quaternion = Quaternion(ego_data['rotation']).inverse
        center -= np.array(ego_data['translation'])
        center = np.dot(quaternion.rotation_matrix, center)
        orientation = quaternion * orientation

        # 从ego vehicle frame转换到sensor frame
        # 从车辆坐标系转换到传感器坐标系
        quaternion = Quaternion(calib_data['rotation']).inverse
        center -= np.array(calib_data['translation'])
        center = np.dot(quaternion.rotation_matrix, center)
        orientation = quaternion * orientation

        # 根据中心点和旋转量生成3D框
        x, y, z = center
        w, l, h = ann['size']
        x_corners = l / 2 * np.array([-1, 1, 1, -1, -1, 1, 1, -1])
        y_corners = w / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        z_corners = h / 2 * np.array([-1, -1, -1, -1, 1, 1, 1, 1])

        # 初始中心为(0, 0, 0)
        box3d = np.vstack((x_corners, y_corners, z_corners))

        # 旋转3D框
        box3d = np.dot(orientation.rotation_matrix, box3d)

        # 平移3D框
        box3d[0, :] = box3d[0, :] + x
        box3d[1, :] = box3d[1, :] + y
        box3d[2, :] = box3d[2, :] + z

        # 设置3D框的数据类型、类型和3D框本身到obj_ann字典中
        obj_ann['data_type'] = ann['data_type']
        obj_ann['type'] = obj_type
        obj_ann['box'] = box3d

        return obj_ann
    
    def project_obj2image(self, obj3d_list, intrinsic):
        """
        将三维对象列表投影到二维图像上，并返回二维对象列表。
        
        Args:
            obj3d_list (list): 三维对象列表，每个对象包含'data_type'、'type'、'box'等属性。
            intrinsic (np.ndarray): 3x3的内在参数矩阵。
        
        Returns:
            list: 二维对象列表，每个对象包含'data_type'、'type'、'box'和'depth'等属性。
        
        """
        # 存储转换后的二维对象列表
        obj2d_list = list()

        # 创建变换矩阵，初始化为4x4的单位矩阵
        trans_mat = np.eye(4)
        # 设置变换矩阵的前3x3部分为内在参数
        trans_mat[:3, :3] = np.array(intrinsic)

        # 遍历每个三维对象
        for obj in obj3d_list:
            # step1: 判断目标是否在图像内(相机坐标系z朝前, x朝右)
            # 判断对象的box是否满足z轴大于0.1，即对象在相机前方
            # step1: 判断目标是否在图像内(相机坐标系z朝前, x朝右)
            in_front = obj['box'][2, :] > 0.1
            if all(in_front) is False:
                continue

            # step2: 转换到像素坐标系
            # 获取对象的box点集
            # step2: 转换到像素坐标系
            points = obj['box']
            # 将点的齐次坐标扩展为4维，并添加一列全为1的向量
            points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)
            # 进行坐标变换，将点从世界坐标系转换到相机坐标系
            transformed_points = np.dot(trans_mat, points)
            # 进行投影变换，将点从相机坐标系转换到像素坐标系
            projected_points = transformed_points[:3, :] / transformed_points[2, :]

            # step3: 计算深度信息，使用中心点的Z坐标
            # 计算box的中心点
            # step3: 计算深度信息，使用中心点的Z坐标
            center_point = np.mean(points[:3, :], axis=1)
            # 提取中心点的Z坐标作为深度信息
            depth = center_point[2]  # 这里假设中心点在物体的中间

            # 创建二维对象字典，包含数据类型、类型、box和深度信息
            obj2d = {'data_type': obj['data_type'], 'type': obj['type'], 'box': projected_points, 'depth': depth}
            # 将二维对象添加到列表中
            obj2d_list.append(obj2d)

        return obj2d_list
    
    
    def plot_annotation_info_camera_only(self, camera_img, obj_list):
        """
        在相机图像上绘制3D包围盒和深度信息文本。
        
        Args:
            camera_img (numpy.ndarray): 相机图像，形状为(H, W, C)，其中H为图像高度，W为图像宽度，C为图像通道数。
            obj_list (list): 对象列表，其中每个对象为一个字典，包含以下键值对：
                - 'type' (str): 对象类型，如'car'、'pedestrian'等。
                - 'box' (numpy.ndarray): 对象在相机坐标系下的3D包围盒顶点坐标，形状为(2, 8)，其中每行分别表示x和y坐标。
                - 'depth' (float): 对象距离相机的深度，单位为米。
                - 'data_type' (str): 数据类型，如'gt'表示真实标注，'pred'表示预测结果。
        
        Returns:
            None
        
        """
        # 遍历对象列表
        for obj in obj_list:
            # 获取对象类型
            obj_type = obj['type']
            # 确保坐标为整数类型
            box = obj['box'].astype(int)  # 确保坐标为整数类型
            # 获取深度信息
            depth = obj['depth']

            # 设置颜色和线条粗细
            # 如果是车辆
            if obj_type == 'car':
                # 设置颜色为黄色
                color = (0, 255, 255)  # 黄色
                thickness = 1
            # 如果是大型车辆
            elif obj_type in ['truck', 'trailer', 'bus', 'construction_vehicle']:
                # 设置颜色为橙色
                color = (64, 128, 255)  # 橙色
                thickness = 1
            # 如果是行人
            elif obj_type == 'pedestrian':
                # 设置颜色为绿色
                color = (0, 255, 0)  # 绿色
                thickness = 1
            # 如果是骑行类
            elif obj_type in ['bicycle', 'motorcycle']:
                # 设置颜色为蓝色
                color = (255, 255, 0)  # 蓝色
                thickness = 1
            else:
                # 其他类型不绘制
                continue  # 其他类型不绘制

            # 设置检测结果和真实标注的颜色差异
            # 如果是真实标注
            if obj['data_type'] == 'gt':
                # 设置颜色为白色
                color = (255, 255, 255)  # 白色表示真实标注
                thickness = 2

            # 绘制3D包围盒到2D图像
            for i in range(4):
                j = (i + 1) % 4
                # 绘制下底面
                # 绘制下底面线条
                cv2.line(camera_img, (box[0, i], box[1, i]), (box[0, j], box[1, j]), color, thickness)
                # 绘制上底面
                # 绘制上底面线条
                cv2.line(camera_img, (box[0, i + 4], box[1, i + 4]), (box[0, j + 4], box[1, j + 4]), color, thickness)
                # 绘制垂直边
                # 绘制垂直边线条
                cv2.line(camera_img, (box[0, i], box[1, i]), (box[0, i + 4], box[1, i + 4]), color, thickness)

            # 计算中心点坐标
            center_x = np.mean(box[0, :4])
            center_y = np.mean(box[1, :4])
            # 在图像上添加深度信息文本
            cv2.putText(camera_img, f'{depth:.0f}m', (int(center_x), int(center_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def visualize_one_sample_single(self, sample, camera_name='CAM_FRONT', results=None, visible_level=1, scale_ratio=1, save_dir=None):
        """
        绘制单个样本的图像并展示或保存
        
        Args:
            sample (dict): 样本数据
            camera_name (str, optional): 相机名称. Defaults to 'CAM_FRONT'.
            results (list, optional): 检测结果列表. Defaults to None.
            visible_level (int, optional): 可见性级别. Defaults to 1.
            scale_ratio (float, optional): 图像缩放比例. Defaults to 1.
            save_dir (str, optional): 图像保存路径. Defaults to None.
        
        Returns:
            None
        
        """
        data_root = self.nusc.dataroot
        # print("data_root: ", data_root)

        # 获取指定相机的数据
        camera_data = self.nusc.get('sample_data', sample['data'][camera_name])

        # 获取图像
        img_path = os.path.join(data_root, camera_data['filename'])
        # print("img_path: ", img_path)
        
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        cv2.putText(img, text=camera_name[4:], org=(50, 80), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=4.0, thickness=3, color=(0, 0, 255))

        # 处理注释信息
        anns_info = []
        if results is not None:
            for res in results:
                if res['detection_score'] < 0.5:
                    continue
                res['visibility_token'] = 4
                res['category_name'] = res['detection_name']
                res['data_type'] = 'result'
                anns_info.append(res)
        else:
            for token in sample['anns']:
                anns_data = self.nusc.get('sample_annotation', token)
                anns_data['data_type'] = 'result'
                anns_info.append(anns_data)

        # 获取校准数据
        calib_data = self.nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
        ego_data = self.nusc.get('ego_pose', camera_data['ego_pose_token'])
        obj3d_list = []
        for ann in anns_info:
            if int(ann['visibility_token']) < visible_level:
                continue
            obj = self.get_obj3d_from_annotation(ann, ego_data, calib_data)
            if obj is not None:
                obj3d_list.append(obj)
        
        # 投影3D标注到图像上
        obj2d_list = self.project_obj2image(obj3d_list, calib_data['camera_intrinsic'])
        self.plot_annotation_info_camera_only(img, obj2d_list)

        # 调整图像大小
        img_h, img_w, _ = img.shape
        img = cv2.resize(img, (int(img_w * scale_ratio), int(img_h * scale_ratio)))

        # 显示或保存图像
        if save_dir is None:
            cv2.imshow('Visualization', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f'{camera_name}.jpg')
            cv2.imwrite(save_path, img)
    
    def find_image_index(self, nusc, image_filename, camera_channel='CAM_FRONT'):
        """
        根据图像文件名在NUScenes数据集中查找对应的样本索引。
        
        Args:
            nusc (nuscenes.Nuscenes): NUScenes数据集对象
            image_filename (str): 图像文件名
            camera_channel (str, optional): 相机通道名称。默认为'CAM_FRONT'。
        
        Returns:
            int: 样本索引，如果图像文件不存在则返回-1。
        
        """
        # 遍历所有样本
        for index, sample in enumerate(nusc.sample):
            # 获取指定相机通道的数据token
            camera_token = sample['data'][camera_channel]
            camera_data = nusc.get('sample_data', camera_token)
            
            # 检查文件名是否匹配
            if camera_data['filename'].endswith(image_filename):
                print(f"Found image '{image_filename}' at sample index: {index}")
                return index
        
        print(f"Image '{image_filename}' not found in the dataset.")
        return -1  # 返回-1表示未找到
    
    def get_annotation_info(self, sample):
        """
        获取样本的注释信息并打印。

        Args:
            sample (dict): 样本数据。

        Returns:
            None
        """
        anns_info = []
        for token in sample['anns']:
            anns_data = self.nusc.get('sample_annotation', token)
            anns_info.append(anns_data)
        
        for ann in anns_info:
            category_name = ann['category_name']
            translation = ann['translation']
            size = ann['size']
            rotation = ann['rotation']
            num_lidar_pts = ann['num_lidar_pts']
            num_radar_pts = ann['num_radar_pts']
            print(f"Category: {category_name}")
            print(f"Translation: {translation}")
            print(f"Size: {size}")
            print(f"Rotation: {rotation}")
            print(f"Number of Lidar Points: {num_lidar_pts}")
            print(f"Number of Radar Points: {num_radar_pts}")
            print("-" * 30)


if '__name__==__main__':
    # 读取YAML配置文件
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    nuscenes_path = config['nuscenes_path']
    image_filename = config['image_filename']
    camera_channel = config['camera_channel']
    save_directory = config['save_directory']

    nusc = NuScenes(version='v1.0-mini', dataroot=nuscenes_path)
    visualizer = nuscenes_visualize(nuscenes_path)
    index = visualizer.find_image_index(nusc, image_filename, camera_channel)
    print("index: ", index)
    
    if index == -1:
        print("Image not found in the dataset.")
    else:
        sample = nusc.sample[index]
        visualizer.get_annotation_info(sample)
        visualizer.visualize_one_sample_single(sample, camera_name=camera_channel, results=None, save_dir=save_directory)