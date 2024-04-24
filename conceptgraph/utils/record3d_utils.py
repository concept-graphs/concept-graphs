import numpy as np
from record3d import Record3DStream
from threading import Event

import cv2
import os
import PyQt5
import torch

# Set the QT_QPA_PLATFORM_PLUGIN_PATH environment variable
pyqt_plugin_path = os.path.join(os.path.dirname(PyQt5.__file__), "Qt", "plugins", "platforms")
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = pyqt_plugin_path

from conceptgraph.utils.geometry import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion

class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])
        
    def correct_pose(self, transformation_matrix):
        '''
        This function corrects the pose of the camera by flipping the y and z axes.
        Since ARKit uses a coordinate frame where two of the axes are flipped in sign.
        '''
        

        # Define the transformation matrix P
        P = torch.tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ]).float()

        # Convert transformation_matrix to a tensor for matrix multiplication
        transformation_tensor = torch.from_numpy(transformation_matrix).float()

        # Apply P to transformation_tensor
        final_transformation = P @ transformation_tensor @ P.T
        
        return final_transformation
        
    def get_frame_data(self):
        if self.event.wait(timeout=1):  # Timeout ensures it doesn't wait forever if no frame is ready
            rgb = self.session.get_rgb_frame()
            depth = self.session.get_depth_frame()
            # intrinsics = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
            camera_pose = self.session.get_camera_pose()  # Quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])

            # print(intrinsic_mat)

            # You can now e.g. create point cloud by projecting the depth map using the intrinsic matrix.
            
             # Postprocess it
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Resize depth to match RGB resolution
            # Note: Ensure depth is a single-channel image before resizing
            depth_resized = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # reformat the intrinsic mat
            intrinsics_full = np.eye(4)
            intrinsics_full[:3, :3] = intrinsic_mat
            intrinsics_tensor =  torch.from_numpy(intrinsics_full).float()
            
            quaternion = [camera_pose.qx, camera_pose.qy, camera_pose.qz, camera_pose.qw]
            rotation_matrix = quaternion_to_rotation_matrix(quaternion)
            # rotation_matrix[1, :] = -rotation_matrix[1, :]  # Negate the second row to invert the y-axis

            transformation_matrix = np.eye(4)  # Create a 4x4 identity matrix
            transformation_matrix[:3, :3] = rotation_matrix  # Set the top-left 3x3 to the rotation matrix
            transformation_matrix[:3, 3] = [camera_pose.tx, camera_pose.ty, camera_pose.tz]  # Set the translation
            # Uncomment the line below to invert the y-component when needed
            # transformation_matrix[1, 3] *= -1  # Negate the y-component in the translation vector
            
            final_transformation = self.correct_pose(transformation_matrix)
 

            # Convert back to numpy if necessary
            final_transformation_matrix = final_transformation.numpy()

            self.event.clear()
            return rgb, depth_resized, intrinsics_tensor, final_transformation_matrix
        return None, None, None, None