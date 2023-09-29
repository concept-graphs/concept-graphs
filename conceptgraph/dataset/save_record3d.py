import argparse
import os
from threading import Event
import time

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from record3d import Record3DStream
from scipy.spatial.transform import Rotation
from tqdm import trange


def write_color(outpath, img):
    cv2.imwrite(outpath, img)


def write_depth(outpath, depth):
    depth = depth * 1000
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(outpath)


def write_pose(outpath, pose):
    # quat_trans: Quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])
    # NB: Record3D / scipy use "scalar-last" format quaternions (x y z w)
    # https://fzheng.me/2017/11/12/quaternion_conventions_en/
    # pose = np.asarray([pose.qx, pose.qy, pose.qz, pose.qw, pose.tx, pose.ty, pose.tz])

    c2w = np.zeros((4, 4))
    c2w[3, 3] = 1.0
    c2w[:3, :3] = Rotation.from_quat(pose[:4]).as_matrix()
    c2w[:3, 3] = pose[4:]
    np.save(outpath, c2w.astype(np.float32))


class DemoApp:
    def __init__(self, savedir_name="saved-record3d", seq_name="debug"):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        self.imgs = []
        self.depths = []
        self.intrinsics = None
        self.poses = []
        self.savedir_name = savedir_name
        self.seq_name = seq_name

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print("Stream stopped")
        # seq_ts = f"{self.seq_name}_{str(int(time.time()))}"
        # savedir = os.path.join(self.savedir_name, seq_ts)
        savedir = os.path.join(self.savedir_name, self.seq_name)
        # add timestamp to savedir
        # savedir = os.path.join(savedir, str(int(time.time())))

        savedir_rgb = os.path.join(savedir, "rgb")
        savedir_depth = os.path.join(savedir, "depth")
        savedir_poses = os.path.join(savedir, "poses")

        os.makedirs(savedir, exist_ok=True)
        os.makedirs(savedir_rgb, exist_ok=True)
        os.makedirs(savedir_depth, exist_ok=True)
        os.makedirs(savedir_poses, exist_ok=True)
        # print(f"Line 69, savedir_poses: {savedir_poses}")

        self.cfg = {}
        self.cfg["dataset_name"] = "record3d"
        self.cfg["camera_params"] = {}
        self.cfg["camera_params"]["image_height"] = self.imgs[0].shape[0]
        self.cfg["camera_params"]["image_width"] = self.imgs[0].shape[1]
        self.cfg["camera_params"]["fx"] = self.intrinsics[0, 0].item()
        self.cfg["camera_params"]["fy"] = self.intrinsics[1, 1].item()
        self.cfg["camera_params"]["cx"] = self.intrinsics[0, 2].item()
        self.cfg["camera_params"]["cy"] = self.intrinsics[1, 2].item()
        self.cfg["camera_params"]["png_depth_scale"] = 1000.0
        print(self.cfg)
        with open(os.path.join(savedir, "dataconfig.yaml"), "w") as f:
            yaml.dump(self.cfg, f)

        for _i in trange(len(self.imgs)):
            fname = str(_i).zfill(5)
            savefile = os.path.join(savedir_rgb, f"{fname}.png")
            write_color(savefile, self.imgs[_i])
            savefile = os.path.join(savedir_depth, f"{fname}.png")
            write_depth(savefile, self.depths[_i])
            savefile = os.path.join(savedir_poses, f"{fname}.npy")
            write_pose(savefile, self.poses[_i])
        
        cv2.destroyallwindows()
        exit()

    def connect_to_device(self, dev_idx):
        print("Searching for devices")
        devs = Record3DStream.get_connected_devices()
        print("{} device(s) found".format(len(devs)))
        for dev in devs:
            print("\tID: {}\n\tUDID: {}\n".format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError(
                "Cannot connect to device #{}, try different index.".format(dev_idx)
            )

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array(
            [[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]]
        )

    def start_processing_stream(self):
        while True:
            # print(f"Line 118, True: {True}")
            self.event.wait()  # Wait for new frame to arrive

            # Copy the newly arrived RGBD frame
            depth = self.session.get_depth_frame()
            # print(f"Line 123, depth: {depth}")
            rgb = self.session.get_rgb_frame()
            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(
                self.session.get_intrinsic_mat()
            )
            pose = (
                self.session.get_camera_pose()
            )  # Quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])

            # print(intrinsic_mat)

            # You can now e.g. create point cloud by projecting the depth map using the intrinsic matrix.

            # Postprocess it
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if self.intrinsics is None:
                self.intrinsics = intrinsic_mat
            self.imgs.append(rgb)
            self.depths.append(depth)
            self.poses.append(
                [pose.qx, pose.qy, pose.qz, pose.qw, pose.tx, pose.ty, pose.tz]
            )

            # Show the RGBD Stream
            cv2.imshow("RGB", rgb)
            cv2.imshow("Depth", depth)
            cv2.waitKey(1)

            self.event.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir_name", type=str, default="saved-record3d", help="Dir to save data to")
    parser.add_argument("--seq_name", type=str, default="debug", help="Unique name for captured trajectory")
    args = parser.parse_args()

    app = DemoApp(savedir_name=args.savedir_name, seq_name=args.seq_name)
    
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()
