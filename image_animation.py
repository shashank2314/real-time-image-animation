import imageio.v2 as imageio  # ✅ fixes DeprecationWarning
import torch
from tqdm import tqdm
from animate import normalize_kp
from demo import load_checkpoints
import numpy as np
from skimage import img_as_ubyte
from skimage.transform import resize
import cv2
import os
import argparse
from google.colab.patches import cv2_imshow  # ✅ for inline display in Colab

# -------------------- Argument Parser --------------------
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image", required=True, help="Path to image to animate")
ap.add_argument("-c", "--checkpoint", required=True, help="Path to checkpoint")
ap.add_argument("-v", "--input_video", required=False, help="Path to video input (optional)")
ap.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")
args = vars(ap.parse_args())

# -------------------- Load Data --------------------
print("[INFO] Loading source image and checkpoint...")
source_path = args["input_image"]
checkpoint_path = args["checkpoint"]
video_path = args["input_video"] if args["input_video"] else None
cpu = args["cpu"]

source_image = imageio.imread(source_path)
source_image = resize(source_image, (256, 256))[..., :3]

# -------------------- Load Model --------------------
generator, kp_detector = load_checkpoints(
    config_path="config/vox-256.yaml",
    checkpoint_path=checkpoint_path,
    cpu=cpu,
)

# -------------------- Output Directory --------------------
if not os.path.exists("output"):
    os.mkdir("output")

relative = True
adapt_movement_scale = True

# -------------------- Load Video or Camera --------------------
if video_path:
    cap = cv2.VideoCapture(video_path)
    print("[INFO] Loading driving video...")
else:
    cap = cv2.VideoCapture(0)
    print("[INFO] Using webcam as driving video (if available)...")

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out1 = cv2.VideoWriter("output/test.avi", fourcc, 12, (256 * 3, 256), True)
cv2_source = cv2.cvtColor(source_image.astype("float32"), cv2.COLOR_BGR2RGB)

# -------------------- Animation Loop --------------------
with torch.no_grad():
    predictions = []
    source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
    if not cpu:
        source = source.cuda()
    kp_source = kp_detector(source)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] No more frames or webcam not available.")
            break

        frame = cv2.flip(frame, 1)

        if not video_path:
            x, y, w, h = 143, 87, 322, 322
            frame = frame[y:y + h, x:x + w]

        frame1 = resize(frame, (256, 256))[..., :3]

        if count == 0:
            source_image1 = frame1
            source1 = torch.tensor(source_image1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            kp_driving_initial = kp_detector(source1)

        frame_test = torch.tensor(frame1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        driving_frame = frame_test
        if not cpu:
            driving_frame = driving_frame.cuda()

        kp_driving = kp_detector(driving_frame)
        kp_norm = normalize_kp(
            kp_source=kp_source,
            kp_driving=kp_driving,
            kp_driving_initial=kp_driving_initial,
            use_relative_movement=relative,
            use_relative_jacobian=relative,
            adapt_movement_scale=adapt_movement_scale,
        )

        out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
        im = np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0]
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        joined_frame = np.concatenate((cv2_source, im, frame1), axis=1)

        # ✅ Display inline in Colab
        cv2_imshow(img_as_ubyte(joined_frame))

        out1.write(img_as_ubyte(joined_frame))
        count += 1

        # Stop after few frames (for Colab demo)
        if count > 10:
            break

    cap.release()
    out1.release()

print("[INFO] Animation complete! Video saved to output/test.avi ✅")
