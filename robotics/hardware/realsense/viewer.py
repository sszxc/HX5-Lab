"""Live RealSense viewer: color | colorized depth side by side. q/Esc quits, s saves a snapshot."""


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--max-depth", type=float, default=2.0, help="meters mapped to the top of the colormap")
    parser.add_argument("--out", default="out/realsense")
    args = parser.parse_args()

    import time
    from pathlib import Path

    import cv2
    import numpy as np
    import pyrealsense2 as rs

    dev = rs.context().query_devices()[0]
    print(dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number),
          "USB", dev.get_info(rs.camera_info.usb_type_descriptor))

    cfg = rs.config()
    if not args.no_color:
        cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if not args.no_depth:
        cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color) if not (args.no_color or args.no_depth) else None

    win = "realsense"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    t_last, n = time.time(), 0
    fps_txt = ""
    try:
        while True:
            frames = pipe.wait_for_frames(5000)
            if align:
                frames = align.process(frames)
            panels = []
            if not args.no_color:
                panels.append(np.asanyarray(frames.get_color_frame().get_data()))
            if not args.no_depth:
                d = np.asanyarray(frames.get_depth_frame().get_data()) * depth_scale
                d8 = np.clip(d / args.max_depth * 255, 0, 255).astype(np.uint8)
                dc = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
                dc[d == 0] = 0  # invalid depth -> black
                panels.append(dc)
            img = np.hstack(panels)

            n += 1
            if time.time() - t_last >= 1.0:
                fps_txt = f"{n / (time.time() - t_last):.1f} fps"
                t_last, n = time.time(), 0
            cv2.putText(img, fps_txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("s"):
                out = Path(args.out)
                out.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(out / f"{stamp}.png"), img)
                if not args.no_depth:
                    np.save(out / f"{stamp}_depth_m.npy", d.astype(np.float32))
                print("saved", out / stamp)
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
