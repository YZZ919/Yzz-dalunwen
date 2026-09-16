#!/usr/bin/env python3
"""Replay an ASL-format EuRoC sequence directly as ROS1 topics.

This avoids creating another multi-gigabyte rosbag.  It publishes the left
camera and IMU streams consumed by the VINS-Mono EuRoC launch file.  When an
enhanced-image directory is supplied it also publishes timestamp-identical
enhanced frames for the clean-room dual-branch tracker.
"""
import argparse
import csv
import os
import sys
import time

import cv2
import rospy
from sensor_msgs.msg import Image, Imu


def rows(path):
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if not row or row[0].lstrip().startswith("#"):
                continue
            yield row


def image_events(root):
    path = os.path.join(root, "mav0", "cam0", "data.csv")
    data_dir = os.path.join(root, "mav0", "cam0", "data")
    for row in rows(path):
        yield int(row[0]), os.path.join(data_dir, row[1])


def imu_events(root):
    path = os.path.join(root, "mav0", "imu0", "data.csv")
    for row in rows(path):
        yield int(row[0]), tuple(float(x) for x in row[1:7])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sequence", help="ASL EuRoC sequence directory")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="wall-clock replay speed (1.0 is real time; 0 means no pacing)")
    ap.add_argument("--start", type=float, default=0.0,
                    help="skip this many seconds from the first sensor timestamp")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after this many sequence seconds (0 means all)")
    ap.add_argument("--enhanced-dir", default="",
                    help="optional directory of enhanced PNGs named like cam0/data entries")
    ap.add_argument("--enhanced-topic", default="/cam0/image_enhanced",
                    help="ROS topic used with --enhanced-dir")
    args = ap.parse_args()
    root = os.path.abspath(args.sequence)
    if not os.path.isfile(os.path.join(root, "mav0", "cam0", "data.csv")):
        raise SystemExit("not an ASL-format EuRoC sequence: " + root)

    rospy.init_node("euroc_asl_replayer", anonymous=True)
    image_pub = rospy.Publisher("/cam0/image_raw", Image, queue_size=30)
    enhanced_dir = os.path.abspath(args.enhanced_dir) if args.enhanced_dir else ""
    if enhanced_dir and not os.path.isdir(enhanced_dir):
        raise SystemExit("enhanced image directory not found: " + enhanced_dir)
    enhanced_pub = (rospy.Publisher(args.enhanced_topic, Image, queue_size=30)
                    if enhanced_dir else None)
    imu_pub = rospy.Publisher("/imu0", Imu, queue_size=1000)

    images = image_events(root)
    imus = imu_events(root)
    try:
        next_image = next(images)
        next_imu = next(imus)
    except StopIteration:
        raise SystemExit("empty EuRoC stream")

    first_ns = min(next_image[0], next_imu[0])
    start_ns = first_ns + int(max(0.0, args.start) * 1e9)
    end_ns = (start_ns + int(args.duration * 1e9)) if args.duration > 0 else None
    wall_start = time.monotonic()
    published_images = 0
    published_enhanced = 0
    published_imus = 0
    skipped = 0

    while not rospy.is_shutdown():
        if next_image is None and next_imu is None:
            break
        use_image = next_imu is None or (next_image is not None and next_image[0] <= next_imu[0])
        if use_image:
            stamp_ns, image_path = next_image
            try:
                next_image = next(images)
            except StopIteration:
                next_image = None
            if stamp_ns < start_ns:
                skipped += 1
                continue
            if end_ns is not None and stamp_ns > end_ns:
                break
            if args.speed > 0:
                target = (stamp_ns - start_ns) * 1e-9 / args.speed
                delay = target - (time.monotonic() - wall_start)
                if delay > 0:
                    rospy.sleep(delay)
            frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if frame is None:
                rospy.logwarn("cannot read image %s", image_path)
                continue
            msg = Image()
            msg.header.stamp = rospy.Time.from_sec(stamp_ns * 1e-9)
            msg.header.frame_id = "cam0"
            msg.height, msg.width = frame.shape[:2]
            msg.encoding = "mono8"
            msg.is_bigendian = 0
            msg.step = msg.width
            msg.data = frame.tobytes()
            image_pub.publish(msg)
            published_images += 1
            if enhanced_pub is not None:
                enhanced_path = os.path.join(enhanced_dir, os.path.basename(image_path))
                enhanced = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)
                if enhanced is None:
                    raise SystemExit("enhanced image missing or unreadable: " + enhanced_path)
                if enhanced.shape != frame.shape:
                    raise SystemExit("enhanced image shape differs from raw: " + enhanced_path)
                enhanced_msg = Image()
                enhanced_msg.header = msg.header
                enhanced_msg.height, enhanced_msg.width = enhanced.shape[:2]
                enhanced_msg.encoding = "mono8"
                enhanced_msg.is_bigendian = 0
                enhanced_msg.step = enhanced_msg.width
                enhanced_msg.data = enhanced.tobytes()
                enhanced_pub.publish(enhanced_msg)
                published_enhanced += 1
            if published_images % 100 == 0:
                rospy.loginfo("published %d images, %d IMU samples", published_images, published_imus)
        else:
            stamp_ns, values = next_imu
            try:
                next_imu = next(imus)
            except StopIteration:
                next_imu = None
            if stamp_ns < start_ns:
                skipped += 1
                continue
            if end_ns is not None and stamp_ns > end_ns:
                break
            if args.speed > 0:
                target = (stamp_ns - start_ns) * 1e-9 / args.speed
                delay = target - (time.monotonic() - wall_start)
                if delay > 0:
                    rospy.sleep(delay)
            wx, wy, wz, ax, ay, az = values
            msg = Imu()
            msg.header.stamp = rospy.Time.from_sec(stamp_ns * 1e-9)
            msg.header.frame_id = "imu0"
            msg.angular_velocity.x = wx
            msg.angular_velocity.y = wy
            msg.angular_velocity.z = wz
            msg.linear_acceleration.x = ax
            msg.linear_acceleration.y = ay
            msg.linear_acceleration.z = az
            imu_pub.publish(msg)
            published_imus += 1

    rospy.loginfo("replay finished: images=%d enhanced=%d imu=%d skipped=%d",
                  published_images, published_enhanced, published_imus, skipped)
    # Give subscribers a short chance to receive the final queued messages.
    rospy.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
