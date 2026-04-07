"""CLI entry point for MuJoCo robot viewer."""

import argparse
from maxcomp.viewer import launch


def main():
    p = argparse.ArgumentParser(description="Launch MuJoCo viewer for a robot model")
    p.add_argument("model", nargs="?", default="robots/my-robot/scene_single_finger.xml",
                   help="Path to MuJoCo XML (scene.xml or robot.xml)")
    p.add_argument("--fast", action="store_true",
                   help="Show only collision geoms (much fewer triangles)")
    args = p.parse_args()

    launch(args.model, fast=args.fast)


if __name__ == "__main__":
    main()
