"""
Interactive MuJoCo viewer for my-robot.

Usage:
    python view_robot.py                          # default: scene.xml (visual meshes)
    python view_robot.py --fast                   # collision-only (much faster)
    python view_robot.py robots/my-robot/robot.xml

Controls:
    Left-click + drag   — rotate camera
    Right-click + drag  — pan camera
    Scroll              — zoom
    Double-click        — track a body
    Ctrl + Right-click  — apply force to a body (perturbation)
    Space               — pause / resume
    Backspace           — reset to initial state
    Tab                 — toggle visual / collision geoms
    ESC                 — quit
"""

import argparse
import mujoco
import mujoco.viewer


def main():
    p = argparse.ArgumentParser(description="Launch MuJoCo viewer for a robot model")
    p.add_argument("model", nargs="?", default="robots/my-robot/scene_single_finger.xml",
                   help="Path to MuJoCo XML (scene.xml or robot.xml)")
    p.add_argument("--fast", action="store_true",
                   help="Show only collision geoms (much fewer triangles)")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    tri_count = model.mesh_facenum.sum()
    print(f"Model: {args.model}")
    print(f"  Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")
    print(f"  Geoms: {model.ngeom}, Mesh triangles: {tri_count}")
    print(f"  nq={model.nq}, nv={model.nv}")

    if args.fast:
        print("\n  --fast: hiding visual geoms (group 2), showing collision (group 3)")

    def setup_viewer(viewer):
        if args.fast:
            # Hide visual meshes (group 2), show collision hulls (group 3)
            viewer.opt.geomgroup[2] = False  # visual off
            viewer.opt.geomgroup[3] = True   # collision on
        # Disable shadows and reflections for speed
        model.vis.quality.shadowsize = 512
        model.vis.quality.offsamples = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        setup_viewer(viewer)
        print("\nViewer launched. Close window or press ESC to quit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
