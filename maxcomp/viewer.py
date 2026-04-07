"""
Interactive MuJoCo viewer for robot models.
"""

import mujoco
import mujoco.viewer


def launch(model_path: str = "robots/my-robot/scene_single_finger.xml",
           fast: bool = False):
    """Launch the MuJoCo passive viewer."""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    tri_count = model.mesh_facenum.sum()
    print(f"Model: {model_path}")
    print(f"  Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")
    print(f"  Geoms: {model.ngeom}, Mesh triangles: {tri_count}")
    print(f"  nq={model.nq}, nv={model.nv}")

    if fast:
        print("\n  --fast: hiding visual geoms (group 2), showing collision (group 3)")

    def setup_viewer(viewer):
        if fast:
            viewer.opt.geomgroup[2] = False
            viewer.opt.geomgroup[3] = True
        model.vis.quality.shadowsize = 512
        model.vis.quality.offsamples = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        setup_viewer(viewer)
        print("\nViewer launched. Close window or press ESC to quit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
