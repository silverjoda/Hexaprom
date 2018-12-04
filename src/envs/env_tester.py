from envs.ant_reach import AntReach
import mujoco_py

# Simulator objects
modelpath = "/home/silverjoda/SW/python-research/Hexaprom/src/envs/exp.xml"
model = mujoco_py.load_model_from_path(modelpath)
sim = mujoco_py.MjSim(model)

viewer = mujoco_py.MjViewer(sim)

q_dim = sim.get_state().qpos.shape[0]
qvel_dim = sim.get_state().qvel.shape[0]

obs_dim  = q_dim + qvel_dim
act_dim = sim.data.actuator_length.shape[0]
print(obs_dim, act_dim)

def render(human=True):
    if not human:
        return sim.render(camera_name=None,
                               width=224,
                               height=224,
                               depth=False)
        #return viewer.read_pixels(width, height, depth=False)
    viewer.render()

while True:
    sim.data.ctrl[:] = [0] * act_dim
    sim.forward()
    sim.step()
    render()

