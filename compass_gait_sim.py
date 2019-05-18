import argparse
import math
import numpy as np

from pydrake.all import (Box,
                         DiagramBuilder,
                         FindResourceOrThrow,
                         FloatingBaseType,
                         Isometry3,
                         RigidBodyTree,
                         SignalLogger,
                         Simulator,
                         VisualElement, 
                         PortDataType,
                         LeafSystem,
                         VectorSystem, 
                         BasicVector)
from pydrake.examples.compass_gait import (CompassGait, CompassGaitParams)
from underactuated import (PlanarRigidBodyVisualizer)

class CompassGaitController(VectorSystem):
    def __init__(self, compass_gait):
        self.compass_gait = compass_gait
        self.num_states = compass_gait.get_output_port(0).size()
        self.compass_gait_context = self.compass_gait.CreateDefaultContext()
        
        VectorSystem.__init__(self, self.num_states, 1)
    
    def _DoCalcVectorOutput(self, context, inp, state, out):
        self.compass_gait_context = context
        print_stuff = False
        if print_stuff:
            print("inp", inp)
            print("state", state)
            print("out", out)
        out[0] = 0

        return 0


def Simulate2dCompassGait(duration):

    tree = RigidBodyTree(FindResourceOrThrow(
                        "drake/examples/compass_gait/CompassGait.urdf"),
                     FloatingBaseType.kRollPitchYaw)
    params = CompassGaitParams()
    R = np.identity(3)
    R[0, 0] = math.cos(params.slope())
    R[0, 2] = math.sin(params.slope())
    R[2, 0] = -math.sin(params.slope())
    R[2, 2] = math.cos(params.slope())
    X = Isometry3(rotation=R, translation=[0, 0, -5.])
    color = np.array([0.9297, 0.7930, 0.6758, 1])
    tree.world().AddVisualElement(VisualElement(Box([100., 1., 10.]), X, color))
    tree.compile()
    
    # initialize builder and compass gait plant
    builder = DiagramBuilder()
    compass_gait = builder.AddSystem(CompassGait())
    
    # Create a logger to log at 30hz
    state_dim = compass_gait.get_output_port(1).size()
    state_log = builder.AddSystem(SignalLogger(state_dim))
    state_log.DeclarePeriodicPublish(0.0333, 0.0) # 30hz logging
    builder.Connect(compass_gait.get_output_port(1), state_log.get_input_port(0))
    
    # Create a controller
    controller = builder.AddSystem(
        CompassGaitController(compass_gait))
    builder.Connect(compass_gait.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), compass_gait.get_input_port(0))

    visualizer = builder.AddSystem(PlanarRigidBodyVisualizer(tree,
                                                             xlim=[-1., 5.],
                                                             ylim=[-1., 2.],
                                                             figsize_multiplier=2))
    builder.Connect(compass_gait.get_output_port(1), visualizer.get_input_port(0))
    #print(compass_gait.get_input_port(0))

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)

    context = simulator.get_mutable_context()
    diagram.Publish(context)  # draw once to get the window open
    context.set_accuracy(1e-4)
    context.SetContinuousState([0., 0., 0.4, -2.])
    #context.SetContinuousState([0, 0, 1, -3.])

    simulator.StepTo(duration)
    return visualizer, state_log
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", "--duration",
                        type=float,
                        help="Duration to run sim.",
                        default=10.0)
    args = parser.parse_args()
    
    Simulate2dCompassGait(args.duration)