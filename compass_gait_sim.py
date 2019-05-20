import argparse
import math

from numpy import sin, cos
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

from pydrake.all import (MathematicalProgram, 
                         Solve, 
                         GetInfeasibleConstraints, 
                         IpoptSolver, 
                         SolverOptions, 
                         SnoptSolver, 
                         SolverType, 
                         Evaluate)

from pydrake.examples.compass_gait import (CompassGait, CompassGaitParams)
from underactuated import (PlanarRigidBodyVisualizer)

class CompassGaitController(VectorSystem):
    def __init__(self, compass_gait, params):
        self.compass_gait = compass_gait
        self.num_states = compass_gait.get_output_port(0).size()
        self.params = params
        
        VectorSystem.__init__(self, self.num_states, 1)
        
        self.prev_state = None
    
    def print_reset(self, state):
        """
        Prints when compass gait resets
        For the no-actuation compass gait, steady-state reset states are:
            start: [-0.219, 0.324, 1.093, 0.376]
            end:   [0.324, -0.219, 1.496, 1.81]
        """
        if self.prev_state is not None and self.prev_state[0] * state[0] < 0 and abs(state[0]) > 0.01:
            print "RESET"
            print self.prev_state
            print state
    
    def compute_trajectory(self, state_initial, state_final, min_time, max_time, test_traj = None):
        """
        Uses DMOC method to compute a trajectory from specified initial state to final state, following 
        compass gait dynamics.
        
        We don't account for foot collisions during the trajectory computation; in particular, 
        this method should only be used to find a trajectory to a final state in the guard function 
        (i.e. right before a foot collision).
        
        It should also be noted that the slope of the ramp does not affect the dynamics of the compass gait, 
        and so is not used in computation.
        
        If test_traj != None, then it should be a tuple (states, u, time_step), i.e. values of the variables.
        The optimizer will then operate in test mode, where instead of adding constraints it will test the 
        values of the constraints on the values.
        """
        test_mode = (test_traj is not None)
        
        mp = MathematicalProgram()
        mp.SetSolverOption(SolverType.kSnopt, "Major iterations limit", 500)
        
        if test_mode:
            num_time_steps = len(test_traj[0]) - 1
        else:
            num_time_steps = 100

        slack = 0.001
        
        ###########################
        # Define variables for MP #
        ###########################
        
        # create states over time as variables
        k = 0
        states = mp.NewContinuousVariables(2, "state_%d" % k)
        for k in range(1, num_time_steps + 1):
            state = mp.NewContinuousVariables(2, "state_%d" % k)
            states = np.vstack((states, state))
        
        # create u over time as variable
        u = mp.NewContinuousVariables(num_time_steps + 1, "u")
        
        # create time_step variable
        time_step = mp.NewContinuousVariables(1, "time_step")
        total_time = time_step[0] * num_time_steps
        mp.AddConstraint(total_time >= min_time)
        mp.AddConstraint(total_time <= max_time)
        
        if test_traj is not None:
            assert len(test_traj) == 3
            states, u, time_step = test_traj
        
        #########################
        # Some helper functions #
        #########################
        
        # define params
        mh = self.params.mass_hip()
        m = self.params.mass_leg()
        l = self.params.length_leg()
        b = self.params.center_of_mass_leg()
        g = self.params.gravity()
        a = l - b
        h = time_step
        
        # Note that the B matrix is [-1, 1], which represents how 
        # u actually affects the dynamics
        def u_eff(k): return np.array([-u[k], u[k]])
        def u_plus(k): return h/4*(u_eff(k) + u_eff(k+1))
        def u_minus(k): return h/4*(u_eff(k) + u_eff(k+1))
        
        def E(th_st, dth_st, th_sw, dth_sw):
            # Lagrangian of compass gait system
            return 0.5*(mh*l*l + m*a*a + m*l*l) * dth_st**2 + 0.5*m*b*b * dth_sw**2 \
                    - m * l * b * dth_st * dth_sw * cos(th_sw - th_st) \
                    + g * (m*a + m*l + mh*l) * cos(th_st) \
                    - m*g*b * cos(th_sw)
        
        def Ed(th_st_0, th_sw_0, th_st_1, th_sw_1):
            return h*E((th_st_0 + th_st_1) / 2, (th_st_1 - th_st_0) / h, (th_sw_0 + th_sw_1) / 2, (th_sw_1 - th_sw_0) / h)
        
        def L(th_st, dth_st, th_sw, dth_sw):
            # Lagrangian of compass gait system
            return 0.5*(mh*l*l + m*a*a + m*l*l) * dth_st**2 + 0.5*m*b*b * dth_sw**2 \
                    - m * l * b * dth_st * dth_sw * cos(th_sw - th_st) \
                    - g * (m*a + m*l + mh*l) * cos(th_st) \
                    + m*g*b * cos(th_sw)
        
        def Ld(th_st_0, th_sw_0, th_st_1, th_sw_1):
            return h*L((th_st_0 + th_st_1) / 2, (th_st_1 - th_st_0) / h, (th_sw_0 + th_sw_1) / 2, (th_sw_1 - th_sw_0) / h)
        
        
        
        # The following raw expressions were generated by compass_gait_expr.py
        def D1_L_raw(th_st, dth_st, th_sw, dth_sw):
            return [b*dth_st*dth_sw*l*m*sin(th_st - th_sw) + g*(a*m + l*m + l*mh)*sin(th_st), -b*m*(dth_st*dth_sw*l*sin(th_st - th_sw) + g*sin(th_sw))]
        
        def D2_L_raw(th_st, dth_st, th_sw, dth_sw):
            return [-b*dth_sw*l*m*cos(th_st - th_sw) + 1.0*dth_st*(a**2*m + l**2*m + l**2*mh), b*m*(1.0*b*dth_sw - dth_st*l*cos(th_st - th_sw))]
        
        def D1_Ld_raw(th_st_0, th_sw_0, th_st_1, th_sw_1):
            return [(b*l*m*(th_st_0 - th_st_1)*(th_sw_0 - th_sw_1)*sin(th_st_0/2 + th_st_1/2 - th_sw_0/2 - th_sw_1/2)/2 - b*l*m*(th_sw_0 - th_sw_1)*cos(th_st_0/2 + th_st_1/2 - th_sw_0/2 - th_sw_1/2) + g*h**2*(a*m + l*m + l*mh)*sin(th_st_0/2 + th_st_1/2)/2 + (th_st_0 - th_st_1)*(a**2*m + l**2*m + l**2*mh))/h, b*m*(2*b*(th_sw_0 - th_sw_1) - g*h**2*sin(th_sw_0/2 + th_sw_1/2) - l*(th_st_0 - th_st_1)*(th_sw_0 - th_sw_1)*sin(th_st_0/2 + th_st_1/2 - th_sw_0/2 - th_sw_1/2) - 2*l*(th_st_0 - th_st_1)*cos(th_st_0/2 + th_st_1/2 - th_sw_0/2 - th_sw_1/2))/(2*h)]
        
        def D2_Ld_raw(th_st_0, th_sw_0, th_st_1, th_sw_1):
            return [(b*l*m*(th_st_0 - th_st_1)*(th_sw_0 - th_sw_1)*sin(th_st_0/2 + th_st_1/2 - th_sw_0/2 - th_sw_1/2)/2 + b*l*m*(th_sw_0 - th_sw_1)*cos(th_st_0/2 + th_st_1/2 - th_sw_0/2 - th_sw_1/2) + g*h**2*(a*m + l*m + l*mh)*sin(th_st_0/2 + th_st_1/2)/2 - 1.0*(th_st_0 - th_st_1)*(a**2*m + l**2*m + l**2*mh))/h, b*m*(-2*b*(th_sw_0 - th_sw_1) - g*h**2*sin(th_sw_0/2 + th_sw_1/2) - l*(th_st_0 - th_st_1)*(th_sw_0 - th_sw_1)*sin(th_st_0/2 + th_st_1/2 - th_sw_0/2 - th_sw_1/2) + 2*l*(th_st_0 - th_st_1)*cos(th_st_0/2 + th_st_1/2 - th_sw_0/2 - th_sw_1/2))/(2*h)]
        
        # define actual functions we use
        def npconcat(lst):
            if np.ndim(lst[0]) == 0:
                return np.array(lst)
            else:
                return np.concatenate(lst)
        
        def D1_L(q, qdot):
            return npconcat(D1_L_raw(q[0], qdot[0], q[1], qdot[1]))
        
        def D2_L(q, qdot):
            return npconcat(D2_L_raw(q[0], qdot[0], q[1], qdot[1]))
        
        def D1_Ld(k):
            return npconcat(D1_Ld_raw(states[k,0], states[k,1], states[k+1,0], states[k+1,1]))
        
        def D2_Ld(k):
            return npconcat(D2_Ld_raw(states[k,0], states[k,1], states[k+1,0], states[k+1,1]))
        
        # helper function for adding constraints
        def AddZeroConstraint(constraint):
            if test_mode:
                print "Zero Constraint value =", constraint
            else:
                for i in range(len(constraint)):
                    mp.AddConstraint(constraint[i] == 0)
        
        ########################
        # Add DMOC constraints #
        ########################
        
        q0, q0_dot = state_initial[:2], state_initial[2:]
        qf, qf_dot = state_final[:2], state_final[2:]
        
        # Add boundary equality conditions
        AddZeroConstraint(q0 - states[0])
        AddZeroConstraint(qf - states[-1])
        
        # Add dynamics constraints
        for k in range(0, num_time_steps + 1):
            #print "ENERGY", Ed(states[k,0], states[k,1], states[k+1,0], states[k+1,1])
            if k == 0:
                print(q0_dot, (states[1]-states[0])/h)
                print(D2_L(q0, q0_dot), D1_Ld(k), u_minus(k))
                AddZeroConstraint(D2_L(q0, q0_dot) + D1_Ld(k) + u_minus(k))
            elif k < num_time_steps:
                AddZeroConstraint(D2_Ld(k-1) + D1_Ld(k) + u_plus(k-1) + u_minus(k))
            else:
                AddZeroConstraint(-D2_L(qf, qf_dot) + D1_Ld(k-1) + u_plus(k-1))
        
        # Add cost function
        mp.AddQuadraticCost(u.dot(u))
        
        if test_mode:
            return
        
        ################
        # Solve the MP #
        ################
        
        result = Solve(mp)
        if not result.is_success():
            infeasible = GetInfeasibleConstraints(mp, result)
            print "Infeasible constraints:"
            print len(infeasible)
            for i in range(len(infeasible)):
                print infeasible[i]
        
        states = result.GetSolution(states)
        u = result.GetSolution(u)
        time_step = result.GetSolution(time_step)
        
        return states, u, time_step
        
    def DoCalcVectorOutput(self, context, state, _, out):
        self.print_reset(state)
        
        self.prev_state = state
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
    
    minimal_state_log = builder.AddSystem(SignalLogger(4))
    minimal_state_log.DeclarePeriodicPublish(1./300, 0.0) # 100hz logging
    builder.Connect(compass_gait.get_output_port(0), minimal_state_log.get_input_port(0))
    
    # Create a controller
    controller = builder.AddSystem(
        CompassGaitController(compass_gait, params))
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
    initial_state = [-0.219, 0.324, 1.093, 0.376]
    #initial_state = [0., 0., 0.4, -2.]
    #initial_state = [0, 0, 1, -3.]
    context.SetContinuousState(initial_state)

    simulator.StepTo(duration)
    return visualizer, state_log, minimal_state_log

def ComputeTrajectory(state_initial, state_final, min_time, max_time, test_traj = None):
    """
    Invokes the controller's trajectory optimization procedure to 
    find a trajectory from state_initial to state_final.
    
    Basically a wrapper function.
    """
    params = CompassGaitParams()
    builder = DiagramBuilder()
    compass_gait = builder.AddSystem(CompassGait())
    controller = builder.AddSystem(
        CompassGaitController(compass_gait, params))
    
    context = compass_gait.CreateDefaultContext()
    print(context)
    
    return controller.compute_trajectory(state_initial, state_final, min_time, max_time, test_traj)
    # return should be states, u, time_step if test_traj is None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", "--duration",
                        type=float,
                        help="Duration to run sim.",
                        default=10.0)
    args = parser.parse_args()
    
    Simulate2dCompassGait(args.duration)