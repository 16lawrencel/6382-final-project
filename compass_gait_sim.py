import argparse
import math

from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, compass_gait, params, reset_state, passive = False, equal_time_steps = False):
        self.compass_gait = compass_gait
        self.num_states = compass_gait.get_output_port(0).size()
        self.params = params
        
        VectorSystem.__init__(self, self.num_states, 1)
        
        self.prev_state = None
        self.u_list = []
        self.time_array = []
        self.num_steps = 0
        self.freq = 1./30000 # frequency of calls to calcvectoroutput, measure empirically
        self.reset_state = reset_state
        self.passive = passive # go default passive mode
        self.equal_time_steps = equal_time_steps # whether to make time steps equal during optimization
        
    def is_reset(self, state):
        """
        Returns whether the current state is the start of a new reset (comparing with self.prev_state)
        """
        return self.prev_state is not None and self.prev_state[0] * state[0] < 0 and abs(state[0]) > 0.01
    
    def print_reset(self, state):
        """
        Prints when compass gait resets
        For the no-actuation compass gait, steady-state reset states are:
            start: [-0.219, 0.324, 1.093, 0.376]
            end:   [0.324, -0.219, 1.496, 1.81]
        """
        if self.is_reset(state):
            print "RESET"
            print self.prev_state
            print state
    
    def cg_dynamics(self, state, u):
        derivs = np.zeros_like(state)
        derivs[:2] = state[2:]
        
        # define params
        mh = self.params.mass_hip()
        m = self.params.mass_leg()
        l = self.params.length_leg()
        b = self.params.center_of_mass_leg()
        g = self.params.gravity()
        a = l - b
        th_st = state[0]
        th_sw = state[1]
        v_st = state[2]
        v_sw = state[3]
        
        M_inv = np.array([
            [b**2*m/(-b**2*l**2*m**2*cos(th_st - th_sw)**2 + b**2*m*(a**2*m + l**2*(m + mh))), b*l*m*cos(th_st - th_sw)/(-b**2*l**2*m**2*cos(th_st - th_sw)**2 + b**2*m*(a**2*m + l**2*(m + mh)))], 
            [b*l*m*cos(th_st - th_sw)/(-b**2*l**2*m**2*cos(th_st - th_sw)**2 + b**2*m*(a**2*m + l**2*(m + mh))), (a**2*m + l**2*(m + mh))/(-b**2*l**2*m**2*cos(th_st - th_sw)**2 + b**2*m*(a**2*m + l**2*(m + mh)))]])
        
        """
        M = np.array([
            [(mh + m)*l*l + m*a*a, -m*l*b*cos(th_st - th_sw)], 
            [-m*l*b*cos(th_st - th_sw), m*b*b]
        ])
        """
        
        C = np.array([
            [0, -m*l*b*sin(th_st - th_sw) * v_sw], 
            [m*l*b*sin(th_st - th_sw) * v_st, 0]
        ])
        
        tau_g = np.array([
            [(mh*l + m*a + m*l)*g*sin(1.*th_st)], 
            [-m*b*g*sin(1.*th_sw)]
        ])
        
        u_eff = np.array([
            [-u],
            [u]
        ])
        
        v = np.reshape(state[2:], (-1, 1))
        derivs[2:] = M_inv.dot(tau_g + u_eff - C.dot(v)).flatten()
        return derivs
    
    def compute_trajectory(self, state_initial, state_final, min_time, max_time, test_traj = None):
        """
        Direct transcription.
        Note: NOT USED
        """
        print("Computing trajectory")
        
        test_mode = (test_traj is not None)
        
        mp = MathematicalProgram()
        #mp.SetSolverOption(SolverType.kSnopt, "Major iterations limit", 2000)
        
        if test_mode:
            num_time_steps = len(test_traj[0]) - 1
        else:
            num_time_steps = 100
        
        # create u_over_time as variables
        k = 0
        u = mp.NewContinuousVariables(num_time_steps + 1, "u")
        
        # create states_over_time as variables
        k = 0
        states_over_time = mp.NewContinuousVariables(4, "state_%d" % k)
        for k in range(1, num_time_steps + 1):
            state = mp.NewContinuousVariables(4, "state_%d" % k)
            states_over_time = np.vstack((states_over_time, state))
        
        # create time_array variables
        k = 0
        time_array = mp.NewContinuousVariables(num_time_steps+1, "time_array")
        mp.AddConstraint(time_array[0] == 0)
        """
        for k in range(1, num_time_steps + 1):
            mp.AddConstraint(time_array[k] - time_array[k-1] >= 0.01)
            mp.AddConstraint(time_array[k] - time_array[k-1] <= 0.3)
        """
        
        for k in range(2, num_time_steps + 1):
            mp.AddConstraint(time_array[k] - time_array[k-1] == time_array[k-1] - time_array[k-2])
        
        mp.AddConstraint(time_array[-1] >= min_time)
        mp.AddConstraint(time_array[-1] <= max_time)
        
        if test_traj is not None:
            assert len(test_traj) == 3
            states, u, time_array = test_traj
        
        def AddZeroConstraint(constraint, eps = 0):
            if test_mode:
                if abs(constraint[0]) > 0.0001 or abs(constraint[1] > 0.0001):
                    print "Zero Constraint value =", constraint
            else:
                for i in range(len(constraint)):
                    if eps == 0:
                        mp.AddConstraint(constraint[i] == 0)
                    else:
                        assert eps > 0
                        mp.AddConstraint(constraint[i] <= eps)
                        mp.AddConstraint(constraint[i] >= -eps)
        
        # constraints for initial and final states
        AddZeroConstraint(states_over_time[0] - state_initial)
        AddZeroConstraint(states_over_time[-1] - state_final)
        
        # we'll now simulate forward in time and add direct transcription constraints
        for k in range(1, num_time_steps + 1):
            time_step = time_array[k] - time_array[k-1]
            state_next = states_over_time[k-1] + time_step * self.cg_dynamics(states_over_time[k-1], u[k-1])
            
            if k < 10:
                slack = 0.001
            else:
                slack = 0
                
            AddZeroConstraint(state_next - states_over_time[k])
        
        # add quadratic cost for u
        mp.AddQuadraticCost(u.dot(u))
        
        # solve and return output
        result = Solve(mp)
        
        if not result.is_success():
            infeasible = GetInfeasibleConstraints(mp, result)
            print "Infeasible constraints:"
            print len(infeasible)
            for i in range(len(infeasible)):
                print infeasible[i]
        
        trajectory = result.GetSolution(states_over_time)
        input_trajectory = result.GetSolution(u)
        time_array = result.GetSolution(time_array)
        
        return trajectory, input_trajectory, time_array
    
    def compute_trajectory_DMOC(self, state_initial, state_final, min_time, max_time, test_traj = None):
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
        print("Computing trajectory")
        test_mode = (test_traj is not None)
        
        mp = MathematicalProgram()
        mp.SetSolverOption(SolverType.kSnopt, "Major iterations limit", 500)
        
        if test_mode:
            num_time_steps = len(test_traj[0]) - 1
        else:
            num_time_steps = 200
        
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
        mp.SetInitialGuess(u, np.zeros(num_time_steps+1))
        
        # create time_step variable
        k = 0
        time_array = mp.NewContinuousVariables(num_time_steps+1, "time_array")
        mp.AddConstraint(time_array[0] == 0)
        
        if self.equal_time_steps:
            for k in range(2, num_time_steps + 1):
                mp.AddConstraint(time_array[k] - time_array[k-1] == time_array[k-1] - time_array[k-2])
        else:
            for k in range(1, num_time_steps + 1):
                mp.AddConstraint(time_array[k] - time_array[k-1] >= 0.01)
                mp.AddConstraint(time_array[k] - time_array[k-1] <= 0.1)
        
        mp.AddConstraint(time_array[-1] >= min_time)
        mp.AddConstraint(time_array[-1] <= max_time)
        
        """
        time_step = mp.NewContinuousVariables(1, "time_step")
        total_time = time_step[0] * num_time_steps
        mp.AddConstraint(total_time >= min_time)
        mp.AddConstraint(total_time <= max_time)
        """
        
        if test_traj is not None:
            assert len(test_traj) == 3
            states, u, time_array = test_traj
        
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
        
        # Note that the B matrix is [-1, 1], which represents how 
        # u actually affects the dynamics
        def geth(k): return time_array[k+1] - time_array[k]
        
        def u_eff(k): return np.array([-u[k], u[k]])
        def u_plus(k): return geth(k)/4*(u_eff(k) + u_eff(k+1))
        def u_minus(k): return geth(k)/4*(u_eff(k) + u_eff(k+1))
        
        def E(th_st, dth_st, th_sw, dth_sw):
            # Energy of compass gait system
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
        
        def Ld(th_st_0, th_sw_0, th_st_1, th_sw_1, h):
            return h*L((th_st_0 + th_st_1) / 2, (th_st_1 - th_st_0) / h, (th_sw_0 + th_sw_1) / 2, (th_sw_1 - th_sw_0) / h)
        
        
        
        # The following raw expressions were generated by compass_gait_expr.py
        def D1_L_raw(th_st, dth_st, th_sw, dth_sw):
            return [b*dth_st*dth_sw*l*m*sin(th_st - th_sw) + g*(a*m + l*m + l*mh)*sin(th_st), -b*m*(dth_st*dth_sw*l*sin(th_st - th_sw) + g*sin(th_sw))]
        
        def D2_L_raw(th_st, dth_st, th_sw, dth_sw):
            return [-b*dth_sw*l*m*cos(th_st - th_sw) + 1.0*dth_st*(a**2*m + l**2*m + l**2*mh), b*m*(1.0*b*dth_sw - dth_st*l*cos(th_st - th_sw))]
        
        def D1_Ld_raw(th_st_0, th_sw_0, th_st_1, th_sw_1, h):
            return [(b*l*m*(th_st_0 - th_st_1)*(th_sw_0 - th_sw_1)*sin(th_st_0/2 + th_st_1/2 - th_sw_0/2 - th_sw_1/2)/2 - b*l*m*(th_sw_0 - th_sw_1)*cos(th_st_0/2 + th_st_1/2 - th_sw_0/2 - th_sw_1/2) + g*h**2*(a*m + l*m + l*mh)*sin(th_st_0/2 + th_st_1/2)/2 + (th_st_0 - th_st_1)*(a**2*m + l**2*m + l**2*mh))/h, b*m*(2*b*(th_sw_0 - th_sw_1) - g*h**2*sin(th_sw_0/2 + th_sw_1/2) - l*(th_st_0 - th_st_1)*(th_sw_0 - th_sw_1)*sin(th_st_0/2 + th_st_1/2 - th_sw_0/2 - th_sw_1/2) - 2*l*(th_st_0 - th_st_1)*cos(th_st_0/2 + th_st_1/2 - th_sw_0/2 - th_sw_1/2))/(2*h)]
        
        def D2_Ld_raw(th_st_0, th_sw_0, th_st_1, th_sw_1, h):
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
            return npconcat(D1_Ld_raw(states[k,0], states[k,1], states[k+1,0], states[k+1,1], geth(k)))
        
        def D2_Ld(k):
            return npconcat(D2_Ld_raw(states[k,0], states[k,1], states[k+1,0], states[k+1,1], geth(k)))
        
        if test_mode:
            # record to plot later
            zero_constraint_list = []
        
        # helper function for adding constraints
        def AddZeroConstraint(constraint, eps = 0):
            if test_mode:
                if abs(constraint[0]) > 0.0001 or abs(constraint[1] > 0.0001):
                    print "Zero Constraint value =", constraint
                    
                for i in range(len(constraint)):
                    zero_constraint_list.append(constraint[i])
            else:
                for i in range(len(constraint)):
                    if eps == 0:
                        mp.AddConstraint(constraint[i] == 0)
                    else:
                        assert eps > 0
                        mp.AddConstraint(constraint[i] <= eps)
                        mp.AddConstraint(constraint[i] >= -eps)
        
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
            slack = 0
            if k < 3 or num_time_steps - k < 3:
                slack = 0.001
                            
            #print "ENERGY", Ed(states[k,0], states[k,1], states[k+1,0], states[k+1,1])
            if k == 0:
                AddZeroConstraint(D2_L(q0, q0_dot) + D1_Ld(k) + u_minus(k), slack)
            elif k < num_time_steps:
                AddZeroConstraint(D2_Ld(k-1) + D1_Ld(k) + u_plus(k-1) + u_minus(k), slack)
            else:
                AddZeroConstraint(-D2_L(qf, qf_dot) + D1_Ld(k-1) + u_plus(k-1))
        
        # Add cost function
        mp.AddQuadraticCost(u.dot(u))
        
        if test_mode:
            # plot the zero constraints then return
            plt.hist(zero_constraint_list)
            plt.savefig('figs/zero_constraint_hist.png')
            
            return
        
        ################
        # Solve the MP #
        ################
        
        result = Solve(mp)
        """
        if not result.is_success():
            infeasible = GetInfeasibleConstraints(mp, result)
            print "Infeasible constraints:"
            print len(infeasible)
            for i in range(len(infeasible)):
                print infeasible[i]
        """
        
        states = result.GetSolution(states)
        u = result.GetSolution(u)
        time_array = result.GetSolution(time_array)
        
        print "Finished computing trajectory"
        return states, u, time_array
    
    def getTime(self):
        """
        Gets the internal clock's time, assuming the frequency of calls to DoCalcVectorOutput as measured
        """
        return self.num_steps * self.freq
    
    def DoCalcVectorOutput(self, context, state, _, out):
        self.print_reset(state)
        
        if self.is_reset(state):
            # reset u_list
            self.u_list = []
        
        self.prev_state = state
        
        self.num_steps += 1
        """
        if self.num_steps % 10000 == 0:
            print self.num_steps
        """
           
        if self.passive:
            # Don't compute and just return 0
            out[0] = 0
            return
        
        min_time = 1
        max_time = 5
        #state_final = np.array([0.324, -0.219, 1.496, 1.81])
        
        if len(self.u_list) <= 1:
            print self.num_steps
            states, self.u_list, self.time_array = self.compute_trajectory_DMOC(state, self.reset_state, min_time, max_time)
        
        self.time_array += self.getTime() # adjust by current time
        
        t0 = self.getTime() - self.time_array[0]
        t1 = self.time_array[1] - self.getTime()

        while len(self.time_array) > 1 and self.time_array[1] <= self.getTime():
            # pop one item from u_list / time_array
            self.u_list = self.u_list[1:]
            self.time_array = self.time_array[1:]

        out[0] = (self.u_list[0] * t1 + self.u_list[1] * t0) / (t0 + t1) # linear interpolation
            
def Simulate2dCompassGait(duration, 
                          passive = True, 
                          passive_controller = False, 
                          minimal_state_freq = 1./30, 
                          equal_time_steps = False):

    tree = RigidBodyTree(FindResourceOrThrow(
                        "drake/examples/compass_gait/CompassGait.urdf"),
                     FloatingBaseType.kRollPitchYaw)
    params = CompassGaitParams()
    
    if not passive: # actuated on flat ground
        params.set_slope(0)
    
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
    state_log.DeclarePeriodicPublish(1./30, 0.0) # 30hz logging
    builder.Connect(compass_gait.get_output_port(1), state_log.get_input_port(0))
    
    minimal_state_log = builder.AddSystem(SignalLogger(4))
    minimal_state_log.DeclarePeriodicPublish(minimal_state_freq, 0.0)
    builder.Connect(compass_gait.get_output_port(0), minimal_state_log.get_input_port(0))
    
    # Create a controller
    
    # define reset state to aim for when optimizing
    # different for passive / actuated
    if passive:
        reset_state = np.array([0.324, -0.219, 1.496, 1.81])
    else:
        reset_state = np.array([-0.2884, 0.2884, -1.6009, -1.9762])

    controller = builder.AddSystem(
        CompassGaitController(compass_gait, params, reset_state, 
                              passive = passive_controller, 
                              equal_time_steps = equal_time_steps))
    builder.Connect(compass_gait.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), compass_gait.get_input_port(0))

    visualizer = builder.AddSystem(PlanarRigidBodyVisualizer(tree,
                                                             xlim=[-1., 5.],
                                                             ylim=[-1., 2.],
                                                             figsize_multiplier=2))
    builder.Connect(compass_gait.get_output_port(1), visualizer.get_input_port(0))

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.0)

    context = simulator.get_mutable_context()
    diagram.Publish(context)  # draw once to get the window open
    context.set_accuracy(1e-4)
    
    if passive:
        initial_state = [0., 0., 0.4, -2.]
        #initial_state = [-0.219, 0.324, 1.093, 0.376]
    else:
        initial_state = np.array([0.2884, -0.2884, -1.1235, -0.2830]) # zero-slope
        
    context.SetContinuousState(initial_state)

    simulator.StepTo(duration)
    print(controller.num_steps)
    return visualizer, state_log, minimal_state_log

def ComputeTrajectory(state_initial, state_final, min_time, max_time, 
                      test_traj = None, 
                      DMOC = False, 
                      slope = None, 
                      equal_time_steps = False):
    """
    Invokes the controller's trajectory optimization procedure to 
    find a trajectory from state_initial to state_final.
    
    Basically a wrapper function.
    """
    params = CompassGaitParams()
    if slope is not None:
        params.set_slope(slope)
    builder = DiagramBuilder()
    compass_gait = builder.AddSystem(CompassGait())
    controller = builder.AddSystem(
        CompassGaitController(compass_gait, params, state_final, equal_time_steps = equal_time_steps))
    
    context = compass_gait.CreateDefaultContext()
    
    if DMOC:
        compute_traj_method = controller.compute_trajectory_DMOC
    else:
        compute_traj_method = controller.compute_trajectory
    
    return compute_traj_method(state_initial, state_final, min_time, max_time, test_traj)
    # return should be states, u, time_step if test_traj is None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", "--duration",
                        type=float,
                        help="Duration to run sim.",
                        default=10.0)
    args = parser.parse_args()
    
    Simulate2dCompassGait(args.duration)