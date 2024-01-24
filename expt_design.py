import numpy as np
import cvxpy as cvx
import random
import argparse

class ExperimentDesign(object):

    MIN_WEIGHT_FOR_SELECTION = 0.3

    '''
    Represents an experiment design object that can be used to setup
    and run experiment design.
    '''
    def __init__(self,
                 depth_min=2, depth_max=5,
                 fanout_min = 5, fanout_max=20, # these four parameters are what we need
                 ):
        '''
        Create an experiment design instance.

        :param self: The object being created
        :type self: ExperimentDesign
        :param parts_min: Minimum number of partitions to use in experiments
        :type parts_min: int
        :param parts_max: Maximum number of partitions to use in experiments
        :type parts_max: int
        :param total_parts: Total number of partitions in the dataset
        :type total_parts: int
        :param mcs_min: Minimum number of machines to use in experiments
        :type mcs_min: int
        :param mcs_max: Maximum number of machines to use in experiments
        :type mcs_max: int
        :param cores_per_mc: Cores or slots available per machine
        :type cores_per_mc: int
        :param budget: Budget for the experiment design problem
        :type budget: float
        :param budget: Number of points to interpolate between parts_min and parts_max 
        :type budget: float
        '''
        # self.parts_min = parts_min
        # self.parts_max = parts_max
        # self.total_parts = total_parts
        # self.mcs_min = mcs_min
        # self.mcs_max = mcs_max
        # self.cores_per_mc = cores_per_mc
        # self.num_parts_interpolate = num_parts_interpolate
        self.budget = 5000

        self.depth_min_ = depth_min
        self.depth_max_ = depth_max
        self.fanout_min = fanout_min
        self.fanout_max = fanout_max
        self.recievers_max = 5_000 # make it fixed for now

    def _construct_constraints(self, lambdas, points):
        '''Construct non-negative lambdas and budget constraints'''
        constraints = []
        constraints.append(0 <= lambdas)
        constraints.append(lambdas <= 1)
        constraints.append(self._get_cost(lambdas, points) <= self.recievers_max)
        return constraints

    def total_nodes(self, D, F):
        # it is used for calculting the number of receivers
        # i.e., leaves of a tree of fanout F and depth D
        # Depth of 1 means a single node, depth of 0 mean no node
        # minimum depth we use is 2
        D = D-1
        return F**D


    def _get_cost(self, lambdas, points):
        '''Estimate the cost of an experiment. Right now this is input_frac/machines'''
        cost = 0
        num_points = len(points)
        for i in range(0, num_points):
            D = points[i][0]
            F = points[i][1]
            cost = cost + (self.total_nodes(D, F)*lambdas[i])
        return cost

        scale_min = float(self.parts_min) / float(self.total_parts)
        for i in range(0, num_points):
            scale = points[i][0]
            mcs = points[i][1]
            cost = cost + (float(scale) / scale_min * 1.0 / float(mcs) * lambdas[i])
        return cost

    def _get_training_points(self):
        '''Enumerate all the training points given the params for experiment design'''
        depth_range = list(range(self.depth_min_, self.depth_max_+1))
        fanout_range = list(range(self.fanout_min, self.fanout_max+1))
        for depth in depth_range:
            for fanout in fanout_range:
                if self.total_nodes(depth, fanout) <= self.recievers_max:
                    yield [depth, fanout]

        # mcs_range = list(range(self.mcs_min, self.mcs_max + 1))

        # scale_min = float(self.parts_min) / float(self.total_parts)
        # scale_max = float(self.parts_max) / float(self.total_parts)
        # scale_range = np.linspace(scale_min, scale_max, self.num_parts_interpolate)

        # for scale in scale_range:
        #     for mcs in mcs_range:
        #         if np.round(scale * self.total_parts) >= self.cores_per_mc * mcs:
        #             yield [scale, mcs]

    def _frac2parts(self, fraction):
        '''Convert input fraction into number of partitions'''
        return int(np.ceil(fraction * self.total_parts))

    def run(self):
        ''' Run experiment design. Returns a list of configurations and their scores'''
        training_points = list(self._get_training_points())
        num_points = len(training_points)

        all_training_features = np.array([_get_features(point) for point in training_points])
        covariance_matrices = list(_get_covariance_matrices(all_training_features))

        lambdas = cvx.Variable(num_points)

        objective = cvx.Minimize(_construct_objective(covariance_matrices, lambdas))
        constraints = self._construct_constraints(lambdas, training_points)

        problem = cvx.Problem(objective, constraints)

        opt_val = problem.solve()
        # TODO: Add debug logging
        # print "solution status ", problem.status
        # print "opt value is ", opt_val

        filtered_lambda_idxs = []
        for i in range(0, num_points):
            if lambdas[i].value > self.MIN_WEIGHT_FOR_SELECTION:
                filtered_lambda_idxs.append((lambdas[i].value, i))

        sorted_by_lambda = sorted(filtered_lambda_idxs, key=lambda t: t[0], reverse=True)
        return [(training_points[idx][0], training_points[idx][1], l) for (l,idx) in sorted_by_lambda]
        return [(self._frac2parts(training_points[idx][0]), training_points[idx][0],
                 training_points[idx][1], l) for (l, idx) in sorted_by_lambda]

def _construct_objective(covariance_matrices, lambdas):
    ''' Constructs the CVX objective function. '''
    num_points = len(covariance_matrices)
    num_dim = int(covariance_matrices[0].shape[0])
    objective = 0
    matrix_part = np.zeros([num_dim, num_dim])
    for j in range(0, num_points):
        matrix_part = matrix_part + covariance_matrices[j] * lambdas[j]

    for i in range(0, num_dim):
        k_vec = np.zeros(num_dim)
        k_vec[i] = 1.0
        objective = objective + cvx.matrix_frac(k_vec, matrix_part)

    return objective

def _get_covariance_matrices(features_arr):
    ''' Returns a list of covariance matrices given expt design features'''
    col_means = np.mean(features_arr, axis=0)
    means_inv = (1.0 / col_means)
    nrows = features_arr.shape[0]
    for i in range(0, nrows):
        feature_row = features_arr[i,]
        ftf = np.outer(feature_row.transpose(), feature_row)
        yield np.diag(means_inv).transpose().dot(ftf.dot(np.diag(means_inv)))

def _get_features(training_point):
    ''' Compute the features for a given point. Point is expected to be [input_frac, machines]'''
    depth = training_point[0]
    fanout = training_point[1]
    return [depth, fanout]
    scale = training_point[0]
    mcs = training_point[1]
    return [1.0, float(scale) / float(mcs), float(mcs), np.log(mcs)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Design')

    # parser.add_argument('--min-parts', type=int, required=True,
    #     help='Minimum number of partitions to use in experiments')
    # parser.add_argument('--max-parts', type=int, required=True,
    #     help='Maximum number of partitions to use in experiments')
    # parser.add_argument('--total-parts', type=int, required=True,
    #     help='Total number of partitions in the dataset')

    # parser.add_argument('--min-mcs', type=int, required=True,
    #     help='Minimum number of machines to use in experiments')
    # parser.add_argument('--max-mcs', type=int, required=True,
    #     help='Maximum number of machines to use in experiments')
    # parser.add_argument('--cores-per-mc', type=int, default=2,
    #     help='Number of cores or slots available per machine, (default 2)')

    parser.add_argument('--depth-min', type=int, default=2,
        help='Minimum depth of the multicast tree in experiments')
    parser.add_argument('--depth-max', type=int, default =5,
        help='Maximum depth of the multicast tree in experiments')
    parser.add_argument('--fanout-min', type=int, default=5,
        help='Minimum fanout factor of the multicast tree in experiments')
    parser.add_argument('--fanout-max', type=int, default=20,
        help='Maximum fanout factor of the multicast tree in experiments')

    # parser.add_argument('--budget', type=float, default=10.0,
    #     help='Budget of experiment design problem, (default 10.0)')
    # parser.add_argument('--num-parts-interpolate', type=int, default=20,
    #     help='Number of points to interpolate between min_parts and max_parts, (default 20)')

    args = parser.parse_args()

    ex = ExperimentDesign(
        args.depth_min, args.depth_max, args.fanout_min,args.fanout_max)

    expts = ex.run()
    print("Depth Fanout Weight")
    for expt in expts:
        # Generate dummy data for predictor test
        print( str(expt[0])+" "+ str(expt[1])+" "+str(round(expt[0]*(1+random.random()%10*0.1),4) ))
        # print(f"{expt[0], expt[1], expt[2]}")
    
    # print ("Machines, Cores, InputFraction, Partitions, Weight")
    # for expt in expts:
    #     print ("%d, %d, %f, %d, %f" % (expt[2], expt[2] * args.cores_per_mc, expt[1], expt[0], expt[3]))
