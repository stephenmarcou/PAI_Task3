"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, DotProduct, WhiteKernel
from scipy.stats import norm


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BOAlgorithm class.
# NOTE: main() is not called by the checker.
class BOAlgorithm():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        # Define the kernels for v(x)
        linear_kernel = C(1.0) * DotProduct() + WhiteKernel(0.0001)
        matern_kernel = Matern(nu=2.5)

        kernel_v = linear_kernel + matern_kernel 
        
        #kernel_v = C(np.sqrt(0.2), constant_value_bounds="fixed")*RBF(length_scale=0.5, length_scale_bounds="fixed")
        
        

        # Define the GP model
        self.vfunc = GaussianProcessRegressor(kernel=kernel_v, alpha=0.0001**2, n_restarts_optimizer=10, normalize_y=True)

        kernel_gp = C(1)*RBF(length_scale=10) + WhiteKernel(0.15)
        self.GP = GaussianProcessRegressor(kernel_gp, alpha=0.15**2, normalize_y=True)

        
        self.x = []
        self.v_val = []
        self.f_val = []
        

    def recommend_next(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        
        
        x_opt = self.optimize_acquisition_function()
        return x_opt

        raise NotImplementedError

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        
        
        # Predict the mean and standard deviation at the given points
        mu, sigma = self.GP.predict(x, return_std=True)
        
        # Predict the mean for v(x)
        mu_v, std_v = self.vfunc.predict(x, return_std=True)
        
        
        # Ensure that sigma is not zero to avoid numerical issues
        sigma = np.maximum(sigma, 1e-9)
        std_v = np.maximum(std_v, 1e-9)
        
        # probability being under safety treshold
        prob = norm.cdf((SAFETY_THRESHOLD - mu_v) / std_v)

        beta = 0.1 
        
        # Calculate the UCB score
        ucb_value = mu + beta * sigma
        
        # Lagrangian penalty for constraint violation
        lambda_penalty = 10  #mu_v/2

        penalty = lambda_penalty * np.maximum(mu_v - SAFETY_THRESHOLD, 0)

        # someone told me that one of the papers talked about multiplying the 
        # probability here
        af_value = ucb_value * prob - penalty
        

        
        return af_value.flatten()
        


        
        
        
        
        # TODO: Implement the acquisition function you want to optimize.
        raise NotImplementedError

    def add_observation(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        
        self.x.append(x)
        self.f_val.append(f)
        self.v_val.append(v)
        

        
    
        x_list = np.array(self.x).reshape(-1,1)
        f_val = np.array(self.f_val).reshape(-1,1)
        v_val = np.array(self.v_val).reshape(-1,1)
        

        
        self.GP.fit(x_list,f_val)
        self.vfunc.fit(x_list,v_val)

       # raise NotImplementedError

    def get_optimal_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
            # Predict f(x) and v(x) for each candidate point
        
        
        x_opt = 0
        f_max = -100000

        for x,f,v in zip(self.x, self.f_val, self.v_val):
            if f > f_max and v < SAFETY_THRESHOLD:
                f_max = f
                x_opt = x
        # print(f"f_max: {f_max}")
        return x_opt
        
        

                
            

        raise NotImplementedError

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BOAlgorithm()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_observation(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.recommend_next()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function recommend_next must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_observation(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_optimal_solution()
    assert check_in_domain(solution), \
        f'The function get_optimal_solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
