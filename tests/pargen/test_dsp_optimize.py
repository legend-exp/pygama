import dsp_optimize
import numpy as np
from matplotlib.figure import Figure


def test_bayesian_opt():

    # Create optimizer
    optimizer = dsp_optimize.BayesianOptimizer(acq_func="ei", batch_size=1)

    assert optimizer.acq_func == "ei"
    assert optimizer.batch_size == 1
    assert isinstance(optimizer, dsp_optimize.BayesianOptimizer)

    # add a dimension
    optimizer.add_dimension("x", parameter=0.0, min_val=-5, max_val=5)

    assert "x" in optimizer.dims
    assert optimizer.dims["x"]["parameter"] == 0.0
    assert optimizer.dims["x"]["min_val"] == -5
    assert optimizer.dims["x"]["max_val"] == 5

    # Add initial observations
    optimizer.add_initial_values(
        np.array([[0], [2], [-1]]),
        np.array([10.0, 5.0, 7.0]),
        np.array([0.1, 0.2, 0.1]),
    )

    assert np.array_equal(optimizer.x_init, np.array([[0], [2], [-1]]))
    assert np.array_equal(optimizer.y_init, np.array([10.0, 5.0, 7.0]))
    assert np.array_equal(optimizer.yerr_init, np.array([0.1, 0.2, 0.1]))

    # Update DSP parameters database with the current optimization point
    db_dict = optimizer.update_db_dict({})

    assert "x" in db_dict

    # Simulate new evaluation and update optimizer
    optimizer.update({"x": [1.0], "y_val": 3.5, "y_val_err": 0.1})

    assert optimizer.optimal_x == 1.0
    assert optimizer.y_min == 3.5

    # Plot the result
    fig = optimizer.plot()

    assert isinstance(fig, Figure)
