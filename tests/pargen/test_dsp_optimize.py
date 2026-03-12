import numpy as np
from lgdo import Table, lh5
from matplotlib.figure import Figure

from pygama.pargen import dsp_optimize


def test_bayesian_opt():

    # Create optimizer
    optimizer = dsp_optimize.BayesianOptimizer(acq_func="ei", batch_size=1)

    assert optimizer.batch_size == 1
    assert isinstance(optimizer, dsp_optimize.BayesianOptimizer)

    # add a dimension
    optimizer.add_dimension("x", parameter=0.0, min_val=-5, max_val=5)

    # Add initial observations
    optimizer.add_initial_values(
        np.array([[0]]),
        np.array([0]),
        np.array([1]),
    )

    # Update DSP parameters database with the current optimization point
    db_dict = optimizer.update_db_dict({})

    assert "x" in db_dict

    # Simulate new evaluation and update optimizer
    optimizer.update({"x": [0.0], "y_val": 0, "y_val_err": 1})

    assert optimizer.optimal_x == 0.0
    assert optimizer.y_min == 0.0

    # Plot the result
    fig = optimizer.plot()

    assert isinstance(fig, Figure)

    acq_fig = optimizer.plot_acq()
    assert isinstance(acq_fig, Figure)

    # test different acq function
    optimizer = dsp_optimize.BayesianOptimizer(acq_func="ucb", batch_size=1)
    optimizer = dsp_optimize.BayesianOptimizer(acq_func="lcb", batch_size=1)


def test_run_one_dsp(raw_test_file):

    tab = lh5.read("ch1057600/raw", raw_test_file)

    config = {
        "outputs": ["tp_min", "tp_max", "wf_min", "wf_max"],
        "processors": {
            "tp_min, tp_max, wf_min, wf_max": {
                "function": "min_max",
                "module": "dspeed.processors",
                "args": ["waveform", "tp_min", "tp_max", "wf_min", "wf_max"],
                "unit": ["ns", "ns", "ADC", "ADC"],
            }
        },
    }
    # first test without fom
    dsp = dsp_optimize.run_one_dsp(tab, config)

    assert isinstance(dsp, Table)

    # now test with a fom
    def fom_func(tb, verbosity, kwargs=None):
        tb_ak = tb.view_as("ak")

        if kwargs is None:
            kwargs = {"a": 1}

        return np.std(tb_ak["tp_min"]) * kwargs["a"]

    dsp = dsp_optimize.run_one_dsp(tab, config, fom_function=fom_func)
    assert isinstance(dsp, float)

    dsp_kw = dsp_optimize.run_one_dsp(
        tab, config, fom_function=fom_func, fom_kwargs={"a": 2}
    )
    assert isinstance(dsp, float)
    assert dsp_kw == 2 * dsp


def test_optimise(raw_test_file):

    tab = lh5.read("ch1057600/raw", raw_test_file)

    config = {
        "outputs": ["tp_min", "tp_max", "wf_min", "wf_max", "pz_mean"],
        "processors": {
            "tp_min, tp_max, wf_min, wf_max": {
                "function": "min_max",
                "module": "dspeed.processors",
                "args": ["waveform", "tp_min", "tp_max", "wf_min", "wf_max"],
                "unit": ["ns", "ns", "ADC", "ADC"],
            },
            "bl_mean , bl_std, bl_slope, bl_intercept": {
                "function": "linear_slope_fit",
                "module": "dspeed.processors",
                "args": [
                    "wf_blsub[0:750]",
                    "bl_mean",
                    "bl_std",
                    "bl_slope",
                    "bl_intercept",
                ],
                "unit": ["ADC", "ADC", "ADC", "ADC"],
            },
            "wf_blsub": {
                "function": "bl_subtract",
                "module": "dspeed.processors",
                "args": ["waveform", "baseline", "wf_blsub"],
                "unit": "ADC",
            },
            "wf_pz": {
                "function": "pole_zero",
                "module": "dspeed.processors",
                "args": ["wf_blsub", "db.pz.tau", "wf_pz"],
                "unit": "ADC",
                "defaults": {"db.pz.tau": "27460.5"},
            },
            "pz_mean , pz_std, pz_slope, pz_intercept": {
                "function": "linear_slope_fit",
                "module": "dspeed.processors",
                "args": [
                    "wf_pz[1500:]",
                    "pz_mean",
                    "pz_std",
                    "pz_slope",
                    "pz_intercept",
                ],
                "unit": ["ADC", "ADC", "ADC", "ADC"],
            },
        },
    }

    def fom_func(tb):
        tb_ak = tb.view_as("ak")

        return {
            "y_val": np.mean(abs(tb_ak["pz_mean"])),
            "y_val_err": np.std(tb_ak["pz_mean"]),
        }

    optim = dsp_optimize.BayesianOptimizer(acq_func="ei", batch_size=10)

    optim.add_dimension("pz", parameter="tau", min_val=0, max_val=50000)
    optim.add_initial_values(
        np.array([[0]]),
        np.array([1000]),
        np.array([1000]),
    )
    opt, best = dsp_optimize.run_bayesian_optimisation(
        tab, config, fom_function=fom_func, db_dict={}, optimisers=optim, n_iter=10
    )
    assert "pz" in opt
    assert "tau" in opt["pz"]

    best = best[0]
    assert "y_val" in best
    assert "y_val_err" in best
