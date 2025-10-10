# Copyright 2025 Jacopo Iollo <jacopo.iollo@inria.fr>, Geoffroy Oudoumanessah <geoffroy.oudoumanessah@inria.fr>
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
import pytest
import matplotlib.pyplot as plt

from .config import get_conditional_test_config


def pytest_addoption(parser):
    parser.addoption(
        "--plot",
        action="store_true",
        default=False,
        help="Generate plots during testing",
    )
    parser.addoption(
        "--plot-wait",
        action="store_true",
        default=False,
        help="Wait for manual plot closure instead of auto-closing after 2s",
    )


@pytest.fixture
def plot_if_enabled(request):
    def _plot_if_enabled(plot_func):
        if request.config.getoption("--plot"):
            plot_func()
            if request.config.getoption("--plot-wait"):
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(1)
                plt.close()
        else:
            plt.close("all")

    return _plot_if_enabled


@pytest.fixture
def test_config(request):
    """
    Main test configuration fixture.

    Usage:
        @pytest.mark.parametrize("test_config", [
            {"schedule_name": "LinearSchedule", "timer_name": "vp"},
            {"schedule_name": "CosineSchedule", "timer_name": "heun"},
        ], indirect=True)
        def test_something(test_config):
            # test_config contains everything needed for the test
    """
    # Get parameters from indirect parametrization
    params = getattr(request, "param", {})
    return get_conditional_test_config(**params)
