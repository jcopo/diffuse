import pytest
import matplotlib.pyplot as plt


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
            fig = plot_func()
            if request.config.getoption("--plot-wait"):
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(1)
                plt.close()
        else:
            plt.close("all")

    return _plot_if_enabled
