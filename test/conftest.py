import pytest
import matplotlib.pyplot as plt

def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", default=False, help="Generate plots during testing")


@pytest.fixture
def plot_if_enabled(request):
    def _plot_if_enabled(plot_func):
        if request.config.getoption("--plot"):
            fig = plot_func()
            plt.show()
        else:
            plt.close("all")

    return _plot_if_enabled