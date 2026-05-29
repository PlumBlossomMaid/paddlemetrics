import paddle
import pytest
from unittests._helpers import seed_all

from paddlemetrics import Metric, MetricCollection
from paddlemetrics.classification import (MulticlassAccuracy,
                                         MulticlassConfusionMatrix,
                                         MulticlassPrecision, MulticlassRecall)
from paddlemetrics.regression import MeanAbsoluteError, MeanSquaredError
from paddlemetrics.utils.imports import _MATPLOTLIB_AVAILABLE
from paddlemetrics.wrappers import (ClasswiseWrapper, MetricTracker,
                                   MultioutputWrapper)

seed_all(42)


def test_raises_error_on_wrong_input():
    """Make sure that input type errors are raised on the wrong input."""
    with pytest.raises(TypeError, match="Metric arg need to be an instance of a .*"):
        MetricTracker([1, 2, 3])
    with pytest.raises(
        ValueError,
        match="Argument `maximize` should either be a single bool or list of bool",
    ):
        MetricTracker(MeanAbsoluteError(), maximize=2)
    with pytest.raises(
        ValueError,
        match="The len of argument `maximize` should match the length of the metric collection",
    ):
        MetricTracker(
            MetricCollection([MeanAbsoluteError(), MeanSquaredError()]),
            maximize=[False, False, False],
        )
    with pytest.raises(
        ValueError,
        match="Argument `maximize` should be a single bool when `metric` is a single Metric",
    ):
        MetricTracker(MeanAbsoluteError(), maximize=[False])


@pytest.mark.parametrize(
    ("method", "method_input"),
    [
        (
            "update",
            (
                paddle.randint(low=0, high=10, shape=(50,)),
                paddle.randint(low=0, high=10, shape=(50,)),
            ),
        ),
        (
            "forward",
            (
                paddle.randint(low=0, high=10, shape=(50,)),
                paddle.randint(low=0, high=10, shape=(50,)),
            ),
        ),
        ("compute", None),
    ],
)
def test_raises_error_if_increment_not_called(method, method_input):
    """Test that error is raised if another method is called before increment."""
    tracker = MetricTracker(MulticlassAccuracy(num_classes=10))
    with pytest.raises(ValueError, match=f"`{method}` cannot be called before .*"):
        if method_input is not None:
            getattr(tracker, method)(*method_input)
        else:
            getattr(tracker, method)()


@pytest.mark.parametrize(
    ("base_metric", "metric_input", "maximize"),
    [
        (
            MulticlassAccuracy(num_classes=10),
            (
                paddle.randint(low=0, high=10, shape=(50,)),
                paddle.randint(low=0, high=10, shape=(50,)),
            ),
            True,
        ),
        (
            MulticlassPrecision(num_classes=10),
            (
                paddle.randint(low=0, high=10, shape=(50,)),
                paddle.randint(low=0, high=10, shape=(50,)),
            ),
            True,
        ),
        (
            MulticlassRecall(num_classes=10),
            (
                paddle.randint(low=0, high=10, shape=(50,)),
                paddle.randint(low=0, high=10, shape=(50,)),
            ),
            True,
        ),
        (MeanSquaredError(), (paddle.randn(50), paddle.randn(50)), False),
        (MeanAbsoluteError(), (paddle.randn(50), paddle.randn(50)), False),
        (
            MetricCollection(
                [
                    MulticlassAccuracy(num_classes=10),
                    MulticlassPrecision(num_classes=10),
                    MulticlassRecall(num_classes=10),
                ]
            ),
            (
                paddle.randint(low=0, high=10, shape=(50,)),
                paddle.randint(low=0, high=10, shape=(50,)),
            ),
            True,
        ),
        (
            MetricCollection(
                [
                    MulticlassAccuracy(num_classes=10),
                    MulticlassPrecision(num_classes=10),
                    MulticlassRecall(num_classes=10),
                ]
            ),
            (
                paddle.randint(low=0, high=10, shape=(50,)),
                paddle.randint(low=0, high=10, shape=(50,)),
            ),
            [True],
        ),
        (
            MetricCollection([MeanSquaredError(), MeanAbsoluteError()]),
            (paddle.randn(50), paddle.randn(50)),
            False,
        ),
        (
            MetricCollection([MeanSquaredError(), MeanAbsoluteError()]),
            (paddle.randn(50), paddle.randn(50)),
            [False, False],
        ),
        (
            ClasswiseWrapper(MulticlassAccuracy(num_classes=3, average=None)),
            (
                paddle.randint(low=0, high=3, shape=(50,)),
                paddle.randint(low=0, high=3, shape=(50,)),
            ),
            True,
        ),
    ],
)
def test_tracker(base_metric, metric_input, maximize):
    """Test that arguments gets passed correctly to child modules."""
    tracker = MetricTracker(base_metric, maximize=maximize)
    for i in range(5):
        tracker.increment()
        for _ in range(5):
            tracker.update(*metric_input)
        for _ in range(5):
            tracker(*metric_input)
        val = tracker.compute()
        if isinstance(val, dict):
            for v in val.values():
                assert v != 0.0
        else:
            assert val != 0.0
        assert tracker.n_steps == i + 1
    assert tracker.n_steps == 5
    all_computed_val = tracker.compute_all()
    if isinstance(all_computed_val, dict):
        for v in all_computed_val.values():
            assert v.size == 5
    else:
        assert all_computed_val.size == 5
    val, idx = tracker.best_metric(return_step=True)
    if isinstance(val, dict):
        for v, i in zip(val.values(), idx.values()):
            assert v != 0.0
            assert i in list(range(5))
    else:
        assert val != 0.0
        assert idx in list(range(5))
    val2 = tracker.best_metric(return_step=False)
    assert val == val2


@pytest.mark.parametrize(
    "base_metric",
    [
        pytest.param(MulticlassConfusionMatrix(3), id="Multiclass-confusion-matrix"),
        pytest.param(
            MetricCollection([MulticlassConfusionMatrix(3), MulticlassAccuracy(3)]),
            id="Metric-collection",
        ),
    ],
)
def test_best_metric_for_not_well_defined_metric_collection(base_metric):
    """Check for user warnings related to best metric.

    Test that if user tries to compute the best metric for a metric that does not have a well defined best, we throw an
    warning and return None.

    """
    tracker = MetricTracker(base_metric, maximize=True)
    for _ in range(3):
        tracker.increment()
        for _ in range(5):
            tracker.update(
                paddle.randint(low=0, high=3, shape=(10,)),
                paddle.randint(low=0, high=3, shape=(10,)),
            )
    with pytest.warns(
        UserWarning,
        match="Encountered the following error when trying to get the best metric.*",
    ):
        best = tracker.best_metric()
    if isinstance(best, dict):
        assert best["MulticlassAccuracy"] is not None
        assert best["MulticlassConfusionMatrix"] is None
    else:
        assert best is None
    with pytest.warns(
        UserWarning,
        match="Encountered the following error when trying to get the best metric.*",
    ):
        best, idx = tracker.best_metric(return_step=True)
    if isinstance(best, dict):
        assert best["MulticlassAccuracy"] is not None
        assert best["MulticlassConfusionMatrix"] is None
        assert idx["MulticlassAccuracy"] is not None
        assert idx["MulticlassConfusionMatrix"] is None
    else:
        assert best is None
        assert idx is None


@pytest.mark.parametrize(
    ("input_to_tracker", "assert_type"),
    [
        (MultioutputWrapper(MeanSquaredError(), num_outputs=2), paddle.Tensor),
        (
            MetricCollection(
                {
                    "mse": MultioutputWrapper(MeanSquaredError(), num_outputs=2),
                    "mae": MultioutputWrapper(MeanAbsoluteError(), num_outputs=2),
                }
            ),
            dict,
        ),
    ],
)
def test_metric_tracker_and_collection_multioutput(input_to_tracker, assert_type):
    """Check that MetricTracker support wrapper inputs and nested structures."""
    tracker = MetricTracker(input_to_tracker, maximize=False)
    for _ in range(5):
        tracker.increment()
        for _ in range(5):
            preds, target = paddle.randn(100, 2), paddle.randn(100, 2)
            tracker.update(preds, target)
    all_res = tracker.compute_all()
    assert isinstance(all_res, assert_type)
    best_metric, which_epoch = tracker.best_metric(return_step=True)
    if isinstance(best_metric, dict):
        for v in best_metric.values():
            assert v is None
        for v in which_epoch.values():
            assert v is None
    else:
        assert best_metric is None
        assert which_epoch is None


@pytest.mark.parametrize(
    "base_metric",
    [
        MeanSquaredError(),
        MeanAbsoluteError(),
        MulticlassAccuracy(num_classes=10),
        MetricCollection([MeanSquaredError(), MeanAbsoluteError()]),
        ClasswiseWrapper(MulticlassAccuracy(num_classes=10, average=None)),
        MetricCollection(
            [ClasswiseWrapper(MulticlassAccuracy(num_classes=10, average=None))]
        ),
    ],
)
def test_tracker_higher_is_better_integration(base_metric):
    """Check that the maximize argument is correctly set based on the metric higher_is_better attribute."""
    tracker = MetricTracker(base_metric, maximize=None)
    if isinstance(base_metric, Metric):
        assert tracker.maximize == base_metric.higher_is_better
    else:
        collection_higher_is_better = []
        for m in base_metric.values():
            if isinstance(m, ClasswiseWrapper):
                collection_higher_is_better.extend(
                    [m.higher_is_better] * m.metric.num_classes
                )
            else:
                collection_higher_is_better.append(m.higher_is_better)
        assert tracker.maximize == collection_higher_is_better


@pytest.mark.skipif(not _MATPLOTLIB_AVAILABLE, reason="matplotlib not available")
def test_plot():
    """Test the plot method of MetricTracker."""
    import matplotlib.pyplot as plt

    tracker = MetricTracker(MulticlassAccuracy(num_classes=10))
    for _ in range(3):
        tracker.increment()
        for _ in range(5):
            tracker.update(
                paddle.randint(low=0, high=10, shape=(50,)),
                paddle.randint(low=0, high=10, shape=(50,)),
            )
    fig, ax = tracker.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close()
    val = tracker.compute_all()
    fig, ax = tracker.plot(val)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close()
    fig, ax = plt.subplots()
    _, ax = tracker.plot(val, ax=ax)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close()
    values = [tracker.compute() for _ in range(3)]
    fig, ax = tracker.plot(values)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close()


def test_compute_all_edge_cases():
    """Test edge cases for compute_all method."""
    tracker = MetricTracker(MulticlassAccuracy(num_classes=10))
    with pytest.raises(ValueError, match="`compute_all` cannot be called before"):
        tracker.compute_all()
    tracker = MetricTracker(
        MetricCollection(
            [MulticlassAccuracy(num_classes=10), MulticlassPrecision(num_classes=10)]
        )
    )
    for _ in range(3):
        tracker.increment()
        for _ in range(5):
            tracker.update(
                paddle.randint(low=0, high=10, shape=(50,)),
                paddle.randint(low=0, high=10, shape=(50,)),
            )
    results = tracker.compute_all()
    assert isinstance(results, dict)
    assert len(results) == 2
    for v in results.values():
        assert v.size == 3


def test_best_metric_edge_cases():
    """Test edge cases for best_metric method."""
    tracker = MetricTracker(MulticlassConfusionMatrix(num_classes=3), maximize=True)
    for _ in range(3):
        tracker.increment()
        for _ in range(5):
            tracker.update(
                paddle.randint(low=0, high=3, shape=(10,)),
                paddle.randint(low=0, high=3, shape=(10,)),
            )
    with pytest.warns(
        UserWarning,
        match="Encountered the following error when trying to get the best metric",
    ):
        best = tracker.best_metric()
    assert best is None
    tracker = MetricTracker(
        MetricCollection(
            [
                MulticlassConfusionMatrix(num_classes=3),
                MulticlassAccuracy(num_classes=3),
            ]
        ),
        maximize=True,
    )
    for _ in range(3):
        tracker.increment()
        for _ in range(5):
            tracker.update(
                paddle.randint(low=0, high=3, shape=(10,)),
                paddle.randint(low=0, high=3, shape=(10,)),
            )
    with pytest.warns(
        UserWarning,
        match="Encountered the following error when trying to get the best metric",
    ):
        best, idx = tracker.best_metric(return_step=True)
    assert best["MulticlassConfusionMatrix"] is None
    assert best["MulticlassAccuracy"] is not None
    assert idx["MulticlassConfusionMatrix"] is None
    assert idx["MulticlassAccuracy"] is not None
