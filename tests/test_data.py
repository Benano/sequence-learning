#!/usr/bin/env python3

import numpy as np
import pytest
from numpy.testing import assert_allclose

from elise.data import (
    CirclePattern,
    ContinuousDataloader,
    Dataloader,
    MultiHotPattern,
    OneHotPattern,
    Pattern,
    ShuffleDataloader,
)

DT = 0.1

# TODO use the factory approach

###############################
# test the Base Pattern class #
###############################


@pytest.fixture()
def base_sequence():
    pattern = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return pattern


@pytest.fixture()
def base_sequence2():
    pattern = np.array(
        [
            [0.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 2.0],
        ]
    )
    return pattern


@pytest.fixture()
def base_sequence3():
    pattern = np.array(
        [
            [0.0, 3.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [3.0, 3.0, 3.0],
        ]
    )
    return pattern


@pytest.fixture()
def pattern_factory(base_sequence):
    def factory(sequence=None, dt=None, duration=None):
        if sequence is None:
            sequence = base_sequence
        if dt is None and duration is None:
            dt = DT
        pattern = Pattern(sequence, dt=dt, duration=duration)
        return pattern

    return factory


@pytest.fixture()
def transform_plus():
    def plus(x, a=2.0):
        return x + a

    return plus


@pytest.fixture()
def transform_mult():
    def mult(x, m=2.0):
        return x * m

    return mult


@pytest.fixture()
def transform_reshape():
    def reshape(x):
        return np.expand_dims(x, -1)

    return reshape


class TestPattern:
    @pytest.mark.parametrize(("dt", "dur"), [(0.1, None), (None, 0.6)])
    def test_init(self, pattern_factory, base_sequence, dt, dur):
        pat = pattern_factory(dt=dt, duration=dur)
        assert pat.dt == pytest.approx(DT)
        assert pat.duration == pytest.approx(len(base_sequence) * DT)
        assert pat.shape == base_sequence.shape
        assert_allclose(base_sequence, pat.pattern)

    def test_init_error(self, pattern_factory):
        with pytest.raises(ValueError):
            _ = pattern_factory(dt=0.3, duration=0.7)

    def test_len(self, pattern_factory, base_sequence):
        pat = pattern_factory()
        assert len(pat) == base_sequence.shape[0]

    @pytest.mark.parametrize(
        "idx", [0, (2, 1), np.s_[4, 2], np.s_[:, 1], slice(0, 4, 2)]
    )
    def test_getitem(self, pattern_factory, base_sequence, idx):
        pat = pattern_factory()
        assert_allclose(pat[idx], base_sequence[idx])

    def test_transform(self, pattern_factory, base_sequence, transform_plus):
        pat = pattern_factory()
        pat.transform(transform_plus)
        expected = transform_plus(base_sequence)
        assert_allclose(pat.pattern, expected)

    def test_pretransform_reshape_error(self, pattern_factory, transform_reshape):
        pat = pattern_factory()
        with pytest.raises(ValueError) as exc_info:  # noqa
            pat.transform(transform_reshape)


################################
# Test the OneHotPattern class #
################################


@pytest.fixture()
def onehot_sequence():
    return np.array([1, 1, 0, 0, 2, 2])


@pytest.fixture()
def onehot_pattern(onehot_sequence):
    return OneHotPattern(onehot_sequence, dt=DT, width=3)


def test_onehot(onehot_pattern, base_sequence):
    assert_allclose(onehot_pattern[:, :], base_sequence)


##################################
# Test the MultiHotPattern class #
##################################


@pytest.fixture()
def multihot_sequence():
    return [2, [1, 4], -1]


@pytest.fixture()
def multihot_pattern(multihot_sequence):
    return MultiHotPattern(multihot_sequence, duration=0.3, width=5)


def test_multihot(multihot_pattern):
    expected = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert_allclose(multihot_pattern[:, :], expected)


###############################
# Test the continuous pattern #
###############################


@pytest.fixture()
def circle_pattern():
    def factory(radius=0.5, center_x=0.5, center_y=0.5, period=1.0):
        return CirclePattern(
            radius=radius, center_x=center_x, center_y=center_y, period=period
        )

    return factory


def test_circle_pattern_attributes(circle_pattern):
    circle = circle_pattern(radius=1.0, center_x=0.5, center_y=-0.5, period=3.0)
    assert circle.radius == 1.0
    assert circle.center_x == 0.5
    assert circle.center_y == -0.5
    assert circle.duration == 3.0


@pytest.mark.parametrize("t", [0.0, 0.5, 0.7])
def test_circle_pattern_call(circle_pattern, t):
    radius, center_x, center_y, period = 1.0, 0.0, 0.0, 1.0
    circle = circle_pattern(radius, center_x, center_y, period)
    exp = np.array([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])
    assert_allclose(circle(t), exp)


###################
# Test Dataloader #
###################


@pytest.fixture()
def dataloader_factory(pattern_factory):
    def factory(pattern=None, pre_transforms=[], online_transforms=[]):
        if pattern is None:
            pattern = pattern_factory()
        return Dataloader(
            pattern, pre_transforms=pre_transforms, online_transforms=online_transforms
        )

    return factory


class TestDataloader:
    @pytest.mark.parametrize("transforms", [[], [transform_plus, transform_mult]])
    def test_init(self, dataloader_factory, base_sequence, transforms):
        dataloader = dataloader_factory(online_transforms=transforms)
        assert_allclose(dataloader.pattern[:, :], base_sequence)
        assert dataloader.dt == pytest.approx(DT)
        assert dataloader.online_transforms == transforms

    def test_pretransform_reshape_error(self, dataloader_factory, transform_reshape):
        with pytest.raises(ValueError) as exc_info:  # noqa
            dataloader = dataloader_factory(pre_transforms=[transform_reshape])  # noqa

    def test_pretransforms(
        self, dataloader_factory, transform_plus, transform_mult, base_sequence
    ):
        dataloader = dataloader_factory(pre_transforms=[transform_plus, transform_mult])
        expected = transform_mult(transform_plus(base_sequence))
        assert_allclose(dataloader.pattern, expected)

    @pytest.mark.parametrize(
        ("t", "idx"),
        [
            (0.1, 1),
            (0.5, 5),
            (0.6, 0),
            (0.7, 1),
            (1.1, -1),
            (1.2, 0),
            (0.05, 0),
            (0.15, 1),
            (0.55, 5),
            (0.61, 0),
        ],
    )
    def test_call(self, dataloader_factory, base_sequence, t, idx):
        dataloader = dataloader_factory()
        assert_allclose(dataloader(t), base_sequence[idx])

    @pytest.mark.parametrize(
        ("T", "idx", "dt"),
        [
            (0.5, 5, 0.1),
            (0.5, 5, 0.23),
            (0.5, 5, 0.01),
            (0.6, 0, 0.1),
            (0.6, 0, 0.23),
            (0.6, 0, 0.01),
            (1.1, -1, 0.1),
            (1.1, -1, 0.23),
            (1.1, -1, 0.01),
        ],
    )
    def test_call_consistency(self, dataloader_factory, base_sequence, T, idx, dt):
        dataloader = dataloader_factory()
        for t in np.arange(0.0, T, dt):
            _ = dataloader(t)
        assert_allclose(dataloader(T), base_sequence[idx])

    @pytest.mark.parametrize(
        ("T", "idx", "dt"),
        [
            (0.5, 5, 0.1),
            (0.6, 0, 0.1),
            (1.1, -1, 0.1),
        ],
    )
    def test_online_transforms(
        self,
        dataloader_factory,
        base_sequence,
        transform_plus,
        transform_mult,
        T,
        idx,
        dt,
    ):
        dataloader = dataloader_factory(
            online_transforms=[transform_plus, transform_mult]
        )
        for t in np.arange(0.0, T, dt):
            _ = dataloader(t)
        expected = transform_mult(transform_plus(base_sequence[idx]))
        assert_allclose(dataloader(T), expected)

    def test_iter(self, dataloader_factory, base_sequence):
        dataloader = dataloader_factory()
        idx = 0
        t_exp = 0.0
        for t, pat in dataloader.iter(0.0, 1.2, 0.1):
            assert t == t_exp
            assert_allclose(pat, base_sequence[idx % len(base_sequence)])
            t_exp += 0.1
            idx += 1

    def test_get_full_pattern(self, dataloader_factory, base_sequence):
        dataloader = dataloader_factory()
        res = dataloader.get_full_pattern(DT)
        assert_allclose(res, base_sequence)

    def test_Dataloader_factory_error(self, dataloader_factory):
        with pytest.raises(TypeError):
            dataloader_factory(pattern="Not a Pattern.")


#############################
# Test ContinuousDataloader #
#############################


@pytest.fixture()
def continuous_dataloader_factory(circle_pattern):
    def factory(pattern=None, pre_transforms=[], online_transforms=[]):
        if pattern is None:
            pattern = circle_pattern()
        return Dataloader(
            pattern, pre_transforms=pre_transforms, online_transforms=online_transforms
        )

    return factory


class TestContiDataloader:
    def test_bla(self, continuous_dataloader_factory):
        dl = continuous_dataloader_factory()
        assert isinstance(dl, ContinuousDataloader)
        assert isinstance(dl(0), np.ndarray)

    @pytest.mark.parametrize("t", [0.0, 0.1, 0.4])
    def test_call(self, continuous_dataloader_factory, circle_pattern, t):
        circle = circle_pattern()
        dataloader = continuous_dataloader_factory()
        res = dataloader(t)
        exp = circle(t)
        assert_allclose(res, exp)

    @pytest.mark.parametrize(
        "t",
        [
            0.0,
            0.1,
            0.4,
        ],
    )
    def test_pretransforms(
        self,
        continuous_dataloader_factory,
        transform_plus,
        transform_mult,
        circle_pattern,
        t,
    ):
        dataloader = continuous_dataloader_factory(
            pre_transforms=[transform_mult, transform_plus]
        )
        circle = circle_pattern()
        res = dataloader(t)
        exp = transform_plus(transform_mult(circle(t)))
        assert_allclose(res, exp)


##########################
# Test ShuffleDataloader #
##########################

SHUFFLE_SIZE = 4
SHUFFLE_SEED = 42


@pytest.fixture()
def shuffled_idx_factory():
    def factory(num_pattern, seed=None, length=None):
        if seed is None:
            seed = SHUFFLE_SEED
        if length is None:
            length = SHUFFLE_SIZE
        rng = np.random.default_rng(SHUFFLE_SEED)
        return rng.choice(num_pattern, size=length)

    return factory


@pytest.fixture()
def multisequence_target_factory():
    def factory(pattern_list, shuffled_idx):
        return np.concatenate([pattern_list[i].pattern for i in shuffled_idx], axis=0)

    return factory


@pytest.fixture()
def multisequence(pattern_factory, base_sequence, base_sequence2, base_sequence3):
    return [
        pattern_factory(base_sequence),
        pattern_factory(base_sequence2),
        pattern_factory(base_sequence3),
    ]


@pytest.fixture()
def shuffledataloader_factory(multisequence):
    def factory(
        pattern=None, length=None, seed=None, pre_transforms=[], online_transforms=[]
    ):
        if pattern is None:
            pattern = multisequence
        if length is None:
            length = SHUFFLE_SIZE
        if seed is None:
            seed = SHUFFLE_SEED
        max_dur = max([pat.duration for pat in pattern])
        return ShuffleDataloader(
            pattern=pattern,
            t_max=max_dur * SHUFFLE_SIZE,
            seed=seed,
            pre_transforms=pre_transforms,
            online_transforms=online_transforms,
        )

    return factory


class TestShuffleDataloader:
    def test_init(self, multisequence, shuffledataloader_factory):
        dataloader = shuffledataloader_factory(
            pattern=multisequence, length=SHUFFLE_SIZE, seed=SHUFFLE_SEED
        )
        assert len(dataloader.pattern) == len(multisequence)
        assert len(dataloader.dts) == len(multisequence)
        assert len(dataloader.durations) == len(multisequence)

    def test_shuffled_idx(
        self, multisequence, shuffledataloader_factory, shuffled_idx_factory
    ):
        dataloader = shuffledataloader_factory(
            pattern=multisequence, length=SHUFFLE_SIZE, seed=SHUFFLE_SEED
        )
        expected_random_idx = shuffled_idx_factory(
            num_pattern=len(multisequence), seed=SHUFFLE_SEED, length=SHUFFLE_SIZE
        )
        assert len(dataloader._pat_ids) >= len(expected_random_idx)

        assert_allclose(
            dataloader._pat_ids[: len(expected_random_idx)], expected_random_idx
        )

    def test_pretransforms(
        self, multisequence, shuffledataloader_factory, transform_plus, transform_mult
    ):
        dataloader = shuffledataloader_factory(
            pattern=multisequence,
            length=SHUFFLE_SIZE,
            seed=SHUFFLE_SEED,
            pre_transforms=[transform_plus, transform_mult],
        )
        for i, pat in enumerate(multisequence):
            expected = transform_mult(transform_plus(pat[:, :]))
            assert_allclose(dataloader.pattern[i], expected)

    @pytest.mark.parametrize(
        ("t", "idx"),
        [
            (0.0, 0),
            (0.1, 1),
            (0.2, 2),
            (0.3, 3),
            (0.4, 4),
            (0.5, 5),
            (0.6, 6),
            (0.7, 7),
            (1.2, 12),
            (1.3, 13),
            (1.4, 14),
            (0.76, 7),
            (0.399, 3),
            (0.05, 0),
        ],
    )
    def test_call(
        self,
        multisequence,
        shuffledataloader_factory,
        shuffled_idx_factory,
        multisequence_target_factory,
        t,
        idx,
    ):
        dataloader = shuffledataloader_factory(
            pattern=multisequence, length=SHUFFLE_SIZE, seed=SHUFFLE_SEED
        )
        expected_random_idx = shuffled_idx_factory(
            num_pattern=len(multisequence), seed=SHUFFLE_SEED, length=SHUFFLE_SIZE
        )
        expected_pattern = multisequence_target_factory(
            multisequence, expected_random_idx
        )

        assert_allclose(dataloader(t), expected_pattern[idx])

    @pytest.mark.parametrize(
        ("t", "idx"),
        [
            (0.0, 0),
            (0.4, 4),
            (0.5, 5),
            (0.7, 7),
            (1.2, 12),
        ],
    )
    def test_online_transforms(
        self,
        multisequence,
        shuffledataloader_factory,
        shuffled_idx_factory,
        multisequence_target_factory,
        transform_plus,
        transform_mult,
        t,
        idx,
    ):
        dataloader = shuffledataloader_factory(
            pattern=multisequence,
            length=SHUFFLE_SIZE,
            seed=SHUFFLE_SEED,
            online_transforms=[transform_plus, transform_mult],
        )
        expected_random_idx = shuffled_idx_factory(
            num_pattern=len(multisequence), seed=SHUFFLE_SEED, length=SHUFFLE_SIZE
        )
        expected_pattern = multisequence_target_factory(
            multisequence, expected_random_idx
        )

        expected = transform_mult(transform_plus(expected_pattern[idx]))
        assert_allclose(dataloader(t), expected)

    def test_iter(
        self,
        multisequence,
        dataloader_factory,
        shuffled_idx_factory,
        multisequence_target_factory,
        shuffledataloader_factory,
    ):
        dataloader = shuffledataloader_factory(
            pattern=multisequence, length=SHUFFLE_SIZE, seed=SHUFFLE_SEED
        )
        expected_random_idx = shuffled_idx_factory(
            num_pattern=len(multisequence), seed=SHUFFLE_SEED, length=SHUFFLE_SIZE
        )
        expected_pattern = multisequence_target_factory(
            multisequence, expected_random_idx
        )

        res_iter = [x[1] for x in dataloader.iter(0.0, 1.0, DT)]

        for res, exp in zip(res_iter, expected_pattern):
            assert_allclose(res, exp)

    def test_get_full_pattern(
        self,
        multisequence,
        shuffledataloader_factory,
        base_sequence,
        base_sequence2,
        base_sequence3,
    ):
        dataloader = shuffledataloader_factory(
            pattern=multisequence, length=SHUFFLE_SIZE, seed=SHUFFLE_SEED
        )

        res = dataloader.get_full_pattern(dt=DT)
        expected = np.concatenate(
            (base_sequence, base_sequence2, base_sequence3), axis=0
        )
        assert_allclose(res, expected)
