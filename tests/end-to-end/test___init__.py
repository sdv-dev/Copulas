import numpy as np

from copulas import random_state


class TestRandomState:
    """Test class for the random state wrapper."""

    def init(self, random_state=None):
        self.random_state = random_state

    def set_random_state(self, random_state):
        self.random_state = random_state

    @random_state
    def sample(self):
        pass


def test_random_state_decorator():
    """Test the ``random_state`` decorator end-to-end.

    Expect that the random state wrapper leaves the global state
    where it left off.

    Setup:
        - A global random state is initialized with a seed of 42.
        - The state is advanced by generating two random values.
    Input:
        - A seed of 0 is given to the test instance.
    Side Effects:
        - Sampling two more random values after the test instance
          method completes is expected to continue where the random
          state left off.
    """
    # Setup
    original_state = np.random.get_state()

    # Get the expected random sequence with a seed of 42.
    _SEED = 42
    np.random.seed(_SEED)
    expected = np.random.random(size=4)

    # Set the global random state.
    new_state = np.random.RandomState(seed=_SEED).get_state()
    np.random.set_state(new_state)

    first_sequence = np.random.random(size=2)

    # Run
    instance = TestRandomState()
    instance.set_random_state(np.random.RandomState(10))
    instance.sample()

    second_sequence = np.random.random(size=2)

    # Assert
    np.testing.assert_array_equal(first_sequence, expected[:2])
    np.testing.assert_array_equal(second_sequence, expected[2:])

    # Cleanup
    np.random.set_state(original_state)
