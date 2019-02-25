from unittest import TestCase
from unittest.mock import MagicMock, patch

from copulas import random_state


class TestRandomStateDecorator(TestCase):

    @patch('copulas.np.random')
    def test_valid_random_state(self, random_mock):
        """The decorated function use the random_seed attribute if present."""
        # Setup
        my_function = MagicMock()
        instance = MagicMock()
        instance.random_seed = 42

        args = ('some', 'args')
        kwargs = {'keyword': 'value'}

        random_mock.get_state.return_value = "random state"

        # Run
        decorated_function = random_state(my_function)
        decorated_function(instance, *args, **kwargs)

        # Check
        my_function.assert_called_once_with(instance, *args, **kwargs)

        instance.assert_not_called
        random_mock.get_state.assert_called_once_with()
        random_mock.seed.assert_called_once_with(42)
        random_mock.set_state.assert_called_once_with("random state")

    @patch('copulas.np.random')
    def test_no_random_state(self, random_mock):
        """If random_seed is None, the decorated function only call to the original."""
        # Setup
        my_function = MagicMock()
        instance = MagicMock()
        instance.random_seed = None

        args = ('some', 'args')
        kwargs = {'keyword': 'value'}

        random_mock.get_state.return_value = "random state"

        # Run
        decorated_function = random_state(my_function)
        decorated_function(instance, *args, **kwargs)

        # Check
        my_function.assert_called_once_with(instance, *args, **kwargs)

        instance.assert_not_called
        random_mock.get_state.assert_not_called()
        random_mock.seed.assert_not_called()
        random_mock.set_state.assert_not_called()
