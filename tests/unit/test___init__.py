"""Top level unit tests."""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

import copulas
from copulas import _find_addons


@pytest.fixture
def mock_copulas():
    copulas_module = sys.modules['copulas']
    copulas_mock = MagicMock()
    sys.modules['copulas'] = copulas_mock
    yield copulas_mock
    sys.modules['copulas'] = copulas_module


@patch.object(copulas, 'entry_points')
def test__find_addons_module(entry_points_mock, mock_copulas):
    """Test loading an add-on."""
    # Setup
    entry_point = MagicMock()
    entry_point.name = 'copulas.submodule.entry_name'
    entry_point.load.return_value = 'entry_point'
    entry_points_mock.return_value = [entry_point]

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    assert mock_copulas.submodule.entry_name == 'entry_point'


@patch.object(copulas, 'entry_points')
def test__find_addons_object(entry_points_mock, mock_copulas):
    """Test loading an add-on."""
    # Setup
    entry_point = MagicMock()
    entry_point.name = 'copulas.submodule:entry_object.entry_method'
    entry_point.load.return_value = 'new_method'
    entry_points_mock.return_value = [entry_point]

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    assert mock_copulas.submodule.entry_object.entry_method == 'new_method'


@patch('warnings.warn')
@patch('copulas.entry_points')
def test__find_addons_bad_addon(entry_points_mock, warning_mock):
    """Test failing to load an add-on generates a warning."""

    # Setup
    def entry_point_error():
        raise ValueError('bad value')

    bad_entry_point = Mock()
    bad_entry_point.name = 'bad_entry_point'
    bad_entry_point.value = 'bad_module'
    bad_entry_point.load.side_effect = entry_point_error
    entry_points_mock.return_value = [bad_entry_point]
    msg = 'Failed to load "bad_entry_point" from "bad_module" with error:\nbad value'

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch('copulas.entry_points')
def test__find_addons_wrong_base(entry_points_mock, warning_mock):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = MagicMock()
    bad_entry_point.name = 'bad_base.bad_entry_point'
    entry_points_mock.return_value = [bad_entry_point]
    msg = (
        "Failed to set 'bad_base.bad_entry_point': expected base module to be 'copulas', found "
        "'bad_base'."
    )

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch('copulas.entry_points')
def test__find_addons_missing_submodule(entry_points_mock, warning_mock):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = MagicMock()
    bad_entry_point.name = 'copulas.missing_submodule.new_submodule'
    entry_points_mock.return_value = [bad_entry_point]
    msg = (
        "Failed to set 'copulas.missing_submodule.new_submodule': module 'copulas' has no "
        "attribute 'missing_submodule'."
    )

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch('copulas.entry_points')
def test__find_addons_module_and_object(entry_points_mock, warning_mock):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = MagicMock()
    bad_entry_point.name = 'copulas.missing_submodule:new_object'
    entry_points_mock.return_value = [bad_entry_point]
    msg = (
        "Failed to set 'copulas.missing_submodule:new_object': cannot add 'new_object' to unknown "
        "submodule 'copulas.missing_submodule'."
    )

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    warning_mock.assert_called_once_with(msg)


@patch('warnings.warn')
@patch.object(copulas, 'entry_points')
def test__find_addons_missing_object(entry_points_mock, warning_mock, mock_copulas):
    """Test incorrect add-on name generates a warning."""
    # Setup
    bad_entry_point = MagicMock()
    bad_entry_point.name = 'copulas.submodule:missing_object.new_method'
    entry_points_mock.return_value = [bad_entry_point]
    msg = "Failed to set 'copulas.submodule:missing_object.new_method': missing_object."

    del mock_copulas.submodule.missing_object

    # Run
    _find_addons()

    # Assert
    entry_points_mock.assert_called_once_with(group='copulas_modules')
    warning_mock.assert_called_once_with(msg)
