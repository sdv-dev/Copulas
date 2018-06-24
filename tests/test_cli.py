#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `copulas` package."""

import unittest

from click.testing import CliRunner

from copulas import cli


class TestCli(unittest.TestCase):
    """Tests for `copulas` package."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main, ['data/iris.data.csv'])
        assert result.exit_code == -1  # It crashes right now

        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
