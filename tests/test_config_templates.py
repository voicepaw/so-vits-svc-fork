"""Test that config templates are accessible and valid."""

from pathlib import Path
from unittest import TestCase


class TestConfigTemplates(TestCase):
    """Test config template accessibility."""

    def test_config_template_dir_exists_in_source(self):
        """Test that the config template directory exists in source code."""
        # Test the source code location directly
        config_template_dir = (
            Path(__file__).parent.parent
            / "src"
            / "so_vits_svc_fork"
            / "preprocessing"
            / "config_templates"
        )

        self.assertTrue(
            config_template_dir.exists(),
            f"Config template directory does not exist: {config_template_dir}",
        )

    def test_config_templates_are_available_in_source(self):
        """Test that config template JSON files are available in source code."""
        config_template_dir = (
            Path(__file__).parent.parent
            / "src"
            / "so_vits_svc_fork"
            / "preprocessing"
            / "config_templates"
        )

        json_files = list(config_template_dir.rglob("*.json"))
        self.assertGreater(
            len(json_files),
            0,
            f"No JSON files found in config template directory: {config_template_dir}",
        )

    def test_expected_config_templates_exist_in_source(self):
        """Test that expected config templates exist in source code."""
        config_template_dir = (
            Path(__file__).parent.parent
            / "src"
            / "so_vits_svc_fork"
            / "preprocessing"
            / "config_templates"
        )

        expected_templates = [
            "so-vits-svc-4.0v1",
            "so-vits-svc-4.0v1-legacy",
            "quickvc",
        ]

        available_templates = [x.stem for x in config_template_dir.rglob("*.json")]

        for template in expected_templates:
            self.assertIn(
                template,
                available_templates,
                f"Expected config template '{template}' not found. Available: {available_templates}",
            )

    def test_cli_config_type_choices_in_source(self):
        """Test that CLI config type choices are properly populated in source code."""
        config_template_dir = (
            Path(__file__).parent.parent
            / "src"
            / "so_vits_svc_fork"
            / "preprocessing"
            / "config_templates"
        )

        # This simulates what happens in __main__.py line 533
        choices = [x.stem for x in config_template_dir.rglob("*.json")]

        self.assertGreater(
            len(choices),
            0,
            "Config type choices list is empty - CLI will fail with 'not one of .' error",
        )
        self.assertIn(
            "so-vits-svc-4.0v1",
            choices,
            f"Default config type 'so-vits-svc-4.0v1' not in choices: {choices}",
        )
