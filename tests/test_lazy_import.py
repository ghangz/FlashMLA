import importlib
import unittest


class LazyImportTest(unittest.TestCase):
    def test_import_package_without_compiled_extension(self):
        module = importlib.import_module("flash_mla")

        self.assertEqual(module.__version__, "1.0.1")
        self.assertTrue(callable(module.get_mla_metadata))


if __name__ == "__main__":
    unittest.main()
