"""Regression checks for the W-space to DKL calibration pipeline."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.color_thres import ColorTransFilenames, color_thresholds


class TestW3DToDKL(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.colors = color_thresholds(3, directory.name)
        self.colors.M_RGBToLMS = np.array([[2., 1., 0.], [0., 3., 1.], [1., 0., 4.]])
        self.colors.M_ConeIncToDKL = np.array([[1., 2., 1.], [1., -1., 0.], [0., 1., -2.]])

    def test_background_is_origin(self):
        bg = [0.2, -0.3, 0.4]
        np.testing.assert_array_equal(self.colors.W3D_to_dkl(bg, bg), np.zeros(3))

    def test_batch_matches_absolute_rgb_pipeline(self):
        colors = self.colors
        stimuli = np.array([[0.2, 0.4, -0.6], [-0.8, 0.1, 0.5], [0., 0.3, 0.7]])
        background = np.array([0.1, -0.2, 0.3])
        for bg in (background, np.array([background, background * 2, -background])):
            with self.subTest(shape=bg.shape):
                expected = []
                for stimulus, ref in zip(stimuli, np.broadcast_to(bg, stimuli.shape)):
                    lms_s = colors.M_RGBToLMS @ colors.W_unit_to_N_unit(stimulus)
                    lms_bg = colors.M_RGBToLMS @ colors.W_unit_to_N_unit(ref)
                    expected.append(colors.M_ConeIncToDKL @ (lms_s - lms_bg))
                np.testing.assert_allclose(colors.W3D_to_dkl(stimuli, bg), expected)
                ref = bg[0] if bg.ndim == 2 else bg
                single = colors.W3D_to_dkl(stimuli[0].tolist(), ref)
                self.assertEqual(single.shape, (3,))
                np.testing.assert_allclose(single, expected[0])

    def test_former_name_is_alias(self):
        self.assertIs(color_thresholds.W3D_to_DKL, color_thresholds.W3D_to_dkl)

    def test_empty_batch(self):
        result = self.colors.W3D_to_dkl(np.empty((0, 3)), [0, 0, 0])
        self.assertEqual(result.shape, (0, 3))

    def test_invalid_inputs(self):
        for s, bg in [
            (0, [0, 0, 0]), ([1, 2], [0, 0, 0]),
            (np.zeros((3, 2)), [0, 0, 0]), (np.zeros((2, 3)), np.zeros((1, 3))),
            ([0, 0, 0], np.zeros((2, 3))), ([np.nan, 0, 0], [0, 0, 0]),
            ([0, 0, 0], [0, np.inf, 0]),
        ]:
            with self.subTest(s=s, bg=bg), self.assertRaises(ValueError):
                self.colors.W3D_to_dkl(s, bg)

    def test_load_pairs_and_clear_on_failure(self):
        names = ColorTransFilenames()
        pairs = (
            ("M_2DWToRGB", "M_RGBTo2DW"),
            ("M_2DWToDKL", "M_DKLTo2DW"),
            ("M_LMSToRGB", "M_RGBToLMS"),
            ("M_ConeIncToDKL", "M_DKLToConeInc"),
        )
        for forward, inverse in pairs:
            np.savetxt(self.path / getattr(names, forward + "_filename"), np.eye(3) * 2, delimiter=",")
            np.savetxt(self.path / getattr(names, inverse + "_filename"), np.eye(3) / 2, delimiter=",")
        self.colors.load_transformation_matrix()
        for forward, inverse in pairs:
            np.testing.assert_array_equal(getattr(self.colors, forward), np.eye(3) * 2)
            np.testing.assert_array_equal(getattr(self.colors, inverse), np.eye(3) / 2)
            for name in (forward, inverse):
                self.assertEqual(getattr(self.colors, "file_path_" + name),
                                 str(self.path / getattr(names, name + "_filename")))

        # Reading the second file fails after the first matrix was read.
        for _, inverse in pairs:
            (self.path / getattr(names, inverse + "_filename")).write_text("")
        self.colors.load_transformation_matrix()
        for pair in pairs:
            for name in pair:
                self.assertIsNone(getattr(self.colors, name))
                self.assertIsNone(getattr(self.colors, "file_path_" + name))

        # Missing calibration files also clear every attribute.
        self.colors.load_transformation_matrix("07292025")
        for pair in pairs:
            for name in pair:
                self.assertIsNone(getattr(self.colors, name))
                self.assertIsNone(getattr(self.colors, "file_path_" + name))

    def test_incomplete_pair_is_not_loaded(self):
        names = ColorTransFilenames()
        np.savetxt(self.path / names.M_ConeIncToDKL_filename, np.eye(3), delimiter=",")
        self.colors.load_transformation_matrix()
        self.assertIsNone(self.colors.M_ConeIncToDKL)
        self.assertIsNone(self.colors.M_DKLToConeInc)
        self.assertIsNone(self.colors.file_path_M_ConeIncToDKL)
        self.assertIsNone(self.colors.file_path_M_DKLToConeInc)


class TestPointAndCovarianceTransforms(unittest.TestCase):
    """Check inverses and that each covariance map is J @ Sigma @ J.T."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.colors = color_thresholds(3, directory.name)
        self.colors.M_RGBToLMS = np.array([[2., 1., 0.], [0., 3., 1.], [1., 0., 4.]])
        self.colors.M_LMSToRGB = np.linalg.inv(self.colors.M_RGBToLMS)
        self.colors.M_ConeIncToDKL = np.array([[1., 2., 1.], [1., -1., 0.], [0., 1., -2.]])
        self.colors.M_DKLToConeInc = np.linalg.inv(self.colors.M_ConeIncToDKL)

        rng = np.random.default_rng(0)
        self.points = rng.uniform(-0.9, 0.9, (5, 3))
        A = rng.normal(size=(5, 3, 3))
        self.covs = A @ np.swapaxes(A, -1, -2) + 0.1 * np.eye(3)
        self.bg = np.array([0.1, -0.2, 0.3])
        self.ambient = np.array([0.05, 0.02, 0.01])
        self.bg_lms = self.colors.W3D_to_lms(self.bg, self.ambient)

    @staticmethod
    def jacobian(f, x, h=1e-6):
        return np.stack([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(3)], axis=-1)

    def test_point_round_trips(self):
        c, W = self.colors, self.points
        np.testing.assert_allclose(c.dkl_to_W3D(c.W3D_to_dkl(W, self.bg), self.bg), W)
        np.testing.assert_allclose(c.dkl_to_W3D(np.zeros(3), self.bg), self.bg)
        np.testing.assert_allclose(c.lms_to_W3D(c.W3D_to_lms(W, self.ambient), self.ambient), W)
        single = c.dkl_to_W3D(c.W3D_to_dkl(W[0], self.bg), self.bg)
        self.assertEqual(single.shape, (3,))

    def test_lms_matches_cone_contrast(self):
        c = self.colors
        lms = c.W3D_to_lms(self.points, self.ambient)
        np.testing.assert_allclose(
            (lms - self.bg_lms) / self.bg_lms,
            c.W3D_to_cc(self.points, self.bg_lms, self.ambient),
        )

    def test_forward_covariances_match_jacobians(self):
        c = self.colors
        cases = {
            "RGB": (c.W_unit_to_N_unit, c.W3D_to_rgb_covM),
            "LMS": (lambda x: c.W3D_to_lms(x, self.ambient), c.W3D_to_lms_covM),
            "cc": (lambda x: c.W3D_to_cc(x, self.bg_lms, self.ambient),
                   lambda S: c.W3D_to_cc_covM(S, self.bg_lms)),
            "DKL": (lambda x: c.W3D_to_dkl(x, self.bg), c.W3D_to_dkl_covM),
        }
        for name, (f, to_cov) in cases.items():
            J = self.jacobian(f, self.points[0])
            with self.subTest(space=name):
                np.testing.assert_allclose(
                    to_cov(self.covs), J @ self.covs @ J.T, rtol=1e-6, atol=1e-9)
                np.testing.assert_allclose(
                    to_cov(self.covs[0]), J @ self.covs[0] @ J.T, rtol=1e-6, atol=1e-9)

    def test_covariance_round_trips(self):
        c, S = self.colors, self.covs
        pairs = {
            "RGB": (c.W3D_to_rgb_covM, c.rgb_to_W3D_covM),
            "LMS": (c.W3D_to_lms_covM, c.lms_to_W3D_covM),
            "cc": (lambda X: c.W3D_to_cc_covM(X, self.bg_lms),
                   lambda X: c.cc_to_W3D_covM(X, self.bg_lms)),
            "DKL": (c.W3D_to_dkl_covM, c.dkl_to_W3D_covM),
        }
        for name, (forward, inverse) in pairs.items():
            with self.subTest(space=name):
                np.testing.assert_allclose(inverse(forward(S)), S)

    def test_cc_covariance_with_per_point_background(self):
        c = self.colors
        bg_lms = c.W3D_to_lms(self.points, self.ambient)
        batch = c.W3D_to_cc_covM(self.covs, bg_lms)
        for i in range(len(self.covs)):
            np.testing.assert_allclose(batch[i], c.W3D_to_cc_covM(self.covs[i], bg_lms[i]))
        np.testing.assert_allclose(c.cc_to_W3D_covM(batch, bg_lms), self.covs)

    def test_invalid_covariance_inputs(self):
        c = self.colors
        for bad in (np.eye(2), np.zeros((2, 2, 3)), np.zeros((1, 1, 3, 3))):
            with self.subTest(shape=np.shape(bad)), self.assertRaises(ValueError):
                c.W3D_to_dkl_covM(bad)
        with self.assertRaises(ValueError):
            c.W3D_to_cc_covM(self.covs, np.ones((2, 3)))


class TestFileLookupAndData(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.colors = color_thresholds(3, directory.name)

    def test_finds_file_added_after_indexing(self):
        with self.assertRaises(FileNotFoundError):
            self.colors._find_exact_path("late.csv")
        (self.path / "sub").mkdir()
        (self.path / "sub" / "late.csv").write_text("1")
        self.assertEqual(self.colors._find_exact_path("late.csv"),
                         str(self.path / "sub" / "late.csv"))

    def test_duplicate_names_warn(self):
        for folder in ("a", "b"):
            (self.path / folder).mkdir()
            (self.path / folder / "dup.csv").write_text("1")
        with self.assertWarns(RuntimeWarning):
            self.colors._find_exact_path("dup.csv")

    def test_failed_matrix_load_warns(self):
        with self.assertWarns(RuntimeWarning):
            self.colors.load_transformation_matrix()

    def test_get_data_lists_available_keys(self):
        self.colors.CIE_data = {"stim3D": 1}
        self.assertEqual(self.colors.get_data("stim3D"), 1)
        with self.assertRaisesRegex(KeyError, "stim3D"):
            self.colors.get_data("missing")
        with self.assertRaises(ValueError):
            self.colors.get_data("stim3D", dataset="other")


if __name__ == "__main__":
    unittest.main()
