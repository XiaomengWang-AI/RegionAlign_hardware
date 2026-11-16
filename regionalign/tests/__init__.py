"""CPU protocol and kernel regression tests of the RegionAlign package.

The package resolves every path it is given, and the tests compare its outputs
with the temporary paths they passed in.  Temporary files are therefore created
under the resolved temporary root, so a symbolic link in ``TMPDIR`` (the default
on some clusters) does not make those comparisons fail.
"""
import os
import tempfile

tempfile.tempdir = os.path.realpath(tempfile.gettempdir())
