#!/usr/bin/env python3

import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
FILTER = ROOT / 'hw/scripts/saif_filter.py'
HEADER = '''(SAIFILE
(SAIFVERSION "2.0")
(DIRECTION "backward")
(DESIGN "TOP")
(DIVIDER / )
(TIMESCALE 1ps)
(DURATION 100)
'''


class SaifFilterTest(unittest.TestCase):
    def run_filter(self, body, *args):
        with tempfile.TemporaryDirectory() as tmp:
            source = pathlib.Path(tmp) / 'trace.saif'
            source.write_text(HEADER + body + ')\n')
            return subprocess.run([str(FILTER), *args, str(source)], text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_root_is_renamed_without_losing_hierarchy(self):
        result = self.run_filter(
            ' (INSTANCE TOP\n'
            '  (INSTANCE dut\n'
            '   (NET a (T0 1) (T1 1) (TX 0) (TC 2) (IG 0))\n'
            '   (INSTANCE cell\n'
            '    (NET y (T0 1) (T1 1) (TX 0) (TC 2) (IG 0))\n'
            '   )\n'
            '  )\n'
            ' )\n',
            '--instance', 'dut', '--top', 'my_dut', '--rename-root')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('(DESIGN "my_dut")', result.stdout)
        self.assertIn('(INSTANCE my_dut', result.stdout)
        self.assertIn('(INSTANCE cell', result.stdout)

    def test_ambiguous_suffix_fails(self):
        result = self.run_filter(
            ' (INSTANCE TOP\n'
            '  (INSTANCE a\n   (INSTANCE dut\n   )\n  )\n'
            '  (INSTANCE b\n   (INSTANCE dut\n   )\n  )\n'
            ' )\n', '--instance', 'dut')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ambiguous (2 matches)', result.stderr)

    def test_activity_parentheses_are_escaped(self):
        result = self.run_filter(
            ' (INSTANCE TOP\n'
            '  (NET\n'
            '   (signal(3) (T0 1) (T1 1) (TX 0) (TC 2) (IG 0))\n'
            '  )\n'
            ' )\n', '--instance', 'TOP')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('(signal_3_ (T0 1)', result.stdout)

    def test_activity_bus_indices_are_unescaped(self):
        result = self.run_filter(
            ' (INSTANCE TOP\n'
            '  (NET\n'
            '   (signal\\[3\\] (T0 1) (T1 1) (TX 0) (TC 2) (IG 0))\n'
            '  )\n'
            ' )\n', '--instance', 'TOP')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('(signal[3] (T0 1)', result.stdout)


if __name__ == '__main__':
    unittest.main()
