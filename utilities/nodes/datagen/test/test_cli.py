import shutil
import subprocess
import time

from datagen.cli import sim_running


def start(tmp_path, program, *args):
    """A process named gym_bridge, running the given program."""
    exe = tmp_path / 'gym_bridge'
    shutil.copy(shutil.which(program), exe)
    return subprocess.Popen([str(exe), *args])


def test_a_running_simulator_counts_as_running(tmp_path):
    proc = start(tmp_path, 'sleep', '10')
    try:
        assert sim_running()
    finally:
        proc.kill()
        proc.wait()


def test_a_finished_simulator_that_nobody_reaped_does_not_count(tmp_path):
    proc = start(tmp_path, 'true')
    time.sleep(0.3)  # it has exited, and stays a zombie until it is waited for
    try:
        assert not sim_running()
    finally:
        proc.wait()
