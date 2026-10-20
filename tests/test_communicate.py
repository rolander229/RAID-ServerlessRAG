"""Self-checks for the slice-to-slice message channels (Sec. II-D, Fig. 8).

Pure standard library, no torch, and no ``/dev/shm`` required: the arena is
created under a temporary directory on a Mac and under ``/dev/shm`` on Linux,
which is where the two transports actually differ in speed but not in behaviour.

Run: python3 tests/test_communicate.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slicer.communicate.channel import (  # noqa: E402
    Arena,
    mkfifo,
    open_fifo,
    open_pipe,
    open_shm,
    open_shm_fifo,
    tmpfs_root,
)


def _peer(target, *args) -> tuple[threading.Thread, list]:
    """Run one endpoint in a thread; a pipe would otherwise deadlock on us."""
    out: list = []

    def run():
        try:
            out.append(target(*args))
        except BaseException as error:  # re-raised so the test sees it
            out.append(error)

    thread = threading.Thread(target=run)
    thread.daemon = True
    thread.start()
    return thread, out


def _join(thread, out):
    thread.join(timeout=10)
    if thread.is_alive():
        raise AssertionError("peer endpoint blocked; the protocol deadlocked")
    if out and isinstance(out[0], BaseException):
        raise out[0]
    return out[0] if out else None


class TestPipe(unittest.TestCase):
    def test_round_trip_of_a_payload_larger_than_the_pipe(self):
        sender, receiver = open_pipe()
        payload = bytes(bytearray(range(256))) * 2000  # ~512 KB, 8x the buffer
        peer, out = _peer(receiver.recv)
        sender.send(payload)
        self.assertEqual(_join(peer, out), payload)
        sender.close()
        receiver.close()

    def test_headers_and_payload_share_one_stream_in_order(self):
        sender, receiver = open_pipe()
        payloads = [b"", b"a", bytes(4096), b"tail"]
        peer, out = _peer(lambda: [receiver.recv() for _ in payloads])
        for payload in payloads:
            sender.send(payload)
        self.assertEqual(_join(peer, out), payloads)
        sender.close()
        receiver.close()

    def test_a_big_payload_costs_many_syscalls(self):
        """Fig. 8's premise, as an assertion: the pipe is the slow transport."""
        sender, receiver = open_pipe()
        payload = bytes(512 * 1024)
        peer, out = _peer(receiver.recv)
        sender.send(payload)
        _join(peer, out)
        self.assertGreaterEqual(sender.sent.writes, len(payload) // 65536,
                                "a 512 KB payload should not cross a 64 KB pipe "
                                "in fewer writes than that")
        sender.close()
        receiver.close()

    def test_closing_the_sender_ends_the_receiver(self):
        sender, receiver = open_pipe()
        sender.close()
        with self.assertRaises(EOFError):
            receiver.recv()
        receiver.close()

    def test_endpoints_refuse_the_wrong_direction(self):
        sender, receiver = open_pipe()
        with self.assertRaises(NotImplementedError):
            sender.recv()
        with self.assertRaises(NotImplementedError):
            receiver.send(b"x")
        sender.close()
        receiver.close()


class TestSharedMemory(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="slicer-shm-")
        self.arena = Arena.of(slots=4, slot_bytes=2 * 1024 * 1024, root=self.root)
        self.addCleanup(self.arena.destroy)

    def _pair(self, slots):
        return open_shm(self.arena, slots)

    def test_payload_arrives_byte_for_byte(self):
        sender, receiver = self._pair([0, 1])
        payload = bytes(bytearray(range(256))) * 200
        peer, out = _peer(receiver.recv)
        sender.send(payload)
        self.assertEqual(_join(peer, out), payload)
        sender.close()
        receiver.close()

    def test_one_write_carries_a_whole_message(self):
        """Fig. 8's claim: the channel moves a header, the arena moves the rest."""
        sender, receiver = self._pair([0, 1])
        payload = bytes(512 * 1024)  # larger than a pipe buffer, smaller than a slot
        self.assertLess(len(payload), self.arena.slot_bytes)
        peer, out = _peer(receiver.recv)
        sender.send(payload)
        _join(peer, out)
        self.assertEqual(sender.sent.writes, 1)
        self.assertEqual(receiver.received.reads, 1)
        self.assertEqual(receiver.received.payload_bytes, len(payload))
        sender.close()
        receiver.close()

    def test_a_second_slot_lets_the_sender_run_ahead(self):
        """Double buffering: message 2 is written while message 1 is still held."""
        sender, receiver = self._pair([0, 1])
        held = threading.Event()
        release = threading.Event()

        def consume():
            with receiver.borrow() as first:
                self.assertEqual(len(first), 4)
                held.set()
                release.wait(10)
            return receiver.recv()

        peer, out = _peer(consume)
        sender.send(b"aaaa")
        held.wait(10)
        sender.send(b"bbbb")  # would block forever on a single-slot ring
        self.assertTrue(held.is_set())
        release.set()
        self.assertEqual(_join(peer, out), b"bbbb")
        sender.close()
        receiver.close()

    def test_a_single_slot_would_have_blocked(self):
        sender, receiver = self._pair([0])
        held = threading.Event()
        release = threading.Event()

        def consume():
            with receiver.borrow():
                held.set()
                release.wait(10)

        peer, _ = _peer(consume)
        sender.send(b"aaaa")
        held.wait(10)
        second, out = _peer(sender.send, b"bbbb")
        second.join(0.5)
        self.assertTrue(second.is_alive(), "one slot must not admit a second message")
        release.set()
        _join(peer, out)
        second.join(10)
        sender.close()
        receiver.close()

    def test_an_oversized_payload_is_refused_not_truncated(self):
        sender, receiver = self._pair([0])
        with self.assertRaises(ValueError):
            sender.send(bytes(self.arena.slot_bytes + 1))
        sender.close()
        receiver.close()

    def test_two_directions_share_one_arena(self):
        """A slice sends on one queue and receives on another without colliding."""
        a_send, b_recv = open_shm(self.arena, [0, 1])
        b_send, a_recv = open_shm(self.arena, [2, 3])
        peer, out = _peer(lambda: (b_recv.recv(), a_recv.recv()))
        a_send.send(b"from-a")
        b_send.send(b"from-b")
        self.assertEqual(_join(peer, out), (b"from-a", b"from-b"))
        for endpoint in (a_send, b_recv, b_send, a_recv):
            endpoint.close()

    def test_the_arena_file_is_created_and_removed(self):
        self.assertTrue(os.path.exists(self.arena.path))
        self.arena.destroy()
        self.assertFalse(os.path.exists(self.arena.path))

    def test_the_default_arena_lands_on_tmpfs(self):
        """The transport is only Fig. 8's idea if the pages never reach a disk."""
        if tmpfs_root() != "/dev/shm":
            self.skipTest("no tmpfs on this box")
        arena = Arena.of(slots=2, slot_bytes=4096)
        self.addCleanup(arena.destroy)
        self.assertTrue(arena.path.startswith("/dev/shm/"), arena.path)
        sender, receiver = open_shm(arena, [0, 1])
        peer, out = _peer(receiver.recv)
        sender.send(b"tmpfs")
        self.assertEqual(_join(peer, out), b"tmpfs")
        sender.close()
        receiver.close()


class TestAcrossProcesses(unittest.TestCase):
    """The mechanism P5's slices actually use: a name for the queue, a path for the arena.

    A spawned child inherits no descriptor the runtime has not duped for itself, so
    every rendezvous here has to be a path both sides can derive, and the payload
    pages have to be mapped twice rather than handed over.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="slicer-fifo-")

    def tearDown(self):
        for entry in os.listdir(self.root):
            os.unlink(os.path.join(self.root, entry))
        os.rmdir(self.root)

    def test_a_named_pipe_moves_a_payload_the_same_way(self):
        sender, receiver = open_fifo("hop", root=self.root)
        self.assertTrue(os.path.exists(mkfifo("hop", root=self.root)))
        payload = bytes(bytearray(range(256))) * 2000
        peer, out = _peer(receiver.recv)
        sender.send(payload)
        self.assertEqual(_join(peer, out), payload)
        sender.close()
        receiver.close()

    def test_an_attached_arena_leaves_the_file_for_its_creator(self):
        arena = Arena.of(slots=2, slot_bytes=4096, root=self.root)
        other = Arena.attach(arena.path, 2, 4096)
        self.addCleanup(arena.destroy)
        arena.slot(0)[:4] = b"seed"
        self.assertEqual(bytes(other.slot(0))[:4], b"seed",
                         "two mappings of one file are Fig. 8's zero-copy hop")
        other.destroy()
        self.assertTrue(os.path.exists(arena.path), "a mapper does not own the name")

    def test_a_child_sends_through_an_arena_it_mapped_itself(self):
        import multiprocessing

        arena = Arena.of(slots=2, slot_bytes=2 * 1024 * 1024, root=self.root)
        payloads = [bytes(n + 1) * 1000 for n in range(5)]
        context = multiprocessing.get_context("spawn")
        child = context.Process(target=_child_producer,
                                args=(self.root, arena.path, arena.slot_bytes, payloads))
        child.start()
        self.addCleanup(child.join, 10)
        _, receiver = open_shm_fifo(arena, "hop", [0, 1], root=self.root)
        received = [receiver.recv() for _ in payloads]
        child.join(10)
        self.assertEqual(child.exitcode, 0, "the producer died in the child")
        self.assertEqual(received, payloads)
        receiver.close()
        arena.destroy()


def _child_producer(root, arena_path, slot_bytes, payloads):
    """The sender's half of :meth:`TestAcrossProcesses.test_a_child_sends...`."""
    arena = Arena.attach(arena_path, 2, slot_bytes)
    sender, _ = open_shm_fifo(arena, "hop", [0, 1], root=root)
    for payload in payloads:
        sender.send(payload)
    sender.close()
    arena.destroy()


class TestEnvironment(unittest.TestCase):
    def test_tmpfs_root_falls_back_to_the_working_directory(self):
        root = tmpfs_root()
        self.assertTrue(os.path.isdir(root))
        if root != "/dev/shm":
            self.skipTest("no tmpfs on this box; the timings here are disk timings")


if __name__ == "__main__":
    unittest.main(verbosity=2)
