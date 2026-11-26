"""Pure scheduling helpers for scripted HRL demonstrations."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from m_dVrk.hrl.constants import BOARD_CENTER


Command = tuple[str, str, str]


class Phase2BimanualScheduler:
    """Advance paired Phase-2 source actions independently per arm.

    The two source reaches start together, but each arm grasps as soon as its
    own reach completes.  This matters because the state machine arms finish
    at different low-level timesteps.  Waiting for both arms to become idle
    loses the earlier arm's ``previous=reach`` mask precondition and makes a
    later joint ``grasp+grasp`` label impossible under the guided mask.

    Centre transfers remain serialized to avoid two arms releasing onto the
    same destination at once.  Unpaired recovery rings are handled by the
    serial tail.
    """

    def __init__(
        self,
        pairs: Iterable[tuple[str, str]],
        serial_tail: Iterable[Command] = (),
    ) -> None:
        self._pairs = deque(pairs)
        self._serial_tail = deque(serial_tail)
        self._left_source: deque[Command] = deque()
        self._right_source: deque[Command] = deque()
        self._destination: deque[Command] = deque()
        self.stage = "pair_setup" if self._pairs else "serial"

    @property
    def complete(self) -> bool:
        return (
            self.stage == "complete"
            and not self._pairs
            and not self._serial_tail
            and not self._left_source
            and not self._right_source
            and not self._destination
        )

    def _begin_pair(self) -> None:
        red_ring, blue_ring = self._pairs.popleft()
        self._left_source = deque(
            [
                ("reach", "left_arm", red_ring),
                ("grasp", "left_arm", red_ring),
            ]
        )
        self._right_source = deque(
            [
                ("reach", "right_arm", blue_ring),
                ("grasp", "right_arm", blue_ring),
            ]
        )
        self._destination = deque(
            [
                ("reach", "left_arm", BOARD_CENTER),
                ("release", "left_arm", BOARD_CENTER),
                ("reach", "right_arm", BOARD_CENTER),
                ("release", "right_arm", BOARD_CENTER),
            ]
        )
        self.stage = "source"

    def issue_ready(
        self,
        *,
        left_idle: bool,
        right_idle: bool,
    ) -> tuple[Command, ...]:
        """Return the next commands legal at the current arm-idle state."""
        all_idle = left_idle and right_idle

        # Several stage transitions do not consume a simulator step.  Loop
        # until commands are issued or physical progress is required.
        for _ in range(4):
            if self.stage == "pair_setup":
                if not all_idle:
                    return ()
                if self._pairs:
                    self._begin_pair()
                    continue
                self.stage = "serial"
                continue

            if self.stage == "source":
                ready: list[Command] = []
                if left_idle and self._left_source:
                    ready.append(self._left_source.popleft())
                if right_idle and self._right_source:
                    ready.append(self._right_source.popleft())
                if ready:
                    return tuple(ready)
                if not self._left_source and not self._right_source and all_idle:
                    self.stage = "destination"
                    continue
                return ()

            if self.stage == "destination":
                if not all_idle:
                    return ()
                if self._destination:
                    return (self._destination.popleft(),)
                self.stage = "pair_setup"
                continue

            if self.stage == "serial":
                if not all_idle:
                    return ()
                if self._serial_tail:
                    return (self._serial_tail.popleft(),)
                self.stage = "complete"
                return ()

            return ()

        raise RuntimeError("Phase-2 bimanual scheduler failed to settle")


class Phase3BimanualScheduler:
    """Pipeline two centre-stack transfers without targeting one ring twice.

    The left arm first removes the current top ring.  Once its grasp has
    completed, the next ring is exposed and the right source transfer can run
    concurrently with the left destination transfer.  Destinations are kept
    distinct, so the overlap is physical concurrency rather than a mask hint.
    """

    def __init__(
        self,
        pairs: Iterable[tuple[str, str, str, str]],
    ) -> None:
        self._pairs = deque(pairs)
        self._left_source: deque[Command] = deque()
        self._left_destination: deque[Command] = deque()
        self._right_source: deque[Command] = deque()
        self._right_destination: deque[Command] = deque()
        self.stage = "pair_setup" if self._pairs else "complete"

    @property
    def complete(self) -> bool:
        return (
            self.stage == "complete"
            and not self._pairs
            and not self._left_source
            and not self._left_destination
            and not self._right_source
            and not self._right_destination
        )

    def _begin_pair(self) -> None:
        left_ring, left_target, right_ring, right_target = self._pairs.popleft()
        self._left_source = deque([
            ("reach", "left_arm", left_ring),
            ("grasp", "left_arm", left_ring),
        ])
        self._left_destination = deque([
            ("reach", "left_arm", left_target),
            ("release", "left_arm", left_target),
        ])
        self._right_source = deque([
            ("reach", "right_arm", right_ring),
            ("grasp", "right_arm", right_ring),
        ])
        self._right_destination = deque([
            ("reach", "right_arm", right_target),
            ("release", "right_arm", right_target),
        ])
        self.stage = "left_source"

    def issue_ready(
        self,
        *,
        left_idle: bool,
        right_idle: bool,
    ) -> tuple[Command, ...]:
        all_idle = left_idle and right_idle
        for _ in range(5):
            if self.stage == "pair_setup":
                if not all_idle:
                    return ()
                if self._pairs:
                    self._begin_pair()
                    continue
                self.stage = "complete"
                return ()

            if self.stage == "left_source":
                if left_idle and self._left_source:
                    return (self._left_source.popleft(),)
                if not self._left_source and left_idle:
                    self.stage = "overlap"
                    continue
                return ()

            if self.stage == "overlap":
                ready: list[Command] = []
                if left_idle and self._left_destination:
                    ready.append(self._left_destination.popleft())
                if right_idle and self._right_source:
                    ready.append(self._right_source.popleft())
                if ready:
                    return tuple(ready)
                if (
                    not self._left_destination
                    and not self._right_source
                    and all_idle
                ):
                    self.stage = "right_destination"
                    continue
                return ()

            if self.stage == "right_destination":
                if right_idle and self._right_destination:
                    return (self._right_destination.popleft(),)
                if not self._right_destination and all_idle:
                    self.stage = "pair_setup"
                    continue
                return ()

            return ()

        raise RuntimeError("Phase-3 bimanual scheduler failed to settle")


class Phase4BimanualScheduler:
    """Run two independent peg-to-outside transfers concurrently.

    Unlike the centre stack in Phase 3, all four Phase-4 source rings are
    exposed from the first step. Each arm has its own physically reachable
    outside region, so source and destination primitives can overlap safely.
    """

    def __init__(
        self,
        pairs: Iterable[tuple[str, str, str, str]],
    ) -> None:
        self._pairs = deque(pairs)
        self._left_commands: deque[Command] = deque()
        self._right_commands: deque[Command] = deque()
        self.stage = "pair_setup" if self._pairs else "complete"

    @property
    def complete(self) -> bool:
        return (
            self.stage == "complete"
            and not self._pairs
            and not self._left_commands
            and not self._right_commands
        )

    def _begin_pair(self) -> None:
        left_ring, left_target, right_ring, right_target = self._pairs.popleft()
        self._left_commands = deque([
            ("reach", "left_arm", left_ring),
            ("grasp", "left_arm", left_ring),
            ("reach", "left_arm", left_target),
            ("release", "left_arm", left_target),
        ])
        self._right_commands = deque([
            ("reach", "right_arm", right_ring),
            ("grasp", "right_arm", right_ring),
            ("reach", "right_arm", right_target),
            ("release", "right_arm", right_target),
        ])
        self.stage = "transfer"

    def issue_ready(
        self,
        *,
        left_idle: bool,
        right_idle: bool,
    ) -> tuple[Command, ...]:
        all_idle = left_idle and right_idle
        for _ in range(3):
            if self.stage == "pair_setup":
                if not all_idle:
                    return ()
                if self._pairs:
                    self._begin_pair()
                    continue
                self.stage = "complete"
                return ()

            if self.stage == "transfer":
                ready: list[Command] = []
                if left_idle and self._left_commands:
                    ready.append(self._left_commands.popleft())
                if right_idle and self._right_commands:
                    ready.append(self._right_commands.popleft())
                if ready:
                    return tuple(ready)
                if not self._left_commands and not self._right_commands and all_idle:
                    self.stage = "pair_setup"
                    continue
                return ()

            return ()

        raise RuntimeError("Phase-4 bimanual scheduler failed to settle")
