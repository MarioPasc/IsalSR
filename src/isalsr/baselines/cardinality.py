"""Constant-memory distinct-count sketch (HyperLogLog).

Reference:
    P. Flajolet, E. Fusy, O. Gandouet and F. Meunier, "HyperLogLog: the analysis
    of a near-optimal cardinality estimation algorithm", *Proc. AofA 2007*,
    DMTCS Proc. AH, pp. 137-156, 2007.

Purpose: replace the ``set[int]`` of seen deduplication keys by a fixed-size
register array.  With ``p = 14`` the sketch occupies ``2**14 = 16384`` bytes and
has a relative standard error of ``1.04 / sqrt(m) ~ 0.81 %``, so three shadow
counters cost ~48 KB instead of the ~1-2 GB that three exact key sets cost on a
long run.

Deviation from the 1970s-vintage original that is worth stating explicitly: the
*small range* correction (linear counting) is applied, but the *large range*
correction ``-2**32 * ln(1 - E / 2**32)`` is **not**.  That correction exists
only to undo the saturation of a 32-bit hash space.  This implementation mixes
every key into a full 64-bit value, so the saturation regime is unreachable, and
applying the 32-bit formula would clamp any estimate above ``2**32 / 30``
(~1.43e8) to a wrong value.  Omitting it with 64-bit hashes is the standard
modern choice (Heule, Nunkesser and Hall, "HyperLogLog in practice", *EDBT
2013*, section 4).

Restriction: this module depends only on the Python standard library.
"""

from __future__ import annotations

import math

__all__ = ["HyperLogLog"]

_MASK64 = (1 << 64) - 1


def _mix64(key: int) -> int:
    """Return a well-distributed 64-bit value derived from *key*.

    The SplitMix64 finalizer (Steele, Lea and Flood, *OOPSLA 2014*).  Applying it
    makes the sketch robust to keys that are not themselves uniform, such as
    small sequential integers or the signed output of CPython's ``hash``.

    Args:
        key: Any Python integer, possibly negative.

    Returns:
        A 64-bit unsigned integer.
    """
    z = (key & _MASK64) + 0x9E3779B97F4A7C15 & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


class HyperLogLog:
    """Fixed-memory estimator of the number of distinct keys in a stream.

    Args:
        p: Log2 of the register count.  Must satisfy ``4 <= p <= 18``.
            ``p = 14`` gives 16384 registers (16 KB) and ~0.81 % relative
            standard error.

    Raises:
        ValueError: If *p* is outside the supported range.
    """

    __slots__ = ("_p", "_m", "_alpha", "_registers")

    def __init__(self, p: int = 14) -> None:
        if not 4 <= p <= 18:
            raise ValueError(f"p must be in [4, 18], got {p}")
        self._p: int = p
        self._m: int = 1 << p
        self._alpha: float = self._alpha_for(self._m)
        self._registers: bytearray = bytearray(self._m)

    @staticmethod
    def _alpha_for(m: int) -> float:
        """Return the bias-correction constant ``alpha_m`` for *m* registers.

        Args:
            m: Register count.

        Returns:
            The constant from Flajolet et al. (2007), Table/eq. for ``alpha_m``.
        """
        if m == 16:
            return 0.673
        if m == 32:
            return 0.697
        if m == 64:
            return 0.709
        return 0.7213 / (1.0 + 1.079 / m)

    @property
    def p(self) -> int:
        """Return the log2 register count."""
        return self._p

    @property
    def n_registers(self) -> int:
        """Return the number of registers."""
        return self._m

    @property
    def relative_standard_error(self) -> float:
        """Return the asymptotic relative standard error ``1.04 / sqrt(m)``."""
        return 1.04 / math.sqrt(self._m)

    def add(self, key: int) -> None:
        """Insert *key* into the sketch.

        Args:
            key: Any Python integer.  It is reduced modulo ``2**64`` and then
                mixed, so process-local signed ``hash`` values and unsigned
                64-bit digests are both valid key sources and both lossless.
                Keys wider than 64 bits are truncated.
        """
        x = _mix64(key)
        index = x >> (64 - self._p)
        # The rank is the position of the leftmost 1 in the remaining
        # ``64 - p`` bits, counting from 1; an all-zero suffix gives 64 - p + 1.
        w = (x << self._p) & _MASK64
        rank = (64 - w.bit_length() + 1) if w else (64 - self._p + 1)
        if rank > self._registers[index]:
            self._registers[index] = rank

    def count(self) -> float:
        """Return the estimated number of distinct keys inserted.

        Returns:
            The HyperLogLog estimate, with linear counting substituted in the
            small-range regime (``E <= 2.5 m`` with at least one empty
            register).
        """
        m = float(self._m)
        raw_sum = 0.0
        n_zero = 0
        for reg in self._registers:
            raw_sum += 2.0**-reg
            if reg == 0:
                n_zero += 1
        estimate = self._alpha * m * m / raw_sum
        if estimate <= 2.5 * m and n_zero > 0:
            return m * math.log(m / n_zero)
        return estimate

    def merge(self, other: HyperLogLog) -> None:
        """Merge *other* into this sketch in place.

        The union of two HyperLogLog sketches over the same register count is
        the elementwise maximum of their registers, so the merge is exact: the
        merged sketch equals the sketch of the concatenated streams.

        Args:
            other: A sketch with the same ``p``.

        Raises:
            ValueError: If the register counts differ.
        """
        if other._p != self._p:
            raise ValueError(f"Cannot merge sketches with p={self._p} and p={other._p}")
        mine = self._registers
        theirs = other._registers
        for i in range(self._m):
            if theirs[i] > mine[i]:
                mine[i] = theirs[i]

    def copy(self) -> HyperLogLog:
        """Return an independent copy of this sketch."""
        clone = HyperLogLog(self._p)
        clone._registers = bytearray(self._registers)
        return clone

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self._p}, count={self.count():.0f})"
