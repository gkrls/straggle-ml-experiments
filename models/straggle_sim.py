import random
import time
import torch
import torch.nn as nn
import torch.distributed as dist

from threading import Lock

class AtomicCounter:
    def __init__(self, initial=0):
        self._value = initial
        self._lock = Lock()
    
    def add(self, n=1):
        with self._lock:
            val = self._value
            self._value += n
            return val
    
    def inc(self): 
        return self.add(1)
    
    def get(self):
        with self._lock:
            return self._value
    
    def set(self, n=0):
        with self._lock:
            old_val = self._value
            self._value = n  # Fixed: was += n
            return old_val

class SlowWorkerPattern:
    """
    Inject delays at:
      - forward_start  (register_forward_pre_hook)
      - forward_end    (register_forward_hook)
      - backward_end   (register_full_backward_hook)

    Parameters
    ----------
    points : int
        Number of hook points (1-3): 1=fwd_start, 2=+fwd_end, 3=+bwd_end
    prob : float
        Probability of applying delay per hook firing (0-100 percent).
    amount : float
        Base delay in seconds (not milliseconds).
    ranks : list[int] or None
        If given, only these ranks will straggle.
    multiplier_range : tuple[float, float]
        Multiplier range applied to 'amount'. Example: (0.5, 2.0).
    seed : int
        RNG seed for reproducibility.
    verbose : bool
        Print when delays are applied.
    """

    def __init__(self, 
                 points: int = 3, 
                 prob: float = 2.0,
                 amount: float = 2.0,
                 ranks: list[int] | None = None, 
                 multiplier_range: tuple[float, float] = (1.0, 1.0), 
                 seed: int = None,
                 verbose: bool = False):
        
        if not (1 <= points <= 3): raise ValueError("points must be in [1, 3]")
        self.points = points

        # probability: accept 0..1 or 0..100 (percent)
        if not (0.0 <= prob <= 100.0): raise ValueError("prob must be in [0,100].")
        self.prob = float(prob / 100 if prob > 1 else prob)

        if amount < 0: raise ValueError("amount must be >= 0.")
        self.amount = float(amount)

        self.active = self.points > 0 and self.prob > 0 and self.amount > 0
        if not self.active: print(f"[warning] straggle_sim created but not active -- points: {self.points}, prob: {self.prob}, amount: {self.amount}")

        self.ranks = set(ranks) if ranks else None

        if multiplier_range is not None:
            a, b = float(multiplier_range[0]), float(multiplier_range[1])
            if not (0 < a <= b): raise ValueError("multiplier_range must satisfy (0 < min_multiplier <= max_multiplier).")
            self.multi_range = (a, b)
        else:
            self.multi_range = (1.0, 1.0)

        # Use different seeds for different RNGs
        base_seed = seed if seed is not None else random.randint(0, 2**32-1)
        self.rng_1 = random.Random(base_seed)
        self.rng_2 = random.Random(base_seed + 42)
        self.verbose = verbose

        self._handles = []
        self._rank = None
        self._step_has_straggled = False  # Track if current step already straggled

        self.stats = {
            "num_straggle_steps" : AtomicCounter(),
            "num_straggle_events": AtomicCounter(),
            "total_straggle_time": AtomicCounter()
        }

    def _reset_stats(self):
        for counter in self.stats.values():
            counter.set(0)

    def _get_rank_safe(self) -> int:
        """Safely get the current rank."""
        try:
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return 0

    def attach(self, root: nn.Module) -> int:
        if not self.active: return 0

        """Attach hooks to the root model. With DDP, use ddp.module."""
        self.detach()  # idempotent
        self._reset_stats()

        if dist.is_available() and dist.is_initialized():
            self._rank = self._get_rank_safe()
        else:
            self._rank = 0
            if self.verbose: print("[straggle_sim] dist unavailable or not initialized, using rank=0")

        # Filter rank
        if self.ranks is not None and self._rank not in self.ranks:
            if self.verbose: print(f"[straggle_sim] rank {self._rank} not in target ranks {self.ranks}, skipping")
            return 0

        # Forward start
        self._handles.append(root.register_forward_pre_hook(self._on_fwd_start))

        # Forward end
        if self.points > 1:
            self._handles.append(root.register_forward_hook(self._on_fwd_end))

        # Backward end
        if self.points > 2:
            try: self._handles.append(root.register_full_backward_hook(self._on_bwd_end))
            except AttributeError: raise ValueError("[straggle_sim] register_full_backward_hook unavailable; use points <= 2")

        if self.verbose: print(f"[straggle_sim] attached {len(self._handles)} straggle hooks to rank {self._rank}")

        return len(self._handles)

    def detach(self):
        """Remove all attached hooks."""
        removed = 0
        for h in self._handles:
            try:
                h.remove()
                removed += 1
            except Exception: pass
        self._handles.clear()
        if removed and self.verbose: print(f"[straggle_sim] detached {removed} hook(s).")

    def _sample_delay_seconds(self) -> float:
        """Sample a delay in seconds."""
        if self.multi_range == (1.0, 1.0): return self.amount
        a, b = self.multi_range
        mult = self.rng_2.uniform(a, b)
        return self.amount * mult

    def _maybe_sleep(self, where: str):
        """Maybe inject a delay, tracking stats."""
        if self.rng_1.random() < self.prob:
            delay_sec = self._sample_delay_seconds()
            
            # Update stats
            if not self._step_has_straggled:
                self.stats["num_straggle_steps"].inc()
                self._step_has_straggled = True
            
            self.stats["num_straggle_events"].inc()
            self.stats["total_straggle_time"].add(delay_sec)
            
            if self.verbose:
                print(f"[straggle_sim][rank {self._rank}] {where}: sleeping {delay_sec:.3f}s "
                      f"(base={self.amount:.3f}s, range={self.multi_range})")
            
            time.sleep(delay_sec)

    # ---- hook callbacks ----
    def _on_fwd_start(self, module, inputs):
        self._step_has_straggled = False  # Reset for new step
        self._maybe_sleep("forward_start")

    def _on_fwd_end(self, module, inputs, outputs):
        self._maybe_sleep("forward_end")

    def _on_bwd_end(self, module, grad_input, grad_output):
        self._maybe_sleep("backward_end")

    def get_stats(self) -> dict:
        """Get current statistics."""
        return {
            "num_straggle_steps": self.stats["num_straggle_steps"].get(),
            "num_straggle_events": self.stats["num_straggle_events"].get(),
            "total_straggle_time_ms": self.stats["total_straggle_time"].get(),
            "avg_straggle_time_ms": self.stats["total_straggle_time"].get() / max(1, self.stats["num_straggle_events"].get())
        }

    def print_stats(self):
        """Print current statistics."""
        stats = self.get_stats()
        print(f"[straggle_sim] Stats for rank {self._rank}:")
        print(f"  Straggle steps: {stats['num_straggle_steps']}")
        print(f"  Straggle events: {stats['num_straggle_events']}")
        print(f"  Total straggle time: {stats['total_straggle_time_ms']:.1f}ms")
        print(f"  Avg straggle time: {stats['avg_straggle_time_ms']:.1f}ms")

    def __repr__(self) -> str:
            """Pretty print configuration in one line."""
            ranks_str = f"ranks={list(self.ranks)}" if self.ranks else "all_ranks"
            multi_str = f"×{self.multi_range[0]:.1f}-{self.multi_range[1]:.1f}" if self.multi_range != (1.0, 1.0) else ""
            return (f"SlowWorkerPattern(points={self.points}, prob={self.prob:.1%}, "
                    f"amount={self.amount:.2f}s{multi_str}, {ranks_str}, enabled={self.enabled})")