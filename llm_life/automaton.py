"""The LLM-as-cellular-automaton map.

State is a fixed-length sequence of token ids ``s in V^L``. One generation is a
single, *synchronous* forward pass: we teacher-force the whole current sequence
and read off the model's next-token distribution at every position in parallel,
then resample. Because the update is synchronous (every site updated from the
*previous* generation at once) this is genuinely cellular-automaton-like rather
than ordinary autoregressive generation.

Neighborhood, honestly stated
-----------------------------
With a causal LM, the logit at position k depends on positions 0..k-1 (plus a
prepended BOS). So a site's "neighborhood" is its entire *left* context, not a
local symmetric window. This is therefore a *directed, long-range* automaton,
not a local CA. That asymmetry is a known limitation of the causal variant; a
masked-LM map (bidirectional, local-ish) is the natural follow-up. We keep the
caveat explicit rather than papering over it.

Absorbing variant
------------------
For the absorbing-state (directed-percolation-style) phase transition to be
real, the all-dead configuration must be a true fixed point with *no
spontaneous birth from vacuum*. We enforce exactly that: a site whose entire
left context is the dead token is forced to the dead token; every other site
samples at temperature T. Activity can spread from live regions but can never
nucleate out of nothing, so all-dead is strictly absorbing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .sampler import gumbel_like, sample


@dataclass
class StepNoise:
    """Container so two replicas can share an identical noise realization."""

    gumbel: torch.Tensor


class BaseAutomaton:
    """Shared iteration machinery. A concrete automaton defines its transition
    by implementing ``logits(state) -> (L, V)`` (the next-gen distribution at
    every site) and ``_vacuum_mask(state)`` (which sites the absorbing rule
    forces to the dead token)."""

    def __init__(self, model, dead_token: int, device: str):
        self.model = model
        self.dead_token = dead_token
        self.device = device
        # Homeostatic "overpopulation death" knob (Conway's balance axis): each
        # site's logit for a token is reduced in proportion to how many times
        # that token already occupies the current generation. 0 = off (default,
        # preserving prior behavior). This is the negative-feedback control that
        # temperature -- a pure disorder knob -- cannot provide.
        self.freq_penalty = 0.0

    def logits(self, state: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _vacuum_mask(self, state: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def _apply_freq_penalty(self, lg: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Subtract freq_penalty * (count of each token in the current grid) from
        every site's logits, so over-represented tokens are suppressed."""
        if not self.freq_penalty:
            return lg
        counts = torch.bincount(state, minlength=lg.shape[-1]).to(lg.dtype)
        return lg - self.freq_penalty * counts.unsqueeze(0)

    def step(
        self,
        state: torch.Tensor,
        temperature: float,
        absorbing: bool,
        noise: StepNoise | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, StepNoise]:
        """Advance one generation. Returns (next_state, noise_used)."""
        lg = self._apply_freq_penalty(self.logits(state), state)
        gumbel = noise.gumbel if noise is not None else gumbel_like(lg, generator=generator)
        nxt = sample(lg, temperature, noise=gumbel)
        if absorbing:
            nxt = torch.where(self._vacuum_mask(state), self.dead_token, nxt)
        return nxt, StepNoise(gumbel=gumbel)

    def random_state(self, length: int, vocab_size: int, generator: torch.Generator) -> torch.Tensor:
        return torch.randint(0, vocab_size, (length,), generator=generator, device=self.device)

    @torch.no_grad()
    def trajectory(
        self,
        init: torch.Tensor,
        steps: int,
        temperature: float,
        absorbing: bool,
        generator: torch.Generator,
        record_noise: bool = False,
        refractory: int = 0,
    ):
        """Iterate the map. Returns (states (steps+1, L), noises or None).

        ``record_noise`` keeps every step's Gumbel tensor so a second replica
        can be driven with identical noise (coupled-noise damage spreading).

        ``refractory`` > 0 adds a decay/refractory rule (the Brian's-Brain
        mechanism that makes CAs glider-rich): a cell that has stayed alive for
        more than ``refractory`` consecutive generations is forced dead. This
        gives the *trailing* death a travelling structure needs -- without it an
        active region can spread but never vacate, so it fills instead of moving.
        """
        L = init.shape[0]
        states = torch.empty(steps + 1, L, dtype=torch.long, device=self.device)
        states[0] = init
        noises = [] if record_noise else None
        cur = init
        age = (init != self.dead_token).long()  # generations a cell has been alive
        for t in range(steps):
            cur, used = self.step(cur, temperature, absorbing, generator=generator)
            if refractory:
                alive = cur != self.dead_token
                age = torch.where(alive, age + 1, torch.zeros_like(age))
                too_old = age > refractory
                cur = torch.where(too_old, torch.full_like(cur, self.dead_token), cur)
                age = torch.where(too_old, torch.zeros_like(age), age)
            states[t + 1] = cur
            if record_noise:
                noises.append(used)
        return states, noises

    @torch.no_grad()
    def replay_with_noise(
        self,
        init: torch.Tensor,
        noises: list[StepNoise],
        temperature: float,
        absorbing: bool,
    ) -> torch.Tensor:
        """Iterate from ``init`` reusing a recorded noise sequence (the coupled
        replica of a damage-spreading pair)."""
        L = init.shape[0]
        states = torch.empty(len(noises) + 1, L, dtype=torch.long, device=self.device)
        states[0] = init
        cur = init
        for t, n in enumerate(noises):
            cur, _ = self.step(cur, temperature, absorbing, noise=n)
            states[t + 1] = cur
        return states


class LLMAutomaton(BaseAutomaton):
    """Causal variant: each site is regenerated from its entire *left* context
    (directed, long-range neighborhood)."""

    def __init__(self, model, dead_token: int, bos_token: int, device: str):
        super().__init__(model, dead_token, device)
        self.bos_token = bos_token

    @torch.no_grad()
    def logits(self, state: torch.Tensor) -> torch.Tensor:
        """Next-gen logits for every site. ``state`` is (L,) -> returns (L, V).

        We prepend BOS so position 0 also has a (vacuum) context, then take the
        predictions made *for* sites 0..L-1.
        """
        L = state.shape[0]
        inp = torch.empty(1, L + 1, dtype=torch.long, device=self.device)
        inp[0, 0] = self.bos_token
        inp[0, 1:] = state
        out = self.model(inp).logits[0]  # (L+1, V)
        return out[:L]                    # predictions for sites 0..L-1

    def _vacuum_mask(self, state: torch.Tensor) -> torch.Tensor:
        """Boolean (L,): True where the entire left context is the dead token.

        Site k's left context is sites 0..k-1. Cumulative-AND of (s == dead),
        shifted right by one so site 0 (empty left context) counts as vacuum.
        """
        is_dead = state == self.dead_token
        prefix_all_dead = torch.cumprod(is_dead.long(), dim=0).bool()  # sites 0..k dead
        vacuum = torch.empty_like(prefix_all_dead)
        vacuum[0] = True                       # site 0: empty left context == vacuum
        vacuum[1:] = prefix_all_dead[:-1]      # site k: sites 0..k-1 all dead
        return vacuum


class MaskedLMAutomaton(BaseAutomaton):
    """Masked-LM variant: the faithful Conway-like rule. Every site is
    recomputed *simultaneously* from the rest of the sequence. To get a
    non-trivial prediction for a site we must mask it, so one generation is a
    single batched forward pass over L copies of the sequence, each with a
    different position masked.

    Neighborhood note: attention here is *bidirectional* (symmetric -- a real
    improvement over the causal leftward rule) but still *global*: each site
    attends to all others, not a local window. A strictly local-window variant
    is left as future work.
    """

    def __init__(self, model, dead_token: int, mask_token: int,
                 cls_token: int, sep_token: int, device: str):
        super().__init__(model, dead_token, device)
        self.mask_token = mask_token
        self.cls_token = cls_token
        self.sep_token = sep_token

    @torch.no_grad()
    def logits(self, state: torch.Tensor) -> torch.Tensor:
        """(L,) -> (L, V): each row is the prediction for that site given all the
        others (that site masked), wrapped in CLS/SEP."""
        L = state.shape[0]
        idx = torch.arange(L, device=self.device)
        batch = state.unsqueeze(0).repeat(L, 1).clone()   # (L, L)
        batch[idx, idx] = self.mask_token                 # mask one distinct site per row
        cls = torch.full((L, 1), self.cls_token, dtype=torch.long, device=self.device)
        sep = torch.full((L, 1), self.sep_token, dtype=torch.long, device=self.device)
        inp = torch.cat([cls, batch, sep], dim=1)         # (L, L+2)
        out = self.model(inp).logits                      # (L, L+2, V)
        return out[idx, idx + 1]                          # masked-site logits per row -> (L, V)

    def _vacuum_mask(self, state: torch.Tensor) -> torch.Tensor:
        """Symmetric 'no birth from vacuum': a site is forced dead iff every
        *other* site is dead (its whole neighborhood is the dead token), so the
        all-dead state is strictly absorbing."""
        live = state != self.dead_token
        others_live = int(live.sum().item()) - live.long()
        return others_live == 0


class LocalMaskedLMAutomaton(BaseAutomaton):
    """Local + symmetric + periodic CA -- the faithful Conway analogue and the
    setup most likely to support gliders.

    Each site is predicted from ONLY its +/- w neighbours on a ring (wrap-around
    boundary), with the centre masked. Because the neighbourhood is finite,
    information has a finite propagation speed (<= w sites/generation) -- the
    prerequisite for localized travelling structures that full attention
    destroys. One generation is a single batched forward over L windows of
    length 2w+1 (cheap: independent of L's effect on context).
    """

    def __init__(self, model, dead_token: int, mask_token: int, cls_token: int,
                 sep_token: int, window: int, device: str):
        super().__init__(model, dead_token, device)
        self.mask_token = mask_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.window = window  # radius w

    def _ring_index(self, L: int) -> torch.Tensor:
        """(L, 2w+1) wrap-around neighbour indices for every site."""
        w = self.window
        offsets = torch.arange(-w, w + 1, device=self.device)
        return (torch.arange(L, device=self.device).unsqueeze(1) + offsets.unsqueeze(0)) % L

    @torch.no_grad()
    def logits(self, state: torch.Tensor) -> torch.Tensor:
        L = state.shape[0]
        w = self.window
        windows = state[self._ring_index(L)].clone()   # (L, 2w+1)
        windows[:, w] = self.mask_token                 # mask each window's centre
        cls = torch.full((L, 1), self.cls_token, dtype=torch.long, device=self.device)
        sep = torch.full((L, 1), self.sep_token, dtype=torch.long, device=self.device)
        inp = torch.cat([cls, windows, sep], dim=1)     # (L, 2w+3)
        out = self.model(inp).logits                    # (L, 2w+3, V)
        return out[:, w + 1]                            # centre prediction (after CLS) -> (L, V)

    def _apply_freq_penalty(self, lg: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Neighbourhood-LOCAL frequency penalty -- Conway's overpopulation-death
        balance, made local. Each site's logit for a token is reduced in
        proportion to how many of that site's +/-w *spatial neighbours* (excluding
        the centre) already hold that token, so a token is discouraged from
        appearing where it is locally over-represented.

        This is the natural pairing with the local rule and the knob the global
        penalty could not provide: the global version (BaseAutomaton, whole-grid
        bincount) can only force global turnover and floods to noise (FINDINGS
        sec 9); a *local* density-dependent death is what lets a structure sit at
        the edge between growth and death -- the precondition for droplet-stable
        localized lifeforms (still-lifes / gliders) rather than fill-or-die.
        """
        if not self.freq_penalty:
            return lg
        L, V = lg.shape
        win = state[self._ring_index(L)]                       # (L, 2w+1) neighbour tokens
        w = self.window
        nbr = torch.cat([win[:, :w], win[:, w + 1:]], dim=1)   # drop centre -> (L, 2w)
        counts = torch.zeros(L, V, dtype=lg.dtype, device=lg.device)
        counts.scatter_add_(1, nbr, torch.ones_like(nbr, dtype=lg.dtype))
        return lg - self.freq_penalty * counts

    def _vacuum_mask(self, state: torch.Tensor) -> torch.Tensor:
        """Local 'no birth from vacuum': a site is forced dead iff none of its
        +/- w neighbours (excluding itself) is alive."""
        L = state.shape[0]
        nbr_live = state[self._ring_index(L)] != self.dead_token  # (L, 2w+1)
        nbr_live[:, self.window] = False                          # don't count self
        return nbr_live.sum(dim=1) == 0
