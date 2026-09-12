"""Check document coverage and exact design arithmetic, not an event engine."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path

feature = Path(__file__).resolve().parent
project = feature.parents[1]
spec = (feature / 'spec.md').read_text(encoding='utf-8')
adr = (project / 'docs/adr/0015-shared-backup-expert-game.md').read_text(encoding='utf-8')
ticket = (feature / 'issues/01-design-shared-backup-game.md').read_text(encoding='utf-8')

for number in range(1, 13):
    assert f'## {number}. ' in spec, number
for phrase in ('Status: Proposed', 'Law(Y_t | F_t^0)', 'epsilon_(N,Pi)',
               'POLICY FUNCTIONS', 'simultaneous one-sided95%',
               'conditional', 'Standard library first'):
    assert phrase in spec, phrase
assert 'Status: Proposed' in adr
assert 'Type: mathematical design' in ticket
for name in ('0004-common-state-failure-semantics',
             '0005-hedge-copy-lifecycle-and-random-keys',
             '0014-scaled-multi-expert-multi-backup'):
    assert (project / 'docs/adr' / (name + '.md')).is_file()
print('PASS: required design sections and local references')

def rate(active, n, c):
    return F(1) if not active else min(F(1), n*c/active)

def fixed_heads(requirements, n, c):
    """Analytic sharing fixture: no arrivals, failures, queues or policies."""
    remaining = {i: F(work) for i, work in enumerate(requirements)}
    completed = {}
    now, executed = F(0), F(0)
    while remaining:
        speed = rate(len(remaining), n, c)
        duration = min(remaining.values()) / speed
        executed += len(remaining) * speed * duration
        assert len(remaining) * speed <= n*c
        now += duration
        remaining = {i: work-speed*duration for i, work in remaining.items()}
        for i in list(remaining):
            if remaining[i] == 0:
                completed[i] = now
                del remaining[i]
    return completed, executed

assert fixed_heads([1], 2, F(1,2)) == ({0:F(1)}, F(1))
assert fixed_heads([1,1], 2, F(1,2)) == ({0:F(2),1:F(2)}, F(2))
assert fixed_heads([1,2], 2, F(1,2)) == ({0:F(2),1:F(3)}, F(3))
assert fixed_heads([1,1,1], 2, F(1,2)) == ({0:F(3),1:F(3),2:F(3)}, F(3))
print('PASS: cross-Expert interference, completion acceleration, dual sharing')

for n in (1,2,8,16,32,64):
    for c in (F(1,4), F(1,2), F(1), F(2)):
        for active in range(2*n+1):
            assert active*rate(active,n,c) == min(active,n*c)
            if c == 2:
                assert rate(active,n,c) == 1
assert rate(1,1,F(1,2)) == F(1,2)
print('PASS: exact capacity identity, zero-head case, uncoupled limit')

cap = F(45,16)
assert cap == F('2.8125')
assert cap//1 == 2 and cap//2 == 1
assert cap-1 < 2 and cap-2 == F('0.8125')
bank = [''.join(row) for row in product('NDSX', repeat=4)]
assert len(set(bank)) == 256 and 'NNNN' in bank and 'NSSN' in bank
assert F('0.45')/F('.5') == F('.9')
print('PASS: reservation arithmetic, 256 rules, nominal F base ratio')
print('PASS: design checks only; production simulation and equilibrium unverified')
