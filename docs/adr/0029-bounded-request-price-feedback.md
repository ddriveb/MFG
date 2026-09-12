# ADR-0029: Bounded requested-Reservation price feedback

Status: Accepted for a bounded diagnostic (confirmed by the user on 2026-09-08)

Date: 2026-09-08

## Decision

Repeat the ADR-0026 256-identity, four-round controller with one isolated
change: the public price charges one unit for every eligible pre-projection D/I
request, rather than charging incremental executed work.  Keep the same
refined bins, random thresholds, NIIN fallback, beta 4, eta 0.2, alpha 0.25,
CRN structure, physics, Reservation, and 5,120-call bound.

The controller still updates from `(requested_work-capacity)/capacity` and all
claims remain false.  Compare its price, requested/admitted/suppressed work and
physics path directly with ADR-0026.  Four rounds are diagnostic, not a
convergence allowance.

## References

- `docs/adr/0028-requested-reservation-price-diagnostic.md`
- `.scratch/token-mfg-restoration/issues/23-run-requested-price-feedback.md`
