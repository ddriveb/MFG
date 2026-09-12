# ADR-0028: Requested Reservation price diagnostic

Status: Accepted for a bounded diagnostic (confirmed by the user on 2026-09-08)

Date: 2026-09-08

## Context

ADR-0026's round controller measures eligible D/I demand before Reservation
projection, but `incremental_executed_work` charges only work that survives the
projection.  The bounded run raised price while requested work also rose.

ADR-0021's `token_extended_reservation_v1` does not repair that mismatch: its
`Q_reserved` is the actual admitted charge and is zero for a denied request.
Relabeling it as a requested-demand price would violate the accepted model.

## Decision

Define a separate diagnostic cost identity:

    C_requested_reservation(i)
      = C_runtime_adr0009(i; p_exec=0)
        + p_req * Q_requested(i)

where `Q_requested=1` exactly when the Token is eligible for protection and
requests D or I before Reservation projection; otherwise it is zero.  A denied
eligible request pays the same request toll as an admitted request.  This is a
demand-side coordination toll, not executed-work billing, an admitted deposit,
or a market-clearing shadow price.

First run a zero-scheduler-call fixed grid on the immutable ADR-0027 panel bank:

- `beta in {1,2,4}`;
- `p_req in {0,0.5,1,2,4}`;
- unchanged supported bins, NIIN population, CRN, and panel weights;
- price zero must reproduce ADR-0027 exactly;
- all best-response, regret, Nash, and MFG claims remain false.

Because this is retrospective conditional re-scoring under a fixed NIIN
population, it may establish only that the request toll has the intended local
demand direction.  Any population feedback run must present the price online,
rerun the full physics, and use a separately frozen protocol.

## References

- `docs/adr/0021-token-private-cost-and-deviation-semantics.md`
- `docs/adr/0026-token-mfg-round-price-feedback.md`
- `docs/adr/0027-refined-token-observation-and-fixed-grid-pilot.md`
- `.scratch/token-mfg-restoration/issues/22-run-requested-reservation-price-grid.md`
