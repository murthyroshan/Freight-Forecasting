"""
Module B, part 1: port constraints and part-load capacity.

The question this answers is not "does this ship fit into this port".
It is "how much cargo can this ship carry into this port", which is a
different and much more useful question.

A laden Capesize draws about 18.2 m. Paradip's coal berth permits 16.0 m.
The naive conclusion is that Capesizes cannot call Paradip. In practice
they arrive PART-LADEN: you load to whatever draft the discharge port
allows and accept a lower payload. So the real trade-off facing a
chartering desk is

    load ~152,000 t into a Capesize and discharge alongside at Paradip
    load ~176,000 t and lighter at Sagar-Sandheads for Haldia
    split the parcel into two Panamaxes at a higher $/t but no penalty

and that is an optimisation, not a yes/no test.

HOW THE PART-LOAD CALCULATION WORKS

Over the operating range, a hull immerses almost linearly with weight.
The constant of proportionality is TPC - tonnes per centimetre - which
is the weight needed to change mean draft by one centimetre:

    TPC = waterplane_area * water_density / 100

Waterplane area is estimated as Lwl * B * Cw, with Cw the waterplane
coefficient. Bulk carriers are full-form; Cw is taken as 0.85 at loaded
draft, and Lwl as 0.97 * LOA. This is an ESTIMATE, not a hydrostatic
table: a real fixture would use the ship's own deadweight scale. The
TPC values it produces (Capesize ~110, Panamax ~61 t/cm) sit inside the
published ranges for those classes, which is the accuracy this decision
needs.

WATER DENSITY

A ship floats deeper in less dense water, so a river berth costs draft.
Haldia sits in the Hooghly, which is brackish. The standard correction
is the Dock Water Allowance:

    FWA_mm = displacement / (4 * TPC)          (fresh water, 1000 kg/m3)
    DWA    = FWA * (1025 - rho_dock) / 25

which is applied wherever a port declares a density below seawater.

SOURCES

Port figures are the declared maxima for the COAL-handling berth, which
is not always the port's deepest berth. Under-keel clearance is already
embedded in a declared permissible draft, so it is not deducted again.
Vessel figures are representative of each class rather than any one
ship - individual vessels vary by a few percent.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src import paths  # noqa: E402,F401

SEAWATER = 1.025          # t/m3
CW = 0.85                 # waterplane coefficient, full-form bulk carrier
LWL_RATIO = 0.97          # waterline length as a fraction of LOA

# Below this share of cargo capacity a fixture stops making commercial
# sense: freight is quoted per tonne but the hire is paid on the ship, so
# a part-laden voyage carries nearly the full cost of a laden one. Used
# only to LABEL an option, never to hide it - the tonnage is still shown.
MIN_UTILISATION = 0.60


# ---------------------------------------------------------------- vessels
# dwt      summer deadweight, tonnes
# draft    summer salt-water draft, metres
# loa/beam metres
# lightship hull steel etc, tonnes - not cargo
# constants bunkers, fresh water and stores for a laden ocean passage;
#           deadweight minus constants is the most cargo that can ever
#           be loaded, before any draft restriction applies.
VESSELS = {
    'Handysize':    dict(dwt=35000,  draft=10.2, loa=180, beam=28.0,
                         lightship=8000,  constants=1200),
    'Supramax':     dict(dwt=58000,  draft=12.8, loa=190, beam=32.3,
                         lightship=11000, constants=1800),
    'Panamax':      dict(dwt=82000,  draft=14.4, loa=229, beam=32.3,
                         lightship=14000, constants=2500),
    'Post-Panamax': dict(dwt=98000,  draft=15.4, loa=250, beam=38.0,
                         lightship=16500, constants=2800),
    'Capesize':     dict(dwt=180000, draft=18.2, loa=289, beam=45.0,
                         lightship=24000, constants=3500),
    'Newcastlemax': dict(dwt=208000, draft=18.5, loa=300, beam=50.0,
                         lightship=27000, constants=3800),
}

# ------------------------------------------------------------------ ports
# max_draft  declared permissible draft at the coal berth, metres
# max_loa    metres; None where not a binding published limit
# max_beam   metres; None where not published
# density    t/m3 at the berth - below 1.025 costs draft
# lighterage True where the location is an anchorage served by floating
#            cranes rather than a quay a ship can lie alongside
PORTS = {
    'Gangavaram': dict(
        max_draft=21.0, max_loa=None, max_beam=None, density=1.025,
        lighterage=False, geometry_verified=False,
        note='Deepest draft on the east coast. Not present in IMF '
             'PortWatch, so it carries no congestion signal here.'),
    'Visakhapatnam': dict(
        max_draft=18.1, max_loa=None, max_beam=None, density=1.025,
        lighterage=False, geometry_verified=False,
        note='Outer harbour general cargo berth, 356 m quay, built for '
             'imported coking coal. Takes vessels to about 200,000 DWT.'),
    'Dhamra': dict(
        max_draft=18.0, max_loa=None, max_beam=None, density=1.025,
        lighterage=False, geometry_verified=False,
        note='Has demonstrated 18.40 m on a 187,000 t coking coal '
             'parcel, so 18.0 m is conservative.'),
    'Paradip': dict(
        max_draft=16.0, max_loa=300.0, max_beam=46.0, density=1.025,
        lighterage=False, geometry_verified=True,
        note='Coal berth 16.0 m against a 16.5 m port envelope. LOA and '
             'beam bind here before draft does for the largest classes.'),
    'Gopalpur': dict(
        max_draft=14.5, max_loa=None, max_beam=None, density=1.025,
        lighterage=False, geometry_verified=False,
        note='Shallowest of the direct-berth options.'),
    'Haldia': dict(
        max_draft=8.5, max_loa=None, max_beam=None, density=1.010,
        lighterage=False, geometry_verified=False,
        note='Riverine, on the Hooghly. Silts to 7.0-7.5 m between '
             'dredging campaigns, and brackish water costs further '
             'draft. The river approach also imposes length and channel '
             'limits that are NOT captured here, so large classes will '
             'show a draft-limited tonnage they could not physically '
             'reach. Treat Haldia volumes as arriving by barge from '
             'Sagar-Sandheads unless a verified LOA limit is added.'),
    'Sagar-Sandheads': dict(
        max_draft=None, max_loa=None, max_beam=None, density=1.025,
        lighterage=True, geometry_verified=True,
        note='An ANCHORAGE in 40-50 m of water, not a berth. Capesizes '
             'anchor and floating cranes discharge into barges for '
             'Haldia. Modelling it as a deep-draft port inverts the '
             'answer for the entire Haldia trade.'),
}


# congestion.py and risk.py key ports in lowercase; this module uses the
# display spelling, because these names are rendered directly. The MILP
# will have to join the two, and a join that guesses at case fails
# silently on the one port whose spelling differs. So state the mapping
# once, here, and let tests/test_ports.py assert it stays exhaustive.
#
# Gangavaram and Sagar-Sandheads are berth-capability entries with no
# PortWatch feed. They map to None rather than being left out, so a
# lookup returns "no activity data" instead of raising a KeyError that
# reads like a typo.
PORTWATCH_KEY = {
    'Paradip':         'paradip',
    'Visakhapatnam':   'visakhapatnam',
    'Dhamra':          'dhamra',
    'Haldia':          'haldia',
    'Gopalpur':        'gopalpur',
    'Gangavaram':      None,
    'Sagar-Sandheads': None,
}


def portwatch_key(port):
    """The congestion/risk key for a port named as this module names it.

    Returns None for a berth we model physically but hold no arrivals
    for. Raises for a port this module does not know at all.
    """
    _port(port)
    return PORTWATCH_KEY[port]


def _vessel(name):
    if name not in VESSELS:
        raise KeyError('unknown vessel class %r; known: %s'
                       % (name, ', '.join(VESSELS)))
    return VESSELS[name]


def _port(name):
    if name not in PORTS:
        raise KeyError('unknown port %r; known: %s'
                       % (name, ', '.join(PORTS)))
    return PORTS[name]


def _positive_density(density):
    # `density <= 0` is False for NaN, so a NaN slips straight through a
    # bare sign test and comes out the far end as a NaN TPC, a NaN
    # allowance and finally a NaN tonnage on screen. Test for finite
    # first.
    if not (isinstance(density, (int, float))
            and math.isfinite(density) and density > 0):
        raise ValueError('water density must be a positive finite number, '
                         'got %r' % (density,))
    return float(density)


def tpc(vessel, density=SEAWATER):
    """Tonnes per centimetre immersion."""
    density = _positive_density(density)
    v = _vessel(vessel)
    waterplane = (v['loa'] * LWL_RATIO) * v['beam'] * CW
    return waterplane * density / 100.0


def dock_water_allowance(vessel, density):
    """Extra draft, in metres, from floating in water lighter than sea
    water. Zero at or above 1.025."""
    density = _positive_density(density)
    if density >= SEAWATER:
        return 0.0
    v = _vessel(vessel)
    displacement = v['lightship'] + v['dwt']
    fwa_mm = displacement / (4.0 * tpc(vessel, SEAWATER))
    dwa_mm = fwa_mm * (1025.0 - density * 1000.0) / 25.0
    return dwa_mm / 1000.0


def max_cargo(vessel, port):
    """Largest cargo this class can load for this destination, in tonnes.

    Returns (tonnes, binding_constraint). Zero means the vessel cannot be
    loaded at all for that berth.
    """
    v = _vessel(vessel)
    p = _port(port)

    # Hard geometry first: no amount of part-loading makes a ship shorter
    # or narrower.
    if p['max_loa'] is not None and v['loa'] > p['max_loa']:
        return 0.0, 'LOA %.0f m exceeds limit %.0f m' % (v['loa'],
                                                         p['max_loa'])
    if p['max_beam'] is not None and v['beam'] > p['max_beam']:
        return 0.0, 'beam %.1f m exceeds limit %.1f m' % (v['beam'],
                                                          p['max_beam'])

    # Cargo capacity before any draft restriction.
    full_cargo = v['dwt'] - v['constants']

    # A lighterage anchorage imposes no draft limit - the ship never
    # comes alongside.
    if p['lighterage'] or p['max_draft'] is None:
        return float(full_cargo), 'deadweight (no draft limit at anchorage)'

    # Draft the port allows this ship, after the density penalty.
    allowed = p['max_draft'] - dock_water_allowance(vessel, p['density'])

    if allowed >= v['draft']:
        return float(full_cargo), 'deadweight (draft not binding)'

    shortfall_cm = (v['draft'] - allowed) * 100.0
    reduction = shortfall_cm * tpc(vessel, p['density'])
    cargo = full_cargo - reduction
    if cargo <= 0:
        return 0.0, 'draft %.2f m allows no cargo' % allowed
    return float(cargo), 'draft %.2f m allowed vs %.2f m laden' % (
        allowed, v['draft'])


def can_serve(vessel, port):
    """Full assessment of one vessel class against one destination."""
    v = _vessel(vessel)
    p = _port(port)
    cargo, binding = max_cargo(vessel, port)
    # Bug guard: a capacity must never round UP. Reporting 41,740 t when
    # the ship can lift 41,739.64 t is an overload instruction, however
    # small. Floor once here so tonnage, utilisation and voyage counts
    # are all derived from the same conservative figure.
    cargo = math.floor(cargo)
    full = v['dwt'] - v['constants']
    util = cargo / full if full else 0.0

    if p['lighterage']:
        verdict = 'lighterage'
    elif cargo <= 0:
        verdict = 'cannot serve'
    elif util < MIN_UTILISATION:
        # Physically loadable, commercially absurd. Freight is quoted per
        # tonne but paid on the ship, so sailing a Capesize 39% full
        # costs roughly the same as sailing it full. This is an ECONOMIC
        # judgement, flagged separately from the physical result rather
        # than folded into it.
        verdict = 'uneconomic'
    else:
        verdict = 'alongside'

    caveats = []
    if not p['geometry_verified'] and verdict in ('alongside', 'uneconomic'):
        caveats.append(
            'draft-only assessment: LOA, beam and channel or lock limits '
            'for this berth are not verified, so the tonnage is an UPPER '
            'BOUND and the largest classes may be excluded outright')

    return {
        'caveats': caveats,
        'vessel': vessel,
        'port': port,
        'verdict': verdict,
        'max_cargo_t': round(cargo),
        'full_cargo_t': full,
        'utilisation': round(util, 4),
        'foregone_t': round(full - cargo) if cargo > 0 else full,
        'binding': binding,
        'laden_draft_m': v['draft'],
        'port_draft_m': p['max_draft'],
        'density_penalty_m': round(dock_water_allowance(
            vessel, p['density']), 3),
        'tpc': round(tpc(vessel, p['density']), 1),
        'note': p['note'],
    }


# Preference order when two options need the same number of voyages.
# Berthing alongside beats lightering, which beats a fixture we have
# already labelled uneconomic.
_VERDICT_RANK = {'alongside': 0, 'lighterage': 1, 'uneconomic': 2}


def options_for(parcel_t, ports=None, vessels=None,
                include_uneconomic=True):
    """Every workable way to move `parcel_t` tonnes, best first.

    Ranked by voyages, then by verdict, then by PARCEL utilisation -
    how much of the chartered capacity this cargo actually fills.

    Ranking on the vessel's own utilisation is the obvious mistake and
    it is wrong: a Newcastlemax and a Capesize both load to 100% of
    their own capacity, so both look perfect, but moving 160,000 t in a
    Newcastlemax fills only 78% of the ship you are paying for against
    91% for the Capesize. Freight is paid on the ship, so the larger
    vessel is the worse fixture even though it "fits".
    """
    if not isinstance(parcel_t, (int, float)) or parcel_t != parcel_t:
        raise ValueError('parcel_t must be a number, got %r' % (parcel_t,))
    if parcel_t <= 0:
        raise ValueError('parcel_t must be positive, got %r' % (parcel_t,))

    out = []
    for vs in (vessels or VESSELS):
        for pt in (ports or PORTS):
            r = dict(can_serve(vs, pt))
            if r['max_cargo_t'] <= 0:
                continue
            if r['verdict'] == 'uneconomic' and not include_uneconomic:
                continue
            voyages = math.ceil(parcel_t / r['max_cargo_t'])
            r['voyages'] = voyages
            r['parcel_t'] = parcel_t
            # Share of the capacity actually chartered that the cargo
            # fills. Never above 1, and lower means more wasted hire.
            r['parcel_utilisation'] = round(
                parcel_t / (voyages * r['max_cargo_t']), 4)
            out.append(r)

    out.sort(key=lambda r: (r['voyages'],
                            _VERDICT_RANK.get(r['verdict'], 9),
                            -r['parcel_utilisation']))
    return out


if __name__ == '__main__':
    print('=' * 78)
    print('  PORT CONSTRAINTS AND PART-LOAD CAPACITY')
    print('=' * 78)

    print('\n  TPC and laden draft by class')
    print('  %-14s %7s %7s %7s %9s %10s' %
          ('class', 'DWT', 'draft', 'beam', 'TPC t/cm', 'cargo cap'))
    print('  ' + '-' * 62)
    for name, v in VESSELS.items():
        print('  %-14s %7d %6.1fm %6.1fm %9.1f %10d'
              % (name, v['dwt'], v['draft'], v['beam'], tpc(name),
                 v['dwt'] - v['constants']))

    print('\n  Max cargo (tonnes) by class and destination')
    ports = list(PORTS)
    print('  %-14s' % 'class' + ''.join('%12s' % p[:11] for p in ports))
    print('  ' + '-' * (14 + 12 * len(ports)))
    for vs in VESSELS:
        row = '  %-14s' % vs
        for pt in ports:
            c, _ = max_cargo(vs, pt)
            row += '%12s' % ('-' if c <= 0 else '{:,}'.format(round(c)))
        print(row)

    print('\n  The trade-off this exists to expose:')
    for vs, pt in [('Capesize', 'Paradip'), ('Capesize', 'Visakhapatnam'),
                   ('Capesize', 'Sagar-Sandheads'), ('Capesize', 'Haldia'),
                   ('Newcastlemax', 'Paradip'), ('Panamax', 'Paradip')]:
        r = can_serve(vs, pt)
        print('\n    %s -> %s' % (vs, pt))
        print('      %-13s %s' % ('verdict', r['verdict']))
        if r['max_cargo_t']:
            print('      %-13s %s t of %s t (%.0f%% utilisation)'
                  % ('cargo', '{:,}'.format(r['max_cargo_t']),
                     '{:,}'.format(r['full_cargo_t']),
                     r['utilisation'] * 100))
            if r['foregone_t']:
                print('      %-13s %s t left on the table'
                      % ('forgone', '{:,}'.format(r['foregone_t'])))
        print('      %-13s %s' % ('binding', r['binding']))
        if r['density_penalty_m']:
            print('      %-13s %.2f m lost to brackish water'
                  % ('density', r['density_penalty_m']))

    print('\n' + '=' * 78)
    parcel = 160000
    print('  Ways to move a %s t parcel' % '{:,}'.format(parcel))
    print('  %-13s %-16s %-11s %4s %9s %8s' %
          ('class', 'port', 'verdict', 'voy', 'cargo/voy', 'fill'))
    print('  ' + '-' * 68)
    for r in options_for(parcel)[:10]:
        print('  %-13s %-16s %-11s %4d %9s %7.0f%%'
              % (r['vessel'], r['port'], r['verdict'], r['voyages'],
                 '{:,}'.format(r['max_cargo_t']),
                 r['parcel_utilisation'] * 100))
    print('\n  "fill" is how much of the capacity you charter this parcel')
    print('  actually uses - the number that decides which ship to take.')
    print('=' * 78)
