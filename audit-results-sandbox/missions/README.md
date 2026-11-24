# Multi-mission telemetry - sandbox verification (2026-11-21/22)

Machine: 2 vCPU Intel Xeon @ 2.30 GHz, 7 GB RAM, no GPU. App run natively (uvicorn) with the pinned torch 2.6.0 /
torchvision 0.21.0 environment; outbound network blocked through a dead proxy; port 8081. Docker could not be used here.

* Clean install (port 8081): starter pack only (SMAP/MSL + complete OPSSAT-AD v1). Real data.
* Fixture instance (port 8083): Hubble and Webb loaded from SYNTHETIC files that follow
  each documented schema (backend/tests/missions_fixtures.py). Verifies code paths and UI only - NOT real data.

Scripts in this folder reproduce each check: opssat_flow.py (browser workflow), fixture_flow.py, kbd.py (keyboard only),
a11y.py (axe-core 4.14 WCAG 2.1 A/AA, light+dark; horizontal overflow at 375/768/1280 px), persist.py (restart).


Note (3.1.8): ESA Missions 1-3, FUNcube-1 and SatNOGS UWE-4 were removed from the app; older results above that mention them are historical.
