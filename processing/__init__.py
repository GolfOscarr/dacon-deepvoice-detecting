"""The data-processing pipeline, implemented from docs/processing/03.

Independent of ``training/``: nothing here is imported by the training loop,
and nothing here modifies it. What is *shared* is the contracts -- the
manifest schema (``training.manifest``), the spec value types
(``training.spec``: ``SampleSpec``, ``ComponentDraw``, ``CELL_TABLE``, the RNG
key), the cell-mix arithmetic (``training.sampler``: ``CellMix``,
``composed_fractions``, ``check_mix``) and the transform registries -- so a
spec drawn here is auditable by ``training.audit`` unchanged and renderable by
either renderer. Integration into the training loop is a later, separate step.

Stages, in the order docs/processing/03 §2 runs them:

    processing.config   DrawConfig -- every knob of the draw, YAML-loadable
    processing.sampler  DRAW-1..DRAW-4: timeline, cell, the take/offset/tile
                        rule (D-3, D-4, D-5), DOSS on every row kind (D-16),
                        lead/tail silence (D-6)
    processing.render   REN-1..REN-4: sample-exact placement, a taper at every
                        joint (tile and sequential), augment, test chain
    processing.ship     SHIP-3..SHIP-6: the train/test-symmetric chain at the
                        model boundary -- channel policy, dc_offset, band limit
"""
