# Survey — Prior Art Index

Reference material for competition 236749. Compiled 2026-09-04. Built for lookup, not reading.

| File | Use it when you need… |
|---|---|
| [01-sota-speech.md](01-sota-speech.md) | Voice-fake SOTA methods, numbers, what generalizes |
| [02-sota-music.md](02-sota-music.md) | Music-fake SOTA, and why 16 kHz changes everything |
| [03-sota-singing-mixed.md](03-sota-singing-mixed.md) | Sung vocals, mixed audio, **should we separate?** |
| [04-sota-presence.md](04-sota-presence.md) | Voice/music presence heads |
| [05-models.md](05-models.md) | Pretrained checkpoint catalog + licenses + sizes |
| [06-datasets.md](06-datasets.md) | Dataset catalog + **redistributability verdict** |
| [07-generators.md](07-generators.md) | TTS / SVS / VC / TTM models for building fake data |
| [08-augmentation.md](08-augmentation.md) | Augmentation + channel simulation recipes and tools |
| [09-challenge-playbooks.md](09-challenge-playbooks.md) | Winning recipes from AT-ADD / ASVspoof 5 / SVDD |
| [10-open-questions.md](10-open-questions.md) | Unresolved gaps; what to verify before relying on it |

**Confidence marks used throughout**: ★ verified from the primary paper/abstract · ☆ from a
secondary summary, re-read before relying on it · ⚠️ risk · 🔴 decision-changing.

---

## The 10 findings that should drive our design

1. 🔴 **16 kHz standardization destroys the shortcut every published AI-music detector uses.**
   Detectors ride bitrate/sample-rate/upsampling artifacts above 8 kHz; our test set has none.
   → off-the-shelf music detectors won't transfer; our training data must go through an
   identical 16 kHz chain. [02](02-sota-music.md)
2. 🔴 **The music head is the highest-EV target.** Weight 0.27 (vs voice 0.18), least-solved
   literature, and accompaniment is *easier* to detect than vocals (97–98% vs 65–80% TPR).
   [02](02-sota-music.md), [03](03-sota-singing-mixed.md)
3. 🔴 **Naive separate-then-detect fails.** Separation artifacts spread across all stems:
   38% FPR (vocals), 94.7% FPR (accompaniment). Use separation for *features* (per-band SNR),
   not as preprocessing. [03](03-sota-singing-mixed.md)
4. **Type-aware routing beat unified detection** in AT-ADD Track 2 (96.10% macro-F1) →
   separate specialist frontends for the voice-fake and music-fake heads.
   [09](09-challenge-playbooks.md)
5. **SSL frontend choice is the single biggest lever**; backbone architecture is the smallest.
   [01](01-sota-speech.md)
6. **Cross-generator generalization is the binding constraint everywhere.** MERT-AASIST hits
   46.4% EER cross-generator; music detectors drop to 3–12% on a third platform.
   → validation must be generator-disjoint. [02](02-sota-music.md)
7. **Augmentation breadth is the second biggest lever**; RawBoost is the strongest single
   family; codec + band-limit augmentation is mandatory given telephone-channel test samples.
   [08](08-augmentation.md)
8. **Multi-crop inference (4–5 crops, median pooling) + segment-level supervision** is the
   consensus inference recipe; whole-file mean pooling dilutes short fake components.
   [01](01-sota-speech.md), [09](09-challenge-playbooks.md)
9. ⚠️ **License gate kills most of this field's datasets.** SONICS, SingFake, MusicCaps,
   AudioCaps, MSD are YouTube-linked → cannot be handed to DACON → cannot be used at all.
   [06](06-datasets.md)
10. **Modern TTS is codec-LM based, not vocoder based.** Detectors trained only on vocoder
    artifacts are blind to it (Codecfake: 41.4% relative EER reduction when codec-trained) —
    but don't *rely* on codec artifacts either. Breadth, not a single artifact family.
    [01](01-sota-speech.md)

## Realistic performance ceilings (for sanity-checking our own numbers)

| Task | Best published | Source |
|---|---|---|
| Speech, unseen generators (ASVspoof 5) | ~4.0–5.0% EER | [01](01-sota-speech.md) |
| Speech, cross-domain (In-the-Wild) | 5.85% EER | [01](01-sota-speech.md) |
| Speech, ASVspoof21 LA / DF | 0.97% / 1.74% EER | [01](01-sota-speech.md) |
| Sung vocals, controlled (CtrSVDD) | 1.65% EER | [03](03-sota-singing-mixed.md) |
| Music, in-distribution | 2.89–4.51% EER | [02](02-sota-music.md) |
| Music, cross-generator | **46.4% EER** ← the real number | [02](02-sota-music.md) |

If our local CV shows <1% EER on the music head, we have a leak or a shortcut, not a model.
