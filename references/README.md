# Reference paper

This project targets the original 2016 AlphaGo system, not AlphaGo Zero or
AlphaZero:

> David Silver, Aja Huang, Chris J. Maddison, et al. “Mastering the game of Go
> with deep neural networks and tree search.” *Nature* 529, 484–489 (2016).
> DOI: [10.1038/nature16961](https://doi.org/10.1038/nature16961).

Primary links:

- [Nature article and citation record](https://www.nature.com/articles/nature16961)
- [Official Google DeepMind-hosted PDF](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)
- [Google Research announcement by David Silver and Demis Hassabis](https://research.google/blog/alphago-mastering-the-ancient-game-of-go-with-machine-learning/)

## Obtain a verified local reading copy

The paper PDF says “All rights reserved,” and the Nature page directs reuse to
its reprints and permissions service. This repository therefore does **not**
redistribute or commit the PDF. The helper downloads a local copy directly from
Google DeepMind and rejects bytes that do not match the pinned digest:

```bash
python3 references/fetch_paper.py
```

To keep the repository clean, remove the local PDF after reading or place the
output outside the repository:

```bash
python3 references/fetch_paper.py /tmp/AlphaGoNaturePaper.pdf
```

Integrity metadata observed from the official object on 2026-07-16:

- SHA-256: `9c9184385a3d37b4f4e9d9715270986c43172747b1d08f29093128c1ef878b60`
- size: `2,682,222` bytes
- Google Cloud Storage generation: `1473689626099000`
- object last modified: `2016-09-12T14:13:46Z`

If Google legitimately replaces the object, the fetcher will fail closed until
the new file is manually checked against the DOI record and this metadata is
updated in review.

## BibTeX

```bibtex
@article{silver2016mastering,
  author  = {Silver, David and Huang, Aja and Maddison, Chris J. and Guez, Arthur
             and Sifre, Laurent and van den Driessche, George and Schrittwieser,
             Julian and Antonoglou, Ioannis and Panneershelvam, Veda and Lanctot,
             Marc and Dieleman, Sander and Grewe, Dominik and Nham, John and
             Kalchbrenner, Nal and Sutskever, Ilya and Lillicrap, Timothy and
             Leach, Madeleine and Kavukcuoglu, Koray and Graepel, Thore and
             Hassabis, Demis},
  title   = {Mastering the game of Go with deep neural networks and tree search},
  journal = {Nature},
  volume  = {529},
  pages   = {484--489},
  year    = {2016},
  doi     = {10.1038/nature16961}
}
```

## Paper used only for contrast

AlphaGo Zero is a later and materially different algorithm. Its primary record
is Silver et al., “Mastering the game of Go without human knowledge,” *Nature*
550, 354–359 (2017), [DOI 10.1038/nature24270](https://doi.org/10.1038/nature24270).
It is cited only to make the boundary explicit; it is not the implementation
target.
