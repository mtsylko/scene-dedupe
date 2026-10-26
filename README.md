# scene-dedupe

Finds photos of the same scene in a folder, including the shots you took one
step to the left. Writes the groups back into the files as XMP tags, so
digiKam, Lightroom, Immich and friends pick them up.

```
230 photos, 45 scenes, 99 photos grouped
threshold 0.85, gap 180s

Scene_20200523_134509 (4 photos, min 0.879, span 11s)
  IMG_20200523_134509.jpg
  IMG_20200523_134516.jpg
  IMG_20200523_134518.jpg
  IMG_20200523_134520.jpg
```

## Why not a perceptual hash

Most duplicate finders compare perceptual hashes. digiKam uses Haar wavelets,
Czkawka uses gradient hashes, and both work the same way underneath. They
shrink the picture, turn it into a short signature and compare signatures.

That is great for exact copies, re-saves and resizes. It falls apart the
moment the camera moves. Take two shots of the same flower bed from half a
step apart and the signatures diverge, even though you can see at a glance
that it is the same thing.

scene-dedupe uses DINOv2 embeddings instead. The vector describes what is in
the frame, not how the pixels are arranged, so a small change of viewpoint
barely moves it. Exact copies still group, because for the model they are the
limiting case of a similar scene.

## Install

```
pipx install git+https://github.com/mtsylko/scene-dedupe
```

This puts a `scene-dedupe` command on your PATH in an environment of its
own, so torch does not land in your system Python. Plain `pip install
git+https://github.com/mtsylko/scene-dedupe` works the same inside a
virtualenv you made yourself.

Torch ships wheels for Python 3.10 to 3.14 and the package declares that
range, so pipx picks a matching interpreter on its own even when its default
is newer. If you have nothing in that range it says so, and you can point it
at one yourself:

```
pipx install --python python3.12 git+https://github.com/mtsylko/scene-dedupe
```

To work on the code instead:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tagging needs [ExifTool](https://exiftool.org). On macOS `brew install
exiftool`, on Debian or Ubuntu `apt install libimage-exiftool-perl`.

Expect about 1.2 GB on disk. Torch and transformers are most of it, and the
model weights, around 350 MB, download on first run into
`~/.cache/huggingface`.

## Tests

The parts that go wrong quietly are covered: the clustering rule, the scene
names and the arithmetic that decides which tags to write. The tests read no
photos and call no exiftool, so they finish in well under a second.

```
python -m unittest test_group_scenes
```

## Use

Look at the report first.

```
scene-dedupe ~/Pictures/some-album
```

Happy with the groups? Write them into the files.

```
scene-dedupe ~/Pictures/some-album --write
```

Then let your photo manager re-read the files. In digiKam that is Tools →
Maintenance → Sync Metadata and Database with the direction set to *From
image metadata to database*. In Lightroom it is Metadata → Read Metadata from
File. Immich picks tags up when it indexes.

A tree called `Scenes` appears, with one child per scene. A scene is named
after the capture time of its earliest photo, `Scene_20200523_134509`. The
name says nothing about position in the album, so a scene keeps it when you
add photos elsewhere and run again.

## Running it again

A second `--write` over the same folder changes nothing. The tool reads the
tags that are already in the files and touches only what differs, so tags do
not pile up on a rerun.

If the grouping really did change, because you added photos or moved the
threshold, the files are brought in line with the new report. A scene tag of
ours that no longer applies is removed, and a photo that is no longer part of
any scene loses its scene tag. Anything that is not ours is left alone,
including your own keywords and other branches of your tag tree.

Ours means a value under `--parent` whose name looks like a scene name, plus
the plain `Scene_01` form that versions before stable names used to write.
Those old tags are replaced on the next `--write`.

## How grouping works

Photos are sorted by capture time. A photo joins the current group only if
both of these hold.

- It was taken within `--gap` seconds of the previous shot.
- Its cosine similarity is at least `--threshold` against **every** photo
  already in the group.

The second rule matters more than it looks. Checking against one member is
enough to build chains, where A matches B, B matches C, and a group quietly
swallows half the album without any two ends having anything in common. On
uniform material like foliage this happens almost immediately. Requiring
every pair to clear the bar makes chains impossible.

The time window is the second guard. Series are shot back to back, so a ten
minute break is a strong signal that the scene changed, whatever the vectors
say. It also speeds things up, since most pairs never get compared.

## How good is it

`eval/labels.json` holds every scene in a 230 photo album marked by hand from
the pictures. Where the tool disagreed with a label, the photos were looked
at again side by side at full size and the label fixed if it was wrong.
`eval/evaluate.py` then scores the grouping pair by pair.

```
python eval/evaluate.py ~/Pictures/some-album
```

| threshold | precision | recall | F1 |
| --- | --- | --- | --- |
| 0.75 | 0.739 | 0.903 | 0.812 |
| 0.80 | 0.816 | 0.861 | 0.838 |
| **0.85** | **0.909** | **0.833** | **0.870** |
| 0.90 | 0.976 | 0.556 | 0.708 |

Precision is how many of the pairs the tool grouped really belong together,
recall is how many of the pairs that belong together it found. They are not
worth the same. A wrong merge writes a tag somebody has to find and undo, a
missed one only leaves a photo untagged, so the default sits where precision
is high.

The harness earns its keep. It found that photos were fed to the model in
their stored orientation, so a portrait and a landscape shot of the same
scene scored far apart. Honouring the EXIF rotation lifted F1 from 0.824 to
0.870, and no amount of staring at the report would have shown that.

One album of 230 photos from one garden on one day is a thin basis. Read the
numbers as a floor for this kind of material, not as a promise.

## Tuning

`--threshold` is the dial worth touching. The default of `0.85` comes from
the labelled set below, where it scores best. Numbers from that run:

- `0.70` grabs almost anything, precision falls to 0.55
- `0.75` still pulls in unrelated frames, precision 0.74
- `0.85` default, precision 0.91 and recall 0.83
- `0.90` precision 0.98 but recall drops to 0.56, half the series are split

Lower it if your series are shot from further apart, raise it if wrong tags
bother you more than missing ones.

The report prints the lowest pair similarity in each group. If that number
sits well above your threshold, the group is solid. The `span` figure tells
you how far apart in time the members were.

Use `--cache groups.npz` while you are experimenting. Embeddings are reused
between runs, so only the clustering is redone and a second pass takes
seconds.

```
scene-dedupe ~/Pictures/album --cache album.npz --threshold 0.80
```

## Options

| Flag | Default | What it does |
| --- | --- | --- |
| `--threshold` | `0.85` | cosine similarity needed to join a group |
| `--gap` | `180` | seconds allowed between consecutive shots |
| `--device` | `auto` | `cpu`, `cuda` or `mps` |
| `--cache` | none | npz file to reuse embeddings |
| `--flat` | off | skip subfolders |
| `--parent` | `Scenes` | parent tag name |
| `--write` | off | write tags with exiftool |

## Limits

**Back up before `--write`.** Tags go into the originals and
`-overwrite_original` is used, so there is no automatic backup copy. A rerun
can also remove scene tags of ours that no longer match the grouping.

**Copies in different folders are missed.** The time window splits anything
further apart than `--gap`, so the same file imported twice years apart will
not group. Run your photo manager's own duplicate search for that, it is good
at exactly this case.

**Uniform scenes are harder.** Forests, beaches and snow give the model less
to work with. Start at a higher threshold there.

**Portrait mode and macro.** Synthetic background blur does not confuse the
grouping, but it does make these shots look unusual to any sharpness metric
you might bolt on later.

**Writing invalidates the embedding cache.** Tagging changes the file size,
and the size is part of the cache key, so the next run re-embeds whatever was
tagged. Tune with `--cache` first and write once at the end.

**Large folders are fine.** Similarity is worked out pair by pair as the
clustering asks for it, so nothing grows with the square of the album. What
you pay for is the embeddings, about 3 KB per photo held in memory and
roughly 60 ms per photo on a laptop GPU. The exception is a very large
`--gap`, which lets groups grow long, and a long group means many pairs to
check against.

Removing a tag later is one command.

```
exiftool -overwrite_original \
  -XMP-lr:HierarchicalSubject-="Scenes|Scene_20200523_134509" \
  -XMP:Subject-=Scene_20200523_134509 *.jpg
```

There is no wildcard form, so clearing the whole tree means deleting both
tags outright, which takes your own keywords with them.

## License

MIT, see [LICENSE](LICENSE).

DINOv2 weights are released by Meta under Apache 2.0 and can be used
commercially. ExifTool is called as an external process and is not bundled.
