# scene-dedupe

Finds photos of the same scene in a folder, including the shots you took one
step to the left. Writes the groups back into the files as XMP tags, so
digiKam, Lightroom, Immich and friends pick them up.

```
230 photos, 56 scenes, 126 photos grouped
threshold 0.75, gap 180s

Scene_20200523_134509 (4 photos, min 0.934, span 11s)
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

## Tuning

`--threshold` is the dial worth touching. Raise it from the default and work
upwards rather than down.

- `0.70` catches loose matches and starts pulling in unrelated frames
- `0.75` default, a reasonable starting point for casual photos
- `0.85` tight, only near identical framing

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
| `--threshold` | `0.75` | cosine similarity needed to join a group |
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
