# Apply the patch for sander

The patch `qmhub_at23.patch` only applies to AmberTools23.

```bash
tar xf AmberTools23.tar.bz2
cd amber22_src
curl -OL https://raw.githubusercontent.com/panxl/qmhub/master/patches/qmhub_at23.patch
patch -p1 < qmhub_at23.patch
```
