"""Deflate64 support for ``zipfile`` (ZIP compression method 9).

Large ZIP archives made with Windows Explorer / some archivers use Deflate64, which Python's ``zipfile`` cannot
read ("That compression method is not supported"). The ESA Anomaly Dataset archives on Zenodo are such files.
``enable_deflate64()`` registers a decompressor backed by the ``inflate64`` package (the decoder py7zr uses), so
``zipfile`` can list and extract those members - from a local file or through ``RemoteFile`` range reads.
"""
from __future__ import annotations

import zipfile

ZIP_DEFLATED64 = 9
_enabled = False


class _Deflate64Decompressor:
    """zipfile's non-zlib decompressor protocol: decompress(data) -> bytes and an ``eof`` flag."""

    def __init__(self):
        import inflate64
        self._d = inflate64.Inflater()

    def decompress(self, data: bytes) -> bytes:
        return self._d.inflate(data)

    @property
    def eof(self) -> bool:
        return bool(self._d.eof)


class _Deflate64Compressor:
    """Writing support (used by tests to build Deflate64 archives)."""

    def __init__(self):
        import inflate64
        self._c = inflate64.Deflater()

    def compress(self, data: bytes) -> bytes:
        return self._c.deflate(data)

    def flush(self) -> bytes:
        return self._c.flush()


def available() -> bool:
    try:
        import inflate64  # noqa: F401
        return True
    except ImportError:
        return False


def enable_deflate64() -> bool:
    """Teach zipfile to read (and, for tests, write) Deflate64 members. Safe to call repeatedly."""
    global _enabled
    if _enabled:
        return True
    if not available():
        return False
    orig_check = zipfile._check_compression
    orig_get_d = zipfile._get_decompressor
    orig_get_c = zipfile._get_compressor

    def check(compression):
        if compression == ZIP_DEFLATED64:
            return
        return orig_check(compression)

    def get_d(compress_type):
        if compress_type == ZIP_DEFLATED64:
            return _Deflate64Decompressor()
        return orig_get_d(compress_type)

    def get_c(compress_type, compresslevel=None):
        if compress_type == ZIP_DEFLATED64:
            return _Deflate64Compressor()
        return orig_get_c(compress_type, compresslevel)

    zipfile._check_compression = check
    zipfile._get_decompressor = get_d
    zipfile._get_compressor = get_c
    zipfile.compressor_names.setdefault(ZIP_DEFLATED64, "deflate64")
    _enabled = True
    return True


METHOD_NAMES = {0: "stored", 8: "deflate", 9: "deflate64", 12: "bzip2", 14: "lzma", 93: "zstandard", 99: "AES-encrypted"}


def check_members(zf: zipfile.ZipFile, names: list[str]) -> None:
    """Raise a readable error before extraction when a member uses a method this installation cannot read."""
    enable_deflate64()
    bad = {}
    for n in names:
        m = zf.getinfo(n).compress_type
        try:
            zipfile._check_compression(m)
            if m == ZIP_DEFLATED64:
                _Deflate64Decompressor()
        except (NotImplementedError, RuntimeError, ImportError):
            bad[n] = METHOD_NAMES.get(m, f"method {m}")
    if bad:
        kinds = sorted(set(bad.values()))
        raise ValueError(f"the archive uses {', '.join(kinds)} compression for {len(bad)} member(s), which this installation "
                         f"cannot decompress" + (" - install the 'inflate64' package (pip install inflate64) and rebuild"
                                                 if "deflate64" in kinds else ""))
