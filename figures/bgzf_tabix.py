# -*- coding: utf-8 -*-
"""Minimal remote BGZF + tabix reader (no pysam).
Usage: tb = RemoteTabix(url); rows = tb.fetch("12", 6_228_757, 6_442_114)
Handles flaky HTTPS with retries and a local block cache.
"""
from __future__ import annotations
import gzip, os, struct, time, zlib
import urllib.request

CACHE = None

def _get(url, start=None, end=None, tries=6, timeout=120):
    hdrs = {"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"}
    if start is not None:
        hdrs["Range"] = f"bytes={start}-{end}"
    last = None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(1.5 * (t + 1))
    raise RuntimeError(f"fetch failed {url} [{start}-{end}]: {last}")

def get_range(url, start, end, slice_size=262144, tries=8, timeout=120):
    """Fetch [start, end] in small slices; proxy drops large range requests."""
    if end < start:
        return b""
    out = []
    pos = start
    while pos <= end:
        sub_end = min(pos + slice_size - 1, end)
        for t in range(tries):
            try:
                hdrs = {"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity",
                        "Range": f"bytes={pos}-{sub_end}"}
                req = urllib.request.Request(url, headers=hdrs)
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    out.append(r.read())
                break
            except Exception as e:
                if t == tries - 1:
                    raise RuntimeError(f"slice {pos}-{sub_end} failed: {e}")
                time.sleep(0.8 * (t + 1))
        pos = sub_end + 1
    return b"".join(out)

def _gunzip_members(buf: bytes) -> bytes:
    """Decompress concatenated gzip/BGZF members, tolerating truncation."""
    out = []
    rest = buf
    while rest:
        d = zlib.decompressobj(31)
        try:
            out.append(d.decompress(rest))
        except zlib.error:
            break
        rest = d.unused_data
        if not d.eof and not rest:
            break
    return b"".join(out)

class RemoteTabix:
    def __init__(self, url: str, cache_dir: str | None = None):
        self.url = url
        self.tbi_url = url + ".tbi"
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            key = os.path.basename(url).replace(".gz", "")
            self.tbi_path = os.path.join(cache_dir, key + ".tbi")
        else:
            self.tbi_path = None
        self._load_index()

    def _load_index(self):
        raw = None
        if self.tbi_path and os.path.exists(self.tbi_path) and os.path.getsize(self.tbi_path) > 1000:
            raw = open(self.tbi_path, "rb").read()
        if not raw:
            raw = _get(self.tbi_url, 0, 5_000_000, tries=8)
            if self.tbi_path:
                open(self.tbi_path, "wb").write(raw)
        data = gzip.decompress(raw)
        assert data[:4] == b"TBI\x01", "not a tabix index"
        n_ref = struct.unpack("<i", data[4:8])[0]
        # tabix header: format, col_seq, col_beg, col_end, meta, skip (6 x int32) then l_nm
        self.header_meta = struct.unpack("<6i", data[8:32])
        l_nm = struct.unpack("<i", data[32:36])[0]
        names = data[36:36 + l_nm].split(b"\x00")
        self.col_seq, self.col_beg, self.col_end = self.header_meta[1:4]
        off = 36 + l_nm
        self.refs = {}
        for i in range(n_ref):
            n_bin = struct.unpack("<i", data[off:off + 4])[0]; off += 4
            bins = {}
            for _ in range(n_bin):
                b = struct.unpack("<I", data[off:off + 4])[0]; off += 4
                n_chunk = struct.unpack("<i", data[off:off + 4])[0]; off += 4
                ch = []
                for _ in range(n_chunk):
                    beg, end = struct.unpack("<QQ", data[off:off + 16]); off += 16
                    ch.append((beg, end))
                bins[b] = ch
            n_intv = struct.unpack("<i", data[off:off + 4])[0]; off += 4
            ioff = list(struct.unpack(f"<{n_intv}Q", data[off:off + 8 * n_intv])); off += 8 * n_intv
            self.refs[names[i].decode()] = (bins, ioff)

    @staticmethod
    def _reg2bins(beg, end):
        end -= 1
        bins = list(range(0, 1))
        for k in range(1, 6):
            bins += list(range(((1 << (3 * (k - 1))) - 1) // 7 + (beg >> (3 * (5 - k))),
                               ((1 << (3 * (k - 1))) - 1) // 7 + (end >> (3 * (5 - k))) + 1))
        return bins

    def fetch(self, chrom: str, beg: int, end: int, max_rounds=40, round_bytes=1_048_576):
        """Linear-index scan: start at the 16-kb linear offset, read forward, stop past `end`."""
        if chrom not in self.refs:
            return []
        bins, ioff = self.refs[chrom]
        idx = beg >> 14
        if idx >= len(ioff):
            idx = len(ioff) - 1
        start = ioff[idx] >> 16
        out, seen = [], set()
        pos_bytes = start
        fails = 0
        for _ in range(max_rounds):
            try:
                buf = get_range(self.url, pos_bytes, pos_bytes + round_bytes - 1,
                                slice_size=131072, tries=8)
                fails = 0
            except RuntimeError:
                fails += 1
                if fails >= 3:
                    break
                pos_bytes = max(start, pos_bytes - 262144)
                continue
            if not buf:
                break
            txt = _gunzip_members(buf).decode("utf-8", "ignore")
            done = False
            for line in txt.split("\n"):
                if not line or line.startswith("#"):
                    continue
                f = line.split("\t")
                if len(f) < 8:
                    continue
                if f[0] != chrom:
                    if out:
                        done = True
                        break
                    continue
                try:
                    pos = int(f[1])
                except ValueError:
                    continue
                if pos < beg - 1:
                    continue
                if pos > end:
                    done = True
                    break
                key = f"{f[0]}:{f[1]}"   # rsids are frequently missing in dense files
                if key in seen:
                    continue
                seen.add(key)
                out.append(f)
            if done:
                break
            # short reads on a flaky proxy are NOT end-of-file; advance only by bytes actually read
            step = len(buf) - 65536
            pos_bytes += step if step > 0 else round_bytes // 2
        return out

def header(url: str, nbytes=4000) -> str:
    buf = _get(url, 0, nbytes - 1, tries=6)
    txt = _gunzip_members(buf).decode("utf-8", "ignore")
    return "\n".join(txt.split("\n")[:3])
