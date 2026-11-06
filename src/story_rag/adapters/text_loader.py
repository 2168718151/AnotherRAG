from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path


class TextDecodeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DecodedText:
    text: str
    encoding: str
    source_name: str
    title: str
    content_hash: str
    document_id: str


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return normalized.replace("\r\n", "\n").replace("\r", "\n")


def decode_text_bytes(data: bytes) -> tuple[str, str]:
    if not data:
        raise TextDecodeError("TXT content is empty")

    if data.startswith(b"\xef\xbb\xbf"):
        try:
            return normalize_text(data.decode("utf-8-sig")), "utf-8-sig"
        except UnicodeDecodeError as exc:
            raise TextDecodeError("Invalid UTF-8 BOM TXT") from exc

    try:
        return normalize_text(data.decode("utf-8")), "utf-8"
    except UnicodeDecodeError:
        pass

    try:
        return normalize_text(data.decode("gb18030")), "gb18030"
    except UnicodeDecodeError as exc:
        raise TextDecodeError("TXT must be UTF-8 BOM, UTF-8, or GB18030") from exc


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_id(source_name: str) -> str:
    canonical = str(Path(source_name)).replace("\\", "/").casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_decoded_text(
    data: bytes,
    *,
    source_name: str,
    title: str | None = None,
) -> DecodedText:
    normalized_source = source_name.strip() or "untitled.txt"
    text, encoding = decode_text_bytes(data)
    resolved_title = (title or Path(normalized_source).stem or "untitled").strip()
    digest = content_hash(text)
    return DecodedText(
        text=text,
        encoding=encoding,
        source_name=normalized_source,
        title=resolved_title,
        content_hash=digest,
        document_id=document_id(normalized_source),
    )


def read_text_file(path: Path) -> DecodedText:
    resolved = path.resolve()
    return build_decoded_text(
        resolved.read_bytes(),
        source_name=str(resolved),
        title=resolved.stem,
    )
