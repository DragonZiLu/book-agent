"""
磁盘缓存：以 md5(text)[:16] 为 key，缓存章节提取结果
重试时未变章节直接读缓存，只重新提取 weak_chapters
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/extractions")


class ExtractionCache:
    """章节提取结果缓存"""

    def __init__(self, cache_dir: Path | str = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # 内存层缓存，避免重复读写磁盘
        self._memory_cache: dict[str, dict] = {}

    def get_key(self, text: str) -> str:
        """计算文本的缓存 key"""
        # 标准化空白字符，减少因格式差异导致 miss
        normalized = " ".join(text.split())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:16]

    def get(self, key: str) -> Optional[dict]:
        """获取缓存，先查内存再查磁盘"""
        if key in self._memory_cache:
            logger.debug(f"缓存命中（内存）: {key}")
            return self._memory_cache[key]

        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._memory_cache[key] = data
                logger.debug(f"缓存命中（磁盘）: {key}")
                return data
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"缓存文件损坏 {key}: {e}")

        return None

    def set(self, key: str, value: dict) -> None:
        """写入缓存"""
        self._memory_cache[key] = value
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.warning(f"缓存写入失败 {key}: {e}")

    def invalidate(self, key: str) -> None:
        """清除指定缓存"""
        self._memory_cache.pop(key, None)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()

    def clear(self) -> None:
        """清空全部缓存"""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
